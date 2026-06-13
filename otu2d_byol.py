import os
import json
import time
import datetime
import math
import random
from typing import List, Tuple

import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models as tv_models
from data import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cv2.setRNGSeed(seed)
def worker_init_fn(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
torch.use_deterministic_algorithms(True, warn_only=True)
g = torch.Generator()
g.manual_seed(seed)

X_train, Y_train, X_valid, Y_valid, X_test,  Y_test = [], [], [], [], [], []
  

with open('/mnt/nvme0/home/utbt/KhoaVM/OTU-2D-Dataset/OTU_2D_850-150-469.json', 'r') as f:
    data = json.load(f)


    for item in data:
        for i in range(len(data[item])):

            if item == 'train':
                X_train.append(OTU_PATH + str(data[item][i]['image']))
                Y_train.append(OTU_PATH + str(data[item][i]['mask']))
            elif item == 'val':
                X_valid.append(OTU_PATH + str(data[item][i]['image']))
                Y_valid.append(OTU_PATH + str(data[item][i]['mask']))
            elif item == 'test':
                X_test.append(OTU_PATH + str(data[item][i]['image']))
                Y_test.append(OTU_PATH + str(data[item][i]['mask']))

IMG_HEIGHT = 256
IMG_WIDTH  = 256
IMG_CHANNELS = 3

# Pre-load all images / masks into NumPy arrays (HWC, float32 in [0, 1]).
def load_image_mask(image_path, mask_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    mask  = cv2.resize(mask,  (IMG_HEIGHT, IMG_WIDTH))
    image = image.astype(np.float32) / 255.0
    mask  = mask.astype(np.float32)  / 255.0
    mask[mask > 0.] = 1.
    mask  = np.expand_dims(mask, axis=-1)
    return image, mask

def preload(image_paths, mask_paths):
    X = np.zeros((len(image_paths), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    Y = np.zeros((len(mask_paths),  IMG_HEIGHT, IMG_WIDTH, 1),            dtype=np.float32)
    for i, (ip, mp) in enumerate(zip(image_paths, mask_paths)):
        X[i], Y[i] = load_image_mask(ip, mp)
    return X, Y

X_train_data, Y_train_data = preload(X_train, Y_train)
X_valid_data, Y_valid_data = preload(X_valid, Y_valid)
X_test_data,  Y_test_data  = preload(X_test,  Y_test)

class ConvBlock(nn.Module):
    """Two (Conv -> BN -> ReLU) layers, same padding."""
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, k, padding=p), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, k, padding=p), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class VGG16Encoder(nn.Module):
    """VGG16 (ImageNet) feature extractor returning 4 skip connections + bottleneck.

    Skip channels = (64, 128, 256, 512), bottleneck = (512, H/16, W/16).
    Replicates Keras' block1_conv2 / block2_conv2 / block3_conv3 / block4_conv3 / block5_conv3.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        feats = vgg.features
        # VGG16 features split (indices match torchvision impl):
        #   block1: 0..3   (conv,relu,conv,relu)  -> idx 3 is block1_conv2 ReLU
        #   pool   : 4
        #   block2: 5..8   -> idx 8  = block2_conv2 ReLU
        #   pool   : 9
        #   block3: 10..15 -> idx 15 = block3_conv3 ReLU
        #   pool   : 16
        #   block4: 17..22 -> idx 22 = block4_conv3 ReLU
        #   pool   : 23
        #   block5: 24..29 -> idx 29 = block5_conv3 ReLU  (bottleneck)
        self.b1 = nn.Sequential(*feats[ 0: 4])
        self.p1 = feats[ 4]
        self.b2 = nn.Sequential(*feats[ 5: 9])
        self.p2 = feats[ 9]
        self.b3 = nn.Sequential(*feats[10:16])
        self.p3 = feats[16]
        self.b4 = nn.Sequential(*feats[17:23])
        self.p4 = feats[23]
        self.b5 = nn.Sequential(*feats[24:30])
    def forward(self, x):
        s1 = self.b1(x)
        s2 = self.b2(self.p1(s1))
        s3 = self.b3(self.p2(s2))
        s4 = self.b4(self.p3(s3))
        b  = self.b5(self.p4(s4))
        return b, [s1, s2, s3, s4]


class DecoderBlockNormal(nn.Module):
    """Upsample (bilinear) -> concat skip -> ConvBlock."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class VGG16UNet(nn.Module):
    """VGG16 -> SPPFast -> normal decoder (paper uses decoder_branch_normal) -> 1x1 sigmoid."""
    def __init__(self, num_classes=1, pretrained_encoder=True):
        super().__init__()
        self.encoder = VGG16Encoder(pretrained=pretrained_encoder)
        # Skip channels: s1=64, s2=128, s3=256, s4=512
        self.dec4 = DecoderBlockNormal(in_ch=512, skip_ch=512, out_ch=512)
        self.dec3 = DecoderBlockNormal(in_ch=512, skip_ch=256, out_ch=256)
        self.dec2 = DecoderBlockNormal(in_ch=256, skip_ch=128, out_ch=128)
        self.dec1 = DecoderBlockNormal(in_ch=128, skip_ch= 64, out_ch= 64)
        self.out_conv = nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        b, [s1, s2, s3, s4] = self.encoder(x)
        d = self.dec4(b, s4)
        d = self.dec3(d, s3)
        d = self.dec2(d, s2)
        d = self.dec1(d, s1)
        return torch.sigmoid(self.out_conv(d))

EPS = 1e-7

def dice_coef(y_true, y_pred):
    y_true = y_true.float().reshape(-1)
    y_pred = y_pred.float().reshape(-1)
    inter = (y_true * y_pred).sum()
    return (2. * inter + EPS) / (y_true.sum() + y_pred.sum() + EPS)

def jaccard(y_true, y_pred):
    y_true = y_true.float().reshape(-1)
    y_pred = y_pred.float().reshape(-1)
    inter = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - inter
    return inter / (union + EPS)

def precision_recall(y_true, y_pred, thr=0.5):
    pred_bin = (y_pred > thr).float().reshape(-1)
    true_bin = (y_true > 0.5).float().reshape(-1)
    tp = (pred_bin * true_bin).sum()
    fp = (pred_bin * (1 - true_bin)).sum()
    fn = ((1 - pred_bin) * true_bin).sum()
    p = tp / (tp + fp + EPS)
    r = tp / (tp + fn + EPS)
    return p, r

# Simplified differentiable SSIM (constant 8x8 mean window).
def _ssim(y_true, y_pred, max_val=1.0):
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    win = 11
    pad = win // 2
    mu_t = F.avg_pool2d(y_true, win, 1, pad)
    mu_p = F.avg_pool2d(y_pred, win, 1, pad)
    mu_t2 = mu_t * mu_t
    mu_p2 = mu_p * mu_p
    mu_tp = mu_t * mu_p
    sig_t2 = F.avg_pool2d(y_true * y_true, win, 1, pad) - mu_t2
    sig_p2 = F.avg_pool2d(y_pred * y_pred, win, 1, pad) - mu_p2
    sig_tp = F.avg_pool2d(y_true * y_pred, win, 1, pad) - mu_tp
    num = (2 * mu_tp + C1) * (2 * sig_tp + C2)
    den = (mu_t2 + mu_p2 + C1) * (sig_t2 + sig_p2 + C2)
    return (num / den).mean()

def ssim_loss(y_true, y_pred):
    return 1.0 - _ssim(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def sim_dice_loss(y_pred, y_true):
    # y_pred / y_true are (B, 1, H, W).
    return 0.5 * (ssim_loss(y_true, y_pred) + dice_loss(y_true, y_pred))

class SegArrayDataset(Dataset):
    """Wraps preloaded NumPy arrays (HWC) into (CHW) torch tensors."""
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).permute(2, 0, 1).contiguous()        # (C,H,W)
        y = torch.from_numpy(self.Y[i]).permute(2, 0, 1).contiguous().float() # (1,H,W)
        return x, y


def evaluate_seg(model, loader, loss_fn, device):
    model.eval()
    losses, dices, jacs, precs, recs = [], [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            p = model(x)
            losses.append(loss_fn(p, y).item())
            dices.append(dice_coef(y, p).item())
            jacs.append(jaccard(y, p).item())
            pr, rc = precision_recall(y, p)
            precs.append(pr.item()); recs.append(rc.item())
    return (np.mean(losses), np.mean(dices), np.mean(jacs), np.mean(precs), np.mean(recs))


def fit_seg(model, train_loader, val_loader, *,
            epochs=200, lr=1e-4, weight_decay=0.0,
            patience_es=30, patience_rl=10, rl_factor=0.1, min_lr=1e-8,
            best_path=None, log_every=1, device=device):
    model = model.to(device)
    opt = torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=rl_factor, patience=patience_rl, min_lr=min_lr)
    best_val = float('inf'); bad = 0
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            opt.zero_grad()
            p = model(x)
            loss = sim_dice_loss(p, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
        run_loss /= len(train_loader.dataset)
        val_loss, val_d, val_j, val_p, val_r = evaluate_seg(model, val_loader, sim_dice_loss, device)
        sched.step(val_loss)
        if (ep + 1) % log_every == 0:
            print(f'Ep {ep+1:3d}/{epochs} | train {run_loss:.4f} | val {val_loss:.4f} '
                  f'dice {val_d:.4f} jac {val_j:.4f} p {val_p:.4f} r {val_r:.4f} '
                  f'| lr {opt.param_groups[0]["lr"]:.2e} | {time.time()-t0:.1f}s')
        if val_loss < best_val - 1e-6:
            best_val = val_loss; bad = 0
            if best_path:
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                torch.save(model.state_dict(), best_path)
                print(f'  ↳ saved best to {best_path} (val_loss={best_val:.4f})')
        else:
            bad += 1
            if bad >= patience_es:
                print(f'Early stopping at epoch {ep+1} (no val improvement in {patience_es} ep).')
                break
    return model

def warmup_cosine_lr(step, base_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))

from PIL import Image

class SSLArrayDataset(Dataset):
    """Returns (view1, view2) tensors from a HWC float32 [0,1] image array."""
    def __init__(self, X, two_view_transform):
        self.X = X
        self.tv = two_view_transform
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        arr = (self.X[i] * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        v1, v2 = self.tv(img)
        # Convert to tensors (PIL ToTensor would auto-scale; but the transforms above
        # operate on PIL images, so we need to convert manually):
        if not torch.is_tensor(v1):
            from torchvision.transforms.functional import to_tensor
            v1 = to_tensor(v1); v2 = to_tensor(v2)
        return v1, v2

# BYOL augmentation pipeline (Grill et al. 2020, Appendix B):
#   Both views: RandomResizedCrop -> HFlip -> ColorJitter(p=0.8) -> Grayscale(p=0.2)
#   View 1 (T) : GaussianBlur p=1.0,  Solarize p=0.0
#   View 2 (T'): GaussianBlur p=0.1,  Solarize p=0.2

import torchvision.transforms as T
from torchvision.transforms import functional as TF

class TwoViewBYOL:
    def __init__(self, H=256, W=256):
        self.t1 = self._build(H, W, blur_p=1.0, solarize_p=0.0)
        self.t2 = self._build(H, W, blur_p=0.1, solarize_p=0.2)
    @staticmethod
    def _build(H, W, blur_p, solarize_p, s=1.0):
        return T.Compose([
            T.RandomResizedCrop((H, W), scale=(0.08, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4*s, 0.4*s, 0.2*s, 0.1*s)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=int(0.1*H) | 1, sigma=(0.1, 2.0))], p=blur_p),
            T.RandomSolarize(threshold=0.5, p=solarize_p),
        ])
    def __call__(self, img_pil):
        return self.t1(img_pil), self.t2(img_pil)

def mlp_head(in_dim, hidden_dim, out_dim, last_bn=False):
    layers = [nn.Linear(in_dim, hidden_dim, bias=False),
              nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
              nn.Linear(hidden_dim, out_dim, bias=True)]
    if last_bn:
        layers.append(nn.BatchNorm1d(out_dim))
    return nn.Sequential(*layers)

class BYOLEncoder(nn.Module):
    """VGG16 backbone -> GAP -> 512-d feature."""
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = VGG16Encoder(pretrained=pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 512
    def forward(self, x):
        b, _ = self.backbone(x)
        return self.gap(b).flatten(1)


class BYOL(nn.Module):
    """Online (encoder f -> projector g -> predictor q) + target (f', g') with EMA.

    Loss: 0.5 * (||q(z1) - sg(t2)||^2 + ||q(z2) - sg(t1)||^2) on L2-normalized vectors,
    written as 2 - 2 * cos(p, sg(t)).
    """
    def __init__(self, proj_hidden=4096, proj_out=256, pred_hidden=4096,
                 base_momentum=0.996, total_steps=None):
        super().__init__()
        self.encoder   = BYOLEncoder(pretrained=True)
        feat = self.encoder.out_dim
        self.projector = mlp_head(feat, proj_hidden, proj_out)
        self.predictor = mlp_head(proj_out, pred_hidden, proj_out)

        self.target_encoder   = BYOLEncoder(pretrained=True)
        self.target_projector = mlp_head(feat, proj_hidden, proj_out)
        for p in self.target_encoder.parameters():   p.requires_grad = False
        for p in self.target_projector.parameters(): p.requires_grad = False
        self._copy_weights(self.encoder,   self.target_encoder)
        self._copy_weights(self.projector, self.target_projector)

        self.base_momentum = base_momentum
        self.total_steps   = total_steps
        self.global_step   = 0

    @staticmethod
    def _copy_weights(src, dst):
        for ps, pd in zip(src.parameters(), dst.parameters()):
            pd.data.copy_(ps.data)
        for bs, bd in zip(src.buffers(), dst.buffers()):
            bd.data.copy_(bs.data)

    def current_tau(self):
        if self.total_steps is None:
            return self.base_momentum
        cos = math.cos(math.pi * self.global_step / self.total_steps)
        return 1.0 - (1.0 - self.base_momentum) * (cos + 1.0) / 2.0

    @torch.no_grad()
    def ema_update(self, tau):
        for p_o, p_t in zip(self.encoder.parameters(),   self.target_encoder.parameters()):
            p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)
        for p_o, p_t in zip(self.projector.parameters(), self.target_projector.parameters()):
            p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)

    @staticmethod
    def regression_loss(p, z):
        p = F.normalize(p, dim=1); z = F.normalize(z, dim=1)
        return (2 - 2 * (p * z).sum(dim=1)).mean()

    def forward(self, x1, x2):
        # Target forward (no grad).
        with torch.no_grad():
            t1 = self.target_projector(self.target_encoder(x1))
            t2 = self.target_projector(self.target_encoder(x2))
        # Online forward.
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        loss = 0.5 * (self.regression_loss(p1, t2) + self.regression_loss(p2, t1))
        return loss

def train_byol(model, loader, *, epochs, base_lr, warmup_epochs, device=device,
               momentum=0.9, save_path=None):
    model = model.to(device)
    steps_per_epoch = max(1, len(loader))
    total_steps  = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    model.total_steps = total_steps
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=base_lr, momentum=momentum, weight_decay=1.5e-6)
    history = []
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        run_loss = 0.0; n = 0
        for v1, v2 in loader:
            v1 = v1.to(device); v2 = v2.to(device)
            lr = warmup_cosine_lr(model.global_step, base_lr, warmup_steps, total_steps)
            for g in opt.param_groups: g['lr'] = lr
            opt.zero_grad()
            loss = model(v1, v2)
            loss.backward()
            opt.step()
            tau = model.current_tau()
            model.ema_update(tau)
            model.global_step += 1
            run_loss += loss.item() * v1.size(0); n += v1.size(0)
        avg = run_loss / max(1, n)
        history.append(avg)
        print(f'Ep {ep+1:3d}/{epochs} | byol loss {avg:.4f} | tau {tau:.4f} '
              f'| lr {lr:.2e} | {time.time()-t0:.1f}s')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.encoder.backbone.state_dict(), save_path)
        print(f'Saved encoder backbone to {save_path}')
    return history

ssl_ds = SSLArrayDataset(X_train_data, TwoViewBYOL(H=IMG_HEIGHT, W=IMG_WIDTH))
ssl_loader = DataLoader(ssl_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, generator=g, worker_init_fn=worker_init_fn, drop_last=True)
print('SSL batches/epoch :', len(ssl_loader))

# Visualize a pair.
v1, v2 = next(iter(ssl_loader))

# Paper: LARS, lr = 0.2 * batch / 256, cosine LR with 10-epoch warmup.
# We use SGD + momentum as an approximation.
BATCH_SIZE    = 8
EPOCHS        = 200
WARMUP_EPOCHS = 10
base_lr = 0.2 * BATCH_SIZE / 256.0

byol = BYOL(proj_hidden=4096, proj_out=256, pred_hidden=4096, base_momentum=0.996)

TRAIN_FLG = 0  # 0 = load, 1 = train
SSL_CKPT = 'weight/VGG16_BYOL_encoder.pt'

if TRAIN_FLG:
    print('===== Training BYOL =====')
    history = train_byol(
        byol, ssl_loader,
        epochs=EPOCHS, base_lr=base_lr, warmup_epochs=WARMUP_EPOCHS,
        save_path=SSL_CKPT,
    )
else:
    print('===== Loading BYOL encoder weights =====')
    byol.encoder.backbone.load_state_dict(torch.load(SSL_CKPT, map_location=device))
    print('Weights loaded successfully.')

def load_pretrained_backbone(seg_model: VGG16UNet, ckpt_path: str, strict=True):
    state = torch.load(ckpt_path, map_location='cpu')
    missing, unexpected = seg_model.encoder.load_state_dict(state, strict=strict)
    print('Loaded pretrained backbone from', ckpt_path)
    if missing:    print('  Missing keys   :', len(missing))
    if unexpected: print('  Unexpected keys:', len(unexpected))
    return seg_model

# ---- 1. Build a fresh segmentation model (untrained decoder + SPP) ----
seg_model = VGG16UNet(num_classes=1, pretrained_encoder=False).to(device)

# ---- 2. Load the SSL-pretrained backbone into seg_model.encoder ----
backbone_ckpt = 'weight/VGG16_BYOL_encoder.pt'
if os.path.exists(backbone_ckpt):
    load_pretrained_backbone(seg_model, backbone_ckpt, strict=True)
else:
    print('NOTE: no SSL checkpoint found at', backbone_ckpt, '-- using ImageNet init only.')

percentage = 0.1
n = int(X_train_data.shape[0] * percentage)
X_train_p, Y_train_p = X_train_data[:n], Y_train_data[:n]
print(f'Using {n} / {X_train_data.shape[0]} train samples ({percentage:.0%}).')

train_ds = SegArrayDataset(X_train_p, Y_train_p)
valid_ds = SegArrayDataset(X_valid_data, Y_valid_data)
test_ds  = SegArrayDataset(X_test_data , Y_test_data)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=4, pin_memory=True, drop_last=False, generator=g, worker_init_fn=worker_init_fn)
valid_loader = DataLoader(valid_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, generator=g, worker_init_fn=worker_init_fn)
test_loader  = DataLoader(test_ds , batch_size=8, shuffle=False, num_workers=4, pin_memory=True, generator=g, worker_init_fn=worker_init_fn)

if TRAIN_FLG:
    fit_seg(seg_model, train_loader, valid_loader,
            epochs=EPOCHS, lr=1e-4,
            patience_es=30, patience_rl=10, rl_factor=0.1, min_lr=1e-8,
            best_path='weight/BYOL_Best.pt')

seg_model.load_state_dict(torch.load('weight/BYOL_Best.pt', map_location=device))

def compute_hd95(y_true, y_pred, chunk_size=2048):
    y_true = y_true.float()
    y_pred = y_pred.float()

    true_points = torch.nonzero(y_true > 0.5)
    pred_points = torch.nonzero(y_pred > 0.5)

    def compute_distances(A, B):
        if A.numel() == 0 or B.numel() == 0:
            return torch.tensor([], dtype=torch.float32, device=A.device)
        return compute_valid_distances_chunked(A, B)

    def compute_valid_distances_chunked(A, B):
        A = A.float()
        B = B.float()
        
        min_distances = []
        
        # Chia nhỏ tập A thành các chunk để tính toán
        for i in range(0, A.shape[0], chunk_size):
            A_chunk = A[i : i + chunk_size]
            
            # Tính khoảng cách cho chunk hiện tại (Kích thước: chunk_size x số_điểm_B)
            # VD: 2048 x 80000 tốn khoảng ~600MB VRAM thay vì 25GB
            distances_chunk = torch.cdist(A_chunk, B, p=2.0) 
            
            # Lấy khoảng cách nhỏ nhất cho từng điểm trong chunk
            min_dist_chunk = torch.min(distances_chunk, dim=1)[0]
            
            min_distances.append(min_dist_chunk)
            
        # Gộp kết quả của tất cả các chunks lại
        return torch.cat(min_distances)

    # Tính khoảng cách hai chiều
    d_true_to_pred = compute_distances(true_points, pred_points)  
    d_pred_to_true = compute_distances(pred_points, true_points)  

    d_all = torch.cat([d_true_to_pred, d_pred_to_true], dim=0)

    if d_all.numel() == 0:
        hd95 = torch.tensor(0.0, dtype=torch.float32, device=y_true.device)
    else:
        hd95 = torch.quantile(d_all, 0.95)  

    return hd95.item()

def last_eval(model, loader, loss_fn, device):
    model.eval()
    losses, dices, jacs, precs, recs, hd95s = [], [], [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            p = model(x)
            losses.append(loss_fn(p, y).item())
            dices.append(dice_coef(y, p).item())
            jacs.append(jaccard(y, p).item())
            pr, rc = precision_recall(y, p)
            precs.append(pr.item()); recs.append(rc.item())
            hd95s.append(compute_hd95(y_pred=p, y_true=y))
    return (np.mean(losses), np.mean(dices), np.mean(jacs), np.mean(precs), np.mean(recs), np.mean(hd95s))

_, dice, iou, prec, rec, hd95 = last_eval(seg_model, test_loader, sim_dice_loss, device)

result = {
    "method": "BYOL",
    "dice": round(dice, 4),
    "iou": round(iou, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "hd95": round(hd95, 4)
}

json_path = "results.json"

# Neu file chua ton tai hoac rong
if (not os.path.exists(json_path)) or os.path.getsize(json_path) == 0:
    data = []
else:
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = []

# Them ket qua moi
data.append(result)

# Ghi lai
with open(json_path, "w") as f:
    json.dump(data, f, indent=4)

print(f'BEST TEST BYOL | Dice: {dice:.4f} | IoU: {iou:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | HD95: {hd95:.4f}')