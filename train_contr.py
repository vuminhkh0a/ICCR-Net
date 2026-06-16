import os
import gc
import json
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models

from data import get_dataloaders

# ========================================================================
# 1. ARCHITECTURE & BACKBONES
# ========================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, k, padding=p), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class VGG16Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = tv_models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = tv_models.vgg16_bn(weights=weights)
        feats = vgg.features
        
        # Slicing cho VGG16_BN (Có thêm các lớp BatchNorm2d)
        # Block 1: Conv, BN, ReLU, Conv, BN, ReLU (0 -> 5) | Pool (6)
        self.b1 = nn.Sequential(*feats[0:6])
        self.p1 = feats[6]
        
        # Block 2: Conv, BN, ReLU, Conv, BN, ReLU (7 -> 12) | Pool (13)
        self.b2 = nn.Sequential(*feats[7:13])
        self.p2 = feats[13]
        
        # Block 3: 3x(Conv, BN, ReLU) (14 -> 22) | Pool (23)
        self.b3 = nn.Sequential(*feats[14:23])
        self.p3 = feats[23]
        
        # Block 4: 3x(Conv, BN, ReLU) (24 -> 32) | Pool (33)
        self.b4 = nn.Sequential(*feats[24:33])
        self.p4 = feats[33]
        
        # Block 5: 3x(Conv, BN, ReLU) (34 -> 42) | Pool (43)
        self.b5 = nn.Sequential(*feats[34:43])
        
    def forward(self, x):
        s1 = self.b1(x)
        s2 = self.b2(self.p1(s1))
        s3 = self.b3(self.p2(s2))
        s4 = self.b4(self.p3(s3))
        b  = self.b5(self.p4(s4))
        return b, [s1, s2, s3, s4]

class DecoderBlockNormal(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class VGG16UNet(nn.Module):
    def __init__(self, num_classes=1, pretrained_encoder=True):
        super().__init__()
        self.encoder = VGG16Encoder(pretrained=pretrained_encoder)
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

# ========================================================================
# 2. CONTRASTIVE LEARNING MODELS
# ========================================================================

# --- SimSiam ---
class SimSiam(nn.Module):
    def __init__(self, proj_hidden=2048, proj_out=2048, pred_hidden=512, pretrained=True):
        super().__init__()
        self.backbone = VGG16Encoder(pretrained=pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Linear(512, proj_hidden, bias=False), nn.BatchNorm1d(proj_hidden), nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_hidden, bias=False), nn.BatchNorm1d(proj_hidden), nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out, bias=False), nn.BatchNorm1d(proj_out, affine=False),  
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_out, pred_hidden, bias=False), nn.BatchNorm1d(pred_hidden), nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_out, bias=True),
        )

    def encode(self, x):
        b, _ = self.backbone(x)
        return self.gap(b).flatten(1)

    @staticmethod
    def D(p, z):
        return -F.cosine_similarity(p, z.detach(), dim=1).mean()

    def forward(self, x1, x2):
        z1, z2 = self.projector(self.encode(x1)), self.projector(self.encode(x2))
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return 0.5 * (self.D(p1, z2) + self.D(p2, z1))

# --- SimCLR ---
class SimCLR(nn.Module):
    def __init__(self, proj_hidden=2048, proj_out=128, temperature=0.5, pretrained=True):
        super().__init__()
        self.backbone = VGG16Encoder(pretrained=pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Linear(512, proj_hidden, bias=False), nn.BatchNorm1d(proj_hidden), nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out, bias=True),
        )
        self.T = temperature

    def encode(self, x):
        b, _ = self.backbone(x)
        return self.gap(b).flatten(1)

    def forward(self, x1, x2):
        z1 = F.normalize(self.projector(self.encode(x1)), dim=1)
        z2 = F.normalize(self.projector(self.encode(x2)), dim=1)
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = (z @ z.T) / self.T
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, float('-inf'))
        pos = torch.cat([(z1 * z2).sum(dim=1), (z2 * z1).sum(dim=1)], dim=0) / self.T
        return (-pos + torch.logsumexp(sim, dim=1)).mean()

# --- MoCo v2 ---
class MoCoEncoder(nn.Module):
    def __init__(self, proj_dim=256, hidden_dim=512, pretrained=True):
        super().__init__()
        self.backbone = VGG16Encoder(pretrained=pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(512, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim, bias=False), nn.BatchNorm1d(proj_dim),
        )
    def forward(self, x):
        b, _ = self.backbone(x)
        return self.proj(self.gap(b).flatten(1))

class MoCo(nn.Module):
    def __init__(self, proj_dim=256, queue_size=4096, momentum=0.999, temperature=0.07):
        super().__init__()
        self.encoder_q = MoCoEncoder(proj_dim=proj_dim)
        self.encoder_k = MoCoEncoder(proj_dim=proj_dim)
        for p in self.encoder_k.parameters(): p.requires_grad = False
        for q, k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k.data.copy_(q.data)

        self.m, self.T, self.K = momentum, temperature, queue_size
        self.register_buffer('queue', F.normalize(torch.randn(proj_dim, queue_size), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self):
        for q, k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k.data.mul_(self.m).add_(q.data, alpha=1.0 - self.m)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        B = keys.size(0)
        ptr = int(self.queue_ptr.item())
        if ptr + B <= self.K:
            self.queue[:, ptr:ptr + B] = keys.T
        else:
            first = self.K - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, :B - first] = keys[first:].T
        self.queue_ptr[0] = (ptr + B) % self.K

    def forward(self, x_q, x_k):
        q = F.normalize(self.encoder_q(x_q), dim=1)
        with torch.no_grad():
            self.momentum_update()
            k = F.normalize(self.encoder_k(x_k), dim=1)
        l_pos = (q * k).sum(dim=1, keepdim=True)
        l_neg = q @ self.queue.clone().detach()
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        self.dequeue_and_enqueue(k)
        return F.cross_entropy(logits, labels)

# --- BYOL ---
def mlp_head(in_dim, hidden_dim, out_dim, last_bn=False):
    layers = [nn.Linear(in_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim, bias=True)]
    if last_bn: layers.append(nn.BatchNorm1d(out_dim))
    return nn.Sequential(*layers)

class BYOLEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = VGG16Encoder(pretrained=pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 512
    def forward(self, x):
        b, _ = self.backbone(x)
        return self.gap(b).flatten(1)

class BYOL(nn.Module):
    def __init__(self, proj_hidden=4096, proj_out=256, pred_hidden=4096, base_momentum=0.996):
        super().__init__()
        self.encoder, self.target_encoder = BYOLEncoder(pretrained=True), BYOLEncoder(pretrained=True)
        self.projector = mlp_head(512, proj_hidden, proj_out)
        self.predictor = mlp_head(proj_out, pred_hidden, proj_out)
        self.target_projector = mlp_head(512, proj_hidden, proj_out)
        
        for p in self.target_encoder.parameters(): p.requires_grad = False
        for p in self.target_projector.parameters(): p.requires_grad = False
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_projector.load_state_dict(self.projector.state_dict())

        self.base_momentum = base_momentum
        self.total_steps, self.global_step = 1, 0

    @torch.no_grad()
    def ema_update(self):
        tau = 1.0 - (1.0 - self.base_momentum) * (math.cos(math.pi * self.global_step / self.total_steps) + 1.0) / 2.0
        for p_o, p_t in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)
        for p_o, p_t in zip(self.projector.parameters(), self.target_projector.parameters()):
            p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)
        return tau

    def forward(self, x1, x2):
        with torch.no_grad():
            t1, t2 = self.target_projector(self.target_encoder(x1)), self.target_projector(self.target_encoder(x2))
        p1, p2 = self.predictor(self.projector(self.encoder(x1))), self.predictor(self.projector(self.encoder(x2)))
        
        def reg_loss(p, z):
            return (2 - 2 * (F.normalize(p, dim=1) * F.normalize(z, dim=1)).sum(dim=1)).mean()
            
        return 0.5 * (reg_loss(p1, t2) + reg_loss(p2, t1))

# ========================================================================
# 3. UTILITIES & LOSSES
# ========================================================================

def warmup_cosine_lr(step, base_lr, warmup_steps, total_steps):
    if step < warmup_steps: return base_lr * step / max(1, warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))

EPS = 1e-7

def dice_coef(y_true, y_pred):
    inter = (y_true.float().reshape(-1) * y_pred.float().reshape(-1)).sum()
    return (2. * inter + EPS) / (y_true.sum() + y_pred.sum() + EPS)

def jaccard(y_true, y_pred):
    y_t, y_p = y_true.float().reshape(-1), y_pred.float().reshape(-1)
    inter = (y_t * y_p).sum()
    return inter / (y_t.sum() + y_p.sum() - inter + EPS)

def precision_recall(y_true, y_pred, thr=0.5):
    pred_bin, true_bin = (y_pred > thr).float().reshape(-1), (y_true > 0.5).float().reshape(-1)
    tp = (pred_bin * true_bin).sum()
    return tp / (tp + (pred_bin * (1 - true_bin)).sum() + EPS), tp / (tp + ((1 - pred_bin) * true_bin).sum() + EPS)

def _ssim(y_true, y_pred):
    C1, C2, win, pad = (0.01) ** 2, (0.03) ** 2, 11, 5
    mu_t, mu_p = F.avg_pool2d(y_true, win, 1, pad), F.avg_pool2d(y_pred, win, 1, pad)
    sig_tp = F.avg_pool2d(y_true * y_pred, win, 1, pad) - (mu_t * mu_p)
    num = (2 * (mu_t * mu_p) + C1) * (2 * sig_tp + C2)
    den = (mu_t * mu_t + mu_p * mu_p + C1) * (F.avg_pool2d(y_true * y_true, win, 1, pad) - mu_t * mu_t + F.avg_pool2d(y_pred * y_pred, win, 1, pad) - mu_p * mu_p + C2)
    return (num / den).mean()

def sim_dice_loss(y_pred, y_true):
    return 0.5 * ((1.0 - _ssim(y_true, y_pred)) + (1.0 - dice_coef(y_true, y_pred)))

def compute_hd95(y_true, y_pred, chunk_size=2048):
    true_points, pred_points = torch.nonzero(y_true.float() > 0.5).float(), torch.nonzero(y_pred.float() > 0.5).float()
    def dists(A, B):
        if A.numel() == 0 or B.numel() == 0: return torch.tensor([], dtype=torch.float32, device=A.device)
        return torch.cat([torch.min(torch.cdist(A[i:i+chunk_size], B, p=2.0), dim=1)[0] for i in range(0, A.shape[0], chunk_size)])
    d_all = torch.cat([dists(true_points, pred_points), dists(pred_points, true_points)], dim=0)
    return torch.quantile(d_all, 0.95).item() if d_all.numel() > 0 else 0.0

# ========================================================================
# 4. TRAINING PIPELINE
# ========================================================================

def pretrain_ssl(method, model, loader, epochs, base_lr, warmup_epochs, device, save_path):
    model.train()
    steps_per_epoch = max(1, len(loader))
    total_steps = epochs * steps_per_epoch
    
    if method == "SimSiam":
        opt = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
    elif method == "SimCLR":
        opt = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
    elif method == "MoCo":
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=base_lr)
    elif method == "BYOL":
        model.total_steps = total_steps
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=base_lr, momentum=0.9, weight_decay=1.5e-6)

    step = 0
    for ep in range(epochs):
        t0 = time.time()
        run_loss, n = 0.0, 0
        
        # Loader from data.py returns 4 variables in train mode. We extract the two augmented views.
        for batch in loader:
            v1, v2 = batch[0].to(device), batch[1].to(device)
            
            if method == "SimSiam":
                lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * (step / max(1, total_steps))))
            else:
                lr = warmup_cosine_lr(step, base_lr, warmup_epochs * steps_per_epoch, total_steps)
                
            for g in opt.param_groups: g['lr'] = lr
            
            opt.zero_grad()
            loss = model(v1, v2)
            loss.backward()
            opt.step()
            
            if method == "BYOL": model.ema_update()
            
            step += 1
            run_loss += loss.item() * v1.size(0); n += v1.size(0)
            
        print(f'Ep {ep+1:3d}/{epochs} | {method} pretrain loss {run_loss/max(1,n):.4f} | lr {lr:.2e} | {time.time()-t0:.1f}s')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if method == "SimSiam": torch.save(model.backbone.state_dict(), save_path)
    elif method == "SimCLR": torch.save(model.backbone.state_dict(), save_path)
    elif method == "MoCo": torch.save(model.encoder_q.backbone.state_dict(), save_path)
    elif method == "BYOL": torch.save(model.encoder.backbone.state_dict(), save_path)
    print(f'Saved {method} encoder backbone to {save_path}')


def fit_seg(model, train_loader, val_loader, epochs, lr, best_path, device, num_patience_epoch):
    opt = torch.optim.NAdam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10, min_lr=1e-8)
    best_val, bad = float('inf'), 0
    
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        run_loss, count = 0.0, 0
        
        for batch in train_loader:
            x, _, y, _ = batch 
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            p = model(x)
            loss = sim_dice_loss(p, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0); count += x.size(0)
            
        model.eval()
        val_loss, val_d, val_j, val_p, val_r = 0.0, 0.0, 0.0, 0.0, 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                x, _, y, _ = batch
                x, y = x.to(device), y.to(device)
                p = model(x)
                val_loss += sim_dice_loss(p, y).item() * x.size(0)
                val_d += dice_coef(y, p).item() * x.size(0)
                val_j += jaccard(y, p).item() * x.size(0)
                pr, rc = precision_recall(y, p)
                val_p += pr.item() * x.size(0); val_r += rc.item() * x.size(0)
                val_count += x.size(0)
                
        val_loss /= val_count
        sched.step(val_loss)
        
        print(f'Ep {ep+1:3d}/{epochs} | Seg train {(run_loss/count):.4f} | val loss {val_loss:.4f} '
              f'dice {(val_d/val_count):.4f} jac {(val_j/val_count):.4f} | lr {opt.param_groups[0]["lr"]:.2e} | {time.time()-t0:.1f}s')
              
        if val_loss < best_val - 1e-6:
            best_val = val_loss; bad = 0
            os.makedirs(os.path.dirname(best_path), exist_ok=True)
            torch.save(model.state_dict(), best_path)
            print(f'  ↳ saved best to {best_path} (val_loss={best_val:.4f})')
        else:
            bad += 1
            if bad >= num_patience_epoch:
                print(f'Early stopping at epoch {ep+1} do loss không cải thiện sau {num_patience_epoch} epoch.')
                break

def run_contrastive_pipeline(method, dataset_name, batch_size, num_workers, pin_memory, device, pre_epochs, epochs, is_train, num_patience_epoch=30, LABELED_RATIO=0.1, run_all=False):
    # Xác định danh sách các phương pháp cần chạy
    methods_to_run = ['SimSiam', 'SimCLR', 'MoCo', 'BYOL'] if run_all else [method]
    
    print(f"========== Chuẩn bị dữ liệu cho {dataset_name} ==========")
    # Dữ liệu chỉ load 1 lần cho tất cả các phương pháp
    labeled_train_loader, train_loader, valid_loader, test_loader = get_dataloaders(dataset_name, batch_size, num_workers, pin_memory, LABELED_RATIO)
    
    for curr_method in methods_to_run:
        print(f"\n========== Khởi chạy pipeline {curr_method} trên {dataset_name} ==========")
        
        ssl_ckpt = f'weight/{dataset_name}/{curr_method}_encoder.pt'
        best_seg_ckpt = f'weight/{dataset_name}/{curr_method}_Best.pt'
        
        if is_train:
            # 1. SETUP SSL MODEL
            if curr_method == 'SimSiam':
                ssl_model = SimSiam(proj_hidden=2048, proj_out=2048, pred_hidden=512).to(device)
                base_lr, warmup = 0.05 * batch_size / 256.0, 0
            elif curr_method == 'SimCLR':
                ssl_model = SimCLR(proj_hidden=2048, proj_out=128, temperature=0.5).to(device)
                base_lr, warmup = 0.3 * batch_size / 256.0, 10
            elif curr_method == 'MoCo':
                ssl_model = MoCo(proj_dim=256, queue_size=4096, momentum=0.999, temperature=0.07).to(device)
                base_lr, warmup = 1e-4, int(pre_epochs * 0.1)
            elif curr_method == 'BYOL':
                ssl_model = BYOL(proj_hidden=4096, proj_out=256, pred_hidden=4096, base_momentum=0.996).to(device)
                base_lr, warmup = 0.2 * batch_size / 256.0, 10
            else:
                print(f"Lỗi: Phương pháp {curr_method} không được hỗ trợ!")
                continue

            # 2. PRETRAINING
            print(f"--- Bắt đầu Pretrain {curr_method} ({pre_epochs} epochs) ---")
            pretrain_ssl(curr_method, ssl_model, train_loader, pre_epochs, base_lr, warmup, device, ssl_ckpt)
            
            # Giải phóng RAM/VRAM
            del ssl_model
            gc.collect()
            torch.cuda.empty_cache()

        # 3. SETUP SEGMENTATION MODEL
        seg_model = VGG16UNet(num_classes=1, pretrained_encoder=False).to(device)

        if is_train:
            print(f"--- Bắt đầu Fine-tune Segmentation {curr_method} ---")
            if os.path.exists(ssl_ckpt):
                missing, unexpected = seg_model.encoder.load_state_dict(torch.load(ssl_ckpt, map_location=device), strict=True)
                print(f"Loaded Backbone. Missing: {len(missing)} | Unexpected: {len(unexpected)}")
            else:
                print(f"Warning: Không tìm thấy checkpoint {ssl_ckpt}, dùng khởi tạo ngẫu nhiên!")

            # TRUYỀN THÊM num_patience_epoch VÀO HÀM fit_seg
            fit_seg(seg_model, labeled_train_loader, valid_loader, epochs, 1e-4, best_seg_ckpt, device, num_patience_epoch)
        else:
            print(f"--- Bỏ qua huấn luyện, chuyển sang chế độ Evaluation cho {curr_method} ---")

        # Load trọng số Segmentation để đánh giá
        if os.path.exists(best_seg_ckpt):
            seg_model.load_state_dict(torch.load(best_seg_ckpt, map_location=device))
            print(f"Đã load trọng số phân vùng tốt nhất từ {best_seg_ckpt}")
        else:
            print(f"Lỗi: Không tìm thấy file trọng số {best_seg_ckpt} để đánh giá!")
            del seg_model
            gc.collect()
            torch.cuda.empty_cache()
            continue
        
        # 4. EVALUATION
        print(f"--- Bắt đầu Evaluation {curr_method} ---")
        seg_model.eval()
        dices, jacs, precs, recs, hd95s = [], [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                x, _, y, _ = batch
                x, y = x.to(device), y.to(device)
                p = seg_model(x)
                dices.append(dice_coef(y, p).item())
                jacs.append(jaccard(y, p).item())
                pr, rc = precision_recall(y, p)
                precs.append(pr.item()); recs.append(rc.item())
                hd95s.append(compute_hd95(y_pred=p, y_true=y))
                
        dice, iou, prec, rec, hd95 = np.mean(dices), np.mean(jacs), np.mean(precs), np.mean(recs), np.mean(hd95s)
        
        result = {
            "dataset": dataset_name,
            "method": curr_method,
            "dice": round(dice, 4),
            "iou": round(iou, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "hd95": round(hd95, 4)
        }

        json_path = "results.json"
        data = []
        if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
            try:
                with open(json_path, "r") as f: data = json.load(f)
            except json.JSONDecodeError:
                data = []
                
        data.append(result)
        with open(json_path, "w") as f: json.dump(data, f, indent=4)

        print(f'BEST TEST {curr_method} | Dice: {dice:.4f} | IoU: {iou:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | HD95: {hd95:.4f}')

        # Dọn dẹp cache trước khi vòng lặp chuyển sang method tiếp theo
        del seg_model
        gc.collect()
        torch.cuda.empty_cache()