import torch
import torch.nn as nn
import pytorch_iou, pytorch_ssim
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import math

def dice_coef(y_true, y_pred, smooth=1e-15):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def jaccard_similarity(y_true, y_pred, smooth=1e-15):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def jacard_loss(y_true, y_pred):
    return 1.0 - jaccard_similarity(y_true, y_pred)


def Ssim_loss(y_true, y_pred, max_val=1.0):
    mean_true = y_true.mean([1,2,3], keepdim=True)
    mean_pred = y_pred.mean([1,2,3], keepdim=True)
    var_true = y_true.var([1,2,3], keepdim=True)
    var_pred = y_pred.var([1,2,3], keepdim=True)
    covar = (y_true * y_pred).mean([1,2,3], keepdim=True) - mean_true * mean_pred
    c1 = (0.01 * max_val)**2
    c2 = (0.03 * max_val)**2
    ssim = ((2 * mean_true * mean_pred + c1) * (2 * covar + c2)) / ((mean_true**2 + mean_pred**2 + c1) * (var_true + var_pred + c2))
    return 1 - ssim.mean()

def focal_loss(y_true, y_pred, alpha=0.26, gamma=2.3):
    BCE = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE
    return focal_loss.mean()

def joint_loss1(y_true, y_pred):
    f_loss = focal_loss(y_true, y_pred)
    s_loss = Ssim_loss(y_true, y_pred)
    j_loss = jacard_loss(y_true, y_pred)
    return (f_loss + s_loss + j_loss) / 3.0

bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)
    return bce_out + ssim_out + iou_out

def MSE_loss(rawA, rawB):
    num_classes = 2.0
    mse = F.mse_loss(rawA, rawB, reduction='none')
    mse_per_image = mse.mean(dim=[1,2,3])
    return (mse_per_image.mean() / num_classes)

def MSE_loss_imbalance(rawA, rawB, n):
    Beta = (n - 1) / (n + 1e-9)
    mse = F.mse_loss(rawA, rawB, reduction='none') 
    w = (1 - Beta) / (1 - Beta ** n)
    w = w.unsqueeze(1)
    mse *= w
    return mse.mean() 

def recall_precision(y_true, y_pred):
    tp = torch.sum(y_true * y_pred)
    fp = torch.sum(y_pred) - tp
    fn = torch.sum(y_true) - tp
    recall = ((tp + 1e-6) / (tp + fn + 1e-6))
    precision = ((tp + 1e-6) / (tp + fp + 1e-6))
    return recall, precision

# def compute_hd95(pred, target, spacing=None):
#     pred = pred.bool()
#     target = target.bool()

#     if pred.sum() == 0 and target.sum() == 0:
#         return 0.0
#     if pred.sum() == 0 or target.sum() == 0:
#         return float('inf')
#     pred_points = torch.nonzero(pred).float()
#     target_points = torch.nonzero(target).float()

#     if spacing is not None:
#         spacing_tensor = torch.tensor(spacing, device=pred.device).float()
#         pred_points = pred_points * spacing_tensor
#         target_points = target_points * spacing_tensor

#     d_pred_to_target = []
#     for batch in pred_points.split(4096):
#         d = torch.cdist(batch, target_points)
#         min_d, _ = torch.min(d, dim=1)
#         d_pred_to_target.append(min_d)
#     d_pred_to_target = torch.cat(d_pred_to_target)
#     d_target_to_pred = []
#     for batch in target_points.split(4096):
#         d = torch.cdist(batch, pred_points)
#         min_d, _ = torch.min(d, dim=1)
#         d_target_to_pred.append(min_d)
#     d_target_to_pred = torch.cat(d_target_to_pred)

#     hd95_val = max(
#         torch.quantile(d_pred_to_target, 0.95).item(),
#         torch.quantile(d_target_to_pred, 0.95).item()
#     )

#     return hd95_val

def compute_hd95(target, pred, chunk_size=2048):
    y_true = target.float()
    y_pred = pred.float()

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


# def compute_hd95(pred, target):
 
#     def get_boundary(mask):
#         # mask: (H, W) bool
#         kernel = torch.ones((1, 1, 3, 3), device=mask.device)
#         mask = mask.float().unsqueeze(0).unsqueeze(0)

#         eroded = F.conv2d(mask, kernel, padding=1) == 9
#         boundary = mask.bool() ^ eroded.bool()

#         return boundary[0, 0]

#     B, _, H, W = pred.shape
#     diag = (H**2 + W**2) ** 0.5

#     results = []

#     for b in range(B):
#         p = pred[b, 0].bool()
#         t = target[b, 0].bool()

#         # --- edge cases ---
#         if not p.any() and not t.any():
#             results.append(torch.tensor(0.0, device=pred.device))
#             continue

#         if not p.any() or not t.any():
#             results.append(torch.tensor(diag, device=pred.device))
#             continue

#         # --- boundary ---
#         p_bd = get_boundary(p)
#         t_bd = get_boundary(t)

#         p_pts = p_bd.nonzero(as_tuple=False).float()  # (Np, 2)
#         t_pts = t_bd.nonzero(as_tuple=False).float()  # (Nt, 2)

#         if p_pts.shape[0] == 0 or t_pts.shape[0] == 0:
#             results.append(torch.tensor(diag, device=pred.device))
#             continue

#         # --- pairwise distance ---
#         dists = torch.cdist(p_pts, t_pts)  # (Np, Nt)

#         d1 = dists.min(dim=1).values
#         d2 = dists.min(dim=0).values

#         all_d = torch.cat([d1, d2])

#         hd95 = torch.quantile(all_d, 0.95)

#         results.append(hd95)

#     return torch.stack(results).mean()

def bce_dice_loss(pred, target):
	return (bce_loss(pred, target) + dice_loss(pred, target)) / 2.0

def muti_bce_loss_fusion(s0, s1, s2, s3, s4, labels_v):
	loss0 = bce_ssim_iou_loss(s0,labels_v)
	loss1 = bce_ssim_iou_loss(s1,labels_v)
	loss2 = bce_ssim_iou_loss(s2,labels_v)
	loss3 = bce_ssim_iou_loss(s3,labels_v)
	loss4 = bce_ssim_iou_loss(s4,labels_v)
	loss = loss0 + loss1 + loss2 + loss3 + loss4
	return loss


def get_bank_weight(capacity, bank_weight_type, k):

    if bank_weight_type == 'Linear':
        return torch.linspace(1.0, 0.0, steps=capacity)

    elif bank_weight_type == 'Exp':
        x = torch.linspace(0, 1, steps=capacity)
        return torch.exp(-k * x)  

    return None

class ContrastiveLoss(nn.Module):
    def __init__(self, capacity, temp=0.1, bank_weight_type=None, device=None, k=None):
        super().__init__()
        self.temp = temp
        self.capacity = capacity
        self.bank_weight_type = bank_weight_type
        self.device = device
        self.k = k
        self.bank_weight = get_bank_weight(self.capacity, self.bank_weight_type, self.k)
        
        if self.bank_weight is not None:
            self.bank_weight = self.bank_weight.to(self.device)
        

    def forward(self, proj, proj_cls, memory_bank, cls_bank):
        proj = F.normalize(proj, dim=1)
        memory_bank = F.normalize(memory_bank, dim=1)

        sim = torch.matmul(proj, memory_bank.T) / self.temp

        proj_cls = proj_cls.unsqueeze(1)
        cls_bank = cls_bank.unsqueeze(0)

    
        valid_mask = (cls_bank != -1)
        pos_mask = (proj_cls == cls_bank) & valid_mask


        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

        if self.bank_weight is None:
            exp_sim = torch.exp(sim) * valid_mask
        else:
            exp_sim = torch.exp(sim) * self.bank_weight * valid_mask

        numerator = (exp_sim * pos_mask).sum(dim=1)
        denominator = exp_sim.sum(dim=1)

        loss = -torch.log(numerator / (denominator + 1e-8))
        return loss.mean()

# def get_bank_weight(capacity, bank_weight_type, n=None, device=None):

#     if bank_weight_type == 'Linear':
#         return torch.linspace(1.0, 0.0, steps=capacity, device=device)

#     elif bank_weight_type == 'Exp':
#         x = torch.linspace(0, 1, steps=capacity, device=device)
#         return torch.exp(-5 * x)

#     elif bank_weight_type == 'Balance':
#         # n: số sample cùng class với proj trong memory_bank
#         # Beta = (n - 1) / (n + 1e-9)
#         # w = (1 - Beta) / (1 - Beta ** n)

#         if n is None:
#             raise ValueError("n must be provided for Balance weighting")

#         beta = (n.float() - 1.0) / (n.float() + 1e-9)
#         w = (1.0 - beta) / (1.0 - beta.pow(n.float()) + 1e-9)

#         return w

#     return None


# class ContrastiveLoss(nn.Module):
#     def __init__(self, capacity, temp=0.1, bank_weight_type=None, device=None):
#         super().__init__()

#         self.temp = temp
#         self.capacity = capacity
#         self.bank_weight_type = bank_weight_type
#         self.device = device

#         if self.bank_weight_type in ['Linear', 'Exp']:
#             self.bank_weight = get_bank_weight(
#                 self.capacity,
#                 self.bank_weight_type,
#                 device=self.device
#             )
#         else:
#             self.bank_weight = None

#     def forward(self, proj, proj_cls, memory_bank, cls_bank):

#         proj = F.normalize(proj, dim=1)
#         memory_bank = F.normalize(memory_bank, dim=1)

#         sim = torch.matmul(proj, memory_bank.T) / self.temp

#         proj_cls = proj_cls.unsqueeze(1)
#         cls_bank = cls_bank.unsqueeze(0)

#         valid_mask = (cls_bank != -1)
#         pos_mask = (proj_cls == cls_bank) & valid_mask

#         sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

#         # -----------------------------
#         # Linear / Exp weighting
#         # -----------------------------
#         if self.bank_weight_type in ['Linear', 'Exp']:

#             exp_sim = (
#                 torch.exp(sim)
#                 * self.bank_weight
#                 * valid_mask
#             )

#         # -----------------------------
#         # Balance weighting
#         # -----------------------------
#         elif self.bank_weight_type == 'Balance':

#             # n: số positive sample của mỗi proj trong memory bank
#             n = pos_mask.sum(dim=1).float()  # [B]

#             balance_weight = get_bank_weight(
#                 self.capacity,
#                 'Balance',
#                 n=n,
#                 device=self.device
#             )  # [B]

#             exp_sim = (
#                 torch.exp(sim)
#                 * balance_weight.unsqueeze(1)
#                 * valid_mask
#             )

#         # -----------------------------
#         # No weighting
#         # -----------------------------
#         else:
#             exp_sim = torch.exp(sim) * valid_mask

#         numerator = (exp_sim * pos_mask).sum(dim=1)
#         denominator = exp_sim.sum(dim=1)

#         loss = -torch.log(numerator / (denominator + 1e-8))

#         return loss.mean()
    
    
def CE_loss_imbalance(rawA, rawB, n):
    
    bce = F.cross_entropy(input=rawA, target=rawB, reduction='none') 
    Beta = (n - 1) / (n + 1e-9)
    w = (1 - Beta) / (1 - Beta ** n)
    bce *= w
    return bce.mean() 

class HardSampleDiscriminativeLoss(nn.Module):
    def __init__(self, capacity, temp=0.1, device=None):
        super().__init__()
        self.temp = temp
        self.capacity = capacity
        self.device = device

    def forward(self, proj, proj_cls, memory_bank, cls_bank):
        proj = F.normalize(proj, dim=1)
        memory_bank = F.normalize(memory_bank, dim=1)

        sim = torch.matmul(proj, memory_bank.T) / self.temp

        proj_cls = proj_cls.unsqueeze(1)
        cls_bank = cls_bank.unsqueeze(0)

    
        valid_mask = (cls_bank != -1)
        pseudo_sim = ((proj_cls == cls_bank) & valid_mask).float()

        sim = sim * valid_mask


        loss = (sim - pseudo_sim) ** 2
        return loss.mean()
    
class HardSampleDiscriminativeLoss2(nn.Module):
    def __init__(self, capacity, temp=0.1, device=None):
        super().__init__()
        self.temp = temp
        self.capacity = capacity
        self.device = device

    def forward(self, proj, proj_cls, memory_bank, cls_bank):
        proj = F.normalize(proj, dim=1)
        memory_bank = F.normalize(memory_bank, dim=1)

        sim = torch.matmul(proj, memory_bank.T) / self.temp

        proj_cls = proj_cls.unsqueeze(1)
        cls_bank = cls_bank.unsqueeze(0)

    
        valid_mask = (cls_bank != -1)

        sim = sim * valid_mask

        pos_mask = (proj_cls == cls_bank) & valid_mask
        neg_mask = (proj_cls != cls_bank) & valid_mask


        loss = torch.sum((1 - sim[pos_mask]) ** 2) + 0.005 * torch.sum(sim[neg_mask] ** 2) 
        return loss.mean()