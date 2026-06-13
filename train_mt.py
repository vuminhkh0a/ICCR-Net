import torch
from loss import *
from ramp import sigmoid_rampup
import pytorch_ssim
import pytorch_iou
import sys

PRE_EPOCHS = 10
EPOCHS = 100
MAX_LAMBDA = 1.0

START_EMA_COEF = 0.99
END_EMA_COEF = 0.999

def get_ema_alpha(step, rampup_length, start=START_EMA_COEF, end=END_EMA_COEF):
    return start + (end - start) * sigmoid_rampup(current=step, rampup_length=rampup_length)

def update_ema_variables(model, ema_model, step, rampup_length, fixed_alpha):
    if fixed_alpha:
        alpha = fixed_alpha
    else:
        alpha = get_ema_alpha(step, rampup_length, START_EMA_COEF, END_EMA_COEF)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

# =================================================================================================================================================================

def pre_train_one_epoch(student, teacher, global_step, rampup_length, loader, device, optimizer):
    student.train()
    teacher.train()

    for student_images, teacher_images, masks, is_labeled in loader:
        LAMBDA = sigmoid_rampup(global_step, rampup_length) * MAX_LAMBDA
        student_images, teacher_images, masks = student_images.to(device), teacher_images.to(device), masks.to(device)
        optimizer.zero_grad()

        s_out = student(student_images)
        t_out = teacher(teacher_images)
        loss1 = bce_dice_loss(s_out, masks)
        loss2 = MSE_loss(s_out, t_out) 
        loss = loss1 + LAMBDA * loss2

        loss.backward()
        optimizer.step()

        global_step += 1
        update_ema_variables(model=student, ema_model=teacher, step=global_step, rampup_length=rampup_length, fixed_alpha=None)



def self_train_one_epoch(student, teacher, loader, device, optimizer, global_step, rampup_length):

    student.train()
    teacher.train()

    for student_images, teacher_images, masks, is_labeled in loader:

        LAMBDA = sigmoid_rampup(global_step, rampup_length) * MAX_LAMBDA
    
     
        student_images_lab = student_images[is_labeled].to(device)
        teacher_images_lab = teacher_images[is_labeled].to(device)
        masks = masks[is_labeled].to(device)
        student_images_unlab = student_images[~is_labeled].to(device)
        teacher_images_unlab = teacher_images[~is_labeled].to(device)

        optimizer.zero_grad()

        # Student output
        s_out = student(student_images_lab)
        s_un_out = student(student_images_unlab)
        
        
        # Teacher output
        t_out = teacher(teacher_images_lab)                           
        t_un_out = teacher(teacher_images_unlab)  
    

        # Loss
        loss1 = bce_dice_loss(s_out, masks)
        loss2 = MSE_loss(s_out, t_out) + MSE_loss(s_un_out, t_un_out)

        
        loss = loss1 + LAMBDA * loss2

        loss.backward()
        optimizer.step()

        global_step += 1
        update_ema_variables(model=student, ema_model=teacher, step=global_step, rampup_length=rampup_length, fixed_alpha=None)

        torch.cuda.empty_cache()
        
# =================================================================================================================================================================
# Evaluate
def evaluate(model, device, loader, with_loss, with_standard_metrics, with_hd95):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_jaccard = 0.0
    running_recall = 0.0
    running_precision = 0.0
    running_hd95 = 0.0


    with torch.no_grad():
        for images, _, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)

            out = model(images)

            if with_loss:
                loss = joint_loss1(masks, out)    
                running_loss += loss.item()
   
            preds = (out > 0.5).float()

            if with_standard_metrics:
                running_dice += dice_coef(masks, preds).item()
                running_jaccard += jaccard_similarity(masks, preds).item()  
                recall, precision = recall_precision(masks, preds)
                running_recall += recall.item()
                running_precision += precision.item()

            if with_hd95:
                running_hd95 += compute_hd95(pred=preds, target=masks)

    avg_loss = running_loss / len(loader)
    avg_dice = running_dice / len(loader)                 
    avg_jaccard = running_jaccard / len(loader)         
    avg_recall = running_recall / len(loader)           
    avg_precision = running_precision / len(loader) 
    avg_hd95 = running_hd95 / len(loader)

    return avg_loss, avg_dice, avg_jaccard, avg_recall, avg_precision, avg_hd95

# =================================================================================================================================================================
def train_one_dataset(is_train, student, teacher, global_step, rampup_length, labeled_train_loader, train_loader, valid_loader, test_loader, device, optimizer, best_model_path):
    
    if is_train:

        print("\n", "--- Pre-train ---")
        best_loss = float('inf')
        for epoch in range(PRE_EPOCHS):
            pre_train_one_epoch(student, teacher, global_step, rampup_length, labeled_train_loader, device, optimizer)
            val_loss, val_dice, val_jaccard, val_precision, val_recall, _ = evaluate(model=teacher, device=device, loader=valid_loader, with_loss=True, with_standard_metrics=True, with_hd95=False)

            print(f"Epoch [{epoch+1}/{PRE_EPOCHS}]")
            print(f"loss: {val_loss:.4f} | dice: {val_dice:.4f} | iou: {val_jaccard:.4f} | precision: {val_precision:.4f} | recall: {val_recall:.4f}")
            sys.stdout.flush()
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(teacher.state_dict(), best_model_path)
                print(f"  → Best models updated (val_loss={val_loss:.4f})")



        print("\n", "--- Self-train ---")
        for epoch in range(EPOCHS):
            self_train_one_epoch(student, teacher, train_loader, device, optimizer, global_step, rampup_length)
            val_loss, val_dice, val_jaccard, val_precision, val_recall, _ = evaluate(model=teacher, device=device, loader=valid_loader, with_loss=True, with_standard_metrics=True, with_hd95=False)

            print(f"Epoch [{epoch+1}/{EPOCHS}]")
            print(f"loss: {val_loss:.4f} | dice: {val_dice:.4f} | iou: {val_jaccard:.4f} | precision: {val_precision:.4f} | recall: {val_recall:.4f}")
            sys.stdout.flush()
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(teacher.state_dict(), best_model_path)
                print(f"  → Best models updated (val_loss={val_loss:.4f})")

    print("\n--- Test Set Evaluation ---")
    teacher.load_state_dict(torch.load(best_model_path))
    _, test_dice, test_jaccard, test_recall, test_precision, test_hd95 = evaluate(model=teacher, device=device, loader=test_loader, with_loss=True, with_standard_metrics=True, with_hd95=True)
    print(f"  Test Dice Coef: {test_dice:.4f}")
    print(f"  Test Jaccard Similarity: {test_jaccard:.4f}")
    print(f"  Test Precision: {test_precision:.4f}")
    print(f"  Test Recall: {test_recall:.4f}")
    print(f"  Test HD95: {test_hd95:.4f}")


