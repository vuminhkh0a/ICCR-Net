import torch
from loss import *
from ramp import sigmoid_rampup
import pytorch_ssim
import pytorch_iou
import sys

EPOCHS = 50


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

def train_one_epoch(model, loader, device, optimizer):
    model.train()

    for student_images, teacher_images, masks, is_labeled in loader:
        student_images, teacher_images, masks = student_images.to(device), teacher_images.to(device), masks.to(device)
        optimizer.zero_grad()

        out = model(student_images)
        loss = bce_loss(out, masks)

        loss.backward()
        optimizer.step()

        
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
def train_one_dataset(is_train, model, labeled_train_loader, valid_loader, test_loader, device, optimizer, best_model_path):
    
    if is_train:

        print("\n", "--- Pre-train ---")
        best_loss = float('inf')
        for epoch in range(EPOCHS):
            train_one_epoch(model, labeled_train_loader, device, optimizer)

            val_loss, val_dice, val_jaccard, val_precision, val_recall, _ = evaluate(model=model, device=device, loader=valid_loader, with_loss=True, with_standard_metrics=True, with_hd95=False)

            print(f"Epoch [{epoch+1}/{EPOCHS}]")
            print(f"loss: {val_loss:.4f} | dice: {val_dice:.4f} | iou: {val_jaccard:.4f} | precision: {val_precision:.4f} | recall: {val_recall:.4f}")
            sys.stdout.flush()
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"  → Best models updated (val_loss={val_loss:.4f})")


    print("\n--- Test Set Evaluation ---")
    model.load_state_dict(torch.load(best_model_path))
    _, test_dice, test_jaccard, test_recall, test_precision, test_hd95 = evaluate(model=model, device=device, loader=test_loader, with_loss=True, with_standard_metrics=True, with_hd95=True)
    print(f"  Test Dice Coef: {test_dice:.4f}")
    print(f"  Test Jaccard Similarity: {test_jaccard:.4f}")
    print(f"  Test Precision: {test_precision:.4f}")
    print(f"  Test Recall: {test_recall:.4f}")
    print(f"  Test HD95: {test_hd95:.4f}")
    


