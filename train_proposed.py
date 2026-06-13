import torch
from loss import *
from ramp import sigmoid_rampup
import time
import sys


MAX_LAMBDA = 1.0
MAX_BETA1 = 1.0
MAX_BETA2 = 2.0
MAX_BETA3 = 1.0
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
        # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

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

def init_weight_proposed(model, with_vgg16bn=False):

    skip_modules = set()
    if with_vgg16bn:
        skip_modules.update([model.down_conv1, model.down_conv2, model.down_conv3, model.down_conv4,])
        if isinstance(model.down_conv5, nn.Sequential):
            skip_modules.add(model.down_conv5[0])

    for module in model.modules():
        should_skip = False
        for skip_module in skip_modules:
            if module is skip_module or module in skip_module.modules():
                should_skip = True
                break
        if should_skip:
            continue

        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
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

            out, _, _ = model(images)

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
                # running_hd95 += compute_hd95(pred=preds, target=masks).item()
                running_hd95 += compute_hd95(pred=preds, target=masks)

    avg_loss = running_loss / len(loader)
    avg_dice = running_dice / len(loader)                 
    avg_jaccard = running_jaccard / len(loader)         
    avg_recall = running_recall / len(loader)           
    avg_precision = running_precision / len(loader) 
    avg_hd95 = running_hd95 / len(loader)

    return avg_loss, avg_dice, avg_jaccard, avg_recall, avg_precision, avg_hd95
# =================================================================================================================================================================

def pre_train_one_epoch(student, teacher, global_step, rampup_length, loader, device, optimizer):
    student.train()
    teacher.train()

    for student_images, teacher_images, masks, is_labeled in loader:
        LAMBDA = sigmoid_rampup(global_step, rampup_length) * MAX_LAMBDA
        student_images, teacher_images, masks = student_images.to(device), teacher_images.to(device), masks.to(device)
        optimizer.zero_grad()

        s_out, _, _ = student(student_images)
        t_out, _, _ = teacher(teacher_images)
        loss1 = bce_ssim_iou_loss(s_out, masks)
        loss2 = bce_dice_loss(s_out, torch.round(t_out)) 
        loss = loss1 + LAMBDA * loss2

        loss.backward()
        optimizer.step()

        global_step += 1
        update_ema_variables(model=student, ema_model=teacher, step=global_step, rampup_length=rampup_length, fixed_alpha=None)
    
    return global_step


def self_train_one_epoch(student, teacher, 
                         loader, device, optimizer, 
                        contr_loss, 
                        global_step, rampup_length, 
                        capacity, memory_bank, cls_bank, cls_proj_bank, 
                        with_imb_loss, with_dis_hard_loss):

    student.train()
    teacher.train()

    for student_images, teacher_images, masks, is_labeled in loader:

        LAMBDA = sigmoid_rampup(global_step, rampup_length) * MAX_LAMBDA
        BETA1 = sigmoid_rampup(global_step, rampup_length) * MAX_BETA1  
        BETA2 = sigmoid_rampup(global_step, rampup_length) * MAX_BETA2      
        BETA3 = sigmoid_rampup(global_step, rampup_length) * MAX_BETA3            
     
        student_images_lab = student_images[is_labeled].to(device)
        teacher_images_lab = teacher_images[is_labeled].to(device)
        masks = masks[is_labeled].to(device)
        student_images_unlab = student_images[~is_labeled].to(device)
        teacher_images_unlab = teacher_images[~is_labeled].to(device)

        print(student_images_lab.shape, teacher_images_lab.shape, masks.shape, student_images_unlab.shape, teacher_images_unlab.shape)

        optimizer.zero_grad()

        # Student output
        s_out, _, _ = student(student_images_lab)
        s_un_out, s_un_proj, s_un_cls = student(student_images_unlab)
        # _, cls = torch.max(s_un_cls, dim=1)
        _, cls = torch.max(torch.softmax(s_un_cls, dim=1), dim=1)
        
        # Teacher output
        t_out, _, _ = teacher(teacher_images_lab)                           
        t_un_out, t_un_proj, t_un_cls = teacher(teacher_images_unlab)  
        # _, pseudo_cls = torch.max(t_un_cls, dim=1)
        _, pseudo_cls = torch.max(torch.softmax(t_un_cls, dim=1), dim=1)

       
        # Bank
        cls_bank = torch.cat((pseudo_cls.detach(), cls_bank), 0)
        cls_bank = torch.cat((cls.detach(), cls_bank), 0)

        memory_bank = torch.cat((t_un_proj.detach(), memory_bank), 0)
        memory_bank = torch.cat((s_un_proj.detach(), memory_bank), 0)

        cls_proj_bank = torch.cat((s_un_cls.detach(), cls_proj_bank), 0)
        cls_proj_bank = torch.cat((t_un_cls.detach(), cls_proj_bank), 0)

        while memory_bank.shape[0] > capacity:
            memory_bank = memory_bank[:-1, :]
        
        while cls_bank.shape[0] > capacity:
            cls_bank = cls_bank[:-1]

        while cls_proj_bank.shape[0] > capacity:
            cls_proj_bank = cls_proj_bank[:-1, :]

        count_match_cls = (cls[:, None] == cls_bank[None, :]).sum(dim=1)
        # Loss
        
        loss1 = bce_ssim_iou_loss(s_out, masks)
        loss2 =  bce_dice_loss(s_out, torch.round(t_out)) + bce_dice_loss(s_un_out, torch.round(t_un_out))
        loss3 = contr_loss(s_un_proj, cls, memory_bank, cls_bank)

        loss = loss1 + LAMBDA * loss2 + BETA1 * loss3

        if with_imb_loss:
            loss4 = CE_loss_imbalance(s_un_cls, pseudo_cls, n=count_match_cls)

            loss += BETA2 * loss4

        if with_dis_hard_loss:
            hard_loss = HardSampleDiscriminativeLoss2(capacity=capacity, device=device)
            loss5 = hard_loss(s_un_proj, cls, t_un_proj, pseudo_cls)

            loss += BETA3 * loss5

        loss.backward()
        optimizer.step()

        global_step += 1
        update_ema_variables(model=student, ema_model=teacher, step=global_step, rampup_length=rampup_length, fixed_alpha=None)
    
        torch.cuda.empty_cache()
    return memory_bank, cls_bank, global_step
        
# =================================================================================================================================================================
def train_one_dataset(is_train, pre_epochs, epochs, student, teacher, global_step, rampup_length, labeled_train_loader, train_loader, valid_loader, test_loader, 
                      device, optimizer, scheduler, best_model_path,
                      contr_loss, capacity, memory_bank, cls_bank, cls_proj_bank, with_imb_loss, with_dis_hard_loss):
    
    start = time.perf_counter()
    if is_train:

        memory_bank, cls_bank, global_step = memory_bank, cls_bank, global_step

        print("\n", "--- Pre-train ---")
        
        best_loss = float('inf')
        for epoch in range(pre_epochs):
            start_time_epoch = time.perf_counter()
            global_step = pre_train_one_epoch(student, teacher, global_step, rampup_length, labeled_train_loader, device, optimizer)
            val_loss, val_dice, val_jaccard, val_precision, val_recall, _ = evaluate(model=teacher, device=device, loader=valid_loader, with_loss=True, with_standard_metrics=True, with_hd95=False)
            end_time_epoch = time.perf_counter()
            print(f"Epoch [{epoch+1}/{pre_epochs}] | Time: {end_time_epoch-start_time_epoch:.2f} seconds")
            print(f"loss: {val_loss:.4f} | dice: {val_dice:.4f} | iou: {val_jaccard:.4f} | precision: {val_precision:.4f} | recall: {val_recall:.4f}")
            sys.stdout.flush()
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(teacher.state_dict(), best_model_path)
                print(f"  → Best models updated (val_loss={val_loss:.4f})")
            
            scheduler.step(val_loss)


        print("\n", "--- Self-train ---")
        for epoch in range(epochs):
            start_time_epoch = time.perf_counter()
            memory_bank, cls_bank, global_step = self_train_one_epoch(student, teacher, 
                                                                    train_loader, device, optimizer, 
                                                                    contr_loss, 
                                                                    global_step, rampup_length, 
                                                                    capacity, memory_bank, cls_bank, cls_proj_bank, 
                                                                    with_imb_loss, with_dis_hard_loss)
            val_loss, val_dice, val_jaccard, val_precision, val_recall, _ = evaluate(model=teacher, device=device, loader=valid_loader, with_loss=True, with_standard_metrics=True, with_hd95=False)
            end_time_epoch = time.perf_counter()
            print(f"Epoch [{epoch+1}/{epochs}] | Time: {end_time_epoch-start_time_epoch:.2f} seconds")
            print(f"loss: {val_loss:.4f} | dice: {val_dice:.4f} | iou: {val_jaccard:.4f} | precision: {val_precision:.4f} | recall: {val_recall:.4f}")
            sys.stdout.flush()
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(teacher.state_dict(), best_model_path)
                print(f"  → Best models updated (val_loss={val_loss:.4f})")
                        
            scheduler.step(val_loss)

    print("\n--- Test Set Evaluation ---")
    teacher.load_state_dict(torch.load(best_model_path))
    _, test_dice, test_jaccard, test_recall, test_precision, test_hd95 = evaluate(model=teacher, device=device, loader=test_loader, with_loss=True, with_standard_metrics=True, with_hd95=True)
    print(f"Test Dice Coef: {test_dice:.4f}")
    print(f"Test Jaccard Similarity: {test_jaccard:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test HD95: {test_hd95:.4f}")

    end = time.perf_counter()

    print(f"Time: {end-start:.2f} seconds")


