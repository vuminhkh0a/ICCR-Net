import numpy as np
import os
import random
import copy
import cv2

import torch
import torch.optim as optim
from data import get_dataloaders
from model import *
from loss import *

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

#====================================================================================================================================

IMAGE_SIZE = 256
NUM_WORKERS = 4
PIN_MEMORY = True
DATASET = 'USOVA'
DEVICE = 'cuda:0'



PRE_EPOCHS = 10
EPOCHS = 100


IS_TRAIN = True

MODEL = 'MT'
WITH_VGG16BN_BACKBONE = True

BATCH_SIZE = 8
CAPACITY = 400
BANK_WEIGHT_TYPE = 'Exp'
WITH_BANK_WEIGHT = True
WITH_IMB_LOSS = True
k = 0.01                

WITH_DIS_HARD_LOSS = False
NUM_CLASS = 2


#====================================================================================================================================

if __name__ == "__main__":
    
    if MODEL == 'Proposed':
        from train_proposed import *
        print("Proposed")
        device = torch.device(DEVICE)
        labeled_train_loader, train_loader, valid_loader, test_loader = get_dataloaders(DATASET, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
        rampup_length = PRE_EPOCHS * len(labeled_train_loader) + EPOCHS * len(train_loader)
        global_step = 0


        best_model_path = f"weight/usova/proposed-b{BATCH_SIZE}-c{CAPACITY}-k{k}"

        if WITH_BANK_WEIGHT:
            best_model_path += f"-BankWeight{BANK_WEIGHT_TYPE}"
            contr_loss = ContrastiveLoss(capacity=CAPACITY, bank_weight_type=BANK_WEIGHT_TYPE, device=device, k=k)

        else:
            contr_loss = ContrastiveLoss(capacity=CAPACITY, device=device)
        
        if WITH_IMB_LOSS:
            best_model_path += f"-IMBLoss"

        if WITH_DIS_HARD_LOSS:
            best_model_path += f"DisHardLoss2"

        
        student = MT_Proposed(with_tsne_emb=False, with_vgg16bn=WITH_VGG16BN_BACKBONE, num_class=NUM_CLASS).to(device)
        teacher = copy.deepcopy(student).to(device)

        best_model_path += ".pth"
        optimizer = torch.optim.NAdam(student.parameters(), lr=1e-4, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-8)


        memory_bank = torch.zeros(CAPACITY, 1024 * 8 * 8).to(device)
        cls_bank = torch.full((CAPACITY,), -1, dtype=torch.long).to(device)
        cls_proj_bank = torch.zeros(CAPACITY, NUM_CLASS).to(device)

        # init_weight(student)
        init_weight_proposed(student, with_vgg16bn=True)
        
        train_one_dataset(IS_TRAIN, PRE_EPOCHS, EPOCHS, student, teacher, global_step, rampup_length, labeled_train_loader, train_loader, valid_loader, test_loader, 
                      device, optimizer, scheduler, best_model_path,
                      contr_loss, CAPACITY, memory_bank, cls_bank, cls_proj_bank, WITH_IMB_LOSS, WITH_DIS_HARD_LOSS)
        
    elif MODEL == 'Unet':
        from train_unet import *
        print("Unet")
        best_model_path = f"weight/usova/unet100.pth"
        device = torch.device(DEVICE)
        model = VGG16BN_Unet(with_tsne_emb=False, with_vgg16bn=True).to(device)
        labeled_train_loader, train_loader, valid_loader, test_loader = get_dataloaders(DATASET, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        train_one_dataset(IS_TRAIN, model, labeled_train_loader, valid_loader, test_loader, device, optimizer, best_model_path)


    elif MODEL == 'MT':
        from train_mt import *
        print("MT")
        best_model_path = f"weight/usova/mt.pth"
        device = torch.device(DEVICE)
        student = VGG16BN_Unet(with_tsne_emb=False, with_vgg16bn=True).to(device)
        teacher = copy.deepcopy(student).to(device)
        labeled_train_loader, train_loader, valid_loader, test_loader = get_dataloaders(DATASET, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
        optimizer = optim.Adam(student.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        rampup_length = PRE_EPOCHS * len(labeled_train_loader) + EPOCHS * len(train_loader)
        global_step = 0
        train_one_dataset(IS_TRAIN, student, teacher, global_step, rampup_length, labeled_train_loader, train_loader, valid_loader, test_loader, device, optimizer, best_model_path)