import numpy as np
import os
import cv2
import random
import json

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import albumentations as A
import torch.backends.cudnn as cudnn

import itertools

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cv2.setRNGSeed(seed)
cudnn.deterministic = True
cudnn.benchmark = False
def worker_init_fn(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
torch.use_deterministic_algorithms(True, warn_only=True)
g = torch.Generator()
g.manual_seed(seed)

IMAGE_SIZE = 256
LABELED_RATIO = 1.0


geometry_transform = A.Compose([
    A.D4(p=0.5),
    A.RandomResizedCrop(scale=(0.5, 1.0), size=(IMAGE_SIZE, IMAGE_SIZE), p=0.5),
    A.Rotate(limit=(-15, 15), p=0.5)
], additional_targets={'image2':'image'})

student_color_transform = A.Compose([
    A.GaussianBlur(p=0.7, blur_limit=10),
    A.ColorJitter(p=0.7, brightness=(0.5, 1.9), contrast=(0.5, 1.9), saturation=(0.5, 1.9), hue=(-0.5, 0.5)),
    A.ToGray(p=0.7),
    A.ToTensorV2()
])

teacher_color_transform = A.Compose([
    A.GaussianBlur(p=0.1, blur_limit=2),
    A.ColorJitter(p=0.1, brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.5, 0.5)),
    A.ToGray(p=0.1),
    A.ToTensorV2()
])

no_transform = A.Compose([
    A.ToTensorV2()
])



class Custom_Dataset(Dataset):
    def __init__(self, images, masks, is_train, is_cls=False, cls=None, name=None, labeled_indices=None):

        self.images = images
        self.masks = masks
        self.is_train = is_train
        self.is_cls = is_cls
        self.cls = cls
        self.name = name

        self.labeled_indices = labeled_indices
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        is_labeled = bool(self.labeled_indices[i]) if self.labeled_indices is not None else False
        image_path = self.images[i]
        image = cv2.resize(cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (IMAGE_SIZE, IMAGE_SIZE)) / 255.0

        mask_path = self.masks[i]
        mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(np.where(mask == 0, 0.0, 1.0), -1)

        if self.is_train:

            geo_aug = geometry_transform(image=image.astype(np.float32), image2=image.astype(np.float32), mask=mask.astype(np.float32))
            student_image, teacher_image, mask = geo_aug['image'], geo_aug['image2'], torch.tensor(geo_aug['mask']).permute(2, 0, 1)

            student_color_aug = student_color_transform(image=student_image)
            teacher_color_aug = teacher_color_transform(image=teacher_image)
            student_image = student_color_aug['image']
            teacher_image = teacher_color_aug['image']

        else:
            t = no_transform(image=image.astype(np.float32), mask=mask.astype(np.float32))
            student_image = t['image']
            teacher_image = t['image']
            mask = t['mask'].permute(2, 0, 1)

        if self.is_cls:
            return student_image.float(), teacher_image.float(), mask.float(), is_labeled, torch.tensor(self.cls[i]) 
        else:

            return student_image.float(), teacher_image.float(), mask.float(), is_labeled

    
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

def get_datasets(name, labeled_indices, with_cls=False):
    # train_x, train_y, valid_x, valid_y, test_x, test_y = [], [], [], [], [], []
    # if name == 'OTU':
    #     with open('/mnt/nvme0/home/utbt/KhoaVM/OTU-2D-Dataset/OTU_2D_annotation.json', 'r') as f:
    #         data = json.load(f)

    #     for item in data:
    #         if item['split'] == 'train':
    #             train_x.append(OTU_PATH + str(item['file_path_img']))
    #             train_y.append(OTU_PATH + str(item['file_path_ann']))
    #         elif item['split'] == 'validation':
    #             valid_x.append(OTU_PATH + str(item['file_path_img']))
    #             valid_y.append(OTU_PATH + str(item['file_path_ann']))
    #         elif item['split'] == 'test':
    #             test_x.append(OTU_PATH + str(item['file_path_img']))
    #             test_y.append(OTU_PATH + str(item['file_path_ann']))

    train_x, train_y, valid_x, valid_y, test_x, test_y = [], [], [], [], [], []
    train_cls, valid_cls, test_cls = [], [], []
    

    if name == 'OTU':
        OTU_PATH = '/mnt/nvme0/home/utbt/KhoaVM/OTU-2D-Dataset/OTU_2D/'
        with open('/mnt/nvme0/home/utbt/KhoaVM/OTU-2D-Dataset/OTU_2D_850-150-469.json', 'r') as f:
            data = json.load(f)

        for item in data:
            for i in range(len(data[item])):

                if item == 'train':
                    train_x.append(OTU_PATH + str(data[item][i]['image']))
                    train_y.append(OTU_PATH + str(data[item][i]['mask']))
                    train_cls.append(int(data[item][i]['label']))
                elif item == 'val':
                    valid_x.append(OTU_PATH + str(data[item][i]['image']))
                    valid_y.append(OTU_PATH + str(data[item][i]['mask']))
                    valid_cls.append(int(data[item][i]['label']))
                elif item == 'test':
                    test_x.append(OTU_PATH + str(data[item][i]['image']))
                    test_y.append(OTU_PATH + str(data[item][i]['mask']))
                    test_cls.append(int(data[item][i]['label']))
    
    elif name == 'USOVA':
        ANNOTATOR = "ovary_r2"
        VARIANT   = "binary"
        DATASET_ROOT = "/mnt/nvme0/home/utbt/KhoaVM/USOVA3D_Dataset"
        s = '/mnt/nvme0/home/utbt/KhoaVM/USOVA3D_Dataset/split.json'

        # ── LOAD JSON ────────────────────────────────────────────────────
        with open(s, encoding="utf-8") as f:
            split = json.load(f)

        def get_split(split_name, annotator=ANNOTATOR, variant=VARIANT):
            imgs, masks = [], []
            for vol, data in split["split"][split_name].items():
                for img_rel in data["images"]:
                    imgs.append(os.path.join(DATASET_ROOT, img_rel))
                for mask_rel in data["labels"][annotator][variant]:
                    masks.append(os.path.join(DATASET_ROOT, mask_rel))
            return imgs, masks

        train_x, train_y = get_split("train")
        valid_x, valid_y = get_split("val")
        test_x,  test_y  = get_split("test")


    print(f"Dataset: {name}")
    print(f"Training data: {len(train_x)}")
    print(f"Validation data: {len(valid_x)}")
    print(f"Testing data: {len(test_x)}")

    if LABELED_RATIO != 1.0:
        labeled_train_dataset = Custom_Dataset(
            [train_x[i] for i, val in enumerate(labeled_indices) if val],
            [train_y[i] for i, val in enumerate(labeled_indices) if val],
            is_train=True,
            labeled_indices=np.ones(np.sum(labeled_indices), dtype=bool),
            name=name)

        train_dataset = Custom_Dataset(
            train_x,
            train_y,
            is_train=True,
            labeled_indices=labeled_indices,
            name=name
        )
        valid_dataset = Custom_Dataset(valid_x, valid_y, is_train=False, name=name)
        test_dataset = Custom_Dataset(test_x, test_y, is_train=False, name=name)
    
    else:
        labeled_train_dataset = Custom_Dataset(train_x, train_y, is_train=True, name=name)
        train_dataset = Custom_Dataset(train_x, train_y, is_train=True, name=name)
        valid_dataset = Custom_Dataset(valid_x, valid_y, is_train=False, name=name)
        test_dataset = Custom_Dataset(test_x, test_y, is_train=False, name=name)

    if with_cls:
        test_dataset = Custom_Dataset(test_x, test_y, is_train=False, is_cls=True, cls=test_cls)
        return test_dataset
    else:
        return labeled_train_dataset, train_dataset, valid_dataset, test_dataset

def get_dataloaders(name, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, with_cls=False):
    
    if name == 'OTU':
        len_train = 850

    else:
        len_train = 2593

    labeled_indices = np.zeros(len_train, dtype=bool)
    labeled_indices[:int(LABELED_RATIO*len_train)] = True
    np.random.shuffle(labeled_indices)

    if with_cls:
        test_dataset = get_datasets(name, with_cls)
        return DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
   
    labeled_train_dataset, train_dataset, valid_dataset, test_dataset = get_datasets(
        name=name,
        labeled_indices=labeled_indices,
        with_cls=with_cls
    )

    if LABELED_RATIO != 1.0:

        primary_indices = np.where(~labeled_indices)[0].tolist()  
        secondary_indices = np.where(labeled_indices)[0].tolist()   
        sampler = TwoStreamBatchSampler(primary_indices=primary_indices, secondary_indices=secondary_indices, batch_size=BATCH_SIZE, secondary_batch_size=int(0.25*BATCH_SIZE))

        labeled_train_loader = DataLoader(labeled_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, generator=g, worker_init_fn=worker_init_fn)
        train_loader = DataLoader(train_dataset, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, batch_sampler=sampler, generator=g, worker_init_fn=worker_init_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, generator=g, worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, generator=g, worker_init_fn=worker_init_fn)
    
    else:
        labeled_train_loader = DataLoader(labeled_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, generator=g, worker_init_fn=worker_init_fn)
        train_loader = DataLoader(train_dataset, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, generator=g, worker_init_fn=worker_init_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, generator=g, worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, generator=g, worker_init_fn=worker_init_fn)


    return labeled_train_loader, train_loader, valid_loader, test_loader


        