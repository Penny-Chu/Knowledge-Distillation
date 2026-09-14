import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# 引入 Config 以取得預設 IMAGE_SIZE
from config import Config

class SkinDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
        # 提取圖片檔名陣列
        self.image_names = df['image'].values
        
        # 取得標準 8 類標籤欄位 (排除 image 與 UNK)
        self.label_cols = [col for col in df.columns if col not in ['image', 'UNK']]
        
        # 預先計算好所有標籤索引 (避免在 __getitem__ 內使用 iloc 造成 CPU 負擔)
        self.labels = df[self.label_cols].values.argmax(axis=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 取得影像路徑並讀取
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        image = Image.open(img_path).convert("RGB")
        
        # 取得預先計算好的類別標籤
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

# 直接以 Config.IMAGE_SIZE 作為預設值，防呆且無須手動傳參
def get_transforms(img_size=Config.IMAGE_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# img_size 同樣預設吃 Config.IMAGE_SIZE
def prepare_dataloaders(train_csv_path, val_csv_path, train_img_dir, val_img_dir, batch_size, num_workers, img_size=Config.IMAGE_SIZE):
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    
    # 若存在 UNK 欄位，自動過濾掉未知外部樣本
    if 'UNK' in train_df.columns:
        train_df = train_df[train_df['UNK'] == 0].reset_index(drop=True)
    if 'UNK' in val_df.columns:
        val_df = val_df[val_df['UNK'] == 0].reset_index(drop=True)
    
    train_transform, val_transform = get_transforms(img_size=img_size)
    
    train_dataset = SkinDataset(train_df, train_img_dir, transform=train_transform)
    val_dataset = SkinDataset(val_df, val_img_dir, transform=val_transform)
    
    # ---------------- 建立 Balanced Sampler (平衡取樣器) ----------------
    # 1. 取得訓練集中每個樣本的類別標籤索引
    train_targets = train_dataset.labels
    
    # 2. 計算各類別的樣本總數
    class_counts = np.bincount(train_targets)
    
    # 3. 計算各類別權重 (樣本數倒數，避免除以 0 加上 epsilon)
    class_weights = 1.0 / (class_counts + 1e-5)
    
    # 4. 將類別權重指派給每一個獨立樣本
    sample_weights = class_weights[train_targets]
    sample_weights = torch.from_numpy(sample_weights).float()
    
    # 5. 建立 WeightedRandomSampler (replacement 必須為 True 允許重複抽取少數類)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # 訓練集使用 sampler 時，shuffle 必須設為 False
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    # 驗證集維持標準循序讀取，反映真實分佈
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader