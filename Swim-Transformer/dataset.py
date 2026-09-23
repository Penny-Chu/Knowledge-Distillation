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
        
        self.image_names = df['image'].values
        
        # 不要動態抓欄位，直接使用 Config 定義的標準順序
        self.label_cols = Config.CLASS_NAMES
        
        # 強制只取這 8 個欄位，並且確保順序與 Config.CLASS_NAMES 完全相同
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
    
    # 訓練集改回標準隨機抽樣 (shuffle=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    # 驗證集維持循序讀取 (shuffle=False)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader