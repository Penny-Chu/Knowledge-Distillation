import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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

def get_transforms(img_size=224):
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

def prepare_dataloaders(train_csv_path, val_csv_path, train_img_dir, val_img_dir, batch_size=32, num_workers=4):
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    
    # 若存在 UNK 欄位，自動過濾掉未知外部樣本
    if 'UNK' in train_df.columns:
        train_df = train_df[train_df['UNK'] == 0].reset_index(drop=True)
    if 'UNK' in val_df.columns:
        val_df = val_df[val_df['UNK'] == 0].reset_index(drop=True)
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = SkinDataset(train_df, train_img_dir, transform=train_transform)
    val_dataset = SkinDataset(val_df, val_img_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader