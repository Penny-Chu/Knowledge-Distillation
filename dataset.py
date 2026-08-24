import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SkinDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
        # 取得標籤欄位 (排除 image 欄位)
        self.label_cols = [col for col in df.columns if col != 'image']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image']
        
        # 讀取對應資料夾中的圖片
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        image = Image.open(img_path).convert("RGB")
        
        # 取得類別標籤 (One-Hot 轉整數索引)
        label = row[self.label_cols].values.argmax()
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def get_transforms(img_size=224):
    # 訓練集資料增強
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 驗證/測試集轉換
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def prepare_dataloaders(train_csv_path, val_csv_path, train_img_dir, val_img_dir, batch_size=32, num_workers=4):
    # 分別讀取訓練與驗證的 CSV
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    
    train_transform, val_transform = get_transforms()
    
    # 將各自獨立的圖片目錄傳入 Dataset
    train_dataset = SkinDataset(train_df, train_img_dir, transform=train_transform)
    val_dataset = SkinDataset(val_df, val_img_dir, transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader