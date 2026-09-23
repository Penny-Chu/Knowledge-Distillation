import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import Config
from dataset import prepare_dataloaders
from model import SkinCancerModel
from trainer import Trainer

def main():
    # 確保輸出權重的資料夾存在
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # 傳入獨立的 CSV 路徑與各自的圖片資料夾路徑
    train_loader, val_loader = prepare_dataloaders(
        train_csv_path=Config.TRAIN_CSV_PATH,
        val_csv_path=Config.VAL_CSV_PATH,
        train_img_dir=Config.TRAIN_IMAGE_DIR,
        val_img_dir=Config.VAL_IMAGE_DIR,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS
    )
    
    model = SkinCancerModel(
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=True
    ).to(Config.DEVICE)
    
    # 根據訓練集各類別樣本數計算反比權重 (Class Weights)
    train_df = pd.read_csv(Config.TRAIN_CSV_PATH)
    # 確保類別名稱排除 UNK
    class_names = Config.CLASS_NAMES
    class_counts = train_df[class_names].sum().values
    # 1. 計算總樣本數與類別總數
    total_samples = np.sum(class_counts)
    num_classes = len(class_names)

    # 將 53 倍的差距壓縮到合理的倍數範圍
    class_weights_np = total_samples / (num_classes * np.sqrt(class_counts + 1e-5))
# 歸一化，維持 Loss 尺度穩定
    class_weights_np = class_weights_np / class_weights_np.sum() * num_classes

    # 3. 轉成 PyTorch FloatTensor 並傳送至指定的裝置 (GPU/CPU)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(Config.DEVICE)
    
    # 使用加權損失函數以處理類別不平衡
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        save_path=Config.WEIGHT_SAVE_PATH
    )
    
    print(f"Start training on {Config.DEVICE} for {Config.EPOCHS} epochs...")
    trainer.fit(Config.EPOCHS)

if __name__ == "__main__":
    main()