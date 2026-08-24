import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import Config
from dataset import prepare_dataloaders
from model import SkinCancerModel
from trainer import Trainer

def main():
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
    
    criterion = nn.CrossEntropyLoss()
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