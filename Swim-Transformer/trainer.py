import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, save_path, patience=20):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_path = os.path.abspath(os.path.normpath(save_path))
        
        # 改以 Balanced Accuracy 為監控指標（越高越好，初始為 0.0）
        self.best_val_bacc = 0.0
        self.patience = patience          # 容忍未改善的最大輪數
        self.patience_counter = 0        # 目前累計未改善的輪數

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(self.train_loader, desc=f"Epoch [{epoch}] Training", leave=False)
        for images, labels in loop:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            loop.set_postfix(loss=loss.item(), acc=correct / total)
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # 蒐集全部預測結果與真實標籤以計算平衡準確率
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # 計算驗證集的 Balanced Accuracy
        val_bacc = balanced_accuracy_score(all_labels, all_preds)
        
        # 根據 Balanced Accuracy 判定是否儲存最佳模型與重置早停計數器
        if val_bacc > self.best_val_bacc:
            self.best_val_bacc = val_bacc
            self.patience_counter = 0  # 表現刷新歷史紀錄，計數器歸零
            
            # 自動建立儲存路徑防呆
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.save_path)
            print(f"--> Saved New Best Model (Best Val Balanced Acc: {self.best_val_bacc:.4f} | Val Acc: {epoch_acc:.4f})")
        else:
            self.patience_counter += 1  # 表現未突破，累計耐心步數
            print(f"--> EarlyStopping counter: {self.patience_counter}/{self.patience}")
            
        return epoch_loss, epoch_acc, val_bacc

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc, val_bacc = self.validate(epoch)
            
            if self.scheduler:
                self.scheduler.step()
                
            print(f"Epoch {epoch:02d}/{epochs:02d} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} BAcc: {val_bacc:.4f}")
            
            # 檢查是否達到早停標準
            if self.patience_counter >= self.patience:
                print(f"\n[Early Stopping] 驗證集 Balanced Accuracy 連續 {self.patience} 輪未提升，提前終止訓練！")
                break