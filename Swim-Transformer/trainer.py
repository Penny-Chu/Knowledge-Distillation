import os
import torch
import numpy as np
import matplotlib.pyplot as plt
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
        
        # 紀錄訓練與驗證的完整歷史數據
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_bacc': []
        }
        
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

    def plot_training_curves(self):
        # 繪製訓練與驗證的 Loss、Accuracy 曲線
        epochs_range = range(1, len(self.history['train_loss']) + 1)
        output_dir = os.path.dirname(self.save_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Loss 曲線 (Train vs Val)
        axes[0].plot(epochs_range, self.history['train_loss'], label='Train Loss', color='royalblue', linewidth=2)
        axes[0].plot(epochs_range, self.history['val_loss'], label='Val Loss', color='darkorange', linewidth=2)
        axes[0].set_title('Loss vs Epochs')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Accuracy 曲線 (Train Acc vs Val Acc vs Val Balanced Acc)
        axes[1].plot(epochs_range, self.history['train_acc'], label='Train Acc', color='royalblue', linewidth=2)
        axes[1].plot(epochs_range, self.history['val_acc'], label='Val Acc', color='darkorange', linewidth=2)
        axes[1].plot(epochs_range, self.history['val_bacc'], label='Val Balanced Acc', color='seagreen', linestyle='--', linewidth=2)
        axes[1].set_title('Accuracy vs Epochs')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        curve_save_path = os.path.join(output_dir, "training_validation_curves.png")
        plt.savefig(curve_save_path, dpi=300)
        plt.close()
        print(f"--> Training and validation curves saved to: {curve_save_path}")

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc, val_bacc = self.validate(epoch)
            
            # 存入本輪指標
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_bacc'].append(val_bacc)
            
            if self.scheduler:
                self.scheduler.step()
                
            print(f"Epoch {epoch:02d}/{epochs:02d} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} BAcc: {val_bacc:.4f}")
            
            # 檢查是否達到早停標準
            if self.patience_counter >= self.patience:
                print(f"\n[Early Stopping] 驗證集 Balanced Accuracy 連續 {self.patience} 輪未提升，提前終止訓練！")
                break
                
        # 訓練完成或觸發早停時輸出折線圖
        self.plot_training_curves()