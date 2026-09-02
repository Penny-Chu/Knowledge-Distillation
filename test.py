import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc as calc_auc,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import label_binarize
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import Config
from model import SkinCancerModel
from dataset import SkinDataset, get_transforms

def reshape_transform(tensor, height=7, width=7):
    if len(tensor.shape) == 4:
        result = tensor.permute(0, 3, 1, 2)
    elif len(tensor.shape) == 3:
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.permute(0, 3, 1, 2)
    else:
        result = tensor
    return result

def run_evaluation_and_gradcam(unk_threshold):
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"--> Using device: {device}")

    eval_output_dir = os.path.join(Config.OUTPUT_DIR, "evaluation_results")
    cam_output_dir = os.path.join(eval_output_dir, "gradcam_visualizations")
    os.makedirs(cam_output_dir, exist_ok=True)

    _, val_transform = get_transforms()
    test_df = pd.read_csv(Config.TEST_CSV_PATH)
    
    # 1. 定義 8 類已知病灶與包含 UNK 的 9 類標籤名稱
    known_classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    has_unk = 'UNK' in test_df.columns
    class_names = known_classes + ['UNK'] if has_unk else known_classes
    n_classes = len(class_names)
    
    test_dataset = SkinDataset(test_df, Config.TEST_IMAGE_DIR, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS
    )
    print(f"--> Loaded {len(test_dataset)} test samples across {n_classes} classes (Included UNK evaluation).")

    # 2. 載入模型權重
    model = SkinCancerModel(model_name=Config.MODEL_NAME, num_classes=Config.NUM_CLASSES, pretrained=False).to(device)
    if not os.path.exists(Config.WEIGHT_SAVE_PATH):
        raise FileNotFoundError(f"找不到權重檔案: {Config.WEIGHT_SAVE_PATH}，請先執行訓練！")
    
    print(f"--> Loading best weights from: {Config.WEIGHT_SAVE_PATH}")
    model.load_state_dict(torch.load(Config.WEIGHT_SAVE_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    # 3. 執行推論與信心門檻判定
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing & Inference"):
            images = images.to(device)
            outputs = model(images)
            
            # 計算 8 類的 Softmax 機率
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, 1)

            # 信心門檻判定：最高機率 < unk_threshold 者歸為第 9 類 UNK (索引 8)
            if has_unk:
                preds = torch.where(max_probs < unk_threshold, torch.tensor(8, device=device), preds)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 4. 計算綜合指標
    acc = accuracy_score(all_labels, all_preds)
    bacc = balanced_accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print("\n" + "=" * 55)
    print("           ISIC 2019 測試集綜合評估指標 (含 UNK 開集)           ")
    print("=" * 55)
    print(f"  * Accuracy (總體準確率)        : {acc:.4f}")
    print(f"  * Balanced Accuracy (平衡準確率): {bacc:.4f}")
    print(f"  * Precision (Macro 精確率)      : {precision:.4f}")
    print(f"  * Recall (Macro 召回率)         : {recall:.4f}")
    print(f"  * F1-Score (Macro F1分數)       : {f1:.4f}")
    print(f"  * UNK 判斷信心門檻 (Threshold)  : {unk_threshold}")
    print("=" * 55)

    print("\n[Classification Report]")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))

    # 5. 繪製混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (with UNK, Thresh={unk_threshold})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_save_path = os.path.join(eval_output_dir, "confusion_matrix_unk.png")
    plt.savefig(cm_save_path, dpi=300)
    plt.close()
    print(f"--> Confusion matrix saved to: {cm_save_path}")

    # 6. Grad-CAM 視覺化
    print("\n--> Generating Grad-CAM heatmaps...")
    target_layers = [model.model.layers[-1].blocks[-1].norm1]
    
    cam = GradCAM(
        model=model, 
        target_layers=target_layers, 
        reshape_transform=reshape_transform
    )

    num_samples_to_plot = min(10, len(test_df))
    for i in range(num_samples_to_plot):
        row = test_df.iloc[i]
        img_name = str(row['image']).strip()
        true_label_idx = row[class_names].values.argmax()
        
        img_path = os.path.join(Config.TEST_IMAGE_DIR, f"{img_name}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(Config.TEST_IMAGE_DIR, img_name)
            
        pil_img = Image.open(img_path).convert("RGB").resize((224, 224))
        rgb_img = np.float32(pil_img) / 255.0

        input_tensor, _ = test_dataset[i]
        input_tensor = input_tensor.unsqueeze(0).to(device)

        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        pred_label_idx = all_preds[i]
        pred_title = class_names[pred_label_idx]
        
        if pred_label_idx < 8:
            pred_prob = all_probs[i][pred_label_idx]
            display_title = f"{pred_title} ({pred_prob:.2f})"
        else:
            max_p = np.max(all_probs[i])
            display_title = f"UNK (Max Conf: {max_p:.2f} < {unk_threshold})"

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(rgb_img)
        axes[0].set_title(f"Original\nTrue: {class_names[true_label_idx]}")
        axes[0].axis('off')

        axes[1].imshow(grayscale_cam, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis('off')

        axes[2].imshow(visualization)
        axes[2].set_title(f"Overlay\nPred: {display_title}")
        axes[2].axis('off')

        plt.tight_layout()
        cam_save_img_path = os.path.join(cam_output_dir, f"gradcam_{i:02d}_{img_name}.png")
        plt.savefig(cam_save_img_path, dpi=300)
        plt.close()

    print(f"--> Grad-CAM visualizations saved to: {cam_output_dir}")
    print("\n✅ All evaluations and visualizations completed successfully!")

if __name__ == "__main__":
    run_evaluation_and_gradcam(unk_threshold=0.6)