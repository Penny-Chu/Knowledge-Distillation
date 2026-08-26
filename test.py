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

def run_evaluation_and_gradcam():
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"--> Using device: {device}")

    eval_output_dir = os.path.join(Config.OUTPUT_DIR, "evaluation_results")
    cam_output_dir = os.path.join(eval_output_dir, "gradcam_visualizations")
    os.makedirs(cam_output_dir, exist_ok=True)

    _, val_transform = get_transforms()
    test_df = pd.read_csv(Config.TEST_CSV_PATH)
    if 'UNK' in test_df.columns:
        test_df = test_df[test_df['UNK'] == 0].reset_index(drop=True)
        test_df = test_df.drop(columns=['UNK'])
    test_dataset = SkinDataset(test_df, Config.TEST_IMAGE_DIR, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS
    )
    class_names = [col for col in test_df.columns if col != 'image']
    n_classes = len(class_names)
    print(f"--> Loaded {len(test_dataset)} test samples across {n_classes} classes.")

    model = SkinCancerModel(model_name=Config.MODEL_NAME, num_classes=Config.NUM_CLASSES, pretrained=False).to(device)
    if not os.path.exists(Config.WEIGHT_SAVE_PATH):
        raise FileNotFoundError(f"找不到權重檔案: {Config.WEIGHT_SAVE_PATH}，請先執行訓練！")
    
    print(f"--> Loading best weights from: {Config.WEIGHT_SAVE_PATH}")
    model.load_state_dict(torch.load(Config.WEIGHT_SAVE_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing & Inference"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 計算數值指標
    acc = accuracy_score(all_labels, all_preds)
    bacc = balanced_accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    try:
        macro_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        macro_auc = float('nan')

    print("\n" + "=" * 55)
    print("           ISIC 2019 測試集綜合評估指標           ")
    print("=" * 55)
    print(f"  * Accuracy (總體準確率)        : {acc:.4f}")
    print(f"  * Balanced Accuracy (平衡準確率): {bacc:.4f}")
    print(f"  * Precision (Macro 精確率)      : {precision:.4f}")
    print(f"  * Recall (Macro 召回率)         : {recall:.4f}")
    print(f"  * F1-Score (Macro F1分數)       : {f1:.4f}")
    print(f"  * ROC-AUC (Macro One-vs-Rest)  : {macro_auc:.4f}")
    print("=" * 55)

    print("\n[Classification Report]")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))

    # 1. 繪製並儲存混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - ISIC 2019')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_save_path = os.path.join(eval_output_dir, "confusion_matrix.png")
    plt.savefig(cm_save_path, dpi=300)
    plt.close()
    print(f"--> Confusion matrix saved to: {cm_save_path}")

    # 2. 繪製並儲存多類別 ROC 曲線圖 (ROC Curves)
    y_test_bin = label_binarize(all_labels, classes=list(range(n_classes)))
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        class_auc = calc_auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {class_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'Multi-class ROC Curve (Macro AUC = {macro_auc:.4f})')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_save_path = os.path.join(eval_output_dir, "roc_curve.png")
    plt.savefig(roc_save_path, dpi=300)
    plt.close()
    print(f"--> ROC curve image saved to: {roc_save_path}")

    # 3. Grad-CAM 視覺化
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
        pred_prob = all_probs[i][pred_label_idx]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(rgb_img)
        axes[0].set_title(f"Original\nTrue: {class_names[true_label_idx]}")
        axes[0].axis('off')

        axes[1].imshow(grayscale_cam, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis('off')

        axes[2].imshow(visualization)
        axes[2].set_title(f"Overlay\nPred: {class_names[pred_label_idx]} ({pred_prob:.2f})")
        axes[2].axis('off')

        plt.tight_layout()
        cam_save_img_path = os.path.join(cam_output_dir, f"gradcam_{i:02d}_{img_name}.png")
        plt.savefig(cam_save_img_path, dpi=300)
        plt.close()

    print(f"--> Grad-CAM visualizations saved to: {cam_output_dir}")
    print("\n✅ All evaluations and visualizations completed successfully!")

if __name__ == "__main__":
    run_evaluation_and_gradcam()