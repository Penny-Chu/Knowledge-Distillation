import os
import torch

class Config:
    # --- 路徑設定 ---
    # 假設目前位於 model 資料夾內，資料夾位於上一層 (../)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # --- 各自獨立的 CSV 標籤檔路徑 (回上一層 ../ 抓取) ---
    TRAIN_CSV_PATH = os.path.join(BASE_DIR, "..", "ISIC_2019_Train_GroundTruth.csv")
    VAL_CSV_PATH   = os.path.join(BASE_DIR, "..", "ISIC_2019_Vaild_GroundTruth.csv")
    TEST_CSV_PATH  = os.path.join(BASE_DIR, "..", "ISIC_2019_Test_GroundTruth.csv")
    
    # --- 各自獨立的圖片資料夾路徑 ---
    TRAIN_IMAGE_DIR = os.path.join(BASE_DIR, "..", "ISIC_2019_Train_Dataset")
    VAL_IMAGE_DIR   = os.path.join(BASE_DIR, "..", "ISIC_2019_Vaild_Dataset")
    TEST_IMAGE_DIR  = os.path.join(BASE_DIR, "..", "ISIC_2019_Test_Dataset")
    
    # --- 權重輸出路徑 ---
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    WEIGHT_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_swin_transformer_base_model.pth")
    
    # --- 模型超參數 ---
    MODEL_NAME = "swin_base_patch4_window7_384.ms_in22k_ft_in1k"  # 使用 timm 預訓練模型
    NUM_CLASSES = 8                        # ISIC 2019 共有 8 類病灶
    IMAGE_SIZE = 384
    
    # --- 訓練超參數 ---
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    EPOCHS = 100
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 1e-2
    SEED = 42
    
    # --- 硬體設定 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")