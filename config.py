class Config:
    # 直接使用相對路徑往上一層抓資料
    train_dir = "../ISIC_2019_Training_Dataset_new"
    val_dir = "../ISIC_2019_Vaild_Dataset"
    test_dir = "../ISIC_2019_Test_Dataset"
    test_csv = "../ISIC_2019_Test_GroundTruth.csv"
    
    # 權重檔存放在當前 model 資料夾內
    save_model_path = "best_model.pth"
    
    # 其他訓練參數
    img_size = (224, 224)
    batch_size = 32
    num_classes = 9
    epochs = 20
    learning_rate = 1e-4
    device = "cuda"