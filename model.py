import torch
import torch.nn as nn
import timm

class SkinCancerModel(nn.Module):
    def __init__(self, model_name="swin_tiny_patch4_window7_224", num_classes=9, pretrained=True):
        super(SkinCancerModel, self).__init__()
        # 使用 timm 載入預訓練的 Vision Transformer 架構
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
    