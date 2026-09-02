import torch
import torch.nn as nn
import timm

class SkinCancerModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(SkinCancerModel, self).__init__()
        # 載入 timm Swin Transformer 預訓練模型
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)