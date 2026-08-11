from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ISICDataModule:
    def __init__(self, config):
        self.config = config
        
    def get_transforms(self):
        train_transform = transforms.Compose([
            transforms.Resize(self.config.img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(self.config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return train_transform, val_transform

    def get_dataloaders(self):
        train_tf, val_tf = self.get_transforms()
        
        train_dataset = datasets.ImageFolder(root=self.config.train_dir, transform=train_tf)
        val_dataset = datasets.ImageFolder(root=self.config.val_dir, transform=val_tf)
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers
        )
        return train_loader, val_loader