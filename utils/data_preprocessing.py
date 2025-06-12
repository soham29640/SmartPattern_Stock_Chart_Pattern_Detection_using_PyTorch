import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os

class ChartPatternDataset(Dataset):
    def __init__(self, image_dir, label_csv, transform=None):
        self.image_dir = image_dir
        self.labels = pd.read_csv(label_csv)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, f"{self.labels.iloc[idx, 0]}.jpg")
        image = Image.open(img_path).convert("RGB")
        label = int(self.labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label
    
image_dir = "data/processed/train_images"
label_csv = "data/processed/train_labels.csv"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = ChartPatternDataset(image_dir=image_dir, label_csv=label_csv, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

