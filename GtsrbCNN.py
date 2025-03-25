import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ---------------------------
# Custom Dataset
# ---------------------------
class SimpleGTSRBDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        """
        Reads the CSV file and image directory. For each row,
        it crops the image using the bounding box and resizes it to 128x128.
        Expected CSV columns:
          Filename, Roi_X1, Roi_Y1, Roi_X2, Roi_Y2, ClassId, ...
        """
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.samples = []
        for _, row in self.data.iterrows():
            filename = row['Filename']
            bbox = (row['Roi_X1'], row['Roi_Y1'], row['Roi_X2'], row['Roi_Y2'])
            label = int(row['ClassId'])
            self.samples.append((filename, bbox, label))
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, bbox, label = self.samples[idx]
        image_path = os.path.join(self.image_dir, filename)
        with Image.open(image_path).convert("RGB") as img:
            cropped = img.crop(bbox)
            resized = cropped.resize((128, 128), Image.BILINEAR)
            if self.transform is not None:
                image = self.transform(resized)
            else:
                image = transforms.ToTensor()(resized)
        return image, label

# ---------------------------
# CNN Model for 128x128 images
# ---------------------------
class GtsrbCNN(nn.Module):
    def __init__(self, n_class, img_size=128):
        super(GtsrbCNN, self).__init__()
        self.color_map = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
        self.module1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.module2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.module3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.flattened_size = self._get_flattened_size(img_size)
        self.fc1 = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Linear(1024, n_class)

    def _get_flattened_size(self, img_size):
        x = torch.zeros(1, 3, img_size, img_size)
        x = self.color_map(x)
        x1 = self.module1(x)
        x2 = self.module2(x1)
        x3 = self.module3(x2)
        return x1.view(1, -1).size(1) + x2.view(1, -1).size(1) + x3.view(1, -1).size(1)

    def forward(self, x):
        x = self.color_map(x)
        branch1 = self.module1(x)
        branch2 = self.module2(branch1)
        branch3 = self.module3(branch2)
        branch1 = branch1.view(branch1.size(0), -1)
        branch2 = branch2.view(branch2.size(0), -1)
        branch3 = branch3.view(branch3.size(0), -1)
        concat = torch.cat([branch1, branch2, branch3], dim=1)
        out = self.fc1(concat)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

# ---------------------------
# Weight Initialization
# ---------------------------
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.05)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0.05)

# ---------------------------
# Training and Evaluation Functions
# ---------------------------
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total += images.size(0)
    return total_loss / total, total_correct / total

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total += images.size(0)
    return total_loss / total, total_correct / total

