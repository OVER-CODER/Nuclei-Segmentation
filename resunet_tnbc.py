import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ---------------------- Data Loading and Processing ----------------------

class NucleiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(128, 128)):
        self.image_size = image_size
        self.image_paths = []
        self.mask_paths = []
        image_files = sorted(os.listdir(image_dir))
        mask_files = sorted(os.listdir(mask_dir))

        for img_file, mask_file in zip(image_files, mask_files):
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)
            self.image_paths.append(img_path)
            self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img = io.imread(img_path)
        mask = io.imread(mask_path)

        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]  # Drop alpha if RGBA

        if mask.ndim == 3:
            if mask.shape[2] == 4:
                mask = mask[:, :, :3]
            mask = color.rgb2gray(mask)
        elif mask.ndim == 2:
            pass  # already grayscale

        img_resized = resize(img, self.image_size, anti_aliasing=True)
        if img_resized.ndim == 3:
            img_gray = color.rgb2gray(img_resized)
        else:
            img_gray = img_resized

        mask_resized = resize(mask, self.image_size, anti_aliasing=True)
        mask_binary = mask_resized > 0.5

        return torch.tensor(np.expand_dims(img_gray.astype(np.float32), axis=0)), \
               torch.tensor(np.expand_dims(mask_binary.astype(np.float32), axis=0)), \
               resize(img, self.image_size, anti_aliasing=True), \
               resize(mask, self.image_size, anti_aliasing=True) # Return resized original images for visualization

# ---------------------- Residual U-Net (ResUNet) Model ----------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

class ResUNet(nn.Module):
    def __init__(self, num_channels=1):
        super(ResUNet, self).__init__()

        self.enc1_res1 = ResidualBlock(num_channels, 64)
        self.enc1_res2 = ResidualBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_res1 = ResidualBlock(64, 128)
        self.enc2_res2 = ResidualBlock(128, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3_res1 = ResidualBlock(128, 256)
        self.enc3_res2 = ResidualBlock(256, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.middle_res1 = ResidualBlock(256, 512)
        self.middle_res2 = ResidualBlock(512, 512)

        self.up3 = UpSample(512, 256)
        self.dec3_res1 = ResidualBlock(256 + 256, 256)
        self.dec3_res2 = ResidualBlock(256, 256)

        self.up2 = UpSample(256, 128)
        self.dec2_res1 = ResidualBlock(128 + 128, 128)
        self.dec2_res2 = ResidualBlock(128, 128)

        self.up1 = UpSample(128, 64)
        self.dec1_res1 = ResidualBlock(64 + 64, 64)
        self.dec1_res2 = ResidualBlock(64, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1_res2(self.enc1_res1(x))
        p1 = self.pool1(e1)

        e2 = self.enc2_res2(self.enc2_res1(p1))
        p2 = self.pool2(e2)

        e3 = self.enc3_res2(self.enc3_res1(p2))
        p3 = self.pool3(e3)

        # Middle
        m = self.middle_res2(self.middle_res1(p3))

        # Decoder
        up3 = self.up3(m)
        d3 = self.dec3_res2(self.dec3_res1(torch.cat([up3, e3], dim=1)))

        up2 = self.up2(d3)
        d2 = self.dec2_res2(self.dec2_res1(torch.cat([up2, e2], dim=1)))

        up1 = self.up1(d2)
        d1 = self.dec1_res2(self.dec1_res1(torch.cat([up1, e1], dim=1)))

        return torch.sigmoid(self.final(d1))

# ---------------------- Training ----------------------

def train_unet(model, train_loader, num_epochs=10, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for img_tensor, mask_tensor, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            img_tensor, mask_tensor = img_tensor.to(device), mask_tensor.to(device)
            optimizer.zero_grad()
            output = model(img_tensor)
            loss = criterion(output, mask_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")
    return model

# ---------------------- Prediction ----------------------

def predict_unet(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    preds = []
    original_images = []
    ground_truth_masks = []
    with torch.no_grad():
        for img_tensor, mask_tensor, original_img, original_mask in tqdm(test_loader, desc="Predicting"):
            img_tensor = img_tensor.to(device)
            output = model(img_tensor).cpu().numpy()
            preds.extend(output)
            original_images.extend(original_img)
            ground_truth_masks.extend(original_mask)
    return np.array(preds), original_images, ground_truth_masks

# ---------------------- Evaluation ----------------------

def compute_metrics(y_true, y_pred, threshold=0.5):
    y_pred_binary = y_pred > threshold
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_binary.flatten()

    TP = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    TN = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
    FP = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    FN = np.sum((y_true_flat == 1) & (y_pred_flat == 0))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1_score, accuracy

# ---------------------- Visualization ----------------------

def visualize_results(original_images, ground_truth_masks, predictions, num_samples=5, save_dir='results/resunet'):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(original_images))):
        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(1, 3, 1)
        if original_images[i].ndim == 3:
            plt.imshow(original_images[i])
        else:
            plt.imshow(original_images[i], cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        # Ground truth mask
        plt.subplot(1, 3, 2)
        if ground_truth_masks[i].ndim == 3:
            plt.imshow(ground_truth_masks[i])
        else:
            plt.imshow(ground_truth_masks[i], cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')

        # Prediction (binarized output)
        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i][0] > 0.5, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f'prediction_result_{i}.png'))
        plt.close()

# ---------------------- Main Execution ----------------------

train_image_dir = 'NucleiSegmentationDataset/all_images'
train_mask_dir = 'NucleiSegmentationDataset/merged_masks'
test_image_dir = 'TestDataset/images'
test_mask_dir = 'TestDataset/masks'

# Create training and test datasets
train_dataset = NucleiDataset(train_image_dir, train_mask_dir)
test_dataset = NucleiDataset(test_image_dir, test_mask_dir)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize model
model = ResUNet(num_channels=1)

# Train model on training dataset
model = train_unet(model, train_loader, num_epochs=10, lr=1e-3)

# Predict on test dataset
preds, original_images, ground_truth_masks = predict_unet(model, test_loader)

# Evaluate predictions
precision, recall, f1_score, accuracy = compute_metrics(np.array(ground_truth_masks) > 0.5, preds > 0.5)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}, Accuracy: {accuracy:.4f}")

# Visualize results
visualize_results(original_images, ground_truth_masks, preds, num_samples=5)