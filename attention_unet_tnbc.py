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
        self.images = []
        self.masks = []
        image_files = sorted(os.listdir(image_dir))
        mask_files = sorted(os.listdir(mask_dir))

        for img_file, mask_file in zip(image_files, mask_files):
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

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

            img_resized = resize(img, image_size, anti_aliasing=True)
            if img_resized.ndim == 3:
                img_gray = color.rgb2gray(img_resized)
            else:
                img_gray = img_resized

            mask_resized = resize(mask, image_size, anti_aliasing=True)
            mask_binary = mask_resized > 0.5

            self.images.append(img_gray.astype(np.float32))
            self.masks.append(mask_binary.astype(np.float32))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.expand_dims(self.images[idx], axis=0)
        mask = np.expand_dims(self.masks[idx], axis=0)
        return torch.tensor(img), torch.tensor(mask)

# ---------------------- Attention U-Net Model ----------------------

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

class AttentionUNet(nn.Module):
    def __init__(self):
        super(AttentionUNet, self).__init__()

        self.enc1 = ConvBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.middle = ConvBlock(256, 512)

        self.up3 = UpConv(512, 256)
        self.attn3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = ConvBlock(256 * 2, 256)

        self.up2 = UpConv(256, 128)
        self.attn2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = ConvBlock(128 * 2, 128)

        self.up1 = UpConv(128, 64)
        self.attn1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec1 = ConvBlock(64 * 2, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        m = self.middle(p3)

        up3 = self.up3(m)
        attn3 = self.attn3(g=up3, x=e3)
        d3 = self.dec3(torch.cat([up3, attn3], dim=1))

        up2 = self.up2(d3)
        attn2 = self.attn2(g=up2, x=e2)
        d2 = self.dec2(torch.cat([up2, attn2], dim=1))

        up1 = self.up1(d2)
        attn1 = self.attn1(g=up1, x=e1)
        d1 = self.dec1(torch.cat([up1, attn1], dim=1))

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
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, mask)
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
    with torch.no_grad():
        for img, _ in test_loader:
            img = img.to(device)
            output = model(img).cpu().numpy()
            preds.extend(output)
    return np.array(preds)

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

def visualize_results(images, masks, predictions, num_samples=5, save_dir='results/attention_unet'):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(images))):
        plt.figure(figsize=(12, 4))

        # Original image in grayscale
        plt.subplot(1, 3, 1)
        plt.imshow(images[i][0], cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        # Ground truth mask (binary)
        plt.subplot(1, 3, 2)
        plt.imshow(masks[i][0], cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        # Prediction (binarized output)
        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i][0] > 0.5, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f'prediction_result_{i}.png'))
        plt.close()

# ---------------------- Main Execution ----------------------

image_dir = 'TNBC_Dataset_Compiled/Slide'
mask_dir = 'TNBC_Dataset_Compiled/Masks'

dataset = NucleiDataset(image_dir, mask_dir)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

attention_unet_model = AttentionUNet()
attention_unet_model = train_unet(attention_unet_model, train_loader, num_epochs=20, lr=0.001)

# Convert test_dataset to numpy for evaluation
test_images = [x[0].numpy() for x in test_dataset]
test_masks = [x[1].numpy() for x in test_dataset]
predictions = predict_unet(attention_unet_model, test_loader)

precision, recall, f1, accuracy = compute_metrics(np.array(test_masks), predictions, threshold=0.5)
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}')

visualize_results(test_images, test_masks, predictions, num_samples=5)