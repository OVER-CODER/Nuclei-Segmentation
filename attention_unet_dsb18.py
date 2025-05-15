import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

# ---------------------- Dataset for DSB 2018 for Attention U-Net ----------------------

class DSB2018DatasetForAttUNet(Dataset):
    def __init__(self, base_path, image_size=(64, 64)):  # smaller image size to speed up
        self.image_size = image_size
        self.samples = []

        for case_id in os.listdir(base_path):
            case_path = os.path.join(base_path, case_id)
            image_path = os.path.join(case_path, "images")
            mask_path = os.path.join(case_path, "masks")

            image_files = os.listdir(image_path)
            mask_files = os.listdir(mask_path)

            if len(image_files) == 0 or len(mask_files) == 0:
                continue

            image_file = os.path.join(image_path, image_files[0])
            mask_files = [os.path.join(mask_path, f) for f in mask_files]

            self.samples.append((image_file, mask_files))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_paths = self.samples[idx]

        img = io.imread(image_path)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        img_gray = color.rgb2gray(img)
        img_resized = resize(img_gray, self.image_size, anti_aliasing=True)

        combined_mask = np.zeros_like(img_gray, dtype=np.float32)
        for m_path in mask_paths:
            mask = io.imread(m_path)
            if mask.ndim == 3:
                mask = color.rgb2gray(mask)
            combined_mask += mask
        combined_mask = np.clip(combined_mask, 0, 1)

        mask_resized = resize(combined_mask, self.image_size, anti_aliasing=True)
        mask_binary = mask_resized > 0.5

        img = np.expand_dims(img_resized.astype(np.float32), axis=0)
        mask = np.expand_dims(mask_binary.astype(np.float32), axis=0)

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

def train_unet(model, loader, num_epochs=10, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(loader):.4f}")
    return model

# ---------------------- Evaluation ----------------------

def evaluate(model, dataset, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=1, shuffle=False):
            x = x.to(device)
            pred = model(x).cpu().numpy()
            y = y.numpy()
            y_true_all.append(y)
            y_pred_all.append(pred)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return compute_metrics(y_true, y_pred, threshold), y_true, y_pred

def compute_metrics(y_true, y_pred, threshold=0.5):
    y_pred_bin = y_pred > threshold
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_bin.flatten()

    TP = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    TN = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
    FP = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    FN = np.sum((y_true_flat == 1) & (y_pred_flat == 0))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    return precision, recall, accuracy, f1

# ---------------------- Visualization ----------------------

def visualize_results(images, masks, preds, save_dir='results/attention_unet_dsb2018', num_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(images))):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(images[i][0], cmap='gray')
        plt.title('Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(masks[i][0], cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(preds[i][0] > 0.5, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f'result_{i}.png'))
        plt.close()

# ---------------------- Main Execution ----------------------

if __name__ == '__main__':
    base_dir = "stage1_train"
    dataset = DSB2018DatasetForAttUNet(base_dir, image_size=(128, 128))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = AttentionUNet()
    trained_model = train_unet(model, train_loader, num_epochs=5, lr=0.001)

    metrics, y_true, y_pred = evaluate(trained_model, test_dataset)
    test_images = [x[0].numpy() for x in test_dataset]
    test_masks = [x[1].numpy() for x in test_dataset]

    visualize_results(test_images, test_masks, y_pred, num_samples=5)