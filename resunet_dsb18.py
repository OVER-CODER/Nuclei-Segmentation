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

# ---------------------- Dataset for DSB 2018 ----------------------

class DSB2018Dataset(Dataset):
    def __init__(self, base_path, image_size=(128, 128)):
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

        img_tensor = np.expand_dims(img_resized.astype(np.float32), axis=0)
        mask_tensor = np.expand_dims(mask_binary.astype(np.float32), axis=0)

        return torch.tensor(img_tensor), torch.tensor(mask_tensor)

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

def train_resunet(model, train_loader, num_epochs=10, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for img, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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

def predict_resunet(model, test_loader):
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

def visualize_results(images, masks, predictions, num_samples=5, save_dir='results/resunet_dsb18'):
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

# ---------------------- Main Execution for DSB 2018 ----------------------

if __name__ == '__main__':
    base_dir = "stage1_train"  # Replace with the actual path to your DSB 2018 training data
    dataset = DSB2018Dataset(base_dir, image_size=(128, 128))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=6)

    resunet_model = ResUNet()
    trained_resunet_model = train_resunet(resunet_model, train_loader, num_epochs=5, lr=0.001)

    # Convert test_dataset to numpy for evaluation and visualization
    test_images_vis = [x[0].numpy() for x in test_dataset]
    test_masks_vis = [x[1].numpy() for x in test_dataset]
    predictions_resunet = predict_resunet(trained_resunet_model, test_loader)

    precision_resunet, recall_resunet, f1_resunet, accuracy_resunet = compute_metrics(
        np.array([x[1].numpy() for x in test_dataset]), predictions_resunet, threshold=0.5
    )
    print("ResUNet Results:")
    print(f'Precision: {precision_resunet:.4f}, Recall: {recall_resunet:.4f}, F1-Score: {f1_resunet:.4f}, Accuracy: {accuracy_resunet:.4f}')

    visualize_results(test_images_vis, test_masks_vis, predictions_resunet, num_samples=5)