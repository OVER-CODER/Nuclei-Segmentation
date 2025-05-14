import os
import numpy as np
from skimage import io, color
from skimage.transform import resize
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------- Data Loading and Processing ----------------------

class HoVerNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(128, 128)):
        self.image_size = image_size
        self.images, self.masks = [], []

        for img_name in sorted(os.listdir(image_dir)):
            img_path = os.path.join(image_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name)

            if not os.path.exists(mask_path):
                continue

            img = io.imread(img_path)
            mask = io.imread(mask_path)

            # Remove alpha channel if present
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]

            # Convert to grayscale and resize
            img = color.rgb2gray(img)
            img = resize(img, image_size, anti_aliasing=True)
            img = np.expand_dims(img.astype(np.float32), axis=0)  # shape: [1, H, W]

            # Process mask
            if mask.ndim == 3:
                mask = color.rgb2gray(mask)
            mask = resize(mask, image_size, anti_aliasing=False)
            mask = (mask > 0.5).astype(np.uint8)
            mask = np.expand_dims(mask, axis=0)  # shape: [1, H, W]

            self.images.append(img)
            self.masks.append(mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)
        return img, mask


# ---------------------- HoVer-Net Wrapper ----------------------

from hovernet_model import HoVerNet

def load_hovernet_model(pretrained=True, nr_types=None, mode='original'):
    model = HoVerNet(num_classes=1)
    if pretrained:
        model.load_state_dict(torch.load('hovernet_fast_pannuke_type_float.pth'))  # download manually
    return model

# ---------------------- Training ----------------------

import torch.nn.functional as F

def train_hovernet(model, dataloader, num_epochs=5, lr=5e-2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for imgs, masks in tqdm(dataloader):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            output = model(imgs)['np']  # nuclear pixel prediction

            # Upsample the output to match the mask size
            output = F.interpolate(output, size=masks.shape[2:], mode='bilinear', align_corners=False)

            loss = criterion(output, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")
    return model

# ---------------------- Prediction ----------------------

def predict_hovernet(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            output = model(imgs)['np'].cpu().numpy()
            preds.extend(output)
    return np.array(preds)

# ---------------------- Evaluation ----------------------

def compute_metrics(y_true, y_pred):
    # Ensure both are flattened to 1D arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Check the shape of the flattened arrays
    print(f"y_true_flat shape: {y_true_flat.shape}")
    print(f"y_pred_flat shape: {y_pred_flat.shape}")

    # Calculate metrics
    TP = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    FP = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    FN = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
    TN = np.sum((y_true_flat == 0) & (y_pred_flat == 0))

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    accuracy = (TP + TN) / (TP + FP + FN + TN)

    return precision, recall, f1, accuracy



# ---------------------- Visualization ----------------------

def visualize_results(images, masks, predictions, save_dir='results/hovernet', num_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(images))):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(np.transpose(images[i], (1, 2, 0)))
        plt.title("Image")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(masks[i][0], cmap='gray')
        plt.title("Ground Truth")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i][0] > 0.5, cmap='gray')
        plt.title("Prediction")
        plt.axis("off")
        plt.savefig(os.path.join(save_dir, f"pred_{i}.png"))
        plt.close()

# ---------------------- Main Execution ----------------------

image_dir = 'TNBC_Dataset_Compiled/Slide'
mask_dir = 'TNBC_Dataset_Compiled/Masks'

dataset = HoVerNetDataset(image_dir, mask_dir)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

hovernet_model = load_hovernet_model(pretrained=False)  # Set True if you have weights
hovernet_model = train_hovernet(hovernet_model, train_loader, num_epochs=10)

test_images = [x[0].numpy() for x in test_dataset]
test_masks = [x[1].numpy() for x in test_dataset]
predictions = predict_hovernet(hovernet_model, test_loader)

precision, recall, f1, acc = compute_metrics(np.array(test_masks), predictions)
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Accuracy: {acc:.4f}')

visualize_results(test_images, test_masks, predictions, num_samples=5)
