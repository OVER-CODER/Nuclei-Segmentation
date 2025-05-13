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
        img = np.expand_dims(self.images[idx], axis=0)  # Expand dims to add channel dimension
        mask = np.expand_dims(self.masks[idx], axis=0)  # Expand dims to add channel dimension
        return torch.tensor(img), torch.tensor(mask)

# ---------------------- Attention U-Net Model ----------------------

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, query, key, value):
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        attention_map = torch.matmul(query, key.transpose(1, 2))  # BxCxHxW -> BxC*HxW
        attention_map = torch.softmax(attention_map, dim=-1)

        output = torch.matmul(attention_map, value)  # BxCxHxW
        output = torch.sigmoid(output)
        return output


class AttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.att1 = AttentionBlock(64)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # Add more encoder and decoder blocks as per your architecture
        self.decoder = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass
        e1 = self.encoder(x)
        
        # Apply attention
        attention_output = self.att1(e1, e1, e1)
        
        # Decoder forward pass (up-sample)
        d1 = self.up1(attention_output)
        
        # Final output
        output = self.decoder(d1)
        
        return output




# ---------------------- Training ----------------------

# Training Function (simplified)
def train_unet(model, train_loader, num_epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        for batch_idx, (img, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
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

def compute_metrics(y_true, y_pred):
    y_true_flat = y_true.numpy().flatten()
    y_pred_flat = y_pred.numpy().flatten()

    TP = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    TN = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
    FP = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    FN = np.sum((y_true_flat == 1) & (y_pred_flat == 0))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    return precision, recall, f1_score, accuracy

# ---------------------- Visualization ----------------------

def visualize_results(images, masks, predictions, num_samples=5, save_dir='results/unet'):
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
        plt.title('Ground Truth Mask')
        plt.axis('off')

        # Predicted mask
        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i][0], cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{i}.png'))
        plt.show()

# ---------------------- Running the Code ----------------------

# Paths to your dataset
image_dir = 'TNBC_Dataset_Compiled/Slide'
mask_dir = 'TNBC_Dataset_Compiled/Masks'

# Dataset and DataLoader
dataset = NucleiDataset(image_dir, mask_dir)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model
model = AttentionUNet(in_channels=1, out_channels=1)

# Training
trained_model = train_unet(model, train_loader, num_epochs=10)

# Testing & Prediction
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
predictions = predict_unet(trained_model, test_loader)

# Compute Metrics
for i, (img, mask) in enumerate(dataset):
    precision, recall, f1_score, accuracy = compute_metrics(mask, predictions[i])
    print(f"Sample {i+1} -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}, Accuracy: {accuracy:.4f}")

# Visualize Results
visualize_results(dataset.images, dataset.masks, predictions, num_samples=5)
