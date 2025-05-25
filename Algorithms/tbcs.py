# Transformer Based Compressed Sensing
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage import io, color
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------- Data Loading with Augmentation ----------------------

class NucleiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(128, 128), grayscale=False, augment=False):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        self.image_size = image_size
        self.grayscale = grayscale
        self.augment = augment
        self.augmentation = A.Compose([
            A.Rotate(limit=90, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            A.Resize(image_size[0], image_size[1]),
            ToTensorV2(),
        ])
        self.resize_only = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        img = io.imread(img_path)
        mask = io.imread(mask_path)

        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if self.grayscale and img.ndim == 3:
            img = color.rgb2gray(img)
        elif img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        img_resized = resize(img, self.image_size, anti_aliasing=True)
        mask_resized = resize(mask, self.image_size, anti_aliasing=True)

        img_normalized = (img_resized - img_resized.mean()) / (img_resized.std() + 1e-8)

        # Convert resized mask to float tensor and then threshold
        mask_tensor = torch.tensor(mask_resized, dtype=torch.float32)
        mask_binary = (mask_tensor > 0.5).float()

        if self.augment:
            augmented = self.augmentation(image=img_normalized, mask=mask_binary.squeeze().numpy()) # Pass numpy for augmentation
            img_tensor = augmented['image'].float()
            mask_tensor_aug = augmented['mask'].float().unsqueeze(0)
            return img_tensor, mask_tensor_aug
        else:
            img_tensor = torch.tensor(img_normalized.transpose(2, 0, 1) if not self.grayscale else img_normalized.unsqueeze(0), dtype=torch.float32)
            return img_tensor, mask_binary.unsqueeze(0)

# ---------------------- Model Definition ----------------------

class CompressedSensingEncoder(nn.Module):
    def __init__(self, img_size, in_channels, measurement_dim):
        super().__init__()
        self.measurement_matrix = nn.Parameter(torch.randn(measurement_dim, in_channels * img_size ** 2, dtype=torch.float32))
        self.in_channels = in_channels
        self.img_size = img_size

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, -1)
        y = torch.matmul(x_flat, self.measurement_matrix.T)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = residual + self.dropout(attn_output)

        residual = x
        x = self.norm2(x)
        mlp_output = self.mlp(x)
        x = residual + mlp_output
        return self.norm3(x)
class TransformerCSNet(nn.Module):
    def __init__(self, measurement_dim, embed_dim=512, img_size=128, heads=8, blocks=4, out_channels=1, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_size = 16
        self.num_patches = (img_size // self.patch_size) ** 2
        self.linear_proj = nn.Linear(measurement_dim, embed_dim * self.num_patches)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.transformers = nn.Sequential(*[TransformerBlock(embed_dim, heads, dropout=dropout) for _ in range(blocks)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, y):
        B = y.shape[0]
        x = self.linear_proj(y).view(B, self.num_patches, self.embed_dim)
        x = self.dropout(x + self.pos_embedding)
        x = self.transformers(x)
        H_out = self.img_size // self.patch_size
        W_out = self.img_size // self.patch_size
        x = x.view(B, H_out, W_out, self.embed_dim).permute(0, 3, 1, 2) # B, C, H, W
        x = self.decoder(x)
        return x

# ---------------------- Training ----------------------

def train(model, encoder, loader, optimizer, criterion, device):
    model.train()
    encoder.train()
    total_loss = 0
    for img, mask in tqdm(loader):
        img, mask = img.to(device), mask.to(device)
        y = encoder(img)
        pred = model(y)
        loss = criterion(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * img.size(0)
    avg_loss = total_loss / len(loader.dataset)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss  # Add this line

# ---------------------- Evaluation ----------------------

def evaluate(model, encoder, loader, device):
    model.eval()
    encoder.eval()
    preds, masks = [], []
    with torch.no_grad():
        for img, mask in loader:
            img = img.to(device)
            y = encoder(img)
            out = model(y).cpu().numpy()
            preds.append(out)
            masks.append(mask.cpu().numpy())
    return np.vstack(preds), np.vstack(masks)

def compute_metrics(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred.flatten() > threshold).astype(np.uint8)
    y_true_binary = y_true.flatten().astype(np.uint8)
    if np.sum(y_true_binary) == 0:
        return 0.0, 0.0, 0.0
    return (
        precision_score(y_true_binary, y_pred_binary, zero_division=0),
        recall_score(y_true_binary, y_pred_binary, zero_division=0),
        f1_score(y_true_binary, y_pred_binary, zero_division=0)
    )

# ---------------------- Visualization ----------------------

def visualize(images, masks, preds, save_dir='results/transformer_cs', count=5):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(count, len(images))):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.imshow(images[i].squeeze().cpu().numpy(), cmap='gray'); plt.title('Image'); plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(masks[i].squeeze(), cmap='gray'); plt.title('Ground Truth'); plt.axis('off') # Removed .cpu().numpy()
        plt.subplot(1, 3, 3); plt.imshow(preds[i].squeeze() > 0.5, cmap='gray'); plt.title('Prediction'); plt.axis('off')
        plt.savefig(f"{save_dir}/prediction_{i}.png")
        plt.close()

# ---------------------- Main ----------------------

if __name__ == "__main__":
    image_dir = 'TNBC_Dataset_Compiled/Slide' # Replace with your actual path
    mask_dir = 'TNBC_Dataset_Compiled/Masks'   # Replace with your actual path

    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print("Error: Image or mask directory not found. Please check the paths.")
        exit()

    grayscale = True
    in_channels = 1 if grayscale else 3
    image_size = (128, 128)
    augment = True # Enable data augmentation
    dataset = NucleiDataset(image_dir, mask_dir, image_size=image_size, grayscale=grayscale, augment=augment)

    train_size = int(0.8 * len(dataset))
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    batch_size = 8 # Increase batch size
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4) # Use multiple workers
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    measurement_dim = 8192 # Try a significantly higher measurement dimension
    encoder = CompressedSensingEncoder(img_size=image_size[0], in_channels=in_channels, measurement_dim=measurement_dim).to(device)
    model = TransformerCSNet(measurement_dim=measurement_dim, img_size=image_size[0], embed_dim=512, heads=8, blocks=4, dropout=0.1).to(device) # Increased embed_dim and decoder capacity

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True) # Learning rate scheduler
    criterion = nn.BCELoss()

    num_epochs = 30 # Train for more epochs
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train(model, encoder, train_loader, optimizer, criterion, device)
        scheduler.step(train_loss) # Step the scheduler based on training loss

        if (epoch + 1) % 5 == 0:
            preds, masks = evaluate(model, encoder, test_loader, device)
            precision, recall, f1 = compute_metrics(masks, preds)
            print(f'Evaluation - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
            visualize([img for img, _ in test_set], masks, preds, count=5)

    print("\nFinal Evaluation:")
    preds, masks = evaluate(model, encoder, test_loader, device)
    precision, recall, f1 = compute_metrics(masks, preds)
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    visualize([img for img, _ in test_set], masks, preds, count=10, save_dir='results/final_transformer_cs')