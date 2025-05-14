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

# ---------------------- Data Loading ----------------------

class NucleiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(128, 128)):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = io.imread(self.image_paths[idx])
        mask = io.imread(self.mask_paths[idx])

        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if img.ndim == 3:
            img = color.rgb2gray(img)

        img = resize(img, self.image_size, anti_aliasing=True)
        mask = resize(mask, self.image_size, anti_aliasing=True) > 0.5

        img = (img - img.mean()) / (img.std() + 1e-8)
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0), torch.tensor(mask, dtype=torch.float32)

# ---------------------- Model Definition ----------------------

class CompressedSensingEncoder(nn.Module):
    def __init__(self, img_size, measurement_dim):
        super().__init__()
        self.measurement_matrix = nn.Parameter(torch.randn(measurement_dim, img_size ** 2))

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, -1)
        y = torch.matmul(x_flat, self.measurement_matrix.T)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        return self.norm2(x + mlp_out)

class TransformerCSNet(nn.Module):
    def __init__(self, measurement_dim, embed_dim=256, img_size=128, heads=8, blocks=4):
        super().__init__()
        self.img_size = img_size
        self.linear = nn.Linear(measurement_dim, embed_dim * (img_size // 16) ** 2)
        self.transformers = nn.Sequential(*[TransformerBlock(embed_dim, heads) for _ in range(blocks)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 2, stride=2), nn.Sigmoid()
        )

    def forward(self, y):
        B = y.shape[0]
        x = self.linear(y).view(B, -1, self.img_size // 16, self.img_size // 16)
        x = x.view(B, (self.img_size // 16) ** 2, -1)
        x = self.transformers(x)
        x = x.transpose(1, 2).view(B, -1, self.img_size // 16, self.img_size // 16)
        return self.decoder(x)

# ---------------------- Training ----------------------

def train(model, encoder, loader, optimizer, criterion, device):
    model.train()
    encoder.train()
    for img, mask in tqdm(loader):
        img, mask = img.to(device), mask.to(device)
        y = encoder(img)
        pred = model(y).squeeze(1)
        loss = criterion(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ---------------------- Evaluation ----------------------

def evaluate(model, encoder, loader, device):
    model.eval()
    encoder.eval()
    preds, masks = [], []
    with torch.no_grad():
        for img, mask in loader:
            img = img.to(device)
            y = encoder(img)
            out = model(y).squeeze(1).cpu().numpy()
            preds.append(out)
            masks.append(mask.numpy())
    return np.vstack(preds), np.vstack(masks)

def compute_metrics(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred.flatten() > threshold).astype(np.uint8)
    y_true = y_true.flatten()
    return (
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred)
    )

# ---------------------- Visualization ----------------------

def visualize(images, masks, preds, save_dir='results/transformer_cs', count=5):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(count, len(images))):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.imshow(images[i][0], cmap='gray'); plt.title('Image'); plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(masks[i], cmap='gray'); plt.title('Ground Truth'); plt.axis('off')
        plt.subplot(1, 3, 3); plt.imshow(preds[i] > 0.5, cmap='gray'); plt.title('Prediction'); plt.axis('off')
        plt.savefig(f"{save_dir}/prediction_{i}.png")
        plt.close()

# ---------------------- Main ----------------------


image_dir = 'TNBC_Dataset_Compiled/Slide'
mask_dir = 'TNBC_Dataset_Compiled/Masks'
dataset = NucleiDataset(image_dir, mask_dir, image_size=(128, 128))

train_size = int(0.8 * len(dataset))
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = CompressedSensingEncoder(img_size=128, measurement_dim=1024).to(device)
model = TransformerCSNet(measurement_dim=1024).to(device)

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(model.parameters()), lr=1e-4)
criterion = nn.BCELoss()

for epoch in range(5):
    print(f"\nEpoch {epoch+1}/5")
    train(model, encoder, train_loader, optimizer, criterion, device)

preds, masks = evaluate(model, encoder, test_loader, device)
precision, recall, f1 = compute_metrics(masks, preds)
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

visualize([img for img, _ in test_set], masks, preds)