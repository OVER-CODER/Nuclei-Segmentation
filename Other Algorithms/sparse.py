import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ---------------------- Data Loading and Augmentation (PyTorch Dataset) ----------------------
class NucleiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(128, 128), augment=False):
        self.image_size = image_size
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        self.augment = augment
        self.augmentation = A.Compose([
            A.Rotate(limit=90, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomCrop(width=int(image_size[0] * 0.9), height=int(image_size[1] * 0.9), p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.Resize(height=image_size[0], width=image_size[1], interpolation=0),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})
        self.transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1], interpolation=0),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img_original = io.imread(img_path)
        mask_original = io.imread(mask_path)

        img = img_original.copy()
        mask = mask_original.copy()

        if img.ndim == 3:
            if img.shape[2] == 4:
                img_original = img[:, :, :3]
            elif img.shape[2] == 1:
                img_original = np.repeat(img, 3, axis=-1)
        elif img.ndim == 2:
            img_original = color.gray2rgb(img)

        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if img.ndim == 3:
            img = color.rgb2gray(img)

        img_resized = resize(img, self.image_size, anti_aliasing=True).astype(np.float32)
        mask_resized = resize(mask, self.image_size, anti_aliasing=True).astype(np.float32)
        mask_binary = (mask_resized > 0.5).astype(np.float32)
        mask_binary = np.expand_dims(mask_binary, axis=-1).astype(np.float32)

        if self.augment:
            augmented = self.augmentation(image=img_resized, mask=mask_binary)
            img_tensor = augmented['image']
            mask_tensor = augmented['mask']
        else:
            transformed = self.transform(image=img_resized, mask=mask_binary)
            img_tensor = transformed['image']
            mask_tensor = transformed['mask']

        if mask_tensor.ndim == 3 and mask_tensor.shape[0] != 1:
            mask_tensor = mask_tensor.permute(2, 0, 1)
        elif mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        return img_tensor, mask_tensor, img_original, mask_original

# ---------------------- Supervised Deep Sparse Coding Network Components ----------------------
class SparseConvEnc(nn.Module):
    def __init__(self, in_channels, num_basis_functions, kernel_size=3, stride=1, padding=1, sparse_threshold=0.1):
        super(SparseConvEnc, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_basis_functions, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(num_basis_functions)
        self.relu = nn.ReLU(inplace=True)
        self.sparse_threshold = sparse_threshold

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        sparse_representation = F.relu(torch.abs(x) - self.sparse_threshold)
        return sparse_representation

class L1Loss(nn.Module):
    def forward(self, x):
        return torch.mean(torch.abs(x))

# ---------------------- Segmentation Head for Sparse Features ----------------------
class SparseSegmentationHead(nn.Module):
    def __init__(self, num_basis_functions, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_basis_functions, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        output = self.conv3(x)
        return output

# ---------------------- SDSCN for Segmentation ----------------------
class SDSCNSegmentation(nn.Module):
    def __init__(self, in_channels, image_size, num_basis_functions, out_channels, sparse_threshold=0.1):
        super().__init__()
        self.sparse_encoder = SparseConvEnc(in_channels, num_basis_functions, sparse_threshold=sparse_threshold)
        self.segmentation_head = SparseSegmentationHead(num_basis_functions, out_channels)
        self.sparsity_loss = L1Loss()

    def forward(self, x):
        sparse_features = self.sparse_encoder(x)
        segmentation = self.segmentation_head(sparse_features)
        sparsity = self.sparsity_loss(sparse_features)
        return segmentation, sparsity

# ---------------------- Combined Loss with Sparsity Penalty ----------------------
def combined_loss(pred, target, sparsity_loss, lambda_sparse=0.0001, smooth=1.0):
    dice = dice_loss(pred, target, smooth)
    bce = F.binary_cross_entropy_with_logits(pred, target)
    return dice + bce + lambda_sparse * sparsity_loss

# ---------------------- Dice Loss ----------------------
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred_flat = pred.contiguous().view(pred.shape[0], -1)
    target_flat = target.contiguous().view(target.shape[0], -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# ---------------------- Training Function (PyTorch) ----------------------
def train_sdscn_torch(model, train_loader, val_loader, num_epochs=10, lr=1e-3, lambda_sparse=0.0001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for img_tensor, mask_tensor, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            img_tensor, mask_tensor = img_tensor.to(device), mask_tensor.to(device)
            optimizer.zero_grad()
            output, sparsity_loss = model(img_tensor)
            loss = combined_loss(output, mask_tensor, sparsity_loss, lambda_sparse)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for img_tensor_val, mask_tensor_val, _, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                img_tensor_val, mask_tensor_val = img_tensor_val.to(device), mask_tensor_val.to(device)
                output_val, sparsity_loss_val = model(img_tensor_val)
                val_loss = combined_loss(output_val, mask_tensor_val, sparsity_loss_val, lambda_sparse)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

    return model

# ---------------------- Prediction Function (PyTorch) ----------------------
def predict_sdscn_torch(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    predictions = []
    original_images = []
    ground_truth_masks = []
    with torch.no_grad():
        for img_tensor, mask_tensor, original_img, original_mask in test_loader:
            img_tensor = img_tensor.to(device)
            output_tuple = model(img_tensor)
            output = output_tuple[0].cpu().numpy() # Access the segmentation output
            predictions.extend(output)
            original_images.extend(original_img)
            ground_truth_masks.extend(original_mask)
    return np.array(predictions), original_images, ground_truth_masks
# ---------------------- Evaluation Metrics ----------------------
def calculate_metrics(predictions, targets, threshold=0.5):
    predictions_binary = (predictions > threshold).flatten()
    targets_binary = targets.flatten().astype(int)
    precision = precision_score(targets_binary, predictions_binary, zero_division=0)
    recall = recall_score(targets_binary, predictions_binary, zero_division=0)
    f1 = f1_score(targets_binary, predictions_binary, zero_division=0)
    accuracy = accuracy_score(targets_binary, predictions_binary)
    return precision, recall, f1, accuracy

def dice_coefficient_numpy(pred, target, smooth=1.):
    pred_flat = (pred > 0.5).flatten()
    target_flat = target.flatten().astype(np.float32)
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def evaluate_model(model, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    dice_scores = []
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for img_tensor, mask_tensor, _, _ in val_loader:
            img_tensor = img_tensor.to(device)
            mask_tensor = mask_tensor.cpu().numpy()
            output_tuple = model(img_tensor)
            output = output_tuple[0].cpu().numpy()  # Access the segmentation output (first element)
            predictions_binary = (output > 0.5)
            dice = dice_coefficient_numpy(output, mask_tensor)
            dice_scores.append(dice)
            all_predictions.extend(predictions_binary)
            all_targets.extend(mask_tensor)

    avg_dice = np.mean(dice_scores)
    precision, recall, f1, accuracy = calculate_metrics(np.array(all_predictions), np.array(all_targets))
    return avg_dice, precision, recall, f1, accuracy
    return avg_dice, precision, recall, f1, accuracy

# ---------------------- Visualization ----------------------
def visualize_results(original_images, original_masks, predictions, num_samples=5, save_dir='results/sdscn_torch'):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(original_images))):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(original_images[i], cmap='gray')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(original_masks[i], cmap='gray')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        pred_mask = torch.sigmoid(torch.tensor(predictions[i])).squeeze().numpy()
        plt.imshow(pred_mask > 0.5, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sample_{i+1}.png"))
        plt.close()

# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    image_dir = 'TNBC_Dataset_Compiled/Slide'
    mask_dir = 'TNBC_Dataset_Compiled/Masks'
    image_size = (128, 128)
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.001
    num_basis_functions = 64
    out_channels = 1
    val_fraction = 0.2
    lambda_sparse = 0.0001
    sparse_threshold = 0.1

    dataset = NucleiDataset(image_dir, mask_dir, image_size=image_size, augment=True)
    train_size = int((1 - val_fraction) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    in_channels = 1
    model = SDSCNSegmentation(in_channels, image_size, num_basis_functions, out_channels, sparse_threshold)
    model = train_sdscn_torch(model, train_loader, val_loader, num_epochs=num_epochs, lr=learning_rate, lambda_sparse=lambda_sparse)

    predictions, original_test_images, original_test_masks = predict_sdscn_torch(model, test_loader)

    avg_dice, precision, recall, f1, accuracy = evaluate_model(model, test_loader)
    print(f'Dice Coefficient on Validation Set: {avg_dice:.4f}')