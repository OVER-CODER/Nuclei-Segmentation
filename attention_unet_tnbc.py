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
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, accuracy_score
import warnings

# Ignore the UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ---------------------- Data Loading and Processing ----------------------

class NucleiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(128, 128)):
        self.image_size = image_size
        self.image_paths = []
        self.mask_paths = []
        self.image_filenames = []
        image_files = sorted(os.listdir(image_dir))
        mask_files = sorted(os.listdir(mask_dir))

        for img_file, mask_file in zip(image_files, mask_files):
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)
            self.image_paths.append(img_path)
            self.mask_paths.append(mask_path)
            self.image_filenames.append(img_file)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        img_filename = self.image_filenames[idx]

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
               resize(mask, self.image_size, anti_aliasing=True), \
               img_filename

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
        for img_tensor, mask_tensor, _, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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
    predictions = []
    original_images = []
    ground_truth_masks = []
    image_filenames = []
    with torch.no_grad():
        for img_tensor, mask_tensor, original_img, original_mask, filename in tqdm(test_loader, desc="Predicting"):
            img_tensor = img_tensor.to(device)
            output = model(img_tensor).cpu().numpy()
            predictions.append((output[0], filename[0]))
            original_images.extend(original_img)
            ground_truth_masks.extend(original_mask)
            image_filenames.extend(filename)
    return predictions, original_images, ground_truth_masks, image_filenames

# ---------------------- Evaluation ----------------------

def evaluate_predictions(predictions, gt_masks):
    dice_scores = []
    iou_scores = []
    precisions = []
    recalls = []
    accuracies = []

    for pred_mask, fname in predictions:
        gt_mask = gt_masks.get(fname)
        if gt_mask is None:
            print(f"Ground truth for {fname} not found.")
            continue

        pred_mask_binary = (pred_mask > 0.5).flatten().astype(np.uint8)
        gt_mask_array = gt_mask.cpu().numpy()
        gt_mask_binary = (gt_mask_array > 0.5).flatten().astype(np.uint8)

        dice = f1_score(gt_mask_binary, pred_mask_binary)
        iou = jaccard_score(gt_mask_binary, pred_mask_binary)
        precision = precision_score(gt_mask_binary, pred_mask_binary, zero_division=0)
        recall = recall_score(gt_mask_binary, pred_mask_binary, zero_division=0)
        accuracy = accuracy_score(gt_mask_binary, pred_mask_binary)

        dice_scores.append(dice)
        iou_scores.append(iou)
        precisions.append(precision)
        recalls.append(recall)
        accuracies.append(accuracy)

    print("\nEvaluation Metrics:")
    print(f"Dice Score (F1):  {np.mean(dice_scores):.4f}")
    print(f"IoU Score:        {np.mean(iou_scores):.4f}")
    print(f"Precision:        {np.mean(precisions):.4f}")
    print(f"Recall:           {np.mean(recalls):.4f}")
    print(f"Accuracy:         {np.mean(accuracies):.4f}")

# ---------------------- Visualization ----------------------

def visualize_results(original_images, ground_truth_masks, predictions, num_samples=5, save_dir='results/attention_unet'):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(original_images))):
        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(1, 3, 1)
        img_to_plot = original_images[i].squeeze(0)
        if img_to_plot.ndim == 3:
            plt.imshow(img_to_plot)
        else:
            plt.imshow(img_to_plot, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        # Ground truth mask
        plt.subplot(1, 3, 2)
        mask_to_plot = ground_truth_masks[i].squeeze(0)
        if mask_to_plot.ndim == 3:
            plt.imshow(mask_to_plot)
        else:
            plt.imshow(mask_to_plot, cmap='gray')
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
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize model
attention_unet_model = AttentionUNet()

# Train model on training dataset
attention_unet_model = train_unet(attention_unet_model, train_loader, num_epochs=20, lr=0.001)

# Predict on test dataset
predictions, original_test_images, original_test_masks, test_filenames = predict_unet(attention_unet_model, test_loader)

# Prepare ground truth masks for evaluation
gt_masks_for_eval = {}
for mask, filename in zip(original_test_masks, test_filenames):
    gt_masks_for_eval[filename] = mask

# Evaluate predictions
evaluate_predictions(predictions, gt_masks_for_eval)

# Visualize results
visualize_results(original_test_images, original_test_masks, predictions, num_samples=5)