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

# ---------------------- U-Net Model ----------------------

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.middle = conv_block(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        m = self.middle(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

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
        gt_mask_binary = (gt_mask > 0.5).flatten().astype(np.uint8)

        dice = f1_score(gt_mask_binary, pred_mask_binary)
        iou = jaccard_score(gt_mask_binary, pred_mask_binary)
        precision = precision_score(gt_mask_binary, pred_mask_binary)
        recall = recall_score(gt_mask_binary, pred_mask_binary)
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

def visualize_results(original_images, ground_truth_masks, predictions, num_samples=5, save_dir='results/unet'):
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
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize model
unet_model = UNet()

# Train model on training dataset
unet_model = train_unet(unet_model, train_loader, num_epochs=5, lr=0.001)

# Predict on test dataset
predictions, original_test_images, original_test_masks, test_filenames = predict_unet(unet_model, test_loader)

# Prepare ground truth masks for evaluation
gt_masks_for_eval = {}
for mask, filename in zip(original_test_masks, test_filenames):
    gt_masks_for_eval[filename] = mask

# Evaluate predictions
evaluate_predictions(predictions, gt_masks_for_eval)

# Visualize results
visualize_results(original_test_images, original_test_masks, predictions, num_samples=5)