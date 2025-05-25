import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import sys
from stardist import models, data
from stardist.models import Config2D
from stardist.plot import render_label, render_label_pred
from stardist.plot import random_label_cmap, draw_polygons
from csbdeep.utils import normalize
from stardist.matching import matching
from skimage.measure import label as sklabel

# ---------------------- Data Loading and Processing ----------------------

class NucleiDatasetStardist(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256, 256)):
        self.image_size = image_size
        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        assert len(self.image_files) == len(self.mask_files), "Number of images and masks should be the same."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        img = io.imread(img_path)
        mask = io.imread(mask_path)

        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if img.ndim == 3:
            img = color.rgb2gray(img)
        img = resize(img, self.image_size, anti_aliasing=True).astype(np.float32)
        img = np.expand_dims(img, axis=-1) # Add channel dimension

        if mask.ndim == 3:
            mask = color.rgb2gray(mask)
        mask_resized = resize(mask, self.image_size, anti_aliasing=False, order=0, preserve_range=True)
        mask_binary = mask_resized > 0.5
        mask_instance = sklabel(mask_binary).astype(np.uint16)

        return img, mask_instance

# ---------------------- Training StarDist ----------------------

def train_stardist(train_loader, val_loader, model_name='stardist_nuclei', n_rays=32, epochs=10, learning_rate=1e-4):
    config = Config2D(
        n_rays=n_rays,
        n_channel_in=1,  # Assuming grayscale input images
        train_epochs=epochs,
        train_learning_rate=learning_rate,
        train_batch_size=4, # Match your DataLoader batch size
        # Add other configuration parameters as needed
    )

    model = models.StarDist2D(config, name=model_name)

    X_train = np.array([item[0] for item in train_loader.dataset])
    Y_train = np.array([item[1] for item in train_loader.dataset])
    X_val = np.array([item[0] for item in val_loader.dataset])
    Y_val = np.array([item[1] for item in val_loader.dataset])

    # Normalize images
    X_train = [normalize(x, 0, 1) for x in X_train]
    X_val = [normalize(x, 0, 1) for x in X_val]

    model.train(X_train, Y_train, validation_data=(X_val, Y_val),
                epochs=epochs) # Remove learning_rate here

    return model

# ---------------------- Prediction with StarDist ----------------------

def predict_stardist(model, test_loader):
    predictions = []
    for img, _ in tqdm(test_loader):
        img_np = img.squeeze().numpy()
        img_norm = normalize(img_np, 0, 1)
        labels, details = model.predict_instances(img_norm)
        predictions.append(labels)
    return predictions

# ---------------------- Evaluation (Instance Segmentation) ----------------------

def evaluate_stardist(ground_truth_masks, predictions, iou_threshold=0.5):
    mean_iou = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    num_samples = len(ground_truth_masks)

    for gt_inst, pred_inst in zip(ground_truth_masks, predictions):
        if gt_inst.max() > 0 or pred_inst.max() > 0:
            stats = matching(gt_inst, pred_inst, thresh=iou_threshold)
            if stats is not None:
                mean_iou += stats.mean_iou if hasattr(stats, 'mean_iou') else stats.iou.mean() if hasattr(stats, 'iou') and len(stats.iou) > 0 else 0

                tp, fp, fn = stats.tp, stats.fp, stats.fn
                total_tp += tp
                total_fp += fp
                total_fn += fn

    mean_iou /= num_samples if num_samples > 0 else 0

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (total_tp + np.sum((ground_truth_masks == 0) & (predictions == 0))) / (ground_truth_masks.size + 1e-8) # This pixel-wise accuracy is likely not meaningful for instance segmentation

    return mean_iou, precision, recall, f1_score, accuracy

# ---------------------- Visualization ----------------------

def visualize_stardist_results(images, ground_truth_masks, predictions, num_samples=5, save_dir='results/stardist'):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(images))):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(ground_truth_masks[i], cmap='nipy_spectral')
        plt.title('Ground Truth (Instance)')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i], cmap='nipy_spectral')
        plt.title('StarDist Prediction (Instance)')
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f'stardist_result_{i}.png'))
        plt.close()

# ---------------------- Main Execution ----------------------

if __name__ == '__main__':
    image_dir = 'NucleiSegmentationDataset/all_images'
    mask_dir = 'NucleiSegmentationDataset/merged_masks'

    dataset = NucleiDatasetStardist(image_dir, mask_dir, image_size=(256, 256))
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Train StarDist model
    stardist_model = train_stardist(train_loader, val_loader, epochs=10, learning_rate=1e-4)

    # Predict on the test set
    test_images = np.array([item[0] for item in test_dataset])
    test_ground_truth_masks = np.array([item[1] for item in test_dataset])
    stardist_predictions = predict_stardist(stardist_model, test_loader)

    # Evaluate (using mean IoU for instance segmentation)
    mean_iou, precision, recall, f1, accuracy = evaluate_stardist(test_ground_truth_masks, stardist_predictions, iou_threshold=0.5)
    print(f'Mean IoU on test set: {mean_iou:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}')

    # Visualize results
    visualize_stardist_results(test_images, test_ground_truth_masks, stardist_predictions, num_samples=5)

    # Optionally save the trained model
    model_save_path = 'stardist_nuclei_model.pth'
    stardist_model.keras_model.save(model_save_path)
    print(f'Trained StarDist model saved to: {model_save_path}')