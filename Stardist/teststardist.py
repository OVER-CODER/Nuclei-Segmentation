import os
import json
from stardist import models
from stardist.models import Config2D
from csbdeep.utils import normalize
from skimage.measure import label as sklabel
from skimage.transform import resize
from skimage import io, color
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from stardist.matching import matching
import matplotlib.pyplot as plt

# ---------------------- Data Loading and Processing for Test Set ----------------------

class TestNucleiDatasetStardist(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256, 256)):
        self.image_size = image_size
        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        assert len(self.image_files) == len(self.mask_files), "Number of test images and masks should be the same."

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
        img_resized = resize(img, self.image_size, anti_aliasing=True).astype(np.float32)
        img_expanded = np.expand_dims(img_resized, axis=-1) # Add channel dimension

        if mask.ndim == 3:
            mask = color.rgb2gray(mask)
        mask_resized = resize(mask, self.image_size, anti_aliasing=False, order=0, preserve_range=True)
        mask_binary = mask_resized > 0.5
        mask_instance = sklabel(mask_binary).astype(np.uint16)

        return img_expanded, mask_instance

# ---------------------- Prediction with Trained StarDist Model ----------------------

def predict_stardist(model, test_loader, prob_threshold=0.5, nms_threshold=0.4):
    predictions = []
    for img, _ in tqdm(test_loader, desc="Predicting on Test Set"):
        img_np = img.squeeze().numpy()
        img_norm = normalize(img_np, 0, 1)
        labels, details = model.predict_instances(img_norm, prob_thresh=prob_threshold, nms_thresh=nms_threshold)
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
    accuracy = (total_tp + np.sum((np.array(ground_truth_masks) == 0) & (np.array(predictions) == 0))) / (np.array(ground_truth_masks).size + 1e-8)

    return mean_iou, precision, recall, f1_score, accuracy

# ---------------------- Visualization ----------------------

def visualize_stardist_results(images, ground_truth_masks, predictions, num_samples=5, save_dir='results/stardist_test'):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(images))):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title('Original Image (Test)')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(ground_truth_masks[i], cmap='nipy_spectral')
        plt.title('Ground Truth (Test)')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i], cmap='nipy_spectral')
        plt.title('StarDist Prediction (Test)')
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f'stardist_test_result_{i}.png'))
        plt.close()

# ---------------------- Main Execution for Prediction and Evaluation ----------------------

if __name__ == '__main__':
    test_image_dir = 'StarTestDataset/images'
    test_mask_dir = 'StarTestDataset/masks'
    model_path = 'stardist_nuclei'  # Path to the directory containing the saved model

    # Create the test dataset and dataloader
    test_dataset = TestNucleiDatasetStardist(test_image_dir, test_mask_dir, image_size=(256, 256))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    try:
        # Load the configuration
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config = Config2D(**config_data)

        # Initialize the model
        stardist_model = models.StarDist2D(config, name='loaded_model')

        # Construct the absolute path to the weights file
        weights_path = os.path.abspath(os.path.join(model_path, 'weights_best.h5'))
        stardist_model.load_weights(weights_path)
        print(f"Loaded weights from: {weights_path}")

        # Predict on the test set
        test_images = np.array([item[0] for item in test_dataset])
        test_ground_truth_masks = np.array([item[1] for item in test_dataset])
        stardist_predictions = predict_stardist(stardist_model, test_loader, prob_threshold=0.3, nms_threshold=0.3) # Adjust thresholds if needed

        # Evaluate on the test set
        mean_iou, precision, recall, f1, accuracy = evaluate_stardist(test_ground_truth_masks, stardist_predictions, iou_threshold=0.5)
        print("\n--- Evaluation on Test Set ---")
        print(f'Mean IoU on test set: {mean_iou:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        print("--- End Evaluation ---")

        # Visualize results on the test set
        visualize_stardist_results(test_images, test_ground_truth_masks, stardist_predictions, num_samples=5)

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
    except Exception as e:
        print(f"An error occurred during loading: {e}")