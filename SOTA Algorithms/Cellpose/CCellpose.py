import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.transform import resize
from tqdm import tqdm
from cellpose import models
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, accuracy_score

def load_images_from_folder(folder, target_size=(128, 128), grayscale=True):
    images = []
    filenames = sorted(os.listdir(folder))
    for filename in filenames:
        img_path = os.path.join(folder, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            continue
        img = skio.imread(img_path)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if grayscale and img.ndim == 3:
            img = np.mean(img, axis=2)
        if not grayscale and img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        img_resized = resize(img, target_size, anti_aliasing=True)
        images.append((img_resized, filename))
    return images

def load_masks_from_folder(folder, target_size=(128, 128)):
    masks = {}
    filenames = sorted(os.listdir(folder))
    for filename in filenames:
        mask_path = os.path.join(folder, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            continue
        mask = skio.imread(mask_path)
        if mask.ndim == 3:
            mask = np.mean(mask, axis=2)
        mask_resized = resize(mask, target_size, anti_aliasing=False) > 0.5
        masks[filename] = mask_resized.astype(np.uint8)
    return masks

def run_segmentation(images, model):
    predictions = []
    for img, fname in tqdm(images, desc="Segmenting"):
        masks, flows, styles = model.eval([img], channels=[0, 0], diameter=30.0)
        predictions.append((masks[0] > 0, fname))  # binarize predicted mask
    return predictions

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

        pred_mask_flat = pred_mask.flatten().astype(np.uint8)
        gt_mask_flat = gt_mask.flatten().astype(np.uint8)

        dice = f1_score(gt_mask_flat, pred_mask_flat)
        iou = jaccard_score(gt_mask_flat, pred_mask_flat)
        precision = precision_score(gt_mask_flat, pred_mask_flat)
        recall = recall_score(gt_mask_flat, pred_mask_flat)
        accuracy = accuracy_score(gt_mask_flat, pred_mask_flat)

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

def visualize_results(original_images, ground_truth_masks, predictions, num_samples=5, save_dir='results/cellpose'):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(original_images))):
        original_img, original_fname = original_images[i]
        pred_mask, pred_fname = predictions[i]
        gt_mask = ground_truth_masks.get(original_fname)

        if gt_mask is None:
            print(f"Ground truth for {original_fname} not found for visualization.")
            continue

        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(1, 3, 1)
        if original_img.ndim == 3:
            plt.imshow(original_img)
        else:
            plt.imshow(original_img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        # Ground truth mask
        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask, cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')

        # Prediction (binarized output)
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prediction_result_{i}.png'))
        plt.close()

    # Display the first result in the notebook like the attached image
    if original_images and predictions and ground_truth_masks:
        original_img, _ = original_images[0]
        pred_mask, _ = predictions[0]
        gt_mask = ground_truth_masks.get(original_images[0][1])

        if gt_mask is not None:
            plt.figure(figsize=(15, 5))

            # Original image
            plt.subplot(1, 3, 1)
            if original_img.ndim == 3:
                plt.imshow(original_img)
            else:
                plt.imshow(original_img, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')

            # Ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask, cmap='gray')
            plt.title('Ground Truth Mask')
            plt.axis('off')

            # Prediction
            plt.subplot(1, 3, 3)
            plt.imshow(pred_mask, cmap='gray')
            plt.title('Prediction')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    image_dir = 'TestDataset/images'
    mask_dir = 'TestDataset/masks'

    images = load_images_from_folder(image_dir)
    masks = load_masks_from_folder(mask_dir)
    gt_masks = load_masks_from_folder(mask_dir)

    # Load standard Cellpose model for nuclei
    model = models.CellposeModel(gpu=True, model_type='nuclei')

    predictions = run_segmentation(images, model)
    visualize_results([img for img, _ in images], masks, predictions)

    # Evaluate predictions
    evaluate_predictions(predictions, gt_masks)