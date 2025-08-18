import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from stardist import models
from stardist.models import Config2D
from csbdeep.utils import normalize
from stardist.matching import matching
from skimage.measure import label as sklabel
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------- Dataset ----------------------
class NucleiDatasetStardist(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(128, 128), train=True):
        self.image_size = image_size
        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        assert len(self.image_files) == len(self.mask_files), "Mismatch in image and mask counts"
        self.train = train
        self.augment = self._get_augmentation(train)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = io.imread(self.image_files[idx])
        mask = io.imread(self.mask_files[idx])

        # Handle channels for image
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if img.ndim == 3:
            img = color.rgb2gray(img)
        img = img.astype(np.float32)

        # Handle channels for mask
        if mask.ndim == 3:
            mask = color.rgb2gray(mask)
        mask = mask.astype(np.float32)
        mask_binary = mask > 0.5

        # Resize image and mask to fixed size
        img_resized = resize(img, self.image_size, anti_aliasing=True, preserve_range=True).astype(np.float32)
        mask_resized = resize(mask_binary.astype(np.float32), self.image_size, order=0, preserve_range=True).astype(np.float32)

        # Label instances in mask
        mask_instance = sklabel(mask_resized > 0.5).astype(np.uint16)

        # Expand dims to add channel axis: (H, W) -> (H, W, 1)
        img_expanded = np.expand_dims(img_resized, axis=-1).astype(np.float32)

        if self.augment and self.train:
            augmented = self.augment(image=img_expanded, mask=mask_instance)
            img_tensor = augmented['image']  # already tensor, shape (C, H, W)
            mask_tensor = augmented['mask'].long()
        else:
            # Normalize image (mean=0, std=1) - do this explicitly since Albumentations Normalize is not applied
            img_expanded = (img_expanded - img_expanded.mean()) / (img_expanded.std() + 1e-8)

            # Convert to tensor with shape (C, H, W)
            img_tensor = torch.tensor(img_expanded.transpose(2, 0, 1)).float()
            mask_tensor = torch.tensor(mask_instance).long()

        return img_tensor, mask_tensor

    def _get_augmentation(self, train):
        transform_list = []
        if train:
            transform_list += [
                A.Resize(*self.image_size),
                A.RandomScale(scale_limit=0.2, p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ElasticTransform(alpha=1, sigma=5, p=0.3),
                A.Normalize(mean=0.0, std=1.0),
                ToTensorV2()
            ]
        else:
            transform_list += [
                A.Resize(*self.image_size),
                A.Normalize(mean=0.0, std=1.0),
                ToTensorV2()
            ]
        return A.Compose(transform_list)

# ---------------------- Train ----------------------
def train_stardist(train_loader, val_loader, model_name='stardist_nuclei', n_rays=64, epochs=5, learning_rate=1e-4, batch_size=8, unet_n_depth=4, unet_n_filter_base=32):
    config = Config2D(
        n_rays=n_rays,
        n_channel_in=1,
        train_epochs=epochs,
        train_learning_rate=learning_rate,
        train_batch_size=batch_size,
        unet_n_depth=unet_n_depth,
        unet_n_filter_base=unet_n_filter_base,
        train_patch_size=(128, 128)
    )
    model = models.StarDist2D(config, name=model_name)

    def preprocess(loader):
        X, Y = [], []
        for imgs, masks in loader:
            for img, mask in zip(imgs, masks):
                X.append(img[0].cpu().numpy())  # img shape (C, H, W), take channel 0
                Y.append(mask.cpu().numpy())
        return np.array(X), np.array(Y)

    X_train, Y_train = preprocess(train_loader)
    X_val, Y_val = preprocess(val_loader)

    model.train(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs)
    return model

# ---------------------- Predict ----------------------
def predict_stardist(model, test_loader):
    predictions = []
    for img, _ in tqdm(test_loader):
        img_np = img.squeeze().cpu().numpy()
        labels, _ = model.predict_instances(img_np)
        predictions.append(labels)
    return predictions

# ---------------------- Evaluate ----------------------
def evaluate_stardist(ground_truth_masks, predictions, iou_threshold=0.5):
    mean_iou, tp, fp, fn = 0, 0, 0, 0
    for gt_mask, pred_mask in zip(ground_truth_masks, predictions):
        if gt_mask.max() > 0 or pred_mask.max() > 0:
            stats = matching(gt_mask, pred_mask, thresh=iou_threshold)
            mean_iou += getattr(stats, 'mean_iou', 0)
            tp += stats.tp
            fp += stats.fp
            fn += stats.fn
    num_samples = len(ground_truth_masks)
    mean_iou /= max(num_samples, 1)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return mean_iou, precision, recall, f1

# ---------------------- Visualize ----------------------
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
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i], cmap='nipy_spectral')
        plt.title('StarDist Prediction')
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f'stardist_result_{i}.png'))
        plt.close()

# ---------------------- Main ----------------------
if __name__ == '__main__':
    image_dir = 'NucleiSegmentationDataset/all_images'
    mask_dir = 'NucleiSegmentationDataset/merged_masks'

    dataset = NucleiDatasetStardist(image_dir, mask_dir, image_size=(128, 128), train=True)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Train
    model = train_stardist(train_loader, val_loader, epochs=5, batch_size=8)

    # Predict
    predictions = predict_stardist(model, test_loader)

    # Evaluate
    gt_masks = []
    for _, mask in test_loader:
        gt_masks.append(mask.squeeze().cpu().numpy())

    mean_iou, precision, recall, f1 = evaluate_stardist(gt_masks, predictions)
    print(f"Mean IoU: {mean_iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Visualize
    imgs_for_viz = []
    for img, _ in test_loader:
        imgs_for_viz.append(img.squeeze().cpu().numpy())
    visualize_stardist_results(imgs_for_viz, gt_masks, predictions)
    
    # image_dir = 'TNBC_Dataset_Compiled/Slide'
    # mask_dir = 'TNBC_Dataset_Compiled/Masks'