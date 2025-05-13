import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from stardist.models import StarDist2D
from csbdeep.utils import normalize

# ---------------------- Dataset Definition ----------------------

class NucleiDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size=(128, 128)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = io.imread(self.image_paths[idx])
        mask = io.imread(self.mask_paths[idx])

        # Drop alpha channel if present
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if mask.ndim == 3:
            if mask.shape[2] == 4:
                mask = mask[:, :, :3]
            mask = color.rgb2gray(mask)

        img_resized = resize(img, self.image_size, anti_aliasing=True)
        img_gray = color.rgb2gray(img_resized) if img_resized.ndim == 3 else img_resized

        mask_resized = resize(mask, self.image_size, anti_aliasing=True)
        mask_binary = mask_resized > 0.5

        return img_gray.astype(np.float32), mask_binary.astype(np.float32)

# ---------------------- Data Preparation ----------------------

def prepare_datasets(image_dir, mask_dir, image_size=(128, 128), test_size=0.2, batch_size=4):
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_files, mask_files, test_size=test_size, random_state=42
    )

    train_dataset = NucleiDataset(train_imgs, train_masks, image_size)
    test_dataset = NucleiDataset(test_imgs, test_masks, image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# ---------------------- StarDist Prediction ----------------------

def predict_stardist(images):
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    preds = []
    for img in tqdm(images):
        img_norm = normalize(img, 1, 99.8, clip=True)
        labels, _ = model.predict_instances(img_norm)
        preds.append((labels > 0).astype(np.float32))  # Convert instance to binary mask
    return preds

# ---------------------- Evaluation ----------------------

def compute_metrics(y_true_list, y_pred_list):
    y_true_flat = np.concatenate([y.flatten() for y in y_true_list])
    y_pred_flat = np.concatenate([y.flatten() for y in y_pred_list])

    TP = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    TN = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
    FP = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    FN = np.sum((y_true_flat == 1) & (y_pred_flat == 0))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1_score, accuracy

# ---------------------- Visualization ----------------------

def visualize_results(images, masks, predictions, num_samples=5, save_dir='results/stardist'):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(images))):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(images[i], cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(masks[i], cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i], cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f'prediction_{i}.png'))
        plt.close()

# ---------------------- Main Execution ----------------------

image_dir = 'TNBC_Dataset_Compiled/Slide'
mask_dir = 'TNBC_Dataset_Compiled/Masks'

train_loader, test_loader = prepare_datasets(image_dir, mask_dir)

test_images, test_masks = [], []
for imgs, masks in test_loader:
    test_images.extend(imgs.numpy())
    test_masks.extend(masks.numpy())

predictions = predict_stardist(test_images)

precision, recall, f1, accuracy = compute_metrics(test_masks, predictions)
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}')

visualize_results(test_images, test_masks, predictions, num_samples=5)
