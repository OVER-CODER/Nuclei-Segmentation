import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
import os
from tqdm import tqdm
from Cellpose.CCellpose import models
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------- Data Loading and Processing ----------------------

class NucleiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(128, 128)):
        self.image_size = image_size
        self.images = []
        self.masks = []
        image_files = sorted(os.listdir(image_dir))
        mask_files = sorted(os.listdir(mask_dir))

        for img_file, mask_file in zip(image_files, mask_files):
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

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

            img_resized = resize(img, image_size, anti_aliasing=True)
            if img_resized.ndim == 3:
                img_gray = color.rgb2gray(img_resized)
            else:
                img_gray = img_resized

            mask_resized = resize(mask, image_size, anti_aliasing=True)
            mask_binary = mask_resized > 0.5

            self.images.append(img_gray.astype(np.float32))
            self.masks.append(mask_binary.astype(np.float32))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.expand_dims(self.images[idx], axis=0)
        mask = np.expand_dims(self.masks[idx], axis=0)
        return torch.tensor(img), torch.tensor(mask)

# ---------------------- Cellpose Model ----------------------

def predict_cellpose(model, image_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    preds = []
    with torch.no_grad():
        for img, _ in image_loader:
            img = img.numpy()  # Convert torch tensor to numpy for Cellpose
            # Cellpose expects a (1, height, width) input in numpy format
            img = np.expand_dims(img, axis=-1)
            pred = model.eval(img)  # [0,0] as single-channel (grayscale)
            preds.extend(pred[0])  # Return the first output, which is the prediction
    return np.array(preds)

# ---------------------- Training (Cellpose) ----------------------

def train_cellpose(image_dir, mask_dir, model_name='cyto'):
    # Load the Cellpose model
    model = models.CellposeModel(gpu=True)
    return model

# ---------------------- Evaluation ----------------------

def compute_metrics(y_true, y_pred, threshold=0.5):
    y_pred_binary = y_pred > threshold
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_binary.flatten()

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

def visualize_results(images, masks, predictions, num_samples=5, save_dir='results/cellpose'):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(images))):
        plt.figure(figsize=(12, 4))

        # Original image in grayscale
        plt.subplot(1, 3, 1)
        plt.imshow(images[i][0], cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        # Ground truth mask (binary)
        plt.subplot(1, 3, 2)
        plt.imshow(masks[i][0], cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        # Prediction (binarized output)
        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i][0] > 0.5, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f'prediction_result_{i}.png'))
        plt.close()

# ---------------------- Main Execution ----------------------

image_dir = 'TNBC_Dataset_Compiled/Slide'
mask_dir = 'TNBC_Dataset_Compiled/Masks'

dataset = NucleiDataset(image_dir, mask_dir)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load Cellpose Model
cellpose_model = train_cellpose(image_dir, mask_dir)

# Convert test_dataset to numpy for evaluation
test_images = [x[0].numpy() for x in test_dataset]
test_masks = [x[1].numpy() for x in test_dataset]

# Predict using Cellpose
predictions = predict_cellpose(cellpose_model, test_loader)

# Compute metrics
precision, recall, f1, acc = compute_metrics(np.array(test_masks), predictions, threshold=0.5)
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Accuracy: {acc:.4f}')

# Visualize results
visualize_results(test_images, test_masks, predictions, num_samples=5)
