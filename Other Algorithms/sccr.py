import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from skimage import io, color
from skimage.transform import resize
from scipy.signal import convolve2d
import os
from tqdm import tqdm
from skimage.filters import sobel_h, sobel_v

# ---------------------- Data Loading and Processing ----------------------

def load_data(image_dir, mask_dir, image_size=(64, 64)):
    images_original = []
    masks_original = []
    images_processed = []
    masks_processed = []
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        img_original = io.imread(img_path)
        mask_original = io.imread(mask_path)

        img = img_original.copy()
        mask = mask_original.copy()

        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if img.ndim == 3:
            img = color.rgb2gray(img)

        img_resized = resize(img, image_size, anti_aliasing=True)
        mask_resized = resize(mask, image_size, anti_aliasing=True)
        mask_binary = mask_resized > 0.5

        images_original.append(img_original)
        masks_original.append(mask_original)
        images_processed.append(img_resized)
        masks_processed.append(mask_binary)

    images_processed = np.array(images_processed)
    images_processed = (images_processed - images_processed.mean()) / (images_processed.std() + 1e-8)  # Normalize
    return images_processed, np.array(masks_processed), images_original, masks_original

# ---------------------- SCCR Model ----------------------

def initialize_filters(num_filters=8, filter_size=5):
    filters = np.random.randn(num_filters, filter_size, filter_size) * 0.01
    if num_filters >= 2:
        # Optional: include Sobel filters for edges
        filters[0] = sobel_h(np.zeros((filter_size, filter_size)))
        filters[1] = sobel_v(np.zeros((filter_size, filter_size)))
    return filters

def extract_features(images, filters):
    num_images = len(images)
    num_filters = filters.shape[0]
    img_height, img_width = images[0].shape[:2]
    features = np.zeros((num_images, num_filters, img_height, img_width))

    for i in range(num_images):
        img = images[i]
        if img.ndim == 3:
            img = color.rgb2gray(img)  # ⚠️ Keep this if filters are 2D
        for j in range(num_filters):
            features[i, j] = convolve2d(img, filters[j], mode='same', boundary='symm')
    return features


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_sccr(images, masks, num_filters=8, filter_size=5, alpha=0.01, beta=0.1, num_iterations=10, learning_rate=0.01):
    filters = initialize_filters(num_filters, filter_size)
    num_images, height, width = images.shape
    w = np.zeros(num_filters)

    for iteration in range(num_iterations):
        print(f"Iteration {iteration+1}/{num_iterations}")
        features = extract_features(images, filters)
        X = features.transpose(0, 2, 3, 1).reshape(-1, num_filters)
        y = masks.reshape(-1)

        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        X_norm = (X - X_mean) / X_std

        clf = LogisticRegression(penalty='l2', max_iter=1000, class_weight='balanced', solver='liblinear')
        clf.fit(X_norm, y)
        w = clf.coef_.flatten()

        y_pred = sigmoid(np.dot(X_norm, w))
        error = y_pred - y
        error_reshaped = error.reshape(num_images, height, width)

        for k in range(num_filters):
            grad = np.zeros_like(filters[k])
            for i in range(num_images):
                grad += convolve2d(images[i], error_reshaped[i], mode='valid')
            filters[k] -= learning_rate * (grad / num_images + beta * filters[k])

    return filters, w, X_mean, X_std

def predict(images, filters, w, X_mean, X_std):
    features = extract_features(images, filters)
    num_images, num_filters, height, width = features.shape
    X = features.transpose(0, 2, 3, 1).reshape(-1, num_filters)
    X_norm = (X - X_mean) / X_std
    y_pred = sigmoid(np.dot(X_norm, w))
    return y_pred.reshape(num_images, height, width)

# ---------------------- Evaluation ----------------------

def compute_metrics(y_true, y_pred, threshold=0.5):
    y_pred_binary = y_pred > threshold
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_binary.flatten()

    TP = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    FP = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    FN = np.sum((y_true_flat == 1) & (y_pred_flat == 0))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1_score

# ---------------------- Visualization ----------------------

def visualize_results(original_images, original_masks, predictions, num_samples=5, save_dir='results/sccr'):
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
        if original_masks[i].ndim == 3:
            plt.imshow(original_masks[i])
        else:
            plt.imshow(original_masks[i], cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        # Prediction (binarized output)
        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i] > 0.5, cmap='gray')  # Binarize
        plt.title('Prediction')
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f'prediction_result_3_{i}.png'))
        plt.close()

# ---------------------- Main Execution ----------------------

image_dir = 'TNBC_Dataset_Compiled/Slide'
mask_dir = 'TNBC_Dataset_Compiled/Masks'

images_processed, masks_processed, images_original, masks_original = load_data(image_dir, mask_dir, image_size=(64, 64))
split_index = int(0.8 * len(images_processed))
train_images_processed, test_images_processed = images_processed[:split_index], images_processed[split_index:]
train_masks_processed, test_masks_processed = masks_processed[:split_index], masks_processed[split_index:]
train_images_original, test_images_original = images_original[:split_index], images_original[split_index:]
train_masks_original, test_masks_original = masks_original[:split_index], masks_original[split_index:]

filters, w, X_mean, X_std = train_sccr(
    train_images_processed, train_masks_processed,
    num_filters=24, filter_size=5,
    alpha=0.0005, beta=0.1,
    num_iterations=100, learning_rate=0.001
)

predictions = predict(test_images_processed, filters, w, X_mean, X_std)
precision, recall, f1 = compute_metrics(test_masks_processed, predictions, threshold=0.5)
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

visualize_results(test_images_original, test_masks_original, predictions, num_samples=5)