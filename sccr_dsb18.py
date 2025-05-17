import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from skimage import io, color
from skimage.transform import resize
from scipy.signal import convolve2d
import os
from tqdm import tqdm
from skimage.filters import sobel_h, sobel_v

# ---------------------- Data Loading and Processing for DSB 2018 ----------------------

def load_dsb_data(base_path, image_size=(128, 128)):
    images = []
    masks = []
    case_ids = os.listdir(base_path)

    for case_id in tqdm(case_ids, desc="Loading DSB Data"):
        case_path = os.path.join(base_path, case_id)
        image_path = os.path.join(case_path, "images")
        mask_path = os.path.join(case_path, "masks")

        image_files = os.listdir(image_path)
        mask_files = os.listdir(mask_path)

        if len(image_files) == 0 or len(mask_files) == 0:
            continue

        image_file = os.path.join(image_path, image_files[0])
        mask_files = [os.path.join(mask_path, f) for f in mask_files]

        img = io.imread(image_file)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if img.ndim == 3:
            img_gray = color.rgb2gray(img)
        else:
            img_gray = img

        img_resized = resize(img_gray, image_size, anti_aliasing=True)

        combined_mask = np.zeros_like(img_gray, dtype=np.float32)
        for m_file in mask_files:
            mask = io.imread(m_file)
            if mask.ndim == 3:
                mask = color.rgb2gray(mask)
            combined_mask += mask
        combined_mask = np.clip(combined_mask, 0, 1)
        mask_resized = resize(combined_mask, image_size, anti_aliasing=True)
        mask_binary = mask_resized > 0.5

        images.append(img_resized)
        masks.append(mask_binary)

    images = np.array(images)
    images = (images - images.mean()) / (images.std() + 1e-8)  # Normalize
    return images, np.array(masks)

# ---------------------- SCCR Model (unchanged) ----------------------

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

# ---------------------- Evaluation (unchanged) ----------------------

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

# ---------------------- Visualization (minor adjustment for save dir) ----------------------

def visualize_results(images, masks, predictions, num_samples=5, save_dir='results/sccr_dsb18'):
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
        plt.imshow(predictions[i] > 0.5, cmap='gray')  # Binarize
        plt.title('Prediction')
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f'prediction_result_{i}.png'))
        plt.close()

# ---------------------- Main Execution for DSB 2018 ----------------------

if __name__ == '__main__':
    base_dir = "stage1_train"  # Replace with the actual path to your DSB 2018 training data
    image_size = (64, 64)
    images, masks = load_dsb_data(base_dir, image_size=image_size)

    split_index = int(0.8 * len(images))
    train_images, test_images = images[:split_index], images[split_index:]
    train_masks, test_masks = masks[:split_index], masks[split_index:]

    num_filters = 24
    filter_size = 5
    alpha = 0.0005
    beta = 0.1
    num_iterations = 10
    learning_rate = 0.001

    filters, w, X_mean, X_std = train_sccr(
        train_images, train_masks,
        num_filters=num_filters, filter_size=filter_size,
        alpha=alpha, beta=beta,
        num_iterations=num_iterations, learning_rate=learning_rate
    )

    predictions = predict(test_images, filters, w, X_mean, X_std)
    precision, recall, f1 = compute_metrics(test_masks, predictions, threshold=0.5)
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

    visualize_results(test_images, test_masks, predictions, num_samples=5)