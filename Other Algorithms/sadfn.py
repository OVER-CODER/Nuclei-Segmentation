import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ---------------------- Data Loading ----------------------

def load_data(image_dir, mask_dir, image_size=(64, 64)):
    images= []
    masks = []
    img_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for img_file, mask_file in zip(img_files, mask_files):
        img = io.imread(os.path.join(image_dir, img_file))
        mask = io.imread(os.path.join(mask_dir, mask_file))

        # Handle RGBA images by dropping alpha
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = img[:, :, :3]
            img = color.rgb2gray(img)

        img_resized = resize(img, image_size, anti_aliasing=True)
        mask_resized = resize(mask, image_size, anti_aliasing=True)
        mask_binary = (mask_resized > 0.5).astype(np.float32)

        images.append(img_resized)
        masks.append(mask_binary)

    images = np.array(images, dtype=np.float32)[..., np.newaxis]
    masks = np.array(masks, dtype=np.float32)[..., np.newaxis]
    images = (images - np.mean(images)) / (np.std(images) + 1e-8)
    return images, masks


# ---------------------- SADFN Model ----------------------

def build_sadfn(input_shape):
    input_img = tf.keras.Input(shape=input_shape)

    # Reconstruction Path
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # Segmentation Path
    y = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    y = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(y)

    # Fusion
    combined = tf.keras.layers.Concatenate()([x, y])
    output = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(combined)

    model = tf.keras.Model(inputs=input_img, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------------------- Evaluation ----------------------

def evaluate_metrics(y_true, y_pred):
    psnr_scores, ssim_scores = [], []
    for i in range(len(y_true)):
        gt = y_true[i].squeeze()
        pred = y_pred[i].squeeze()
        psnr_scores.append(psnr(gt, pred, data_range=1.0))
        ssim_scores.append(ssim(gt, pred, data_range=1.0))
    return np.mean(psnr_scores), np.mean(ssim_scores)

# ---------------------- Visualization ----------------------

def visualize_results(images, ground_truth, predictions, num_samples=5, save_dir='results/sadfn'):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(images))):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(ground_truth[i].squeeze(), cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i].squeeze(), cmap='gray')
        plt.title('Reconstructed')
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f'prediction_result_{i}.png'))
        plt.close()

# ---------------------- Main Script ----------------------

image_dir = 'TNBC_NucleiSegmentation/TNBC_NucleiSegmentation/Slide_05'
mask_dir = 'TNBC_NucleiSegmentation/TNBC_NucleiSegmentation/GT_05'

images, masks = load_data(image_dir, mask_dir, image_size=(64, 64))
split_idx = int(0.8 * len(images))
x_train, y_train = images[:split_idx], masks[:split_idx]
x_test, y_test = images[split_idx:], masks[split_idx:]

model = build_sadfn(input_shape=(64, 64, 1))
model.fit(x_train, y_train, validation_split=0.1, epochs=30, batch_size=4, verbose=1)

preds = model.predict(x_test)
psnr_avg, ssim_avg = evaluate_metrics(y_test, preds)
print(f"Avg PSNR: {psnr_avg:.2f}, Avg SSIM: {ssim_avg:.4f}")

visualize_results(x_test, y_test, preds, num_samples=5)
