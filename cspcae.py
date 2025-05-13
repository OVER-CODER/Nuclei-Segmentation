import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

# ---------------------- Sparsity Loss Function ----------------------

def sparsity_loss(rho=0.05):
    def loss(y_true, y_pred):
        reconstruction_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        hidden = K.mean(K.abs(y_pred), axis=[0, 1, 2])  # mean activation per feature map
        kl_loss = rho * K.log(rho / (hidden + 1e-10)) + (1 - rho) * K.log((1 - rho) / (1 - hidden + 1e-10))
        return reconstruction_loss + 1e-3 * K.sum(kl_loss)  # weighted sparsity loss
    return loss

# ---------------------- Data Loading and Preprocessing ----------------------

def load_data(image_dir, mask_dir, image_size=(64, 64)):
    images, masks = [], []
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for img_file, mask_file in zip(image_files, mask_files):
        img = io.imread(os.path.join(image_dir, img_file))
        mask = io.imread(os.path.join(mask_dir, mask_file))

        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        img = color.rgb2gray(img)

        img_resized = resize(img, image_size, anti_aliasing=True)
        mask_resized = resize(mask, image_size, anti_aliasing=True)
        mask_binary = mask_resized > 0.5

        images.append(img_resized)
        masks.append(mask_binary)

    images = np.expand_dims(np.array(images, dtype=np.float32), axis=-1)
    masks = np.expand_dims(np.array(masks, dtype=np.float32), axis=-1)
    return images, masks

# ---------------------- CSP-CAE Model ----------------------

def build_csp_cae(input_shape):
    input_img = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same', name="encoded")(x)

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss=sparsity_loss(rho=0.05))
    return autoencoder

# ---------------------- Evaluation ----------------------

def evaluate_model(y_true, y_pred, threshold=0.5):
    y_true_flat = y_true.flatten()
    y_pred_binary = (y_pred > threshold).astype(np.uint8).flatten()
    precision = precision_score(y_true_flat, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_binary, zero_division=0)
    return precision, recall, f1

# ---------------------- Visualization ----------------------

def visualize_predictions(images, masks, predictions, num_samples=5, save_dir='results/cspcae'):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(images))):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(masks[i].squeeze(), cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i].squeeze() > 0.5, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f'prediction_result_{i}.png'))
        plt.close()

# ---------------------- Main Execution ----------------------

image_dir = 'TNBC_NucleiSegmentation/TNBC_NucleiSegmentation/Slide_05'
mask_dir = 'TNBC_NucleiSegmentation/TNBC_NucleiSegmentation/GT_05'

images, masks = load_data(image_dir, mask_dir, image_size=(64, 64))
split_idx = int(0.8 * len(images))
train_images, test_images = images[:split_idx], images[split_idx:]
train_masks, test_masks = masks[:split_idx], masks[split_idx:]

model = build_csp_cae(input_shape=(64, 64, 1))
model.fit(train_images, train_masks, epochs=50, batch_size=8, shuffle=True, validation_split=0.1)

predictions = model.predict(test_images)
precision, recall, f1 = evaluate_model(test_masks, predictions)
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

visualize_predictions(test_images, test_masks, predictions, num_samples=5)
