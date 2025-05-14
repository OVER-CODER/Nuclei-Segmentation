import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# --- Create sensing matrix ---
def create_sensing_matrix(output_size, compressed_size, seed=42):
    np.random.seed(seed)
    sensing_matrix = np.random.randn(compressed_size, output_size)
    sensing_matrix /= np.linalg.norm(sensing_matrix, axis=1, keepdims=True)
    return sensing_matrix

# --- Compress labels ---
def compress_labels(labels, sensing_matrix):
    return np.dot(labels, sensing_matrix.T)

# --- Build CNN model ---
def build_cnn(input_shape, compressed_size):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(compressed_size, activation='linear'))
    model.compile(optimizer=Adam(1e-4), loss='mse')
    return model

# --- Recover labels using Lasso ---
def recover_labels_batch(predicted_vectors, sensing_matrix, alpha=0.0001):
    recovered_labels = []
    for vec in predicted_vectors:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(sensing_matrix, vec)
        recovered_labels.append(lasso.coef_)
    return np.array(recovered_labels)

# --- Evaluation ---
def evaluate(y_true, y_pred, threshold=0.5):
    y_true_bin = (y_true > threshold).astype(int)
    y_pred_bin = (y_pred > threshold).astype(int)
    precision = precision_score(y_true_bin.flatten(), y_pred_bin.flatten(), zero_division=0)
    recall = recall_score(y_true_bin.flatten(), y_pred_bin.flatten(), zero_division=0)
    f1 = f1_score(y_true_bin.flatten(), y_pred_bin.flatten(), zero_division=0)
    return precision, recall, f1

# --- Visualization ---
def visualize_results(images, masks, predictions, num_samples=5, save_dir='results/cnncs'):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(images))):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(masks[i].reshape(64, 64), cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i].reshape(64, 64) > 0.5, cmap='gray')
        plt.title('Prediction (Thresh > 0.5)')
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f'cnncs_prediction_{i}.png'))
        plt.close()

# ------------------------- MAIN PIPELINE -------------------------

# Dummy data simulation (replace with actual data loading)
n_samples = 100
input_shape = (64, 64, 1)
output_size = 64 * 64  # 4096 pixels
compressed_size = 200  # Compressed dimension

# Generate dummy input images and masks
X_train = np.random.rand(n_samples, *input_shape).astype(np.float32)
Y_train_sparse = np.random.randint(0, 2, size=(n_samples, output_size))  # binary mask

X_val = np.random.rand(10, *input_shape).astype(np.float32)
Y_val_sparse = np.random.randint(0, 2, size=(10, output_size))

# Compressed sensing
A = create_sensing_matrix(output_size, compressed_size)
Y_train_compressed = compress_labels(Y_train_sparse, A)
Y_val_compressed = compress_labels(Y_val_sparse, A)

# Train model
model = build_cnn(input_shape, compressed_size)
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(X_train, Y_train_compressed, validation_data=(X_val, Y_val_compressed),
                    epochs=50, batch_size=8, callbacks=[early_stop])

# Predict and recover
predicted_compressed = model.predict(X_val)
Y_pred_sparse = recover_labels_batch(predicted_compressed, A)

# Evaluate
precision, recall, f1 = evaluate(Y_val_sparse, Y_pred_sparse)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Visualize and save
visualize_results(X_val, Y_val_sparse, Y_pred_sparse, num_samples=5, save_dir='results/cnncs')
