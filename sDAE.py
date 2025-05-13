import os
import numpy as np
from skimage import io, color
from skimage.transform import resize
from sklearn.decomposition import DictionaryLearning
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# ---------------------- Data Loading ----------------------
def load_data(image_dir, mask_dir, image_size=(32, 32)):
    images = []
    masks = []
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        img = io.imread(img_path)
        mask = io.imread(mask_path)

        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        img = color.rgb2gray(img)

        img_resized = resize(img, image_size, anti_aliasing=True)
        mask_resized = resize(mask, image_size, anti_aliasing=True)
        mask_binary = (mask_resized > 0.5).astype(np.float32)

        images.append(img_resized)
        masks.append(mask_binary)

    images = np.array(images)
    masks = np.array(masks)
    return images, masks

# ---------------------- Sparse Coding ----------------------
def perform_sparse_coding(images, n_components=100):
    n_samples = len(images)
    data_flat = images.reshape((n_samples, -1))
    dict_learner = DictionaryLearning(n_components=n_components, transform_algorithm='lasso_lars', max_iter=100)
    sparse_codes = dict_learner.fit_transform(data_flat)
    return sparse_codes, dict_learner.components_

# ---------------------- SDAE Model ----------------------
def build_sdae(input_dim):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary output
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ---------------------- Evaluation ----------------------
def evaluate_model(y_true, y_pred, threshold=0.5):
    y_true_flat = y_true.flatten()
    y_pred_binary = (y_pred.flatten() > threshold).astype(np.uint8)
    precision = precision_score(y_true_flat, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_binary, zero_division=0)
    return precision, recall, f1

# ---------------------- Main ----------------------
image_dir = 'TNBC_NucleiSegmentation/TNBC_NucleiSegmentation/Slide_05'
mask_dir = 'TNBC_NucleiSegmentation/TNBC_NucleiSegmentation/GT_05'

images, masks = load_data(image_dir, mask_dir, image_size=(32, 32))
sparse_codes, dictionary = perform_sparse_coding(images, n_components=100)

split_idx = int(0.8 * len(images))
X_train, X_test = sparse_codes[:split_idx], sparse_codes[split_idx:]
y_train = masks[:split_idx].reshape((len(X_train), -1)).mean(axis=1)  # Simplified label: % of mask
y_test = masks[split_idx:].reshape((len(X_test), -1)).mean(axis=1)

model = build_sdae(input_dim=100)
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.1, verbose=1)

y_pred = model.predict(X_test)
precision, recall, f1 = evaluate_model(y_test > 0.1, y_pred > 0.1)
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
