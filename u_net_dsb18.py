import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ---------------------- Dataset for DSB 2018 ----------------------

class DSB2018Dataset(Dataset):
    def __init__(self, base_path, image_size=(64, 64)):  # smaller image size to speed up
        self.image_size = image_size
        self.samples = []

        for case_id in os.listdir(base_path):
            case_path = os.path.join(base_path, case_id)
            image_path = os.path.join(case_path, "images")
            mask_path = os.path.join(case_path, "masks")

            image_files = os.listdir(image_path)
            mask_files = os.listdir(mask_path)

            if len(image_files) == 0 or len(mask_files) == 0:
                continue

            image_file = os.path.join(image_path, image_files[0])
            mask_files = [os.path.join(mask_path, f) for f in mask_files]

            self.samples.append((image_file, mask_files))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_paths = self.samples[idx]

        img = io.imread(image_path)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        img_gray = color.rgb2gray(img)
        img_resized = resize(img_gray, self.image_size, anti_aliasing=True)

        combined_mask = np.zeros_like(img_gray, dtype=np.float32)
        for m_path in mask_paths:
            mask = io.imread(m_path)
            if mask.ndim == 3:
                mask = color.rgb2gray(mask)
            combined_mask += mask
        combined_mask = np.clip(combined_mask, 0, 1)

        mask_resized = resize(combined_mask, self.image_size, anti_aliasing=True)
        mask_binary = mask_resized > 0.5

        img = np.expand_dims(img_resized.astype(np.float32), axis=0)
        mask = np.expand_dims(mask_binary.astype(np.float32), axis=0)

        return torch.tensor(img), torch.tensor(mask)

# ---------------------- U-Net Model ----------------------

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.middle = conv_block(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        m = self.middle(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))

# ---------------------- Training ----------------------

def train_unet(model, loader, num_epochs=10, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Loss: {total_loss / len(loader):.4f}")
    return model

# ---------------------- Evaluation ----------------------

def evaluate(model, dataset, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=1, shuffle=False):
            x = x.to(device)
            pred = model(x).cpu().numpy()
            y = y.numpy()
            y_true_all.append(y)
            y_pred_all.append(pred)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return compute_metrics(y_true, y_pred, threshold), y_true, y_pred

def compute_metrics(y_true, y_pred, threshold=0.5):
    y_pred_bin = y_pred > threshold
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_bin.flatten()

    TP = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    TN = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
    FP = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    FN = np.sum((y_true_flat == 1) & (y_pred_flat == 0))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    return precision, recall, accuracy, f1

# ---------------------- Visualization ----------------------

def visualize_results(images, masks, preds, save_dir='results/dsb2018', num_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(images))):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(images[i][0], cmap='gray')
        plt.title('Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(masks[i][0], cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(preds[i][0] > 0.5, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f'result_{i}.png'))
        plt.close()

# ---------------------- Main ----------------------

if __name__ == '__main__':
    base_dir = "stage1_train"
    dataset = DSB2018Dataset(base_dir, image_size=(128, 128))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = UNet()
    trained_model = train_unet(model, train_loader, num_epochs=10, lr=1e-3)

    metrics, y_true, y_pred = evaluate(trained_model, test_dataset)
    test_images = [x[0].numpy() for x in test_dataset]
    test_masks = [x[1].numpy() for x in test_dataset]

    visualize_results(test_images, test_masks, y_pred, num_samples=5)







# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import io, color
# from skimage.transform import resize
# from torch.utils.data import Dataset, DataLoader
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm

# # ---------------------- Dataset for DSB 2018 ----------------------

# class DSB2018Dataset(Dataset):
#     def __init__(self, base_path, image_size=(128, 128)):
#         self.image_size = image_size
#         self.samples = []

#         for case_id in os.listdir(base_path):
#             case_path = os.path.join(base_path, case_id)
#             image_path = os.path.join(case_path, "images")
#             mask_path = os.path.join(case_path, "masks")

#             image_files = os.listdir(image_path)
#             mask_files = os.listdir(mask_path)

#             if len(image_files) == 0 or len(mask_files) == 0:
#                 continue

#             image_file = os.path.join(image_path, image_files[0])
#             mask_files = [os.path.join(mask_path, f) for f in mask_files]

#             self.samples.append((image_file, mask_files))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         image_path, mask_paths = self.samples[idx]

#         img = io.imread(image_path)
#         if img.ndim == 3 and img.shape[2] == 4:
#             img = img[:, :, :3]
#         img_gray = color.rgb2gray(img)
#         img_resized = resize(img_gray, self.image_size, anti_aliasing=True)

#         # Combine all instance masks
#         combined_mask = np.zeros_like(img_gray, dtype=np.float32)
#         for m_path in mask_paths:
#             mask = io.imread(m_path)
#             if mask.ndim == 3:
#                 mask = color.rgb2gray(mask)
#             combined_mask += mask
#         combined_mask = np.clip(combined_mask, 0, 1)  # binary

#         mask_resized = resize(combined_mask, self.image_size, anti_aliasing=True)
#         mask_binary = mask_resized > 0.5

#         img = np.expand_dims(img_resized.astype(np.float32), axis=0)
#         mask = np.expand_dims(mask_binary.astype(np.float32), axis=0)

#         return torch.tensor(img), torch.tensor(mask)

# # ---------------------- U-Net Model ----------------------

# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         def conv_block(in_ch, out_ch):
#             return nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, 3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(out_ch, out_ch, 3, padding=1),
#                 nn.ReLU(inplace=True),
#             )

#         self.enc1 = conv_block(1, 64)
#         self.pool1 = nn.MaxPool2d(2)
#         self.enc2 = conv_block(64, 128)
#         self.pool2 = nn.MaxPool2d(2)
#         self.enc3 = conv_block(128, 256)
#         self.pool3 = nn.MaxPool2d(2)

#         self.middle = conv_block(256, 512)

#         self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#         self.dec3 = conv_block(512, 256)
#         self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.dec2 = conv_block(256, 128)
#         self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.dec1 = conv_block(128, 64)

#         self.final = nn.Conv2d(64, 1, kernel_size=1)

#     def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.pool1(e1))
#         e3 = self.enc3(self.pool2(e2))
#         m = self.middle(self.pool3(e3))
#         d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
#         d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
#         d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
#         return torch.sigmoid(self.final(d1))

# # ---------------------- Training ----------------------

# def train_unet(model, loader, num_epochs=20, lr=1e-3):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.BCELoss()

#     model.train()
#     for epoch in range(num_epochs):
#         total_loss = 0
#         for x, y in tqdm(loader):
#             x, y = x.to(device), y.to(device)
#             optimizer.zero_grad()
#             pred = model(x)
#             loss = criterion(pred, y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(loader):.4f}")
#     return model

# # ---------------------- Evaluation ----------------------

# def evaluate(model, dataset):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.eval()
#     y_true_all, y_pred_all = [], []
#     with torch.no_grad():
#         for x, y in tqdm(DataLoader(dataset, batch_size=1, shuffle=False)):
#             x = x.to(device)
#             pred = model(x).cpu().numpy()
#             y = y.numpy()
#             y_true_all.append(y)
#             y_pred_all.append(pred)
#     y_true = np.concatenate(y_true_all)
#     y_pred = np.concatenate(y_pred_all)

#     return compute_metrics(y_true, y_pred)

# def compute_metrics(y_true, y_pred, threshold=0.5):
#     y_pred_bin = y_pred > threshold
#     y_true_flat = y_true.flatten()
#     y_pred_flat = y_pred_bin.flatten()

#     TP = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
#     TN = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
#     FP = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
#     FN = np.sum((y_true_flat == 1) & (y_pred_flat == 0))

#     precision = TP / (TP + FP + 1e-8)
#     recall = TP / (TP + FN + 1e-8)
#     accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
#     f1 = 2 * precision * recall / (precision + recall + 1e-8)

#     print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
#     return precision, recall, accuracy, f1

# # ---------------------- Execution ----------------------

# base_dir = r"stage1_train"
# dataset = DSB2018Dataset(base_dir)

# loader = DataLoader(dataset, batch_size=4, shuffle=True)

# model = UNet()
# trained_model = train_unet(model, loader, num_epochs=20)
# evaluate(trained_model, dataset)
