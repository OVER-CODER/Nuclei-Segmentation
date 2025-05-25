import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ---------------------- Hierarchical Multi-Scale Attention Module (HMSAM) ----------------------

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels_1x1, red_channels_3x3, out_channels_3x3, red_channels_5x5, out_channels_5x5, pool_proj):
        super(InceptionModule, self).__init__()
        # Branch 1: 1x1 convolution
        self.branch1 = nn.Conv2d(in_channels, out_channels_1x1, kernel_size=1)

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2_red = nn.Conv2d(in_channels, red_channels_3x3, kernel_size=1)
        self.branch2_conv = nn.Conv2d(red_channels_3x3, out_channels_3x3, kernel_size=3, padding=1)

        # Branch 3: 1x1 convolution followed by 5x5 convolution
        self.branch3_red = nn.Conv2d(in_channels, red_channels_5x5, kernel_size=1)
        self.branch3_conv = nn.Conv2d(red_channels_5x5, out_channels_5x5, kernel_size=5, padding=2)

        # Branch 4: Max pooling followed by 1x1 convolution
        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_proj = nn.Conv2d(in_channels, pool_proj, kernel_size=1)

    def forward(self, x):
        branch1_out = F.relu(self.branch1(x))
        branch2_out = F.relu(self.branch2_conv(F.relu(self.branch2_red(x))))
        branch3_out = F.relu(self.branch3_conv(F.relu(self.branch3_red(x))))
        branch4_out = F.relu(self.branch4_proj(self.branch4_pool(x)))
        
        # Concatenate outputs along the channel dimension
        outputs = [branch1_out, branch2_out, branch3_out, branch4_out]
        return torch.cat(outputs, 1)

class HierarchicalAttentionMechanism(nn.Module):
    def __init__(self, in_channels):
        super(HierarchicalAttentionMechanism, self).__init__()
        # Channel Attention Mechanism
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Global Average Pooling
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1), # Reduce channels
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1), # Restore channels
            nn.Sigmoid() # Activation for attention weights
        )
        # Spatial Attention Mechanism
        self.spatial_attention = nn.Conv2d(in_channels, 1, kernel_size=1) # 1x1 conv to get spatial weights

    def forward(self, x):
        # Apply channel attention
        channel_attn = self.channel_attention(x)
        
        # Apply spatial attention
        spatial_attn = torch.sigmoid(self.spatial_attention(x))
        
        # Combine attention features with the input
        # Note: x * channel_attn will broadcast channel_attn across spatial dimensions
        # x * spatial_attn will broadcast spatial_attn across channel dimensions
        out = x * channel_attn + x * spatial_attn
        return out

class HMSAModule(nn.Module):
    def __init__(self, in_channels, out_channels_1x1, red_channels_3x3, out_channels_3x3, red_channels_5x5, out_channels_5x5, pool_proj):
        super(HMSAModule, self).__init__()
        # Initialize Inception Module
        self.inception = InceptionModule(in_channels, out_channels_1x1, red_channels_3x3, out_channels_3x3, red_channels_5x5, out_channels_5x5, pool_proj)
        
        # Calculate the total output channels from the Inception Module
        # This will be the input channels for the Hierarchical Attention Mechanism
        inception_output_channels = out_channels_1x1 + out_channels_3x3 + out_channels_5x5 + pool_proj
        
        # Initialize Hierarchical Attention Mechanism
        self.attention = HierarchicalAttentionMechanism(inception_output_channels)

        # Ensure the output of attention matches the input for residual connection
        # If inception_output_channels != in_channels, we need a projection layer
        if inception_output_channels != in_channels:
            self.projection = nn.Conv2d(inception_output_channels, in_channels, kernel_size=1)
        else:
            self.projection = None

    def forward(self, x):
        inception_out = self.inception(x)
        attention_out = self.attention(inception_out)
        
        # Apply projection if necessary to match input channels for residual connection
        if self.projection:
            attention_out = self.projection(attention_out)
            
        # Residual connection: add input 'x' to the attention-weighted output
        # Now, x and attention_out should have the same number of channels
        hmsam_output = x + attention_out 
        return hmsam_output

# ---------------------- HMSAM-UNet Model ----------------------

class HMSAM_UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(HMSAM_UNet, self).__init__()
        # Helper function for double convolution block
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        # Encoder (Downsampling Path)
        self.enc1 = double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2) # Output of pool4 has 512 channels

        # HMSAM module at the bottleneck
        # The output channels of InceptionModule must sum to 512 for the residual connection
        # Example channel distribution for Inception to sum to 512:
        # 1x1: 128, 3x3: 192, 5x5: 64, Pool: 128 => 128+192+64+128 = 512
        self.hmsam = HMSAModule(in_channels=512, 
                                out_channels_1x1=128, 
                                red_channels_3x3=96, out_channels_3x3=192, # red_channels_3x3 < out_channels_3x3
                                red_channels_5x5=32, out_channels_5x5=64, # red_channels_5x5 < out_channels_5x5
                                pool_proj=128) # pool_proj should be <= in_channels

        # Decoder (Upsampling Path)
        # First up-convolution from HMSAM output (512 channels)
        self.upconv4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        # Concatenation with enc4 (512 channels) -> 512 (upconv) + 512 (enc4) = 1024
        self.dec4 = double_conv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # Concatenation with enc3 (256 channels) -> 256 (upconv) + 256 (enc3) = 512
        self.dec3 = double_conv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Concatenation with enc2 (128 channels) -> 128 (upconv) + 128 (enc2) = 256
        self.dec2 = double_conv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Concatenation with enc1 (64 channels) -> 64 (upconv) + 64 (enc1) = 128
        self.dec1 = double_conv(128, 64)

        # Output layer
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        bridge = self.pool4(enc4) # Bridge has 512 channels

        # HMSAM bottleneck
        hmsam_out = self.hmsam(bridge) # hmsam_out also has 512 channels due to projection/matching

        # Decoder
        # Upsample hmsam_out and concatenate with enc4
        dec4 = self.dec4(torch.cat([self.upconv4(hmsam_out), enc4], dim=1))
        # Upsample dec4 and concatenate with enc3
        dec3 = self.dec3(torch.cat([self.upconv3(dec4), enc3], dim=1))
        # Upsample dec3 and concatenate with enc2
        dec2 = self.dec2(torch.cat([self.upconv2(dec3), enc2], dim=1))
        # Upsample dec2 and concatenate with enc1
        dec1 = self.dec1(torch.cat([self.upconv1(dec2), enc1], dim=1))

        # Final output layer with sigmoid for binary segmentation
        out = torch.sigmoid(self.outc(dec1))
        return out

# ---------------------- Data Loading and Processing (Adapt for your dataset) ----------------------

class NucleiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(128, 128)):
        self.image_size = image_size
        self.image_paths = []
        self.mask_paths = []
        image_files = sorted(os.listdir(image_dir))
        mask_files = sorted(os.listdir(mask_dir))

        for img_file, mask_file in zip(image_files, mask_files):
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)
            self.image_paths.append(img_path)
            self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

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

        img_resized = resize(img, self.image_size, anti_aliasing=True)
        if img_resized.ndim == 3:
            img_gray = color.rgb2gray(img_resized)
        else:
            img_gray = img_resized

        mask_resized = resize(mask, self.image_size, anti_aliasing=True)
        mask_binary = mask_resized > 0.5

        return torch.tensor(np.expand_dims(img_gray.astype(np.float32), axis=0)), \
               torch.tensor(np.expand_dims(mask_binary.astype(np.float32), axis=0)), \
               img, mask # Return original images for visualization

# ---------------------- Training ----------------------

def train_model(model, train_loader, num_epochs=10, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for img_tensor, mask_tensor, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            img_tensor, mask_tensor = img_tensor.to(device), mask_tensor.to(device)
            optimizer.zero_grad()
            output = model(img_tensor)
            loss = criterion(output, mask_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")
    return model

# ---------------------- Prediction ----------------------

def predict_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    preds = []
    original_images = []
    ground_truth_masks = []
    with torch.no_grad():
        for img_tensor, mask_tensor, original_img, original_mask in tqdm(test_loader, desc="Predicting"):
            img_tensor = img_tensor.to(device)
            output = model(img_tensor).cpu().numpy()
            preds.extend(output)
            original_images.extend(original_img)
            ground_truth_masks.extend(original_mask)
    return np.array(preds), original_images, ground_truth_masks

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

def visualize_results(original_images, ground_truth_masks, predictions, num_samples=5, save_dir='results/hmsam_unet'):
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
        if ground_truth_masks[i].ndim == 3:
            plt.imshow(ground_truth_masks[i])
        else:
            plt.imshow(ground_truth_masks[i], cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')

        # Prediction (binarized output)
        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i][0] > 0.5, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f'prediction_result_{i}.png'))
        plt.close()

# ---------------------- Main Execution ----------------------

if __name__ == '__main__':
    # Define your data directories
    image_dir = 'TNBC_Dataset_Compiled/Slide'  # Replace with your image directory
    mask_dir = 'TNBC_Dataset_Compiled/Masks'    # Replace with your mask directory

    # Ensure the directories exist
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"Error: Image directory '{image_dir}' or mask directory '{mask_dir}' not found. Please check the paths.")
    else:
        dataset = NucleiDataset(image_dir, mask_dir, image_size=(128, 128))
        
        if len(dataset) == 0:
            print(f"Error: No images found in '{image_dir}' or masks in '{mask_dir}'. Please ensure files are present.")
        else:
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            
            if train_size == 0:
                print("Error: Not enough data to create a training set. Please provide more images.")
            elif test_size == 0:
                print("Error: Not enough data to create a test set. Please provide more images.")
            else:
                train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

                # Use shuffle=True for training loader for better generalization
                train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True) 
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

                # Instantiate the HMSAM-UNet model
                hmsam_unet_model = HMSAM_UNet(in_channels=1, out_channels=1) # Assuming grayscale input

                # Train the model
                print("Starting model training...")
                hmsam_unet_model = train_model(hmsam_unet_model, train_loader, num_epochs=20, lr=0.001)
                print("Model training complete.")

                # Get predictions and original images for evaluation and visualization
                print("Starting prediction on test set...")
                predictions, original_test_images, original_test_masks = predict_model(hmsam_unet_model, test_loader)
                print("Prediction complete.")

                # Convert original masks to binary for metric calculation
                binary_original_test_masks = [resize(mask, (128, 128), anti_aliasing=True) > 0.5 for mask in original_test_masks]
                binary_original_test_masks_np = np.array([np.expand_dims(mask.astype(np.float32), axis=0) for mask in binary_original_test_masks])

                # Compute evaluation metrics
                precision, recall, f1, accuracy = compute_metrics(binary_original_test_masks_np, predictions, threshold=0.5)
                print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}')

                # Visualize the results
                print("Saving visualization results...")
                visualize_results(original_test_images, original_test_masks, predictions, num_samples=5)
                print("Visualization complete. Check 'results/hmsam_unet' directory.")