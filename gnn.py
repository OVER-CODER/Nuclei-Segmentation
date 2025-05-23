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
from torch_geometric.nn import GATConv, GCNConv # Keep both for flexibility
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split

# ---------------------- Data Loading and Processing with Augmentation ----------------------

class NucleiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(128, 128), augment=False):
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

        self.augment = augment
        if self.augment:
            self.augmentation = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=90, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, p=0.3), # Removed alpha_affine as it's deprecated
                A.Resize(image_size[0], image_size[1]),
                ToTensorV2(),
            ])
        else:
            self.resize_transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img = io.imread(img_path)
        mask = io.imread(mask_path)

        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        if mask.ndim == 3:
            if mask.shape[2] == 4:
                mask = mask[:, :, :3]
            mask = color.rgb2gray(mask)
        elif mask.ndim == 2:
            pass

        mask_binary = (resize(mask, self.image_size, anti_aliasing=True) > 0.5).astype(np.float32)
        img_gray = resize(img, self.image_size, anti_aliasing=True)
        if img_gray.ndim == 3:
            img_gray = color.rgb2gray(img_gray).astype(np.float32)
        else:
            img_gray = img_gray.astype(np.float32)

        if self.augment:
            augmented = self.augmentation(image=img_gray, mask=mask_binary)
            img_tensor = augmented['image'] # ToTensorV2 already adds channel dim for grayscale [1, H, W]
            mask_tensor = augmented['mask'].unsqueeze(0) # [H, W] -> [1, H, W]
        else:
            resized = self.resize_transform(image=img_gray, mask=mask_binary)
            img_tensor = resized['image'] # Already [1, H, W]
            mask_tensor = resized['mask'].unsqueeze(0) # [H, W] -> [1, H, W]

        return img_tensor, mask_tensor, img, mask

# ---------------------- Dice Loss ----------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred_flat = y_pred.contiguous().view(-1)
        y_true_flat = y_true.contiguous().view(-1)
        intersection = (y_pred_flat * y_true_flat).sum()
        score = (2.0 * intersection + self.smooth) / (y_pred_flat.sum() + y_true_flat.sum() + self.smooth)
        return 1.0 - score

# ---------------------- Improved GNN Integrated Segmentation Model ----------------------

class UNetEncoderGNNImproved(nn.Module):
    def __init__(self, in_channels, initial_filters=64, dropout_rate=0.1):
        super(UNetEncoderGNNImproved, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, initial_filters, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(initial_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(initial_filters, initial_filters * 2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(initial_filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(dropout_rate)

        self.conv3 = nn.Conv2d(initial_filters * 2, initial_filters * 4, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(initial_filters * 4)
        self.pool3 = nn.MaxPool2d(2) # Correctly initialized
        self.dropout3 = nn.Dropout2d(dropout_rate)

        self.conv4 = nn.Conv2d(initial_filters * 4, initial_filters * 8, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(initial_filters * 8)
        self.pool4 = nn.MaxPool2d(2) # Correctly initialized
        self.dropout4 = nn.Dropout2d(dropout_rate)

        self.conv_mid = nn.Conv2d(initial_filters * 8, initial_filters * 16, kernel_size=3, padding=1)
        self.relu_mid = nn.ReLU(inplace=True)
        self.bn_mid = nn.BatchNorm2d(initial_filters * 16)
        self.dropout_mid = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x1 = self.dropout1(self.bn1(self.relu1(self.conv1(x))))
        p1 = self.pool1(x1)
        x2 = self.dropout2(self.bn2(self.relu2(self.conv2(p1))))
        p2 = self.pool2(x2)
        x3 = self.dropout3(self.bn3(self.relu3(self.conv3(p2))))
        p3 = self.pool3(x3)
        x4 = self.dropout4(self.bn4(self.relu4(self.conv4(p3))))
        p4 = self.pool4(x4)
        features = self.dropout_mid(self.bn_mid(self.relu_mid(self.conv_mid(p4))))
        return features, x1, x2, x3, x4 # Return intermediate features

class GNNNodeClassifierImproved(nn.Module):
    def __init__(self, feature_dim, gnn_hidden_dim, num_gnn_layers, num_classes, dropout_rate=0.1, gat=True, num_heads=4):
        super(GNNNodeClassifierImproved, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.gat = gat
        self.num_heads = num_heads

        if gat:
            self.convs.append(GATConv(feature_dim, gnn_hidden_dim // num_heads, heads=num_heads, dropout=dropout_rate)) # GATConv has its own dropout
        else:
            self.convs.append(GCNConv(feature_dim, gnn_hidden_dim))
        self.bns.append(nn.BatchNorm1d(gnn_hidden_dim))
        # self.dropouts.append(nn.Dropout(dropout_rate)) # Dropout handled by GATConv

        for _ in range(num_gnn_layers - 1):
            if gat:
                self.convs.append(GATConv(gnn_hidden_dim, gnn_hidden_dim // num_heads, heads=num_heads, dropout=dropout_rate))
            else:
                self.convs.append(GCNConv(gnn_hidden_dim, gnn_hidden_dim))
            self.bns.append(nn.BatchNorm1d(gnn_hidden_dim))
            # self.dropouts.append(nn.Dropout(dropout_rate)) # Dropout handled by GATConv

        self.fc = nn.Linear(gnn_hidden_dim, num_classes)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x) # Using ELU activation
            # if not self.gat: # Apply dropout only if not GAT (GAT has internal dropout)
            #     x = self.dropouts[i](x)
        x = self.fc(x)
        return x

class GNNIntegratedSegmentationImproved(nn.Module):
    def __init__(self, in_channels, feature_dim, gnn_hidden_dim, num_gnn_layers, num_classes, image_size=(128, 128), graph_connectivity='8-connectivity', use_gat=True, gat_heads=4):
        super(GNNIntegratedSegmentationImproved, self).__init__()
        self.encoder = UNetEncoderGNNImproved(in_channels, initial_filters=feature_dim) # Pass initial_filters
        
        # Determine actual downsampled sizes from a dummy pass
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, image_size[0], image_size[1])
            deep_features_dummy, x1_dummy, x2_dummy, x3_dummy, x4_dummy = self.encoder(dummy_input)
            
            self.deep_gnn_graph_size = (deep_features_dummy.shape[2], deep_features_dummy.shape[3]) # (8, 8)
            self.early_gnn_graph_size = (x2_dummy.shape[2], x2_dummy.shape[3]) # (32, 32)

        self.main_upsample_factor = image_size[0] // self.deep_gnn_graph_size[0] # 128 / 8 = 16
        self.early_upsample_factor = image_size[0] // self.early_gnn_graph_size[0] # 128 / 32 = 4

        self.gnn = GNNNodeClassifierImproved(feature_dim * 16, gnn_hidden_dim, num_gnn_layers, num_classes, gat=use_gat, num_heads=gat_heads)
        self.upsample_deep = nn.Upsample(scale_factor=self.main_upsample_factor, mode='bilinear', align_corners=False)
        self.graph_connectivity = graph_connectivity
        self.image_size = image_size
        
        self.edge_index_deep = self._precompute_graph(self.deep_gnn_graph_size[0], self.deep_gnn_graph_size[1])

        # Early GNN integration
        self.gnn_early_feature_dim = self.encoder.conv2.out_channels # 128
        self.gnn_early = GNNNodeClassifierImproved(self.gnn_early_feature_dim, gnn_hidden_dim // 2, num_gnn_layers // 2 if num_gnn_layers > 1 else 1, num_classes=num_classes, gat=use_gat, num_heads=gat_heads) # num_classes is 1
        self.upsample_early = nn.Upsample(scale_factor=self.early_upsample_factor, mode='bilinear', align_corners=False)
        
        self.edge_index_early = self._precompute_graph(self.early_gnn_graph_size[0], self.early_gnn_graph_size[1])

    def _precompute_graph(self, h, w):
        edges = []
        for i in range(h):
            for j in range(w):
                current_node_index = i * w + j
                if self.graph_connectivity == '4-connectivity':
                    neighbors = [(i, j + 1), (i, j - 1), (i + 1, j), (i - 1, j)]
                elif self.graph_connectivity == '8-connectivity':
                    neighbors = [(i, j + 1), (i, j - 1), (i + 1, j), (i - 1, j),
                                 (i + 1, j + 1), (i + 1, j - 1), (i - 1, j + 1), (i - 1, j - 1)]

                for ni, nj in neighbors:
                    if 0 <= ni < h and 0 <= nj < w:
                        neighbor_node_index = ni * w + nj
                        edges.append((current_node_index, neighbor_node_index))

        edge_index = torch.tensor(list(set([(u, v) if u <= v else (v, u) for u, v in edges])), dtype=torch.long).t().contiguous().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return edge_index

    def forward(self, x):
        encoder_features, x1, x2, x3, x4 = self.encoder(x)
        
        # Deep GNN Path
        batch_size_deep, _, h_deep, w_deep = encoder_features.shape
        nodes_deep = encoder_features.permute(0, 2, 3, 1).reshape(batch_size_deep * h_deep * w_deep, -1)
        num_nodes_per_graph_deep = h_deep * w_deep
        batched_edge_index_deep = torch.cat([self.edge_index_deep + i * num_nodes_per_graph_deep for i in range(batch_size_deep)], dim=1)
        
        node_logits_deep = self.gnn(nodes_deep, batched_edge_index_deep)
        segmentation_logits_deep = node_logits_deep.view(batch_size_deep, num_classes, h_deep, w_deep) # num_classes from GNN output
        segmentation_logits_deep = self.upsample_deep(segmentation_logits_deep)

        # Early GNN Path
        early_features = x2 # Features after second pooling (e.g., 32x32)
        batch_size_early, _, h_early, w_early = early_features.shape
        nodes_early = early_features.permute(0, 2, 3, 1).reshape(batch_size_early * h_early * w_early, -1)
        num_nodes_per_graph_early = h_early * w_early
        batched_edge_index_early = torch.cat([self.edge_index_early + i * num_nodes_per_graph_early for i in range(batch_size_early)], dim=1)

        node_logits_early = self.gnn_early(nodes_early, batched_edge_index_early)
        segmentation_logits_early = node_logits_early.view(batch_size_early, num_classes, h_early, w_early) # num_classes is 1
        segmentation_logits_early = self.upsample_early(segmentation_logits_early)
        
        # Combine predictions by simple addition
        # You might want to experiment with a learnable fusion layer here (e.g., concatenation + Conv2d)
        return segmentation_logits_deep + segmentation_logits_early

# ---------------------- Training ----------------------

def train_gnn_unet(model, train_loader, val_loader=None, num_epochs=100, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    pos_weight = torch.tensor([calculate_positive_weight(train_loader)]).to(device)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_dice = DiceLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True) # Increased patience

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for img_tensor, mask_tensor, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            img_tensor, mask_tensor = img_tensor.to(device), mask_tensor.to(device)
            optimizer.zero_grad()
            output = model(img_tensor)
            loss_bce = criterion_bce(output, mask_tensor)
            loss_dice = criterion_dice(torch.sigmoid(output), mask_tensor)
            loss = loss_bce + loss_dice # Combine losses
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        if val_loader:
            val_loss = evaluate_model(model, val_loader, criterion_bce, criterion_dice, device)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_gnn_unet_improved_model.pth') # Save best model
                print("Saved best model.")

    return model

def evaluate_model(model, loader, criterion_bce, criterion_dice, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for img_tensor, mask_tensor, _, _ in loader:
            img_tensor, mask_tensor = img_tensor.to(device), mask_tensor.to(device)
            output = model(img_tensor)
            loss_bce = criterion_bce(output, mask_tensor)
            loss_dice = criterion_dice(torch.sigmoid(output), mask_tensor)
            loss = loss_bce + loss_dice
            total_loss += loss.item()
    return total_loss / len(loader)

def calculate_positive_weight(loader):
    total_pixels = 0
    positive_pixels = 0
    for _, masks, _, _ in loader:
        positive_pixels += torch.sum(masks == 1).item()
        total_pixels += masks.numel()
    if total_pixels == 0:
        return 1.0
    return (total_pixels - positive_pixels) / (positive_pixels + 1e-6)

# ---------------------- Prediction ----------------------

def predict_gnn_unet(model, test_loader, device):
    model.eval()
    preds = []
    original_images = []
    ground_truth_masks = []
    with torch.no_grad():
        for img_tensor, mask_tensor, original_img, original_mask in test_loader:
            img_tensor = img_tensor.to(device)
            output = torch.sigmoid(model(img_tensor)).cpu().numpy()
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

def visualize_results(original_images, ground_truth_masks, predictions, num_samples=5, save_dir='results/gnn_unet_improved'):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(original_images))):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        if original_images[i].ndim == 3:
            plt.imshow(original_images[i])
        else:
            plt.imshow(original_images[i], cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        if ground_truth_masks[i].ndim == 3:
            plt.imshow(ground_truth_masks[i])
        else:
            plt.imshow(ground_truth_masks[i], cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i][0] > 0.5, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f'prediction_result_{i}.png'))
        plt.close()

# ---------------------- Main Execution ----------------------

if __name__ == '__main__':
    image_dir = 'TNBC_Dataset_Compiled/Slide'
    mask_dir = 'TNBC_Dataset_Compiled/Masks'
    image_size = (128, 128)
    batch_size = 5
    num_epochs = 100 # Increased epochs
    learning_rate = 5e-5 # Fine-tuned learning rate
    feature_dim = 64 # Initial filters for encoder
    gnn_hidden_dim = 512
    num_gnn_layers = 2 # Reduced GNN layers for stability
    num_classes = 1
    graph_connectivity = '8-connectivity'
    use_gat = True # Use Graph Attention Network
    gat_heads = 8

    dataset = NucleiDataset(image_dir, mask_dir, image_size, augment=True)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_unet_model = GNNIntegratedSegmentationImproved(
        in_channels=1,
        feature_dim=feature_dim, # Initial filters for encoder
        gnn_hidden_dim=gnn_hidden_dim,
        num_gnn_layers=num_gnn_layers,
        num_classes=num_classes,
        image_size=image_size,
        graph_connectivity=graph_connectivity,
        use_gat=use_gat,
        gat_heads=gat_heads
    )

    gnn_unet_model = train_gnn_unet(gnn_unet_model, train_loader, val_loader, num_epochs, learning_rate)

    # Load the best model if saved during training
    if os.path.exists('best_gnn_unet_improved_model.pth'):
        gnn_unet_model.load_state_dict(torch.load('best_gnn_unet_improved_model.pth', map_location=device))
        print("Loaded best model for evaluation.")
    gnn_unet_model.eval()

    predictions, original_test_images, original_test_masks = predict_gnn_unet(gnn_unet_model, test_loader, device)

    binary_original_test_masks = [resize(mask, image_size, anti_aliasing=True) > 0.5 for mask in original_test_masks]
    binary_original_test_masks_np = np.array([np.expand_dims(mask.astype(np.float32), axis=0).astype(np.float32) for mask in binary_original_test_masks])
    predictions_np = np.array(predictions).astype(np.float32)

    precision, recall, f1, accuracy = compute_metrics(binary_original_test_masks_np, predictions_np, threshold=0.5)
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}')

    visualize_results(original_test_images, original_test_masks, predictions, num_samples=10, save_dir='results/gnn_unet_improved')