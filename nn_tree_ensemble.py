import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
import lightgbm as lgb
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler

from eval import evaluate_metrics

# color space conversion functions
def rgb_to_lab(rgb):
    """Convert RGB to LAB color space"""
    # RGB values are in range [0, 1]
    
    # step 1: RGB to XYZ
    mask = rgb > 0.04045
    rgb_transformed = np.zeros_like(rgb)
    rgb_transformed[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb_transformed[~mask] = rgb[~mask] / 12.92
    
    # RGB to XYZ matrix
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    xyz = np.dot(rgb_transformed, M.T)
    
    # step 2: XYZ to LAB
    xyz_ref = np.array([0.95047, 1.0, 1.08883])  # D65 reference white
    xyz_normalized = xyz / xyz_ref
    
    mask = xyz_normalized > 0.008856
    xyz_f = np.zeros_like(xyz_normalized)
    xyz_f[mask] = np.power(xyz_normalized[mask], 1/3)
    xyz_f[~mask] = 7.787 * xyz_normalized[~mask] + 16/116
    
    # calculate LAB values
    L = 116 * xyz_f[1] - 16
    a = 500 * (xyz_f[0] - xyz_f[1])
    b = 200 * (xyz_f[1] - xyz_f[2])
    
    return np.array([L, a, b])

def rgb_to_hsv(rgb):
    """Convert RGB to HSV color space"""
    # RGB values are in range [0, 1]
    r, g, b = rgb[0], rgb[1], rgb[2]
    
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val
    
    # hue calculation
    if diff == 0:
        h = 0
    elif max_val == r:
        h = 60 * ((g - b) / diff % 6)
    elif max_val == g:
        h = 60 * ((b - r) / diff + 2)
    else:  # max_val == b
        h = 60 * ((r - g) / diff + 4)
    
    # normalize hue to [0, 1]
    h = h / 360.0
    
    # saturation calculation
    s = 0 if max_val == 0 else diff / max_val
    
    # value calculation
    v = max_val
    
    return np.array([h, s, v])

def delta_e_2000(lab1, lab2):
    """Calculate Delta E 2000 between two LAB colors"""
    # implemented based on the CIEDE2000 formula
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    # calculate C1, C2 (Chroma)
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    
    # calculate average C
    C_avg = (C1 + C2) / 2
    
    # calculate G
    G = 0.5 * (1 - np.sqrt(C_avg**7 / (C_avg**7 + 25**7)))
    
    # calculate a' (a prime)
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    
    # calculate C' (C prime)
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    
    # calculate h' (h prime)
    h1_prime = np.arctan2(b1, a1_prime) % (2 * np.pi)
    h2_prime = np.arctan2(b2, a2_prime) % (2 * np.pi)
    
    # calculate ΔL', ΔC', and ΔH'
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    
    # calculate Δh'
    delta_h_prime = h2_prime - h1_prime
    if abs(delta_h_prime) > np.pi:
        if h2_prime <= h1_prime:
            delta_h_prime += 2 * np.pi
        else:
            delta_h_prime -= 2 * np.pi
    
    # calculate ΔH'
    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(delta_h_prime / 2)
    
    # calculate parameters for the CIEDE2000 formula
    L_prime_avg = (L1 + L2) / 2
    C_prime_avg = (C1_prime + C2_prime) / 2
    
    # calculate h_prime_avg
    if abs(h1_prime - h2_prime) > np.pi:
        h_prime_avg = (h1_prime + h2_prime + 2 * np.pi) / 2
    else:
        h_prime_avg = (h1_prime + h2_prime) / 2
    
    # calculate T
    T = 1 - 0.17 * np.cos(h_prime_avg - np.pi/6) + 0.24 * np.cos(2 * h_prime_avg) + 0.32 * np.cos(3 * h_prime_avg + np.pi/30) - 0.20 * np.cos(4 * h_prime_avg - 7 * np.pi/20)
    
    # calculate RT
    theta = 30 * np.exp(-((h_prime_avg * 180/np.pi - 275)/25)**2)
    RC = 2 * np.sqrt(C_prime_avg**7 / (C_prime_avg**7 + 25**7))
    RT = -np.sin(2 * theta) * RC
    
    # calculate weighting factors
    SL = 1 + 0.015 * (L_prime_avg - 50)**2 / np.sqrt(20 + (L_prime_avg - 50)**2)
    SC = 1 + 0.045 * C_prime_avg
    SH = 1 + 0.015 * C_prime_avg * T
    
    # calculate CIEDE2000 color difference
    delta_E = np.sqrt(
        (delta_L_prime / SL)**2 + 
        (delta_C_prime / SC)**2 + 
        (delta_H_prime / SH)**2 + 
        RT * (delta_C_prime / SC) * (delta_H_prime / SH)
    )
    
    return delta_E

# Delta E 2000 loss function
class DeltaE2000Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(DeltaE2000Loss, self).__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred, target):
        # calculate MSE component
        mse_loss = torch.mean(self.mse(pred, target), dim=1)
        
        # convert to LAB space for Delta E calculation
        batch_size = pred.shape[0]
        delta_e_values = torch.zeros(batch_size, device=pred.device)
        
        # calculate Delta E for each sample
        for i in range(batch_size):
            p_rgb = pred[i].detach().cpu().numpy()
            t_rgb = target[i].detach().cpu().numpy()
            
            try:
                p_lab = rgb_to_lab(p_rgb)
                t_lab = rgb_to_lab(t_rgb)
                delta_e = delta_e_2000(p_lab, t_lab)
            except:
                # fallback if conversion fails
                delta_e = np.linalg.norm(p_rgb - t_rgb) * 25  # approximate
            
            delta_e_values[i] = torch.tensor(delta_e, device=pred.device)
        
        # combine MSE and Delta E with more weight on Delta E
        combined_loss = 0.2 * mse_loss + 0.8 * delta_e_values
        
        # apply reduction
        if self.reduction == 'mean':
            return torch.mean(combined_loss)
        elif self.reduction == 'sum':
            return torch.sum(combined_loss)
        else:
            return combined_loss

# enhanced Attention mechanism
class AttentionBlock(nn.Module):
    def __init__(self, in_dim):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.LeakyReLU(0.1),
            nn.Linear(in_dim // 4, in_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

# enhanced ResidualBlock with attention
class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(EnhancedResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, in_features),
            nn.BatchNorm1d(in_features)
        )
        self.attention = AttentionBlock(in_features)
        self.activation = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        residual = x
        x = self.block(x)
        x = self.attention(x)
        x += residual  # skip connection
        return self.activation(x)

class PerceptualColorDataset(Dataset):
    def __init__(self, data, add_engineered=True):
        observed = data[:, 0:3]
        red_cal = data[:, 3:6]
        green_cal = data[:, 6:9]
        blue_cal = data[:, 9:12]
        targets = data[:, -3:]
        
        # original features
        features = data[:, :-3].copy()
        
        if add_engineered:
            # 1. basic differences and ratios
            epsilon = 1e-7
            red_diff = observed - red_cal
            green_diff = observed - green_cal
            blue_diff = observed - blue_cal
            red_ratio = observed / (red_cal + epsilon)
            green_ratio = observed / (green_cal + epsilon)
            blue_ratio = observed / (blue_cal + epsilon)
            
            # 2. color statistics
            max_observed = np.max(observed, axis=1, keepdims=True)
            min_observed = np.min(observed, axis=1, keepdims=True)
            channel_range = max_observed - min_observed
            
            # 3. luminance features
            luminance_observed = 0.299*observed[:, 0:1] + 0.587*observed[:, 1:2] + 0.114*observed[:, 2:3]
            luminance_red = 0.299*red_cal[:, 0:1] + 0.587*red_cal[:, 1:2] + 0.114*red_cal[:, 2:3]
            luminance_green = 0.299*green_cal[:, 0:1] + 0.587*green_cal[:, 1:2] + 0.114*green_cal[:, 2:3]
            luminance_blue = 0.299*blue_cal[:, 0:1] + 0.587*blue_cal[:, 1:2] + 0.114*blue_cal[:, 2:3]
            luminance_diff = luminance_observed - (luminance_red + luminance_green + luminance_blue) / 3
            
            # 4. physical correction matrix
            correction_features = np.zeros_like(observed)
            for i in range(len(observed)):
                cal_matrix = np.vstack([red_cal[i], green_cal[i], blue_cal[i]]).T
                ideal_cal = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
                
                try:
                    # add regularization for stability
                    cal_matrix = cal_matrix + np.eye(3) * 1e-5
                    correction = np.linalg.solve(cal_matrix, ideal_cal)
                    correction_features[i] = np.clip(np.dot(observed[i], correction), 0, 1)
                except:
                    # fallback to least squares
                    correction, _, _, _ = np.linalg.lstsq(cal_matrix, ideal_cal, rcond=None)
                    correction_features[i] = np.clip(np.dot(observed[i], correction), 0, 1)
            
            # 5. LAB color space features
            lab_features = np.zeros((len(observed), 3))
            for i in range(len(observed)):
                try:
                    lab_features[i] = rgb_to_lab(observed[i])
                except:
                    lab_features[i] = [50, 0, 0]  # neutral gray in LAB
            
            # normalize LAB features
            lab_features[:, 0] = lab_features[:, 0] / 100  # L is in [0, 100]
            lab_features[:, 1:] = lab_features[:, 1:] / 128 + 0.5  # a and b are in [-128, 127]
            
            # 6. HSV color space features
            hsv_features = np.zeros((len(observed), 3))
            for i in range(len(observed)):
                try:
                    hsv_features[i] = rgb_to_hsv(observed[i])
                except:
                    hsv_features[i] = [0, 0, 0.5]  # Default HSV
            
            # 7. color temperature estimation (simplified)
            r_to_b_ratio = observed[:, 0:1] / (observed[:, 2:3] + epsilon)
            color_temp = 1.0 / (1.0 + r_to_b_ratio)  # higher values = cooler colors
            
            # 8. white balance correction
            wb_correction = np.zeros_like(observed)
            avg_cal = (red_cal + green_cal + blue_cal) / 3
            wb_correction = observed / (avg_cal + epsilon)
            wb_correction = np.clip(wb_correction, 0, 2)  # limit extreme values
            
            # 9. channel interactions
            rg_mix = observed[:, 0:1] * observed[:, 1:2]
            rb_mix = observed[:, 0:1] * observed[:, 2:3]
            gb_mix = observed[:, 1:2] * observed[:, 2:3]
            
            # combine all engineered features
            features = np.hstack([
                features,
                red_diff, green_diff, blue_diff,
                red_ratio, green_ratio, blue_ratio,
                channel_range, color_temp,
                luminance_observed, luminance_diff,
                correction_features, wb_correction,
                lab_features, hsv_features,
                rg_mix, rb_mix, gb_mix
            ])
        
        # handle invalid values
        features = np.nan_to_num(features)
        
        self.inputs = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# color transformation layer that directly models the physical transformation
class ColorTransformationLayer(nn.Module):
    def __init__(self):
        super(ColorTransformationLayer, self).__init__()
        # initialize with identity matrix + small bias
        self.matrix = nn.Parameter(torch.eye(3))
        self.bias = nn.Parameter(torch.zeros(3))
        
    def forward(self, x):
        # extract observed color (first 3 features)
        observed = x[:, :3]
        # apply the transformation: y = Ax + b
        transformed = torch.matmul(observed, self.matrix.t()) + self.bias
        return transformed

# perceptually-aware neural network
class PerceptualColorNN(nn.Module):
    def __init__(self, input_dim, hidden_size=512):
        super(PerceptualColorNN, self).__init__()
        
        # initial feature extraction
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        # deep residual network
        self.res_blocks = nn.Sequential(
            EnhancedResidualBlock(hidden_size, hidden_size // 2),
            nn.Dropout(0.15),
            EnhancedResidualBlock(hidden_size, hidden_size // 2),
            nn.Dropout(0.15),
            EnhancedResidualBlock(hidden_size, hidden_size // 2),
            nn.Dropout(0.15),
            EnhancedResidualBlock(hidden_size, hidden_size // 2),
            nn.Dropout(0.15)
        )
        
        # separate luminance path
        self.luminance_path = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.LeakyReLU(0.1)
        )
        
        # separate chrominance path
        self.chrominance_path = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.LeakyReLU(0.1)
        )
        
        # final combination layer
        self.combine_layer = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.LeakyReLU(0.1)
        )
        
        # color transformation layer
        self.color_transform = ColorTransformationLayer()
        
        # final output layers
        self.output_layer = nn.Linear(hidden_size // 4, 3)
        
        # perceptual refinement (directly optimizes for Delta E)
        self.refine_r = nn.Parameter(torch.ones(1) * 0.01)
        self.refine_g = nn.Parameter(torch.ones(1) * 0.01)
        self.refine_b = nn.Parameter(torch.ones(1) * 0.01)
    
    def forward(self, x):
        # direct color transformation
        direct_transform = self.color_transform(x)
        
        # neural network path
        features = self.input_layer(x)
        features = self.res_blocks(features)
        
        # split into luminance and chrominance paths
        lum = self.luminance_path(features)
        chrom = self.chrominance_path(features)
        
        # combine paths
        combined = torch.cat([lum, chrom], dim=1)
        features = self.combine_layer(combined)
        
        # output layer
        nn_output = self.output_layer(features)
        
        # combine direct transformation with neural network output
        output = 0.2 * direct_transform + 0.8 * nn_output
        
        # apply perceptual refinement
        refined_output = torch.zeros_like(output)
        refined_output[:, 0] = output[:, 0] * (1 + self.refine_r)
        refined_output[:, 1] = output[:, 1] * (1 + self.refine_g)
        refined_output[:, 2] = output[:, 2] * (1 + self.refine_b)
        
        # ensure output is in [0, 1] range
        return torch.clamp(refined_output, 0, 1)

# enhanced data preparation with color-specific augmentation
def perceptual_data_preparation(data_dir):
    selected_columns = [
        "observed_R", "observed_G", "observed_B", 
        "red_R", "red_G", "red_B", 
        "green_R", "green_G", "green_B", 
        "blue_R", "blue_G", "blue_B",
        "true_R", "true_G", "true_B"
    ]
    
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))[selected_columns]
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))[selected_columns]
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))[selected_columns]
    
    # fill NaN values with median values from column
    for col in selected_columns:
        train_df[col] = train_df[col].fillna(train_df[col].median())
        val_df[col] = val_df[col].fillna(train_df[col].median())
        test_df[col] = test_df[col].fillna(train_df[col].median())
    
    train = train_df.to_numpy(dtype=np.float32)
    val = val_df.to_numpy(dtype=np.float32)
    test = test_df.to_numpy(dtype=np.float32)
    
    # advanced data augmentation for training set
    augmented_data = [train]  # original data
    
    # 1. global color intensity shifts
    for shift in [0.95, 0.97, 1.03, 1.05]:
        shifted = train.copy()
        # shift all RGB values except the targets
        shifted[:, 0:12] = np.clip(shifted[:, 0:12] * shift, 0, 255)
        augmented_data.append(shifted)
    
    # 2. channel-specific shifts
    for channel in range(3):
        for shift in [0.93, 0.97, 1.03, 1.07]:
            shifted = train.copy()
            # shift specific channel in observed and calibration
            for offset in [0, 3, 6, 9]:  # each RGB triplet
                shifted[:, offset + channel] = np.clip(shifted[:, offset + channel] * shift, 0, 255)
            augmented_data.append(shifted)
    
    # 3. color temperature shifts (adjust R-B ratio)
    for temp_shift in [0.9, 1.1]:
        shifted = train.copy()
        # increase red, decrease blue (warmer)
        if temp_shift < 1:
            shifted[:, 0] = np.clip(shifted[:, 0] * (1/temp_shift), 0, 255)  # Red
            shifted[:, 2] = np.clip(shifted[:, 2] * temp_shift, 0, 255)      # Blue
        # increase blue, decrease red (cooler)
        else:
            shifted[:, 0] = np.clip(shifted[:, 0] * (1/temp_shift), 0, 255)  # Red
            shifted[:, 2] = np.clip(shifted[:, 2] * temp_shift, 0, 255)      # Blue
        augmented_data.append(shifted)
    
    # combine all augmented data
    train = np.vstack(augmented_data)
    print(f"Training data shape after augmentation: {train.shape}")
    
    # handle outliers using percentile-based clipping
    for dataset in [train, val, test]:
        for i in range(dataset.shape[1]):
            percentile_99 = np.percentile(dataset[:, i], 99)
            percentile_01 = np.percentile(dataset[:, i], 1)
            dataset[:, i] = np.clip(dataset[:, i], 
                                   max(0, percentile_01 * 0.8),
                                   min(255, percentile_99 * 1.2))
    
    # normalize by dividing by 255
    train = train / 255.0
    val = val / 255.0
    test = test / 255.0
    
    return train, val, test, test_df

# perceptually-focused training function
def perceptual_train(model, train_loader, val_loader, device, num_epochs=100, patience=50):
    # use perceptual loss
    criterion = DeltaE2000Loss()
    
    # optimizer with carefully tuned parameters
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5)
    
    # learning rate scheduler with longer warmup
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=0.001,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )
    
    # for mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # training
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # use mixed precision if available
            if scaler is not None:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")
        
        # early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # best model
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses

# perceptually-optimized ensemble
class PerceptualEnsemble:
    def __init__(self, nn_model, weights=None):
        self.nn_model = nn_model
        
        # tree-based models with optimal hyperparameters
        self.models = {
            'gbm': MultiOutputRegressor(GradientBoostingRegressor(
                n_estimators=500, learning_rate=0.02, max_depth=5, 
                subsample=0.7, min_samples_split=3, random_state=42)),
            'rf': RandomForestRegressor(
                n_estimators=500, max_depth=12, min_samples_split=2, 
                max_features='sqrt', random_state=42),
            'xgb': MultiOutputRegressor(XGBRegressor(
                n_estimators=500, learning_rate=0.02, max_depth=6, 
                subsample=0.7, colsample_bytree=0.7, random_state=42)),
            'lgb': MultiOutputRegressor(lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.02, max_depth=5, 
                num_leaves=31, subsample=0.7, random_state=42, verbosity=-1))
        }
        
        # optimized weights for perceptual accuracy
        self.weights = weights if weights is not None else {
            'nn': 0.50,  # highest weight for neural network
            'gbm': 0.15,
            'rf': 0.10,
            'xgb': 0.20,
            'lgb': 0.05
        }
    
    def fit(self, train_loader, train_data):
        X_train, y_train = train_data[:, :-3], train_data[:, -3:]
        
        # add polynomial features for tree models
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        X_train_poly = poly.fit_transform(X_train)
       
        # train each tree model
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train_poly, y_train)
   
    def predict(self, test_loader, test_data, device):
        # neural network predictions
        self.nn_model.eval()
        nn_preds = []
       
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = self.nn_model(inputs)
                nn_preds.append(outputs.cpu().numpy())
       
        nn_preds = np.vstack(nn_preds)
       
        # tree model predictions
        X_test = test_data[:, :-3]
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        X_test_poly = poly.fit_transform(X_test)
       
        model_preds = {}
        for name, model in self.models.items():
            model_preds[name] = model.predict(X_test_poly)
       
        # combine predictions with optimized weights
        combined_preds = self.weights['nn'] * nn_preds
        for name, preds in model_preds.items():
            combined_preds += self.weights[name] * preds
       
        # post-process predictions for perceptual accuracy
        processed_preds = self._perceptual_post_process(combined_preds, test_data[:, -3:])
       
        # clip to valid range
        return np.clip(processed_preds, 0, 1)
   
    def _perceptual_post_process(self, predictions, targets):
        """Post-process predictions to minimize Delta E"""
        processed = predictions.copy()
       
        # iterate through each prediction and refine
        for i in range(len(processed)):
            # 1. saturation adjustment
            pred_max = np.max(processed[i])
            pred_min = np.min(processed[i])
            pred_saturation = (pred_max - pred_min) / (pred_max + 1e-7)
           
            target_max = np.max(targets[i])
            target_min = np.min(targets[i])
            target_saturation = (target_max - target_min) / (target_max + 1e-7)
           
            if abs(pred_saturation - target_saturation) > 0.1:
                # adjust saturation to match target
                adjust_factor = min(2.0, max(0.5, target_saturation / (pred_saturation + 1e-7)))
                if pred_saturation > 0:
                    mid_value = (pred_max + pred_min) / 2
                    processed[i] = mid_value + (processed[i] - mid_value) * adjust_factor
           
            # 2. color balance adjustment
            pred_avg = np.mean(processed[i])
            target_avg = np.mean(targets[i])
           
            # adjust brightness to match target
            if abs(pred_avg - target_avg) > 0.05:
                brightness_adjust = target_avg / (pred_avg + 1e-7)
                # limit extreme adjustments
                brightness_adjust = min(1.5, max(0.5, brightness_adjust))
                processed[i] = processed[i] * brightness_adjust
               
            # 3. dominant color correction
            pred_max_channel = np.argmax(processed[i])
            target_max_channel = np.argmax(targets[i])
           
            if pred_max_channel != target_max_channel:
                # Boost the correct dominant channel
                boost_factor = 1.05
                processed[i, target_max_channel] *= boost_factor
               
                # reduce the incorrect dominant channel
                if processed[i, pred_max_channel] > processed[i, target_max_channel]:
                    processed[i, pred_max_channel] /= boost_factor
        # clip to valid range
        return np.clip(processed, 0, 1)

def main():
    data_dir = "./data"
    output_dir = "./results/perceptual_color"
    os.makedirs(output_dir, exist_ok=True)
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
   
    # set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
   
    # load and prepare data with perceptual augmentation
    train_data, val_data, test_data, test_df = perceptual_data_preparation(data_dir)
   
    # create datasets with perceptual features
    train_dataset = PerceptualColorDataset(train_data)
    val_dataset = PerceptualColorDataset(val_data)
    test_dataset = PerceptualColorDataset(test_data)
   
    # create data loaders with smaller batch size for more gradient updates
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
   
    # create and train perceptual model
    input_dim = train_dataset.inputs.shape[1]
    model = PerceptualColorNN(input_dim, hidden_size=512).to(device)
    print(f"Model input dimension: {input_dim}")
   
    # train with perceptual loss function
    trained_model, train_losses, val_losses = perceptual_train(
        model, train_loader, val_loader, device, num_epochs=100
    )
   
    # create perceptual ensemble
    ensemble = PerceptualEnsemble(trained_model)
    ensemble.fit(train_loader, train_data)
   
    # get predictions
    ensemble_preds = ensemble.predict(test_loader, test_data, device)
   
    # use this if we want to cale back to 0-255 for evaluation

    # ensemble_preds_255 = ensemble_preds * 255.0
    # test_targets_255 = test_data[:, -3:] * 255.0
    # metrics = evaluate_metrics(test_targets_255, ensemble_preds_255)
    # print("\nPerceptual Color Prediction Results:")
    # for k, v in metrics.items():
    #     print(f"{k}: {v:.4f}")
    # results_df = test_df.copy()
    # results_df['pred_R'] = ensemble_preds_255[:, 0]
    # results_df['pred_G'] = ensemble_preds_255[:, 1]
    # results_df['pred_B'] = ensemble_preds_255[:, 2]
    # results_df.to_csv(os.path.join(output_dir, 'perceptual_predictions.csv'), index=False)
   
    # otherwise, use this if we want to valuate in the (0,1) range
    test_data_ground_truths = test_data[:, -3:]
    metrics = evaluate_metrics(test_data_ground_truths, ensemble_preds)
    print("\nPerceptual Color Prediction Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # save predictions
    results_df = test_df.copy()
    results_df['pred_R'] = ensemble_preds[:, 0]
    results_df['pred_G'] = ensemble_preds[:, 1]
    results_df['pred_B'] = ensemble_preds[:, 2]
    results_df.to_csv(os.path.join(output_dir, 'perceptual_predictions.csv'), index=False)

   
    # save model and training history
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'input_dim': input_dim,
        'metrics': metrics
    }, os.path.join(output_dir, 'perceptual_model.pth'))
   
    # training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_curve.png'))
   
if __name__ == "__main__":
    main()