import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 320

class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder for anomaly detection"""
    def __init__(self, latent_dim=128):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 128x128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, latent_dim, 4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    


class FeatureExtractor:
    """Extract features using pretrained ResNet"""
    def __init__(self, device='cuda'):
        self.device = device
        # Use pretrained ResNet18
        resnet = models.resnet18(pretrained=True)
        # Modify first layer for grayscale
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove final FC layer
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model.to(device)
        self.model.eval()
    
    def extract(self, images):
        with torch.no_grad():
            features = self.model(images)
            features = features.view(features.size(0), -1)
        return features.cpu().numpy()
    

class AnomalyDetector:
    def __init__(self, device='cuda'):
        self.device = device
        self.autoencoder = ConvAutoencoder().to(device)
        self.feature_extractor = FeatureExtractor(device)
        self.threshold = None
        self.healthy_feature_mean = None
        self.healthy_feature_cov = None
        
    def train_autoencoder(self, train_loader, epochs=100, lr=0.001):
        """Train autoencoder on healthy cells only"""
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        criterion = nn.L1Loss()
        
        self.autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for images in train_loader:
                images = images.to(self.device)
                
                optimizer.zero_grad()
                reconstructed, _ = self.autoencoder(images)
                loss = criterion(reconstructed, images)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    
    def compute_reconstruction_error(self, images):
        """Compute reconstruction error for anomaly scoring"""
        self.autoencoder.eval()
        with torch.no_grad():
            reconstructed, _ = self.autoencoder(images)
            # Per-sample MSE
            error = torch.mean((images - reconstructed) ** 2, dim=[1, 2, 3])
        return error.cpu().numpy()
    
    def compute_mahalanobis_distance(self, images):
        """Compute Mahalanobis distance in feature space"""
        features = self.feature_extractor.extract(images)
        diff = features - self.healthy_feature_mean
        inv_cov = np.linalg.pinv(self.healthy_feature_cov)
        distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
        return distances
    
    def fit_threshold(self, val_loader, percentile=95):
        """Compute threshold on validation healthy cells"""
        all_scores = []
        
        for images in val_loader:
            images = images.to(self.device)
            
            # Reconstruction error
            recon_error = self.compute_reconstruction_error(images)
            
            # Mahalanobis distance
            mahal_dist = self.compute_mahalanobis_distance(images)
            
            # Combined score
            combined_score = recon_error + 0.5 * mahal_dist
            all_scores.extend(combined_score)
        
        # Set threshold at percentile
        self.threshold = np.percentile(all_scores, percentile)
        print(f"Threshold set at {percentile}th percentile: {self.threshold:.4f}")
    
    def fit_feature_distribution(self, train_loader):
        """Fit Gaussian distribution on healthy features"""
        all_features = []
        
        for images in train_loader:
            images = images.to(self.device)
            features = self.feature_extractor.extract(images)
            all_features.append(features)
        
        all_features = np.vstack(all_features)
        self.healthy_feature_mean = np.mean(all_features, axis=0)
        self.healthy_feature_cov = np.cov(all_features.T) + np.eye(all_features.shape[1]) *  0.01 #1e-6
    
    def predict(self, images):
        """Predict if cells are anomalous"""
        images = images.to(self.device)
        
        # Reconstruction error
        recon_error = self.compute_reconstruction_error(images)
        
        # Mahalanobis distance
        mahal_dist = self.compute_mahalanobis_distance(images)
        
        # Combined anomaly score
        anomaly_score = recon_error + 0.5 * mahal_dist
        
        # Binary prediction
        predictions = (anomaly_score > self.threshold).astype(int)
        
        return predictions, anomaly_score


def load_trained_model(checkpoint_path, device='cuda'):
    """
    Loads the trained model weights and statistical parameters 
    into the AnomalyDetector class.
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Initialize the detector structure
    detector = AnomalyDetector(device=device)
    
    # Load the saved dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 1. Load Autoencoder weights
    detector.autoencoder.load_state_dict(checkpoint['autoencoder'])
    detector.autoencoder.eval() # Set to evaluation mode
    
    # 2. Load Statistical Params (Threshold, Mean, Covariance)
    detector.threshold = checkpoint['threshold']
    detector.healthy_feature_mean = checkpoint['feature_mean']
    detector.healthy_feature_cov = checkpoint['feature_cov']
    
    print("Model loaded successfully.")
    return detector, checkpoint['threshold']

def get_augmentation_pipeline(is_training=True):
    """Heavy augmentation for limited healthy data"""
    if is_training:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            # Geometric transforms
            A.Rotate(limit=15, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
            
            # Intensity transforms (crucial for EL images)
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            
            # Normalize
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])

def preprocess_image(input_data):
    """
    Reads and preprocesses a single image to match the training format.
    """
    # Read as Grayscale (Crucial: Model expects 1 channel)
    image = None

    # CASE 1: Input is a File Path (String or Path object)
    if isinstance(input_data, (str, Path)):
        # Read as Grayscale immediately
        image = cv2.imread(str(input_data), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image at {input_data}")

    # CASE 2: Input is already an Image (Numpy Array)
    elif isinstance(input_data, np.ndarray):
        image = input_data
        
        # SAFETY CHECK: Ensure it is Grayscale (1 Channel)
        # If the array is (H, W, 3), convert it.
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
    else:
        raise ValueError(f"Unsupported input type: {type(input_data)}")
        
    # Get the validation augmentation pipeline (No flips/rotations, just resize/norm)
    transform = get_augmentation_pipeline(is_training=False)
    
    # Apply transforms
    augmented = transform(image=image)
    image_tensor = augmented['image']
    
    # Add batch dimension (C, H, W) -> (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, image  # Return original for visualization if needed

# ============================================
# 2. RUNNING INFERENCE
# ============================================

def predict_single_image(model, thresh, image_path, device='cuda'):
    # 1. Preprocess
    img_tensor, original_img = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # 2. Predict
    # The predict method in your class returns (binary_pred, anomaly_score)
    # However, your class's predict method expects a batch. 
    # Since we passed a batch of size 1, we extract the first result.
    prediction, score = model.predict(img_tensor)
    
    # Extract scalar values
    is_defect = prediction[0]  # 1 for defect, 0 for healthy
    anomaly_score = score[0]
    
    # 3. Interpret results
    result_text = "DEFECTIVE" if anomaly_score > thresh else "HEALTHY"   # "DEFECTIVE" if is_defect == 1 else "HEALTHY"
    color = (0, 0, 255) if is_defect == 1 else (0, 255, 0) # Red or Green
    
    print(f"--- Inference Results ---")
    # print(f"File: {image_path}")
    print(f"Prediction: {result_text}")
    print(f"Anomaly Score: {anomaly_score:.4f}")
    # print(f"Threshold: {model.threshold:.4f}")
    
    return result_text, anomaly_score
