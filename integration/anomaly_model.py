from sklearn.random_projection import SparseRandomProjection
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
from dataclasses import dataclass
from PIL import Image
from torchvision import transforms
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

# to solve certify problems
# import ssl
# import certifi
# import torch
# import torchvision.models as models
# from torch.hub import download_url_to_file

# ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())

class RobustPreprocessor:
    def __init__(self, target_size: int = 320, use_clahe: bool = False):
        self.target_size = target_size
        self.use_clahe = use_clahe
        if use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(img)

    def resize_with_padding(self, img, target_size=(320, 320)):
        if img is None: return None
        h, w = img.shape[:2]
        if h > w:
            pad_top, pad_bottom = 0, 0
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
        elif w > h:
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
            pad_left, pad_right = 0, 0
        else:
            pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
            
        padded_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
        if isinstance(target_size, int): target_size = (target_size, target_size)
        return cv2.resize(padded_img, target_size)

    def preprocess_for_inference(self, image_input):
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input), cv2.IMREAD_GRAYSCALE)
        else:
            img = image_input.copy()
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.use_clahe:
            img = self.apply_clahe(img)
        
        img = self.resize_with_padding(img, target_size=(self.target_size, self.target_size))
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

patchcore_preprocessor = RobustPreprocessor(target_size=320, use_clahe=False)

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
        resnet = models.resnet18(weights=None)  # pretrained=True
        # Modify first layer for grayscale
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)
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
                print(
                    f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

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
        print(
            f"Threshold set at {percentile}th percentile: {self.threshold:.4f}")

    def fit_feature_distribution(self, train_loader):
        """Fit Gaussian distribution on healthy features"""
        all_features = []

        for images in train_loader:
            images = images.to(self.device)
            features = self.feature_extractor.extract(images)
            all_features.append(features)

        all_features = np.vstack(all_features)
        self.healthy_feature_mean = np.mean(all_features, axis=0)
        self.healthy_feature_cov = np.cov(
            all_features.T) + np.eye(all_features.shape[1]) * 0.01  # 1e-6

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
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)

    # 1. Load Autoencoder weights
    detector.autoencoder.load_state_dict(checkpoint['autoencoder'])
    detector.autoencoder.eval()  # Set to evaluation mode

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
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),

            # Intensity transforms (crucial for EL images)
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.7),
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
    # "DEFECTIVE" if is_defect == 1 else "HEALTHY"
    result_text = "DEFECTIVE" if anomaly_score > thresh else "HEALTHY"
    color = (0, 0, 255) if is_defect == 1 else (0, 255, 0)  # Red or Green

    print(f"--- Inference Results ---")
    # print(f"File: {image_path}")
    print(f"Prediction: {result_text}")
    print(f"Anomaly Score: {anomaly_score:.4f}")
    # print(f"Threshold: {model.threshold:.4f}")

    return result_text, anomaly_score


# A dummy result class to mimic Anomalib's output

@dataclass
class JITResult:
    anomaly_map: torch.Tensor
    pred_score: float


class JITInferencer:
    def __init__(self, path, device="cpu"):
        self.device = device
        print(f"⏳ Loading JIT model from {path}...")
        # Load the TorchScript model
        self.model = torch.jit.load(path, map_location=device)
        self.model.eval()

        # Define the transforms (Hardcoded for EfficientAD)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        print("✅ Model loaded successfully.")

    def predict(self, image):
        """
        Args:
            image: Can be a file path (str) or a numpy array (OpenCV image)
        """
        # 1. Handle Input Types
        if isinstance(image, str):
            image_pil = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image).convert('RGB')
        else:
            image_pil = image  # Assume it's already PIL

        # 2. Preprocess
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        # 3. Inference
        with torch.no_grad():
            # The JIT model returns the anomaly map directly
            anomaly_map = self.model(input_tensor)

        # 4. Format Output
        # Extract the map and calculate the max score
        amap = anomaly_map[0, 0].cpu()  # Shape (256, 256)
        score = amap.max().item()

        # Return object compatible with your pipeline
        return JITResult(anomaly_map=amap, pred_score=score)


# ==========================================
# 1. The Patch-Based Feature Extractor
# ==========================================


class PatchFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load ResNet50
        # self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet = resnet50(weights=None)

        # Extract layers.
        # Layer 2: Higher resolution, low-level features (good for texture)
        # Layer 3: Lower resolution, mid-level features (good for structural defects)
        self.layer2 = nn.Sequential(*list(self.resnet.children())[:6])
        self.layer3 = nn.Sequential(*list(self.resnet.children())[6:7])

        # Average pooling to smooth out noise in the feature map
        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)

        # Freeze everything
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 1. Get intermediate features
        with torch.no_grad():
            # Shape: [B, 512, 28, 28] (assuming 224 input)
            f2 = self.layer2(x)
            f3 = self.layer3(f2)     # Shape: [B, 1024, 14, 14]

        # 2. Resize f3 to match f2's spatial size so we can stack them
        # Bilinear interpolation is standard for feature upsampling
        f3_resized = F.interpolate(
            f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)

        # 3. Concatenate along channel dimension
        # Result: [B, 512+1024, 28, 28] = [B, 1536, 28, 28]
        features = torch.cat([f2, f3_resized], dim=1)

        # 4. Local Smoothing (Optional but recommended for stability)
        features = self.avg_pool(features)

        return features

# ==========================================
# 2. The Anomaly Detector Class
# ==========================================


class PatchCoreSystem:
    def __init__(self, sampling_ratio=0.01):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = PatchFeatureExtractor().to(self.device)
        self.memory_bank = None
        self.sampling_ratio = sampling_ratio  # Only keep 1% of patches to save RAM
        self.patch_paths = []

        # Transformation pipeline
        self.transform = transforms.Compose([
            # Fixed size is crucial for PatchCore
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, img, image_path=None):
        # img = #Image.open(image_path).convert('RGB')
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        return self.transform(img).unsqueeze(0).to(self.device)

    def fit(self, train_folder):
        """
        Builds the memory bank with ON-THE-FLY subsampling to save RAM.
        """
        print(f"Training on {train_folder}...")
        features_list = []
        train_path = Path(train_folder)

        # Calculate how many patches to keep per image to match your target ratio
        # Standard PatchCore keeps ~10% to 1% of total data.
        # We target a specific number of patches to prevent explosion.

        # If we assume 28x28 = 784 patches per image
        # and we want sampling_ratio 0.01 (1%), that is ~8 patches per image.
        patches_per_image = int(28 * 28 * self.sampling_ratio)

        # Safety: Ensure we keep at least 1 patch per image if ratio is tiny
        if patches_per_image < 1:
            patches_per_image = 1

        print(f"Target: Keeping {patches_per_image} random patches per image.")

        for pth in tqdm(list(train_path.iterdir())):
            if not pth.is_file():
                continue

            data = self.preprocess(pth)

            with torch.no_grad():
                # Extract features: [1, 1536, 28, 28]
                features = self.model(data)

            # Reshape to [N_patches, Channels] -> [784, 1536]
            features = features.permute(
                0, 2, 3, 1).reshape(-1, features.shape[1])

            # --- CRITICAL FIX: Random Subsampling HERE, not later ---
            n_patches = features.shape[0]

            # Pick random indices
            if n_patches > patches_per_image:
                indices = torch.randperm(n_patches)[:patches_per_image]
                selected_features = features[indices]
            else:
                selected_features = features

            # Move to CPU immediately to free GPU memory
            features_list.append(selected_features.cpu())
            self.patch_paths.extend([str(pth)] * selected_features.shape[0])

        # Now stacking is safe because we only have the small subset
        self.memory_bank = torch.cat(features_list, dim=0)

        # Move bank to GPU for fast calculation (if it fits)
        # If this still OOMs on GPU, keep it on CPU: .to('cpu')
        try:
            self.memory_bank = self.memory_bank.to(self.device)
            print(f"Memory Bank moved to GPU. Shape: {self.memory_bank.shape}")
        except RuntimeError:  # GPU OOM
            self.memory_bank = self.memory_bank.cpu()
            print(
                f"GPU OOM. Memory Bank kept on CPU. Shape: {self.memory_bank.shape}")

    def predict(self, test_folder):
        """
        Scores images in the test folder.
        """
        print(f"Testing on {test_folder}...")
        test_path = Path(test_folder)
        scores = {}

        for pth in tqdm(list(test_path.iterdir())):
            if not pth.is_file():
                continue

            data = self.preprocess(pth)

            # 1. Extract test features
            features = self.model(data)

            # 2. Reshape [1, 1536, 28, 28] -> [784, 1536]
            # These are the 784 "patches" of the test image
            test_patches = features.permute(
                0, 2, 3, 1).reshape(-1, features.shape[1])

            # 3. Compute Distance to Nearest Memory Bank Patch
            # For every test patch, find the closest patch in the memory bank.
            # L2 Distance calculation

            # We compute distances in batches if memory bank is huge,
            # but for simple implementation we do it directly:
            # dist matrix: [TestPatches, MemoryBankSize]
            dist_matrix = torch.cdist(test_patches, self.memory_bank, p=2)

            # Min distance for each patch (how "normal" is this specific patch?)
            # values: [784], indices: [784]
            min_dists, _ = torch.min(dist_matrix, dim=1)

            # 4. Anomaly Score = The MAXIMUM distance found in the image.
            # Logic: If even ONE patch is very far from the memory bank, the image is anomalous.
            image_score = torch.max(min_dists).item()

            scores[pth.name] = image_score

        return scores
