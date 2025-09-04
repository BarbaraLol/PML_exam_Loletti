import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
import random
from sklearn.preprocessing import LabelEncoder
import librosa
import soundfile as sf


class ImprovedSpectrogramDataset(Dataset):
    """Improved dataset with robust loading and preprocessing"""
    
    def __init__(self, file_paths, label_encoder=None, transform=None, conditional=False, target_shape=(128, 938)):
        print(f"Initializing improved dataset with {len(file_paths)} files...")
        
        self.file_paths = []
        self.target_shape = target_shape  # (freq_bins, time_frames)
        self.conditional = conditional
        self.label_encoder = label_encoder
        self.transform = transform
        
        # Validate files
        invalid_files = 0
        for i, fp in enumerate(file_paths):
            if i % 500 == 0:
                print(f"Validating file {i+1}/{len(file_paths)}...")
            
            if self._is_valid_file(fp):
                self.file_paths.append(fp)
            else:
                invalid_files += 1
        
        print(f"Found {len(self.file_paths)} valid files ({invalid_files} invalid)")
        
        if len(self.file_paths) == 0:
            raise ValueError("No valid spectrogram files found!")
        
        # Compute normalization statistics
        self._compute_normalization_stats()
    
    def _is_valid_file(self, filepath):
        """Check if file is valid"""
        try:
            if not os.path.exists(filepath):
                return False
            
            # Try to load metadata
            data = torch.load(filepath, map_location='cpu', weights_only=False)
            
            if not isinstance(data, dict) or 'spectrogram' not in data:
                return False
            
            spec = data['spectrogram']
            if not isinstance(spec, torch.Tensor):
                return False
            
            if spec.dim() < 2 or spec.numel() == 0:
                return False
            
            # Check for reasonable values
            if torch.isnan(spec).any() or torch.isinf(spec).any():
                return False
            
            return True
            
        except Exception:
            return False
    
    def _compute_normalization_stats(self):
        """Compute dataset statistics for normalization"""
        print("Computing normalization statistics...")
        
        # Sample files for statistics
        sample_size = min(200, len(self.file_paths))
        sample_files = random.sample(self.file_paths, sample_size)
        
        all_values = []
        for file_path in sample_files:
            try:
                data = torch.load(file_path, map_location='cpu')
                spec = data['spectrogram'].float()
                
                # Resize to target shape
                spec = self._resize_spectrogram(spec)
                
                # Sample values to avoid memory issues
                flat_values = spec.flatten()
                if len(flat_values) > 5000:
                    indices = torch.randperm(len(flat_values))[:5000]
                    sampled_values = flat_values[indices]
                else:
                    sampled_values = flat_values
                
                all_values.append(sampled_values)
                
            except Exception as e:
                print(f"Error processing {file_path} for stats: {e}")
                continue
        
        if all_values:
            all_values = torch.cat(all_values)
            self.dataset_mean = float(all_values.mean())
            self.dataset_std = float(all_values.std())
            self.dataset_min = float(all_values.min())
            self.dataset_max = float(all_values.max())
        else:
            # Fallback values
            self.dataset_mean = 0.5
            self.dataset_std = 0.2
            self.dataset_min = 0.0
            self.dataset_max = 1.0
        
        print(f"Dataset statistics:")
        print(f"  Mean: {self.dataset_mean:.4f}")
        print(f"  Std: {self.dataset_std:.4f}")
        print(f"  Min: {self.dataset_min:.4f}")
        print(f"  Max: {self.dataset_max:.4f}")
    
    def _resize_spectrogram(self, spectrogram):
        """Resize spectrogram to target shape"""
        if spectrogram.shape[-2:] == self.target_shape:
            return spectrogram
        
        # Add batch and channel dimensions if needed
        original_shape = spectrogram.shape
        if spectrogram.dim() == 2:
            spec_4d = spectrogram.unsqueeze(0).unsqueeze(0)
        elif spectrogram.dim() == 3:
            spec_4d = spectrogram.unsqueeze(0)
        else:
            spec_4d = spectrogram
        
        # Resize using interpolation
        resized = F.interpolate(
            spec_4d,
            size=self.target_shape,
            mode='bilinear',
            align_corners=False
        )
        
        # Return to original dimensionality
        if len(original_shape) == 2:
            return resized.squeeze(0).squeeze(0)
        elif len(original_shape) == 3:
            return resized.squeeze(0)
        else:
            return resized
    
    def _normalize_spectrogram(self, spectrogram):
        """Normalize spectrogram"""
        # Ensure values are in [0, 1] range
        spec_min = spectrogram.min()
        spec_max = spectrogram.max()
        
        if spec_max > spec_min:
            spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)
        
        # Additional smoothing to reduce artifacts
        spectrogram = torch.clamp(spectrogram, 0, 1)
        
        return spectrogram
    
    def _extract_label(self, filepath):
        """Extract label from filepath"""
        try:
            data = torch.load(filepath, map_location='cpu')
            if 'label' in data and data['label']:
                return data['label']
        except:
            pass
        
        # Fallback: extract from path
        path_str = str(filepath).lower()
        if 'chick' in path_str:
            return 'chick'
        elif 'adult' in path_str:
            return 'adult'
        elif 'noise' in path_str:
            return 'noise'
        else:
            return 'unknown'
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Select file (with fallback on retry)
                file_idx = (idx + attempt) % len(self.file_paths)
                file_path = self.file_paths[file_idx]
                
                # Load data
                data = torch.load(file_path, map_location='cpu')
                spectrogram = data['spectrogram'].float()
                
                # Preprocess spectrogram
                spectrogram = self._resize_spectrogram(spectrogram)
                spectrogram = self._normalize_spectrogram(spectrogram)
                
                # Add channel dimension if needed
                if spectrogram.dim() == 2:
                    spectrogram = spectrogram.unsqueeze(0)
                
                # Apply transforms
                if self.transform:
                    spectrogram = self.transform(spectrogram)
                
                # Validate result
                if torch.isnan(spectrogram).any() or torch.isinf(spectrogram).any():
                    raise ValueError("NaN/Inf in spectrogram")
                
                # Return based on conditional mode
                if self.conditional:
                    label = self._extract_label(file_path)
                    if self.label_encoder is not None:
                        try:
                            encoded_label = self.label_encoder.transform([label])[0]
                            return spectrogram, torch.tensor(encoded_label, dtype=torch.long)
                        except:
                            return spectrogram, torch.tensor(0, dtype=torch.long)
                    else:
                        return spectrogram, torch.tensor(0, dtype=torch.long)
                else:
                    return spectrogram
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to load {self.file_paths[file_idx]}: {e}")
                    # Return dummy data
                    dummy_spec = torch.rand(1, *self.target_shape) * 0.5 + 0.25
                    if self.conditional:
                        return dummy_spec, torch.tensor(0, dtype=torch.long)
                    else:
                        return dummy_spec
                continue
        
        # This shouldn't be reached
        dummy_spec = torch.rand(1, *self.target_shape) * 0.5 + 0.25
        if self.conditional:
            return dummy_spec, torch.tensor(0, dtype=torch.long)
        else:
            return dummy_spec


class AudioDataAugmentation:
    """Audio-specific data augmentation for spectrograms"""
    
    @staticmethod
    def get_transforms(augment_prob=0.3):
        """Get augmentation transforms"""
        return AudioAugmentationCompose([
            AddGaussianNoise(prob=augment_prob, noise_level=0.02),
            FrequencyMask(prob=augment_prob, max_mask_size=8),
            TimeMask(prob=augment_prob, max_mask_size=15),
            SpecAugment(prob=augment_prob),
        ])


class AudioAugmentationCompose:
    """Compose multiple augmentations"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, spectrogram):
        for transform in self.transforms:
            spectrogram = transform(spectrogram)
        return spectrogram


class AddGaussianNoise:
    """Add Gaussian noise to spectrogram"""
    def __init__(self, prob=0.3, noise_level=0.02):
        self.prob = prob
        self.noise_level = noise_level
    
    def __call__(self, spectrogram):
        if random.random() < self.prob:
            noise = torch.randn_like(spectrogram) * self.noise_level
            return torch.clamp(spectrogram + noise, 0, 1)
        return spectrogram


class FrequencyMask:
    """Mask frequency bands"""
    def __init__(self, prob=0.2, max_mask_size=8):
        self.prob = prob
        self.max_mask_size = max_mask_size
    
    def __call__(self, spectrogram):
        if random.random() < self.prob:
            freq_size = spectrogram.shape[-2]
            mask_size = random.randint(1, min(self.max_mask_size, freq_size // 8))
            mask_start = random.randint(0, freq_size - mask_size)
            
            spec_masked = spectrogram.clone()
            spec_masked[..., mask_start:mask_start + mask_size, :] = 0
            return spec_masked
        return spectrogram


class TimeMask:
    """Mask time frames"""
    def __init__(self, prob=0.2, max_mask_size=15):
        self.prob = prob
        self.max_mask_size = max_mask_size
    
    def __call__(self, spectrogram):
        if random.random() < self.prob:
            time_size = spectrogram.shape[-1]
            mask_size = random.randint(1, min(self.max_mask_size, time_size // 10))
            mask_start = random.randint(0, time_size - mask_size)
            
            spec_masked = spectrogram.clone()
            spec_masked[..., mask_start:mask_start + mask_size] = 0
            return spec_masked
        return spectrogram


class SpecAugment:
    """SpecAugment-style augmentation"""
    def __init__(self, prob=0.2):
        self.prob = prob
    
    def __call__(self, spectrogram):
        if random.random() < self.prob:
            # Apply random scaling
            scale = random.uniform(0.8, 1.2)
            spectrogram = spectrogram * scale
            spectrogram = torch.clamp(spectrogram, 0, 1)
            
            # Apply random time warping (simple version)
            if random.random() < 0.5:
                # Time stretch/compress
                time_factor = random.uniform(0.9, 1.1)
                new_time_size = int(spectrogram.shape[-1] * time_factor)
                
                if new_time_size != spectrogram.shape[-1]:
                    spectrogram = F.interpolate(
                        spectrogram.unsqueeze(0),
                        size=(spectrogram.shape[-2], new_time_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    
                    # Crop or pad to original size
                    if new_time_size > spectrogram.shape[-1]:
                        # Crop
                        start_idx = (new_time_size - spectrogram.shape[-1]) // 2
                        spectrogram = spectrogram[..., start_idx:start_idx + spectrogram.shape[-1]]
                    elif new_time_size < spectrogram.shape[-1]:
                        # Pad
                        pad_size = spectrogram.shape[-1] - new_time_size
                        spectrogram = F.pad(spectrogram, (0, pad_size))
        
        return spectrogram


def load_file_paths(data_dir):
    """Load all .pt file paths from directory"""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    file_paths = list(data_path.glob('*.pt'))
    
    if not file_paths:
        raise ValueError(f"No .pt files found in {data_dir}")
    
    print(f"Found {len(file_paths)} .pt files in {data_dir}")
    return file_paths


def extract_labels_from_files(file_paths):
    """Extract all unique labels from files"""
    labels = set()
    
    for file_path in file_paths[:100]:  # Sample first 100 files
        try:
            data = torch.load(file_path, map_location='cpu')
            if isinstance(data, dict) and 'label' in data and data['label']:
                labels.add(data['label'])
            else:
                # Extract from path
                path_str = str(file_path).lower()
                if 'chick' in path_str:
                    labels.add('chick')
                elif 'adult' in path_str:
                    labels.add('adult')
                elif 'noise' in path_str:
                    labels.add('noise')
                else:
                    labels.add('unknown')
        except:
            continue
    
    labels = list(labels)
    print(f"Found labels: {labels}")
    return labels


def create_improved_vae_datasets(data_dir, conditional=False, train_ratio=0.7, val_ratio=0.2, augment=True):
    """Create improved datasets for VAE training"""
    
    print("Creating improved VAE datasets...")
    
    # Load file paths
    file_paths = load_file_paths(data_dir)
    
    # Setup label encoder for conditional VAE
    label_encoder = None
    num_classes = 0
    
    if conditional:
        labels = extract_labels_from_files(file_paths)
        if labels:
            label_encoder = LabelEncoder()
            label_encoder.fit(labels)
            num_classes = len(label_encoder.classes_)
            print(f"Conditional VAE: {num_classes} classes - {list(label_encoder.classes_)}")
        else:
            print("No labels found, switching to standard VAE")
            conditional = False