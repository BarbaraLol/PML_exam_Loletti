import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from torchvision import transforms
import random

class SpectrogramVAEDataset(Dataset):
    """Dataset for VAE training - focuses on spectrograms for reconstruction"""
    def __init__(self, file_paths, label_encoder=None, transform=None, conditional=False):
        self.label_encoder = label_encoder
        self.transform = transform
        self.conditional = conditional
        self.file_paths = [fp for fp in file_paths if self._is_valid_file(fp)]

    def _is_valid_file(self, filepath):
        try:
            data = torch.load(filepath)
            return isinstance(data, dict) and 'spectrogram' in data
        except:
            print(f"Skipping corrupted file: {filepath}")
            return False

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx])
        spectrogram = data['spectrogram'].float()
        
        # FIXED: Better normalization approach
        # Remove extreme outliers first
        spectrogram = torch.clamp(spectrogram, 
                                percentile=torch.quantile(spectrogram, 0.01), 
                                max=torch.quantile(spectrogram, 0.99))
        
        # Normalize to [0, 1] first, then to [-1, 1]
        spec_min = spectrogram.min()
        spec_max = spectrogram.max()
        if spec_max > spec_min:
            spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)
        spectrogram = 2 * spectrogram - 1  # Scale to [-1, 1]
        
        # Check for NaN/Inf and handle
        if torch.isnan(spectrogram).any() or torch.isinf(spectrogram).any():
            print(f"NaN/Inf detected in: {self.file_paths[idx]}")
            spectrogram = torch.zeros_like(spectrogram)
        
        # Apply transforms (data augmentation)
        if self.transform:
            spectrogram = self.transform(spectrogram)
            
        # Add channel dimension if needed [1, height, width]
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)
        
        if self.conditional and 'label' in data and self.label_encoder is not None:
            label = torch.tensor(
                self.label_encoder.transform([data['label']])[0],
                dtype=torch.long
            )
            return spectrogram, label
        else:
            return spectrogram, spectrogram

        


class SpectrogramDataAugmentation:
    """Data augmentation transforms for spectrograms"""
    
    @staticmethod
    def get_transforms():
        return transforms.Compose([
            SpectrogramDataAugmentation.AddNoise(),
            SpectrogramDataAugmentation.FrequencyMask(),
            SpectrogramDataAugmentation.TimeMask(),
        ])
    
    class AddNoise:
        def __init__(self, noise_level=0.01):
            self.noise_level = noise_level
            
        def __call__(self, spectrogram):
            if random.random() < 0.3:  # 30% chance
                noise = torch.randn_like(spectrogram) * self.noise_level
                return torch.clamp(spectrogram + noise, -1, 1)
            return spectrogram
    
    class FrequencyMask:
        def __init__(self, max_mask_size=10):
            self.max_mask_size = max_mask_size
            
        def __call__(self, spectrogram):
            if random.random() < 0.2:  # 20% chance
                freq_size = spectrogram.shape[-2]
                mask_size = random.randint(1, min(self.max_mask_size, freq_size // 4))
                mask_start = random.randint(0, freq_size - mask_size)
                
                spectrogram_masked = spectrogram.clone()
                spectrogram_masked[..., mask_start:mask_start + mask_size, :] = 0
                return spectrogram_masked
            return spectrogram
    
    class TimeMask:
        def __init__(self, max_mask_size=20):
            self.max_mask_size = max_mask_size
            
        def __call__(self, spectrogram):
            if random.random() < 0.2:  # 20% chance
                time_size = spectrogram.shape[-1]
                mask_size = random.randint(1, min(self.max_mask_size, time_size // 4))
                mask_start = random.randint(0, time_size - mask_size)
                
                spectrogram_masked = spectrogram.clone()
                spectrogram_masked[..., :, mask_start:mask_start + mask_size] = 0
                return spectrogram_masked
            return spectrogram


def load_file_paths(data_dir):
    """Load all .pt file paths from directory"""
    return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]


def encode_labels(file_paths):
    """Extract labels from files for label encoder"""
    labels = []
    for fp in file_paths:
        try:
            data = torch.load(fp)
            if isinstance(data, dict) and 'label' in data:
                labels.append(data['label'])
        except:
            continue
    return labels


def get_spectrogram_shape(file_paths):
    """Get the shape of spectrograms from the first valid file"""
    for fp in file_paths:
        try:
            data = torch.load(fp)
            if isinstance(data, dict) and 'spectrogram' in data:
                return data['spectrogram'].shape
        except:
            continue
    raise ValueError("No valid spectrogram files found")


def create_vae_datasets(data_dir, label_encoder=None, conditional=False, 
                       train_ratio=0.7, val_ratio=0.15, augment=True):
    """
    Create train, validation, and test datasets for VAE training
    
    Args:
        data_dir: Directory containing .pt files
        label_encoder: LabelEncoder for conditional VAE
        conditional: Whether to use conditional VAE
        train_ratio: Ratio for training set16
        val_ratio: Ratio for validation set
        augment: Whether to apply data augmentation to training set
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, spectrogram_shape, num_classes)
    """
    from torch.utils.data import random_split
    
    file_paths = load_file_paths(data_dir)
    spectrogram_shape = get_spectrogram_shape(file_paths)
    
    # Get transforms
    train_transform = SpectrogramDataAugmentation.get_transforms() if augment else None
    val_transform = None  # No augmentation for validation/test
    
    # Create full dataset
    full_dataset = SpectrogramVAEDataset(
        file_paths, 
        label_encoder=label_encoder, 
        conditional=conditional
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Apply transforms to training dataset
    if train_transform:
        train_dataset.dataset.transform = train_transform
    
    num_classes = len(label_encoder.classes_) if label_encoder else 0
    
    return train_dataset, val_dataset, test_dataset, spectrogram_shape, num_classes