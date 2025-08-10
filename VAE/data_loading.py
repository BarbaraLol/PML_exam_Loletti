import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import os
from torchvision import transforms
import random


class SpectrogramVAEDataset(Dataset):
    """Corrected dataset with efficient statistics computation"""
    
    def __init__(self, file_paths, label_encoder=None, transform=None, conditional=False):
        self.file_paths = [fp for fp in file_paths if self._is_valid_file(fp)]
        self.label_encoder = label_encoder
        self.transform = transform
        self.conditional = conditional
        
        # Compute dataset statistics for proper normalization
        self._compute_dataset_stats()
    
    def _is_valid_file(self, filepath):
        """Check if file is valid"""
        try:
            data = torch.load(filepath, map_location='cpu')
            return isinstance(data, dict) and 'spectrogram' in data
        except:
            print(f"Skipping corrupted file: {filepath}")
            return False
    
    def _compute_dataset_stats(self):
        """Compute dataset-wide statistics efficiently using streaming approach"""
        print("Computing dataset statistics for normalization...")
        
        # Use streaming statistics to avoid memory issues
        running_min = float('inf')
        running_max = float('-inf')
        running_sum = 0.0
        running_sum_sq = 0.0
        total_samples = 0
        
        # Sample fewer files but more systematically
        sample_size = min(50, len(self.file_paths))  # Reduced sample size
        step = max(1, len(self.file_paths) // sample_size)
        
        # Sample files uniformly across the dataset
        sampled_files = self.file_paths[::step][:sample_size]
        
        print(f"Computing stats from {len(sampled_files)} sampled files...")
        
        for i, fp in enumerate(sampled_files):
            if i % 10 == 0:
                print(f"Processing {i}/{len(sampled_files)} files for stats...")
                
            try:
                data = torch.load(fp, map_location='cpu')
                spec = data['spectrogram'].float()
                
                # Convert to magnitude if complex
                if torch.is_complex(spec):
                    spec = torch.abs(spec)
                
                # Remove negatives and apply log transform
                spec = torch.clamp(spec, min=1e-8)
                spec = torch.log(spec)
                
                # Update streaming statistics
                spec_flat = spec.flatten()
                batch_min = spec_flat.min().item()
                batch_max = spec_flat.max().item()
                batch_sum = spec_flat.sum().item()
                batch_sum_sq = (spec_flat ** 2).sum().item()
                batch_count = spec_flat.numel()
                
                # Update running statistics
                running_min = min(running_min, batch_min)
                running_max = max(running_max, batch_max)
                running_sum += batch_sum
                running_sum_sq += batch_sum_sq
                total_samples += batch_count
                
            except Exception as e:
                print(f"Error processing {fp} for stats: {e}")
                continue
        
        if total_samples > 0:
            # Compute final statistics
            self.dataset_mean = running_sum / total_samples
            variance = (running_sum_sq / total_samples) - (self.dataset_mean ** 2)
            self.dataset_std = max(np.sqrt(variance), 1e-6)  # Prevent division by zero
            
            # Use slightly more conservative bounds to handle outliers
            margin = 2 * self.dataset_std
            self.dataset_min = max(running_min, self.dataset_mean - 4 * self.dataset_std)
            self.dataset_max = min(running_max, self.dataset_mean + 4 * self.dataset_std)
        else:
            # Fallback values if no data processed
            print("Warning: Could not compute statistics, using fallback values")
            self.dataset_min = -10.0
            self.dataset_max = 5.0
            self.dataset_mean = -2.0
            self.dataset_std = 3.0
        
        print(f"Dataset stats computed from {total_samples:,} total values:")
        print(f"  Min: {self.dataset_min:.3f}")
        print(f"  Max: {self.dataset_max:.3f}")
        print(f"  Mean: {self.dataset_mean:.3f}")
        print(f"  Std: {self.dataset_std:.3f}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        try:
            data = torch.load(self.file_paths[idx], map_location='cpu')
            spectrogram = data['spectrogram'].float()
            
            # Proper preprocessing
            spectrogram = self._preprocess_spectrogram(spectrogram)
            
            # Add channel dimension if needed [1, freq, time]
            if spectrogram.dim() == 2:
                spectrogram = spectrogram.unsqueeze(0)
            
            # Apply transforms if any
            if self.transform:
                spectrogram = self.transform(spectrogram)
            
            # Handle conditional case
            if self.conditional and 'label' in data and self.label_encoder:
                label = torch.tensor(
                    self.label_encoder.transform([data['label']])[0],
                    dtype=torch.long
                )
                return spectrogram, label
            else:
                return spectrogram, spectrogram
                
        except Exception as e:
            print(f"Error loading {self.file_paths[idx]}: {e}")
            # Return zeros as fallback
            dummy_shape = (1, 1025, 938)  # Your spectrogram shape
            dummy_tensor = torch.zeros(dummy_shape)
            return dummy_tensor, dummy_tensor
    
    def _preprocess_spectrogram(self, spectrogram):
        """Apply proper preprocessing using dataset statistics"""
        
        # Handle complex spectrograms
        if torch.is_complex(spectrogram):
            spectrogram = torch.abs(spectrogram)
        
        # Remove negatives and apply log transform
        spectrogram = torch.clamp(spectrogram, min=1e-8)
        spectrogram = torch.log(spectrogram)
        
        # Normalize to [0, 1] using dataset statistics (robust approach)
        range_val = self.dataset_max - self.dataset_min
        if range_val > 1e-6:  # Avoid division by zero
            spectrogram = (spectrogram - self.dataset_min) / range_val
        else:
            spectrogram = spectrogram - self.dataset_min
        
        spectrogram = torch.clamp(spectrogram, 0, 1)
        
        # Check for NaN/Inf and handle
        if torch.isnan(spectrogram).any() or torch.isinf(spectrogram).any():
            print(f"NaN/Inf detected after preprocessing, replacing with zeros")
            spectrogram = torch.zeros_like(spectrogram)
        
        return spectrogram


class SpectrogramDataAugmentation:
    """Light data augmentation for spectrograms"""
    
    @staticmethod
    def get_transforms(augment_prob=0.3):
        """Get augmentation transforms with specified probability"""
        return transforms.Compose([
            SpectrogramDataAugmentation.AddNoise(augment_prob),
            SpectrogramDataAugmentation.FrequencyMask(augment_prob),
            SpectrogramDataAugmentation.TimeMask(augment_prob),
        ])
    
    class AddNoise:
        def __init__(self, prob=0.3, noise_level=0.01):
            self.prob = prob
            self.noise_level = noise_level
            
        def __call__(self, spectrogram):
            if random.random() < self.prob:
                noise = torch.randn_like(spectrogram) * self.noise_level
                return torch.clamp(spectrogram + noise, 0, 1)
            return spectrogram
    
    class FrequencyMask:
        def __init__(self, prob=0.2, max_mask_size=15):
            self.prob = prob
            self.max_mask_size = max_mask_size
            
        def __call__(self, spectrogram):
            if random.random() < self.prob:
                freq_size = spectrogram.shape[-2]
                mask_size = random.randint(1, min(self.max_mask_size, freq_size // 8))
                mask_start = random.randint(0, freq_size - mask_size)
                
                spectrogram_masked = spectrogram.clone()
                spectrogram_masked[..., mask_start:mask_start + mask_size, :] = 0
                return spectrogram_masked
            return spectrogram
    
    class TimeMask:
        def __init__(self, prob=0.2, max_mask_size=25):
            self.prob = prob
            self.max_mask_size = max_mask_size
            
        def __call__(self, spectrogram):
            if random.random() < self.prob:
                time_size = spectrogram.shape[-1]
                mask_size = random.randint(1, min(self.max_mask_size, time_size // 8))
                mask_start = random.randint(0, time_size - mask_size)
                
                spectrogram_masked = spectrogram.clone()
                spectrogram_masked[..., :, mask_start:mask_start + mask_size] = 0
                return spectrogram_masked
            return spectrogram


def load_file_paths(data_dir):
    """Load all .pt file paths from directory"""
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    file_paths = []
    for f in os.listdir(data_dir):
        if f.endswith('.pt'):
            file_paths.append(os.path.join(data_dir, f))
    
    if not file_paths:
        raise ValueError(f"No .pt files found in {data_dir}")
    
    print(f"Found {len(file_paths)} .pt files")
    return file_paths


def encode_labels(file_paths):
    """Extract labels from files for label encoder"""
    labels = []
    for fp in file_paths:
        try:
            data = torch.load(fp, map_location='cpu')
            if isinstance(data, dict) and 'label' in data:
                labels.append(data['label'])
        except:
            continue
    
    unique_labels = list(set(labels))
    print(f"Found labels: {unique_labels}")
    return labels


def get_spectrogram_shape(file_paths):
    """Get the shape of spectrograms from the first valid file"""
    for fp in file_paths:
        try:
            data = torch.load(fp, map_location='cpu')
            if isinstance(data, dict) and 'spectrogram' in data:
                shape = data['spectrogram'].shape
                print(f"Spectrogram shape: {shape}")
                return shape
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
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        augment: Whether to apply data augmentation to training set
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, spectrogram_shape, num_classes)
    """
    
    file_paths = load_file_paths(data_dir)
    spectrogram_shape = get_spectrogram_shape(file_paths)
    
    print(f"Creating datasets with {len(file_paths)} files")
    print(f"Spectrogram shape: {spectrogram_shape}")
    
    # Create base dataset
    base_dataset = SpectrogramVAEDataset(
        file_paths,
        label_encoder=label_encoder,
        conditional=conditional
    )
    
    # Split dataset
    total_size = len(base_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Dataset splits: train={train_size}, val={val_size}, test={test_size}")
    
    # Set random seed for reproducible splits
    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        base_dataset, [train_size, val_size, test_size]
    )
    
    # Apply augmentation to training set only
    if augment:
        print("Applying data augmentation to training set")
        # Create augmented training dataset
        train_file_paths = [file_paths[i] for i in train_dataset.indices]
        train_dataset = SpectrogramVAEDataset(
            train_file_paths,
            label_encoder=label_encoder,
            conditional=conditional,
            transform=SpectrogramDataAugmentation.get_transforms()
        )
    
    num_classes = len(label_encoder.classes_) if label_encoder else 0
    
    return train_dataset, val_dataset, test_dataset, spectrogram_shape, num_classes