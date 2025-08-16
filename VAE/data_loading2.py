import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import os
from torchvision import transforms
import random


class SpectrogramVAEDataset(Dataset):
    """Simplified dataset with robust preprocessing for simple VAE"""
    
    def __init__(self, file_paths, label_encoder=None, transform=None, conditional=False):
        print(f"Initializing dataset with {len(file_paths)} files...")
        
        self.file_paths = []
        invalid_files = 0
        
        # Validate files
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
        
        self.label_encoder = label_encoder
        self.transform = transform
        self.conditional = conditional
        
        # Get a sample to determine data statistics and shape
        self._determine_data_properties()
    
    def _is_valid_file(self, filepath):
        """Check if file is valid"""
        try:
            if not os.path.exists(filepath):
                return False
                
            # Load and check basic properties
            data = torch.load(filepath, map_location='cpu', weights_only=True)
            
            if not isinstance(data, dict) or 'spectrogram' not in data:
                return False
            
            spec = data['spectrogram']
            if not isinstance(spec, torch.Tensor):
                return False
                
            if spec.dim() < 2 or spec.numel() == 0:
                return False
                
            # Check for basic validity
            if torch.isnan(spec).all() or torch.isinf(spec).all():
                return False
            
            return True
            
        except Exception as e:
            return False
    
    def _determine_data_properties(self):
        """Determine data properties from sample files"""
        print("Analyzing data properties...")
        
        sample_values = []
        sample_shapes = []
        
        # Sample a few files to understand the data
        sample_size = min(10, len(self.file_paths))
        sample_files = random.sample(self.file_paths, sample_size)
        
        for filepath in sample_files:
            try:
                data = torch.load(filepath, map_location='cpu')
                spec = data['spectrogram'].float()
                
                # Handle complex spectrograms
                if torch.is_complex(spec):
                    spec = torch.abs(spec)
                
                sample_shapes.append(spec.shape)
                
                # Sample some values for statistics
                flat = spec.flatten()
                if flat.numel() > 1000:
                    indices = torch.randperm(flat.numel())[:1000]
                    values = flat[indices]
                else:
                    values = flat
                
                sample_values.append(values)
                
            except Exception as e:
                print(f"Error sampling {filepath}: {e}")
                continue
        
        # Calculate statistics
        if sample_values:
            all_values = torch.cat(sample_values)
            self.data_min = all_values.min().item()
            self.data_max = all_values.max().item()
            self.data_mean = all_values.mean().item()
            self.data_std = all_values.std().item()
        else:
            # Fallback values
            self.data_min = -80.0
            self.data_max = 0.0
            self.data_mean = -40.0
            self.data_std = 20.0
        
        # Determine common shape
        if sample_shapes:
            raw_shape = sample_shapes[0]
            # Store as tuple for model compatibility
            if len(raw_shape) >= 2:
                self.expected_shape = (raw_shape[-2], raw_shape[-1])  # (height, width)
            else:
                self.expected_shape = raw_shape
        else:
            raise ValueError("Could not determine spectrogram shape")
        
        print(f"Data properties determined:")
        print(f"  Shape: {self.expected_shape}")
        print(f"  Min: {self.data_min:.2f}")
        print(f"  Max: {self.data_max:.2f}")
        print(f"  Mean: {self.data_mean:.2f}")
        print(f"  Std: {self.data_std:.2f}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Try the requested index first, then fallbacks
                file_idx = idx if attempt == 0 else (idx + attempt) % len(self.file_paths)
                
                data = torch.load(self.file_paths[file_idx], map_location='cpu')
                spectrogram = data['spectrogram'].float()
                
                # Ensure spectrogram is at least 2D
                if spectrogram.dim() == 1:
                    spectrogram = spectrogram.unsqueeze(0)
                    
                # Convert to 4D: [1, 1, H, W] for Conv2d
                if spectrogram.dim() == 2:
                    spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                elif spectrogram.dim() == 3:
                    spectrogram = spectrogram.unsqueeze(1)  # [1, C, H, W] -> [1, 1, H, W] if C==1
                
                # Verify final shape
                assert spectrogram.dim() == 4, f"Expected 4D tensor, got {spectrogram.dim()}D"
                assert spectrogram.shape[1] == 1, f"Expected 1 channel, got {spectrogram.shape[1]}"
                
                # Preprocess the spectrogram
                spectrogram = self._preprocess_spectrogram(spectrogram)
                
                # Apply transforms if any
                if self.transform:
                    spectrogram = self.transform(spectrogram)
                
                # Final validation
                if torch.isnan(spectrogram).any() or torch.isinf(spectrogram).any():
                    raise ValueError("NaN/Inf in preprocessed spectrogram")
                
                # Handle conditional case
                if self.conditional and 'label' in data and self.label_encoder:
                    try:
                        label = self.label_encoder.transform([data['label']])[0]
                        return spectrogram, torch.tensor(label, dtype=torch.long)
                    except:
                        return spectrogram, torch.tensor(0, dtype=torch.long)
                else:
                    return spectrogram
                    
            except Exception as e:
                print(f"Error loading {self.file_paths[file_idx]} (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    # Last resort: return a dummy tensor with correct shape
                    print(f"Creating dummy data for index {idx}")
                    dummy_shape = (1, 1, *self.expected_shape)  # [1, 1, 1025, 469]
                    dummy_tensor = torch.rand(dummy_shape) * 0.5 + 0.25  # Values in [0.25, 0.75]
                    
                    if self.conditional:
                        return dummy_tensor, torch.tensor(0, dtype=torch.long)
                    else:
                        return dummy_tensor
                continue

             # Force consistent output shape
            if spectrogram.shape[-2:] != self.expected_shape:
                spectrogram = F.interpolate(
                    spectrogram.unsqueeze(0).unsqueeze(0),
                    size=self.expected_shape,
                    mode='bilinear'
                ).squeeze()
            
            # Ensure 4D output [1,1,H,W]
            if spectrogram.dim() == 2:
                spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)
            elif spectrogram.dim() == 3:
                spectrogram = spectrogram.unsqueeze(1)
            
            return spectrogram
    
    def _preprocess_spectrogram(self, spectrogram):
        """Simplified preprocessing for spectrograms"""
        # Handle complex spectrograms
        if torch.is_complex(spectrogram):
            spectrogram = torch.abs(spectrogram)
        
        # Clean up NaN/Inf values
        if torch.isnan(spectrogram).any() or torch.isinf(spectrogram).any():
            spectrogram = torch.nan_to_num(spectrogram, nan=self.data_mean, posinf=self.data_max, neginf=self.data_min)
        
        # Simple normalization to [0, 1]
        # Assuming spectrogram is in dB scale
        spec_min = spectrogram.min()
        spec_max = spectrogram.max()
        
        if spec_max > spec_min:
            # Min-max normalization
            spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)
        else:
            # If constant values, set to 0.5
            spectrogram = torch.full_like(spectrogram, 0.5)
        
        # Ensure values are in [0, 1]
        spectrogram = torch.clamp(spectrogram, 0, 1)
        
        return spectrogram
    
    def _resize_spectrogram(self, spectrogram, target_shape):
        """Resize spectrogram to target shape if needed"""
        if len(target_shape) != 2:
            return spectrogram
            
        target_h, target_w = target_shape
        current_h, current_w = spectrogram.shape[-2:]
        
        # Simple resize using interpolation
        if current_h != target_h or current_w != target_w:
            # Add batch and channel dimensions for interpolation
            spec_4d = spectrogram.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            # Resize using bilinear interpolation
            resized = torch.nn.functional.interpolate(
                spec_4d, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Remove added dimensions
            spectrogram = resized.squeeze(0).squeeze(0)
        
        return spectrogram


class SpectrogramDataAugmentation:
    """Lightweight data augmentation for spectrograms"""
    
    @staticmethod
    def get_transforms(augment_prob=0.2):
        """Get augmentation transforms with lower probability for stability"""
        return transforms.Compose([
            SpectrogramDataAugmentation.AddNoise(augment_prob * 0.5),
            SpectrogramDataAugmentation.FrequencyMask(augment_prob),
            SpectrogramDataAugmentation.TimeMask(augment_prob),
        ])
    
    class AddNoise:
        def __init__(self, prob=0.1, noise_level=0.005):  # Reduced noise
            self.prob = prob
            self.noise_level = noise_level
            
        def __call__(self, spectrogram):
            if random.random() < self.prob:
                noise = torch.randn_like(spectrogram) * self.noise_level
                return torch.clamp(spectrogram + noise, 0, 1)
            return spectrogram
    
    class FrequencyMask:
        def __init__(self, prob=0.15, max_mask_size=10):  # Smaller masks
            self.prob = prob
            self.max_mask_size = max_mask_size
            
        def __call__(self, spectrogram):
            if random.random() < self.prob:
                freq_size = spectrogram.shape[-2]
                mask_size = random.randint(1, min(self.max_mask_size, freq_size // 10))
                mask_start = random.randint(0, freq_size - mask_size)
                
                spectrogram_masked = spectrogram.clone()
                spectrogram_masked[..., mask_start:mask_start + mask_size, :] *= 0.1  # Don't zero completely
                return spectrogram_masked
            return spectrogram
    
    class TimeMask:
        def __init__(self, prob=0.15, max_mask_size=15):  # Smaller masks
            self.prob = prob
            self.max_mask_size = max_mask_size
            
        def __call__(self, spectrogram):
            if random.random() < self.prob:
                time_size = spectrogram.shape[-1]
                mask_size = random.randint(1, min(self.max_mask_size, time_size // 10))
                mask_start = random.randint(0, time_size - mask_size)
                
                spectrogram_masked = spectrogram.clone()
                spectrogram_masked[..., :, mask_start:mask_start + mask_size] *= 0.1  # Don't zero completely
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
    print("Extracting labels from files...")
    
    for i, fp in enumerate(file_paths):
        if i % 1000 == 0:
            print(f"Processing file {i+1}/{len(file_paths)} for labels...")
            
        try:
            data = torch.load(fp, map_location='cpu')
            if isinstance(data, dict) and 'label' in data:
                labels.append(data['label'])
        except:
            continue
    
    unique_labels = list(set(labels))
    print(f"Found {len(labels)} labeled files with {len(unique_labels)} unique labels: {unique_labels}")
    return labels


def inspect_spectrogram_files(data_dir, num_samples=5):
    """Debug function to inspect your spectrogram files"""
    print(f"🔍 INSPECTING SPECTROGRAM FILES")
    print("="*50)
    
    file_paths = load_file_paths(data_dir)
    sample_files = file_paths[:num_samples]
    
    for i, fp in enumerate(sample_files):
        try:
            print(f"\nFile {i+1}: {os.path.basename(fp)}")
            data = torch.load(fp, map_location='cpu')
            
            print(f"  Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            
            if isinstance(data, dict) and 'spectrogram' in data:
                spec = data['spectrogram']
                print(f"  Shape: {spec.shape}")
                print(f"  Dtype: {spec.dtype}")
                print(f"  Is complex: {torch.is_complex(spec)}")
                
                if torch.is_complex(spec):
                    spec_real = torch.abs(spec)
                else:
                    spec_real = spec.float()
                
                print(f"  Min: {spec_real.min().item():.3f}")
                print(f"  Max: {spec_real.max().item():.3f}")
                print(f"  Mean: {spec_real.mean().item():.3f}")
                print(f"  Std: {spec_real.std().item():.3f}")
                
                # Check for problematic values
                nan_count = torch.isnan(spec_real).sum().item()
                inf_count = torch.isinf(spec_real).sum().item()
                zero_count = (spec_real == 0).sum().item()
                
                print(f"  NaN values: {nan_count}")
                print(f"  Inf values: {inf_count}")
                print(f"  Zero values: {zero_count}")
                
                if spec_real.numel() > 0:
                    unique_vals = torch.unique(spec_real).numel()
                    print(f"  Unique values: {unique_vals}/{spec_real.numel()}")
                    
                    if unique_vals == 1:
                        print(f"  ⚠️  WARNING: All values are identical!")
                
            if 'label' in data:
                print(f"  Label: {data['label']}")
                
        except Exception as e:
            print(f"  ❌ Error loading: {e}")
    
    print("\n" + "="*50)


def get_spectrogram_shape(file_paths):
    """Get the shape of spectrograms from the first valid file"""
    print("Determining spectrogram shape...")
    
    for fp in file_paths[:10]:  # Check first 10 files
        try:
            data = torch.load(fp, map_location='cpu')
            if isinstance(data, dict) and 'spectrogram' in data:
                shape = data['spectrogram'].shape
                print(f"Found spectrogram shape: {shape} from {os.path.basename(fp)}")
                # Return as tuple for model compatibility
                if len(shape) >= 2:
                    return (shape[-2], shape[-1])  # Return (height, width)
                else:
                    return shape
        except Exception as e:
            print(f"Error checking {fp}: {e}")
            continue
    
    raise ValueError("No valid spectrogram files found to determine shape")


def create_vae_datasets(data_dir, label_encoder=None, conditional=False, 
                       train_ratio=0.7, val_ratio=0.15, augment=True):
    """
    Create train, validation, and test datasets for simple VAE training
    """
    
    print("🔍 Starting dataset creation...")
    
    # First, inspect a few files
    inspect_spectrogram_files(data_dir, num_samples=3)
    
    file_paths = load_file_paths(data_dir)
    spectrogram_shape = get_spectrogram_shape(file_paths)
    
    print(f"Creating datasets with {len(file_paths)} files")
    print(f"Spectrogram shape: {spectrogram_shape}")
    print(f"Conditional: {conditional}")
    
    # Create base dataset
    try:
        print("Creating base dataset...")
        base_dataset = SpectrogramVAEDataset(
            file_paths,
            label_encoder=label_encoder,
            conditional=conditional
        )
        
        print(f"Successfully created base dataset with {len(base_dataset)} samples")
        
        # Test loading a few samples
        print("Testing dataset loading...")
        test_samples = min(3, len(base_dataset))
        for i in range(test_samples):
            try:
                sample = base_dataset[i]
                if conditional:
                    if isinstance(sample, tuple) and len(sample) == 2:
                        spec, label = sample
                        print(f"  Sample {i}: spec shape {spec.shape}, label = {label.item()}")
                    else:
                        print(f"  Sample {i}: unexpected format - {type(sample)}")
                else:
                    if isinstance(sample, torch.Tensor):
                        print(f"  Sample {i}: spec shape {sample.shape}")
                    else:
                        print(f"  Sample {i}: unexpected format - {type(sample)}")
                        
            except Exception as e:
                print(f"    ❌ Error loading sample {i}: {e}")
        
    except Exception as e:
        print(f"❌ Failed to create base dataset: {e}")
        raise
    
    # Split dataset
    total_size = len(base_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Dataset splits: train={train_size}, val={val_size}, test={test_size}")
    
    # Ensure minimum sizes
    if train_size < 1:
        raise ValueError(f"Training set too small: {train_size}")
    if val_size < 1:
        val_size = 1
        test_size = total_size - train_size - val_size
    
    # Set random seed for reproducible splits
    torch.manual_seed(42)
    
    try:
        train_dataset, val_dataset, test_dataset = random_split(
            base_dataset, [train_size, val_size, test_size]
        )
    except Exception as e:
        print(f"Error splitting dataset: {e}")
        # Fallback: create minimal splits
        train_dataset = base_dataset
        val_dataset = base_dataset
        test_dataset = base_dataset
    
    # Apply augmentation to training set only
    if augment and hasattr(train_dataset, 'indices'):
        print("Applying data augmentation to training set...")
        try:
            # Create augmented training dataset
            train_file_paths = [file_paths[i] for i in train_dataset.indices]
            train_dataset = SpectrogramVAEDataset(
                train_file_paths,
                label_encoder=label_encoder,
                conditional=conditional,
                transform=SpectrogramDataAugmentation.get_transforms()
            )
            print("Augmentation applied successfully")
        except Exception as e:
            print(f"Warning: Could not apply augmentation: {e}")
    
    num_classes = len(label_encoder.classes_) if label_encoder else 0
    
    print(f"✓ Dataset creation complete:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples") 
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Classes: {num_classes}")
    
    return train_dataset, val_dataset, test_dataset, spectrogram_shape, num_classes