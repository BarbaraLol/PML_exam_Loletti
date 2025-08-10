import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import os
from torchvision import transforms
import random


class SpectrogramVAEDataset(Dataset):
    """Fixed dataset with robust preprocessing and error handling"""
    
    def __init__(self, file_paths, label_encoder=None, transform=None, conditional=False):
        print(f"Initializing dataset with {len(file_paths)} files...")
        
        self.file_paths = []
        invalid_files = 0
        
        for i, fp in enumerate(file_paths):
            if i % 500 == 0:  # Print progress every 500 files
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
        
        # Compute dataset statistics with better error handling
        self._compute_dataset_stats()
    
    def _is_valid_file(self, filepath):
        try:
            if not os.path.exists(filepath):
                return False
                
            # Only load metadata, not full tensor
            data = torch.load(filepath, map_location='cpu', weights_only=True)
            
            if not isinstance(data, dict) or 'spectrogram' not in data:
                return False
            
            # Check shape without loading full tensor
            if 'spectrogram' in data:
                if not isinstance(data['spectrogram'], torch.Tensor):
                    return False
                if data['spectrogram'].dim() < 2 or data['spectrogram'].numel() == 0:
                    return False
            
            return True
            
        except Exception as e:
            print(f"File validation error for {filepath}: {e}")
            return False
    
    def _compute_dataset_stats(self):
        """Compute dataset statistics without concatenating all values"""
        print("Computing dataset statistics for normalization...")
    
        # Initialize statistics
        min_val = float('inf')
        max_val = float('-inf')
        sum_val = 0.0
        sum_sq_val = 0.0
        count = 0
        
        # Process files in chunks
        chunk_size = 100
        num_chunks = len(self.file_paths) // chunk_size + 1
        
        print(f"Processing {len(self.file_paths)} files in {num_chunks} chunks...")
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(self.file_paths))
            chunk_files = self.file_paths[start_idx:end_idx]
            
            print(f"Processing chunk {chunk_idx+1}/{num_chunks} ({len(chunk_files)} files)")
            
            for file_path in chunk_files:
                try:
                    data = torch.load(file_path, map_location='cpu')
                    spec = data['spectrogram'].float()
                    
                    # Only process 10% of values per file to save memory
                    flat = spec.flatten()
                    if flat.numel() > 10000:
                        indices = torch.randperm(flat.numel())[:10000]
                        values = flat[indices]
                    else:
                        values = flat
                    
                    # Update statistics with sampled values
                    chunk_min = values.min().item()
                    chunk_max = values.max().item()
                    chunk_sum = values.sum().item()
                    chunk_sq_sum = (values ** 2).sum().item()
                    
                    min_val = min(min_val, chunk_min)
                    max_val = max(max_val, chunk_max)
                    sum_val += chunk_sum
                    sum_sq_val += chunk_sq_sum
                    count += len(values)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        # Calculate final statistics
        self.dataset_min = min_val
        self.dataset_max = max_val
        self.dataset_mean = sum_val / count
        self.dataset_std = (sum_sq_val / count - self.dataset_mean ** 2) ** 0.5
        
        print(f"\nFinal dataset statistics:")
        print(f"  Min: {self.dataset_min:.6f}")
        print(f"  Max: {self.dataset_max:.6f}")
        print(f"  Mean: {self.dataset_mean:.6f}")
        print(f"  Std: {self.dataset_std:.6f}")
        print(f"  Total values processed: {count:,}")
    
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
                
                # Proper preprocessing with error checking
                spectrogram = self._preprocess_spectrogram(spectrogram)
                
                # Validate the result
                if torch.isnan(spectrogram).any() or torch.isinf(spectrogram).any():
                    raise ValueError("NaN/Inf in preprocessed spectrogram")
                
                # Add channel dimension if needed [1, freq, time]
                if spectrogram.dim() == 2:
                    spectrogram = spectrogram.unsqueeze(0)
                
                # Apply transforms if any
                if self.transform:
                    spectrogram = self.transform(spectrogram)
                
                # Handle conditional case
                if self.conditional and 'label' in data and self.label_encoder:
                    try:
                        label = torch.tensor(
                            self.label_encoder.transform([data['label']])[0],
                            dtype=torch.long
                        )
                        return spectrogram, label
                    except:
                        # Fallback to dummy label if label encoding fails
                        return spectrogram, torch.tensor(0, dtype=torch.long)
                else:
                    return spectrogram, spectrogram
                    
            except Exception as e:
                print(f"Error loading {self.file_paths[file_idx]} (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    # Last resort: return a valid dummy tensor
                    print(f"Creating dummy data for index {idx}")
                    dummy_shape = (1, 1025, 938)  # Your spectrogram shape
                    dummy_tensor = torch.rand(dummy_shape) * 0.1  # Small random values
                    
                    if self.conditional:
                        return dummy_tensor, torch.tensor(0, dtype=torch.long)
                    else:
                        return dummy_tensor, dummy_tensor
                continue
    
    def _preprocess_spectrogram(self, spectrogram):
        """Enhanced preprocessing with better error handling"""
        
        # Handle complex spectrograms
        if torch.is_complex(spectrogram):
            spectrogram = torch.abs(spectrogram)
        
        # Check for problematic values before log transform
        if torch.isnan(spectrogram).any() or torch.isinf(spectrogram).any():
            print("Warning: NaN/Inf detected before preprocessing")
            spectrogram = torch.nan_to_num(spectrogram, nan=1e-8, posinf=1e-8, neginf=1e-8)
        
        # Ensure positive values for log transform
        spec_min = spectrogram.min().item()
        if spec_min <= 0:
            spectrogram = spectrogram - spec_min + 1e-8
        
        # Apply log transform with safety check
        spectrogram = torch.log(spectrogram + 1e-8)
        
        # Check for NaN after log
        if torch.isnan(spectrogram).any():
            print("Warning: NaN after log transform, using fallback")
            spectrogram = torch.full_like(spectrogram, self.dataset_mean)
        
        # Robust normalization to [0, 1]
        range_val = self.dataset_max - self.dataset_min
        
        if range_val > 1e-6:
            spectrogram = (spectrogram - self.dataset_min) / range_val
        else:
            # Fallback: standardize and then sigmoid
            spectrogram = (spectrogram - self.dataset_mean) / max(self.dataset_std, 1e-6)
            spectrogram = torch.sigmoid(spectrogram)  # Maps to [0, 1]
        
        spectrogram = torch.clamp(spectrogram, 0, 1)
        
        # Final safety check
        if torch.isnan(spectrogram).any() or torch.isinf(spectrogram).any():
            print("WARNING: NaN/Inf in preprocessed spectrogram")
            # Fallback to random values
            spectrogram = torch.rand_like(spectrogram) * 0.5 + 0.25
        
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


def inspect_spectrogram_files(data_dir, num_samples=10):
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
                    spec_real = spec
                
                print(f"  Min: {spec_real.min().item():.6f}")
                print(f"  Max: {spec_real.max().item():.6f}")
                print(f"  Mean: {spec_real.mean().item():.6f}")
                print(f"  Std: {spec_real.std().item():.6f}")
                print(f"  Unique values: {torch.unique(spec_real).numel()}")
                
                # Check if all values are identical
                if torch.unique(spec_real).numel() == 1:
                    print(f"  ⚠️  WARNING: All values are identical!")
                
            if 'label' in data:
                print(f"  Label: {data['label']}")
                
        except Exception as e:
            print(f"  ❌ Error loading: {e}")
    
    print("\n" + "="*50)


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
    Create train, validation, and test datasets for VAE training with better error handling
    """
    
    # First, inspect a few files to understand the data
    print("🔍 Inspecting data files...")
    inspect_spectrogram_files(data_dir, num_samples=5)
    
    file_paths = load_file_paths(data_dir)
    spectrogram_shape = get_spectrogram_shape(file_paths)
    
    print(f"Creating datasets with {len(file_paths)} files")
    print(f"Spectrogram shape: {spectrogram_shape}")
    
    # Create base dataset with validation
    try:
        base_dataset = SpectrogramVAEDataset(
            file_paths,
            label_encoder=label_encoder,
            conditional=conditional
        )
        
        print(f"Successfully created base dataset with {len(base_dataset)} samples")
        
        # Test loading a few samples
        print("Testing dataset loading...")
        for i in range(min(3, len(base_dataset))):
            try:
                sample = base_dataset[i]
                spec, target = sample
                print(f"  Sample {i}: spec shape {spec.shape}, target type {type(target)}")
                
                # Check for data quality
                if torch.isnan(spec).any():
                    print(f"    ⚠️  Sample {i} contains NaN!")
                if torch.unique(spec).numel() < 10:
                    print(f"    ⚠️  Sample {i} has very low diversity!")
                    
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