import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import os
from torchvision import transforms
import random
import torchaudio.transforms as T


class EnhancedSpectrogramDataset(Dataset):
    """Enhanced dataset with better preprocessing and robust error handling"""
    
    def __init__(self, file_paths, label_encoder=None, transform=None, conditional=False, 
                 target_shape=(1025, 469), normalize_per_sample=False):
        print(f"Initializing enhanced dataset with {len(file_paths)} files...")
        
        self.file_paths = []
        self.target_shape = target_shape
        self.normalize_per_sample = normalize_per_sample
        invalid_files = 0
        
        # Validate files with progress reporting
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
        
        # Compute robust dataset statistics
        self._compute_robust_stats()
    
    def _is_valid_file(self, filepath):
        try:
            if not os.path.exists(filepath):
                return False
                
            data = torch.load(filepath, map_location='cpu', weights_only=True)
            
            if not isinstance(data, dict) or 'spectrogram' not in data:
                return False
            
            spec = data['spectrogram']
            if not isinstance(spec, torch.Tensor):
                return False
            if spec.dim() < 2 or spec.numel() == 0:
                return False
            
            # Check for reasonable value ranges
            if torch.isnan(spec).any() or torch.isinf(spec).any():
                return False
                
            return True
            
        except Exception:
            return False
    
    def _compute_robust_stats(self):
        """Compute robust dataset statistics using sampling"""
        print("Computing robust dataset statistics...")
        
        # Sample files for statistics (max 1000 files)
        sample_files = random.sample(self.file_paths, min(1000, len(self.file_paths)))
        
        all_values = []
        valid_samples = 0
        
        for i, file_path in enumerate(sample_files):
            if i % 100 == 0:
                print(f"Processing sample {i+1}/{len(sample_files)} for statistics...")
            
            try:
                data = torch.load(file_path, map_location='cpu')
                spec = data['spectrogram'].float()
                
                # Sample 10% of values from each file
                flat = spec.flatten()
                if flat.numel() > 10000:
                    indices = torch.randperm(flat.numel())[:10000]
                    values = flat[indices]
                else:
                    values = flat
                
                # Remove outliers (beyond 3 standard deviations)
                mean_val = values.mean()
                std_val = values.std()
                mask = torch.abs(values - mean_val) < 3 * std_val
                clean_values = values[mask]
                
                if len(clean_values) > 0:
                    all_values.append(clean_values)
                    valid_samples += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if valid_samples == 0:
            raise ValueError("No valid samples found for statistics computation")
        
        # Combine all values
        combined_values = torch.cat(all_values, dim=0)
        
        # Compute robust statistics using percentiles
        self.dataset_min = torch.quantile(combined_values, 0.01).item()  # 1st percentile
        self.dataset_max = torch.quantile(combined_values, 0.99).item()  # 99th percentile
        self.dataset_mean = combined_values.mean().item()
        self.dataset_std = combined_values.std().item()
        
        # Compute additional robust statistics
        self.dataset_median = combined_values.median().item()
        self.dataset_q25 = torch.quantile(combined_values, 0.25).item()
        self.dataset_q75 = torch.quantile(combined_values, 0.75).item()
        
        print(f"\nRobust dataset statistics:")
        print(f"  Min (1%): {self.dataset_min:.6f}")
        print(f"  Max (99%): {self.dataset_max:.6f}")
        print(f"  Mean: {self.dataset_mean:.6f}")
        print(f"  Std: {self.dataset_std:.6f}")
        print(f"  Median: {self.dataset_median:.6f}")
        print(f"  Q25-Q75: {self.dataset_q25:.6f} - {self.dataset_q75:.6f}")
        print(f"  Valid samples: {valid_samples:,}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                file_idx = idx if attempt == 0 else (idx + attempt) % len(self.file_paths)
                
                data = torch.load(self.file_paths[file_idx], map_location='cpu')
                spectrogram = data['spectrogram'].float()
                
                # Enhanced preprocessing
                spectrogram = self._enhanced_preprocess(spectrogram)
                
                # Resize to target shape if needed
                if spectrogram.shape[-2:] != self.target_shape:
                    spectrogram = self._resize_spectrogram(spectrogram, self.target_shape)
                
                # Add channel dimension if needed
                if spectrogram.dim() == 2:
                    spectrogram = spectrogram.unsqueeze(0)
                
                # Apply transforms
                if self.transform:
                    spectrogram = self.transform(spectrogram)
                
                # Final validation
                if torch.isnan(spectrogram).any() or torch.isinf(spectrogram).any():
                    raise ValueError("NaN/Inf in final spectrogram")
                
                # Handle conditional case
                if self.conditional and 'label' in data and self.label_encoder:
                    label = self.label_encoder.transform([data['label']])[0]
                    return spectrogram, torch.tensor(label, dtype=torch.long)
                else:
                    return spectrogram
                    
            except Exception as e:
                print(f"Error loading {self.file_paths[file_idx]} (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    # Return dummy data as last resort
                    print(f"Creating dummy data for index {idx}")
                    dummy_tensor = self._create_dummy_spectrogram()
                    
                    if self.conditional:
                        return dummy_tensor, torch.tensor(0, dtype=torch.long)
                    else:
                        return dummy_tensor
                continue
    
    def _enhanced_preprocess(self, spectrogram):
        """Enhanced preprocessing pipeline for spectrograms"""
        
        # Handle complex spectrograms
        if torch.is_complex(spectrogram):
            spectrogram = torch.abs(spectrogram)
        
        # Remove invalid values
        spectrogram = torch.nan_to_num(spectrogram, nan=1e-8, posinf=1e8, neginf=-1e8)
        
        # Handle dB scale conversion if needed (common for spectrograms)
        if spectrogram.min() < 0 and spectrogram.max() < 100:  # Likely in dB
            # Convert dB to linear scale
            spectrogram = torch.pow(10.0, spectrogram / 20.0)
        
        # Ensure positive values
        spectrogram = torch.clamp(spectrogram, min=1e-8)
        
        # Apply log compression for better dynamic range
        spectrogram = torch.log(spectrogram + 1e-8)
        
        # Normalize based on strategy
        if self.normalize_per_sample:
            # Per-sample normalization (z-score)
            mean = spectrogram.mean()
            std = spectrogram.std()
            if std > 1e-6:
                spectrogram = (spectrogram - mean) / std
            spectrogram = torch.tanh(spectrogram)  # Bounded normalization
        else:
            # Global dataset normalization
            spectrogram = (spectrogram - self.dataset_mean) / max(self.dataset_std, 1e-6)
            # Apply robust clipping using percentiles
            spectrogram = torch.clamp(spectrogram, 
                                    (self.dataset_q25 - self.dataset_mean) / self.dataset_std,
                                    (self.dataset_q75 - self.dataset_mean) / self.dataset_std)
        
        # Final normalization to [0, 1]
        min_val, max_val = spectrogram.min(), spectrogram.max()
        if max_val > min_val:
            spectrogram = (spectrogram - min_val) / (max_val - min_val)
        else:
            spectrogram = torch.zeros_like(spectrogram) + 0.5
        
        return spectrogram
    
    def _resize_spectrogram(self, spectrogram, target_shape):
        """Intelligently resize spectrogram preserving important characteristics"""
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(0)  # Add batch dim
        
        # Use bilinear interpolation for smooth resizing
        resized = torch.nn.functional.interpolate(
            spectrogram, 
            size=target_shape, 
            mode='bilinear', 
            align_corners=False
        )
        
        return resized.squeeze(0)  # Remove batch dim
    
    def _create_dummy_spectrogram(self):
        """Create a realistic dummy spectrogram"""
        # Create a spectrogram with realistic characteristics
        freq_bins, time_bins = self.target_shape
        
        # Generate realistic spectral content
        dummy = torch.zeros(1, freq_bins, time_bins)
        
        # Add some low-frequency energy (typical in audio)
        dummy[0, :freq_bins//4, :] = 0.3 + 0.2 * torch.rand(freq_bins//4, time_bins)
        
        # Add some mid-frequency content
        dummy[0, freq_bins//4:freq_bins//2, :] = 0.1 + 0.1 * torch.rand(freq_bins//4, time_bins)
        
        # Add sparse high-frequency content
        high_freq_mask = torch.rand(freq_bins//2, time_bins) < 0.1
        dummy[0, freq_bins//2:, :][high_freq_mask] = 0.1 + 0.1 * torch.rand(high_freq_mask.sum())
        
        return dummy


class AdvancedSpectrogramAugmentation:
    """Advanced augmentation techniques for spectrograms"""
    
    @staticmethod
    def get_transforms(augment_prob=0.3, strong_augment=False):
        """Get comprehensive augmentation pipeline"""
        if strong_augment:
            return transforms.Compose([
                AdvancedSpectrogramAugmentation.SpectralAugment(augment_prob * 1.5),
                AdvancedSpectrogramAugmentation.AddNoise(augment_prob, noise_level=0.02),
                AdvancedSpectrogramAugmentation.FrequencyMask(augment_prob, max_mask_size=25),
                AdvancedSpectrogramAugmentation.TimeMask(augment_prob, max_mask_size=40),
                AdvancedSpectrogramAugmentation.VolumeAugment(augment_prob * 0.8),
                AdvancedSpectrogramAugmentation.SpectralRoll(augment_prob * 0.5),
            ])
        else:
            return transforms.Compose([
                AdvancedSpectrogramAugmentation.SpectralAugment(augment_prob),
                AdvancedSpectrogramAugmentation.AddNoise(augment_prob, noise_level=0.01),
                AdvancedSpectrogramAugmentation.FrequencyMask(augment_prob * 0.7, max_mask_size=15),
                AdvancedSpectrogramAugmentation.TimeMask(augment_prob * 0.7, max_mask_size=25),
                AdvancedSpectrogramAugmentation.VolumeAugment(augment_prob * 0.5),
            ])
    
    class SpectralAugment:
        """Comprehensive spectral augmentation"""
        def __init__(self, prob=0.3):
            self.prob = prob
            
        def __call__(self, spectrogram):
            if random.random() < self.prob:
                # Apply random combination of augmentations
                if random.random() < 0.5:
                    # Frequency warping
                    spectrogram = self._frequency_warp(spectrogram)
                if random.random() < 0.5:
                    # Time stretching simulation
                    spectrogram = self._time_stretch(spectrogram)
                    
            return spectrogram
        
        def _frequency_warp(self, spec):
            """Simulate frequency warping"""
            if spec.dim() == 3:  # [C, F, T]
                freq_size = spec.shape[1]
                # Create warping indices
                warp_factor = 1.0 + 0.1 * (2 * random.random() - 1)  # ±10%
                indices = torch.linspace(0, freq_size - 1, freq_size) * warp_factor
                indices = torch.clamp(indices, 0, freq_size - 1).long()
                return spec[:, indices, :]
            return spec
        
        def _time_stretch(self, spec):
            """Simulate time stretching"""
            if spec.dim() == 3:  # [C, F, T]
                time_size = spec.shape[2]
                stretch_factor = 1.0 + 0.1 * (2 * random.random() - 1)  # ±10%
                new_size = int(time_size * stretch_factor)
                if new_size != time_size:
                    spec_stretched = torch.nn.functional.interpolate(
                        spec.unsqueeze(0), size=(spec.shape[1], new_size), 
                        mode='bilinear', align_corners=False
                    ).squeeze(0)
                    # Crop or pad to original size
                    if new_size > time_size:
                        start = (new_size - time_size) // 2
                        return spec_stretched[:, :, start:start + time_size]
                    else:
                        pad = (time_size - new_size) // 2
                        return torch.nn.functional.pad(spec_stretched, (pad, time_size - new_size - pad))
            return spec
    
    class AddNoise:
        def __init__(self, prob=0.3, noise_level=0.01):
            self.prob = prob
            self.noise_level = noise_level
            
        def __call__(self, spectrogram):
            if random.random() < self.prob:
                # Add different types of noise
                noise_type = random.choice(['gaussian', 'uniform', 'pink'])
                
                if noise_type == 'gaussian':
                    noise = torch.randn_like(spectrogram) * self.noise_level
                elif noise_type == 'uniform':
                    noise = (torch.rand_like(spectrogram) - 0.5) * 2 * self.noise_level
                else:  # pink noise approximation
                    noise = self._generate_pink_noise(spectrogram.shape) * self.noise_level
                
                return torch.clamp(spectrogram + noise, 0, 1)
            return spectrogram
        
        def _generate_pink_noise(self, shape):
            """Generate pink noise (1/f noise)"""
            if len(shape) == 3:  # [C, F, T]
                _, freq_size, time_size = shape
                # Create frequency-dependent noise
                freqs = torch.arange(1, freq_size + 1).float()
                freq_weights = 1.0 / torch.sqrt(freqs)
                freq_weights = freq_weights.view(1, -1, 1)
                
                white_noise = torch.randn(shape)
                pink_noise = white_noise * freq_weights
                return pink_noise
            return torch.randn(shape)
    
    class FrequencyMask:
        def __init__(self, prob=0.2, max_mask_size=15, num_masks=1):
            self.prob = prob
            self.max_mask_size = max_mask_size
            self.num_masks = num_masks
            
        def __call__(self, spectrogram):
            if random.random() < self.prob:
                if spectrogram.dim() == 3:  # [C, F, T]
                    freq_size = spectrogram.shape[1]
                    spec_masked = spectrogram.clone()
                    
                    for _ in range(self.num_masks):
                        mask_size = random.randint(1, min(self.max_mask_size, freq_size // 4))
                        mask_start = random.randint(0, freq_size - mask_size)
                        
                        # Use different masking strategies
                        mask_type = random.choice(['zero', 'mean', 'noise'])
                        
                        if mask_type == 'zero':
                            spec_masked[:, mask_start:mask_start + mask_size, :] = 0
                        elif mask_type == 'mean':
                            mean_val = spectrogram[:, mask_start:mask_start + mask_size, :].mean()
                            spec_masked[:, mask_start:mask_start + mask_size, :] = mean_val
                        else:  # noise
                            noise = torch.rand_like(spec_masked[:, mask_start:mask_start + mask_size, :]) * 0.1
                            spec_masked[:, mask_start:mask_start + mask_size, :] = noise
                    
                    return spec_masked
            return spectrogram
    
    class TimeMask:
        def __init__(self, prob=0.2, max_mask_size=25, num_masks=1):
            self.prob = prob
            self.max_mask_size = max_mask_size
            self.num_masks = num_masks
            
        def __call__(self, spectrogram):
            if random.random() < self.prob:
                if spectrogram.dim() == 3:  # [C, F, T]
                    time_size = spectrogram.shape[2]
                    spec_masked = spectrogram.clone()
                    
                    for _ in range(self.num_masks):
                        mask_size = random.randint(1, min(self.max_mask_size, time_size // 4))
                        mask_start = random.randint(0, time_size - mask_size)
                        
                        # Use different masking strategies
                        mask_type = random.choice(['zero', 'interpolate', 'repeat'])
                        
                        if mask_type == 'zero':
                            spec_masked[:, :, mask_start:mask_start + mask_size] = 0
                        elif mask_type == 'interpolate':
                            # Linear interpolation between boundaries
                            if mask_start > 0 and mask_start + mask_size < time_size:
                                left_val = spec_masked[:, :, mask_start - 1:mask_start]
                                right_val = spec_masked[:, :, mask_start + mask_size:mask_start + mask_size + 1]
                                for i in range(mask_size):
                                    alpha = i / (mask_size - 1) if mask_size > 1 else 0
                                    interpolated = left_val * (1 - alpha) + right_val * alpha
                                    spec_masked[:, :, mask_start + i:mask_start + i + 1] = interpolated
                        else:  # repeat
                            if mask_start > 0:
                                repeat_val = spec_masked[:, :, mask_start - 1:mask_start]
                                spec_masked[:, :, mask_start:mask_start + mask_size] = repeat_val.repeat(1, 1, mask_size)
                    
                    return spec_masked
            return spectrogram
    
    class VolumeAugment:
        def __init__(self, prob=0.3, gain_range=(-6, 6)):
            self.prob = prob
            self.gain_range = gain_range  # in dB
            
        def __call__(self, spectrogram):
            if random.random() < self.prob:
                # Random gain in dB
                gain_db = random.uniform(*self.gain_range)
                gain_linear = 10 ** (gain_db / 20)
                
                # Apply gain and clip
                return torch.clamp(spectrogram * gain_linear, 0, 1)
            return spectrogram
    
    class SpectralRoll:
        def __init__(self, prob=0.2, max_roll=10):
            self.prob = prob
            self.max_roll = max_roll
            
        def __call__(self, spectrogram):
            if random.random() < self.prob:
                if spectrogram.dim() == 3:  # [C, F, T]
                    # Circular roll in frequency dimension
                    roll_amount = random.randint(-self.max_roll, self.max_roll)
                    return torch.roll(spectrogram, shifts=roll_amount, dims=1)
            return spectrogram


def create_enhanced_vae_datasets(data_dir, label_encoder=None, conditional=False, 
                               train_ratio=0.7, val_ratio=0.15, augment=True,
                               strong_augment=False, target_shape=(1025, 469),
                               normalize_per_sample=False):
    """
    Create enhanced datasets with better preprocessing and augmentation
    """
    
    print("🔍 Creating enhanced VAE datasets...")
    
    file_paths = load_file_paths(data_dir)
    
    print(f"📊 Creating enhanced base dataset with {len(file_paths)} files")
    print(f"🎯 Target shape: {target_shape}")
    print(f"📏 Normalization: {'Per-sample' if normalize_per_sample else 'Global'}")
    
    # Create base dataset
    try:
        base_dataset = EnhancedSpectrogramDataset(
            file_paths,
            label_encoder=label_encoder,
            conditional=conditional,
            target_shape=target_shape,
            normalize_per_sample=normalize_per_sample
        )
        
        print(f"✅ Successfully created base dataset with {len(base_dataset)} samples")
        
        # Test loading samples
        print("🧪 Testing dataset loading...")
        for i in range(min(3, len(base_dataset))):
            try:
                sample = base_dataset[i]
                if conditional:
                    spec, target = sample
                    print(f"  Sample {i}: spec shape {spec.shape}, target = {target}")
                else:
                    spec = sample
                    print(f"  Sample {i}: spec shape {spec.shape}")
                    print(f"    Range: [{spec.min():.4f}, {spec.max():.4f}]")
                    
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
    
    print(f"📊 Dataset splits: train={train_size}, val={val_size}, test={test_size}")
    
    # Set random seed for reproducible splits
    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        base_dataset, [train_size, val_size, test_size]
    )
    
    # Apply augmentation to training set only
    if augment:
        print(f"🎨 Applying {'strong' if strong_augment else 'standard'} augmentation to training set")
        
        # Get file paths for training indices
        train_file_paths = [file_paths[i] for i in train_dataset.indices]
        
        # Create augmented training dataset
        train_dataset = EnhancedSpectrogramDataset(
            train_file_paths,
            label_encoder=label_encoder,
            conditional=conditional,
            target_shape=target_shape,
            normalize_per_sample=normalize_per_sample,
            transform=AdvancedSpectrogramAugmentation.get_transforms(
                augment_prob=0.4 if strong_augment else 0.3,
                strong_augment=strong_augment
            )
        )
        
        print(f"✅ Augmented training dataset created with {len(train_dataset)} samples")
    
    # Get dataset info
    num_classes = len(label_encoder.classes_) if label_encoder else 0
    
    return train_dataset, val_dataset, test_dataset, target_shape, num_classes


def load_file_paths(data_dir):
    """Load all .pt file paths from directory with better error handling"""
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    print(f"🔍 Scanning directory: {data_dir}")
    
    file_paths = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.pt'):
                file_paths.append(os.path.join(root, f))
    
    if not file_paths:
        raise ValueError(f"No .pt files found in {data_dir}")
    
    print(f"📁 Found {len(file_paths)} .pt files")
    return sorted(file_paths)  # Sort for reproducibility


def encode_labels(file_paths):
    """Extract labels from files with better error handling"""
    labels = []
    successful_reads = 0
    
    print("🏷️ Extracting labels from files...")
    
    for i, fp in enumerate(file_paths):
        if i % 1000 == 0:
            print(f"  Processing file {i+1}/{len(file_paths)}")
            
        try:
            data = torch.load(fp, map_location='cpu', weights_only=True)
            if isinstance(data, dict) and 'label' in data:
                labels.append(data['label'])
                successful_reads += 1
        except Exception:
            continue
    
    if not labels:
        print("⚠️ No labels found in any files")
        return []
    
    unique_labels = sorted(list(set(labels)))
    print(f"📊 Found {len(labels)} labeled samples from {successful_reads} files")
    print(f"🏷️ Unique labels ({len(unique_labels)}): {unique_labels}")
    
    return labels


def inspect_enhanced_files(data_dir, num_samples=5):
    """Enhanced file inspection with detailed analysis"""
    print(f"🔍 ENHANCED FILE INSPECTION")
    print("=" * 60)
    
    file_paths = load_file_paths(data_dir)
    sample_files = random.sample(file_paths, min(num_samples, len(file_paths)))
    
    stats = {
        'shapes': [],
        'dtypes': [],
        'value_ranges': [],
        'has_labels': 0,
        'complex_files': 0
    }
    
    for i, fp in enumerate(sample_files):
        try:
            print(f"\n📄 File {i+1}: {os.path.basename(fp)}")
            data = torch.load(fp, map_location='cpu')
            
            print(f"  📋 Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            
            if isinstance(data, dict) and 'spectrogram' in data:
                spec = data['spectrogram']
                stats['shapes'].append(spec.shape)
                stats['dtypes'].append(str(spec.dtype))
                
                print(f"  📐 Shape: {spec.shape}")
                print(f"  🔢 Dtype: {spec.dtype}")
                print(f"  🔧 Is complex: {torch.is_complex(spec)}")
                
                if torch.is_complex(spec):
                    stats['complex_files'] += 1
                    spec_real = torch.abs(spec)
                else:
                    spec_real = spec
                
                min_val, max_val = spec_real.min().item(), spec_real.max().item()
                mean_val, std_val = spec_real.mean().item(), spec_real.std().item()
                stats['value_ranges'].append((min_val, max_val))
                
                print(f"  📊 Range: [{min_val:.6f}, {max_val:.6f}]")
                print(f"  📈 Mean±Std: {mean_val:.6f} ± {std_val:.6f}")
                print(f"  🎯 Unique values: {torch.unique(spec_real).numel():,}")
                
                # Check for problematic values
                if torch.isnan(spec_real).any():
                    print(f"  ⚠️ Contains NaN values!")
                if torch.isinf(spec_real).any():
                    print(f"  ⚠️ Contains Inf values!")
                if torch.unique(spec_real).numel() == 1:
                    print(f"  ⚠️ All values are identical!")
                
            if isinstance(data, dict) and 'label' in data:
                stats['has_labels'] += 1
                print(f"  🏷️ Label: {data['label']}")
                
        except Exception as e:
            print(f"  ❌ Error loading: {e}")
    
    # Summary statistics
    print(f"\n📊 SUMMARY STATISTICS")
    print("=" * 60)
    print(f"🔍 Files inspected: {len(sample_files)}")
    print(f"🏷️ Files with labels: {stats['has_labels']}")
    print(f"🔧 Complex files: {stats['complex_files']}")
    
    if stats['shapes']:
        unique_shapes = list(set(stats['shapes']))
        print(f"📐 Unique shapes: {unique_shapes}")
        
    if stats['dtypes']:
        unique_dtypes = list(set(stats['dtypes']))
        print(f"🔢 Data types: {unique_dtypes}")
        
    if stats['value_ranges']:
        all_mins = [r[0] for r in stats['value_ranges']]
        all_maxs = [r[1] for r in stats['value_ranges']]
        print(f"📊 Value ranges: [{min(all_mins):.6f}, {max(all_maxs):.6f}]")
    
    print("\n" + "=" * 60)