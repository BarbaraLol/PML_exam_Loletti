import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import torchaudio
import random

class SpectrogramNormalization:
    """Enhanced normalization specifically for spectrograms"""
    
    @staticmethod
    def robust_normalize(spectrogram, percentile_range=(1, 99)):
        """Robust normalization using percentiles to handle outliers"""
        # Flatten for percentile calculation
        flat = spectrogram.flatten()
        
        # Calculate robust min/max using percentiles
        p_low, p_high = torch.quantile(flat, torch.tensor(percentile_range) / 100.0)
        
        # Clip and normalize
        normalized = torch.clamp(spectrogram, p_low, p_high)
        normalized = (normalized - p_low) / (p_high - p_low + 1e-8)
        
        return normalized
    
    @staticmethod
    def frequency_wise_normalize(spectrogram):
        """Normalize each frequency bin separately"""
        # spectrogram shape: [channels, freq_bins, time_frames]
        if spectrogram.dim() == 3:
            # Normalize each frequency bin across time
            mean = spectrogram.mean(dim=-1, keepdim=True)
            std = spectrogram.std(dim=-1, keepdim=True)
            normalized = (spectrogram - mean) / (std + 1e-8)
        else:
            # Handle 2D case
            mean = spectrogram.mean(dim=-1, keepdim=True)
            std = spectrogram.std(dim=-1, keepdim=True)
            normalized = (spectrogram - mean) / (std + 1e-8)
        
        # Apply sigmoid to constrain range
        return torch.sigmoid(normalized)
    
    @staticmethod
    def adaptive_normalize(spectrogram):
        """Adaptive normalization based on spectrogram content"""
        # Calculate dynamic range
        db_range = spectrogram.max() - spectrogram.min()
        
        if db_range > 80:  # High dynamic range
            # Use robust percentile normalization
            return SpectrogramNormalization.robust_normalize(spectrogram, (2, 98))
        elif db_range < 20:  # Low dynamic range
            # Use frequency-wise normalization to enhance details
            return SpectrogramNormalization.frequency_wise_normalize(spectrogram)
        else:
            # Standard min-max normalization
            return (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-8)

class EnhancedSpectrogramAugmentation:
    """More sophisticated augmentation for spectrograms"""
    
    def __init__(self, augment_prob=0.5):
        self.augment_prob = augment_prob
        
        # Create augmentation pipeline
        self.augmentations = [
            self.freq_mask,
            self.time_mask,
            self.pitch_shift,
            self.time_stretch,
            self.add_structured_noise,
            self.frequency_dropout,
            self.mixup_augment
        ]
    
    def __call__(self, spectrogram):
        if random.random() < self.augment_prob:
            # Apply 1-3 random augmentations
            num_augs = random.randint(1, 3)
            selected_augs = random.sample(self.augmentations, num_augs)
            
            for aug in selected_augs:
                spectrogram = aug(spectrogram)
        
        return spectrogram
    
    def freq_mask(self, spectrogram, max_mask_pct=0.15):
        """Frequency masking with variable mask size"""
        if random.random() < 0.3:
            freq_size = spectrogram.shape[-2]
            max_mask_size = int(freq_size * max_mask_pct)
            mask_size = random.randint(1, max_mask_size)
            mask_start = random.randint(0, freq_size - mask_size)
            
            masked = spectrogram.clone()
            masked[..., mask_start:mask_start + mask_size, :] *= random.uniform(0.0, 0.3)
            return masked
        return spectrogram
    
    def time_mask(self, spectrogram, max_mask_pct=0.2):
        """Time masking with variable mask size"""
        if random.random() < 0.3:
            time_size = spectrogram.shape[-1]
            max_mask_size = int(time_size * max_mask_pct)
            mask_size = random.randint(1, max_mask_size)
            mask_start = random.randint(0, time_size - mask_size)
            
            masked = spectrogram.clone()
            masked[..., :, mask_start:mask_start + mask_size] *= random.uniform(0.0, 0.3)
            return masked
        return spectrogram
    
    def pitch_shift(self, spectrogram, max_shift=5):
        """Simulate pitch shifting by frequency bin shifting"""
        if random.random() < 0.2:
            shift = random.randint(-max_shift, max_shift)
            if shift != 0:
                shifted = torch.roll(spectrogram, shift, dims=-2)
                # Zero out wrapped regions
                if shift > 0:
                    shifted[..., :shift, :] = 0
                else:
                    shifted[..., shift:, :] = 0
                return shifted
        return spectrogram
    
    def time_stretch(self, spectrogram, stretch_range=(0.85, 1.15)):
        """Simulate time stretching through interpolation"""
        if random.random() < 0.2:
            stretch_factor = random.uniform(*stretch_range)
            if stretch_factor != 1.0:
                # Interpolate along time axis
                new_width = int(spectrogram.shape[-1] * stretch_factor)
                stretched = F.interpolate(
                    spectrogram.unsqueeze(0),
                    size=(spectrogram.shape[-2], new_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
                # Crop or pad to original size
                if stretched.shape[-1] > spectrogram.shape[-1]:
                    # Crop
                    start = (stretched.shape[-1] - spectrogram.shape[-1]) // 2
                    stretched = stretched[..., start:start + spectrogram.shape[-1]]
                elif stretched.shape[-1] < spectrogram.shape[-1]:
                    # Pad
                    pad_size = spectrogram.shape[-1] - stretched.shape[-1]
                    stretched = F.pad(stretched, (pad_size // 2, pad_size - pad_size // 2))
                
                return stretched
        return spectrogram
    
    def add_structured_noise(self, spectrogram, noise_types=['pink', 'white', 'bandpass']):
        """Add structured noise patterns"""
        if random.random() < 0.2:
            noise_type = random.choice(noise_types)
            intensity = random.uniform(0.01, 0.05)
            
            if noise_type == 'pink':
                # Pink noise (1/f spectrum)
                noise = torch.randn_like(spectrogram)
                # Apply frequency-dependent scaling
                freq_weights = torch.arange(1, spectrogram.shape[-2] + 1).float().unsqueeze(-1)
                freq_weights = 1.0 / torch.sqrt(freq_weights)
                noise = noise * freq_weights.unsqueeze(0) * intensity
                
            elif noise_type == 'white':
                # White noise
                noise = torch.randn_like(spectrogram) * intensity
                
            else:  # bandpass
                # Bandpass noise (only certain frequency ranges)
                noise = torch.randn_like(spectrogram) * intensity
                freq_size = spectrogram.shape[-2]
                # Random frequency band
                band_start = random.randint(0, freq_size // 2)
                band_width = random.randint(freq_size // 10, freq_size // 3)
                band_end = min(band_start + band_width, freq_size)
                
                # Zero out noise outside the band
                noise[..., :band_start, :] = 0
                noise[..., band_end:, :] = 0
            
            return torch.clamp(spectrogram + noise, 0, 1)
        return spectrogram
    
    def frequency_dropout(self, spectrogram, dropout_prob=0.1):
        """Randomly drop entire frequency bins"""
        if random.random() < 0.15:
            mask = torch.rand(spectrogram.shape[-2]) > dropout_prob
            masked = spectrogram.clone()
            masked[..., ~mask, :] = 0
            return masked
        return spectrogram
    
    def mixup_augment(self, spectrogram, alpha=0.2):
        """MixUp augmentation (when used with batch processing)"""
        # This would typically be applied at batch level
        # For now, just return the original
        return spectrogram

class ImprovedSpectrogramDataset(torch.utils.data.Dataset):
    """Enhanced dataset with better preprocessing and augmentation"""
    
    def __init__(self, file_paths, label_encoder=None, transform=None, 
                 conditional=False, use_enhanced_preprocessing=True):
        
        self.file_paths = self._validate_files(file_paths)
        self.label_encoder = label_encoder
        self.conditional = conditional
        self.use_enhanced_preprocessing = use_enhanced_preprocessing
        
        # Enhanced augmentation
        if transform is True:  # If True, use enhanced augmentation
            self.transform = EnhancedSpectrogramAugmentation(augment_prob=0.6)
        else:
            self.transform = transform
        
        # Compute enhanced dataset statistics
        if use_enhanced_preprocessing:
            self._compute_enhanced_stats()
    
    def _validate_files(self, file_paths):
        """More robust file validation"""
        valid_files = []
        
        for fp in file_paths:
            try:
                data = torch.load(fp, map_location='cpu', weights_only=True)
                
                # Enhanced validation
                if not isinstance(data, dict) or 'spectrogram' not in data:
                    continue
                    
                spec = data['spectrogram']
                if not isinstance(spec, torch.Tensor):
                    continue
                    
                # Check for reasonable spectrogram properties
                if spec.dim() < 2 or spec.numel() < 1000:  # Too small
                    continue
                    
                # Check for problematic values
                if torch.isnan(spec).any() or torch.isinf(spec).any():
                    continue
                    
                # Check dynamic range
                if (spec.max() - spec.min()) < 1e-6:  # No variation
                    continue
                    
                valid_files.append(fp)
                
            except Exception as e:
                print(f"Skipping invalid file {fp}: {e}")
                continue
        
        print(f"Validated {len(valid_files)}/{len(file_paths)} files")
        return valid_files
    
    def _compute_enhanced_stats(self):
        """Compute enhanced statistics for better normalization"""
        print("Computing enhanced dataset statistics...")
        
        # Sample subset for statistics (more efficient)
        sample_size = min(1000, len(self.file_paths))
        sampled_files = random.sample(self.file_paths, sample_size)
        
        all_percentiles = []
        freq_means = []
        freq_stds = []
        
        for fp in sampled_files[:100]:  # Use first 100 for detailed stats
            try:
                data = torch.load(fp, map_location='cpu')
                spec = data['spectrogram'].float()
                
                # Compute percentiles for robust normalization
                flat = spec.flatten()
                percentiles = torch.quantile(flat, torch.linspace(0, 1, 21))
                all_percentiles.append(percentiles)
                
                # Frequency-wise statistics
                if spec.dim() >= 2:
                    freq_mean = spec.mean(dim=-1)  # Average across time
                    freq_std = spec.std(dim=-1)
                    
                    if spec.dim() == 3:  # [C, F, T]
                        freq_mean = freq_mean.mean(dim=0)  # Average across channels
                        freq_std = freq_std.mean(dim=0)
                    
                    freq_means.append(freq_mean)
                    freq_stds.append(freq_std)
                    
            except Exception as e:
                continue
        
        # Aggregate statistics
        if all_percentiles:
            self.dataset_percentiles = torch.stack(all_percentiles).mean(dim=0)
            self.robust_min = self.dataset_percentiles[1]  # 5th percentile
            self.robust_max = self.dataset_percentiles[19]  # 95th percentile
        else:
            self.robust_min = 0.0
            self.robust_max = 1.0
        
        if freq_means and freq_stds:
            self.freq_means = torch.stack(freq_means).mean(dim=0)
            self.freq_stds = torch.stack(freq_stds).mean(dim=0)
        else:
            # Fallback
            dummy_size = 1025  # Typical spectrogram frequency bins
            self.freq_means = torch.zeros(dummy_size)
            self.freq_stds = torch.ones(dummy_size)
        
        print(f"Enhanced stats computed:")
        print(f"  Robust range: [{self.robust_min:.4f}, {self.robust_max:.4f}]")
        print(f"  Freq means range: [{self.freq_means.min():.4f}, {self.freq_means.max():.4f}]")
    
    def _enhanced_preprocessing(self, spectrogram):
        """Enhanced preprocessing pipeline"""
        # Handle complex spectrograms
        if torch.is_complex(spectrogram):
            spectrogram = torch.abs(spectrogram)
        
        # Remove NaN/Inf
        spectrogram = torch.nan_to_num(spectrogram, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Convert dB to linear if needed (detect dB range)
        if spectrogram.min() < -10 and spectrogram.max() > 10:
            # Likely in dB, convert to linear
            spectrogram = torch.pow(10.0, spectrogram / 20.0)
            spectrogram = torch.clamp(spectrogram, min=1e-7)
            spectrogram = torch.log(spectrogram + 1e-7)
        
        # Apply adaptive normalization
        spectrogram = SpectrogramNormalization.adaptive_normalize(spectrogram)
        
        return spectrogram
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Retry mechanism with fallback
        for attempt in range(3):
            try:
                file_idx = (idx + attempt) % len(self.file_paths)
                data = torch.load(self.file_paths[file_idx], map_location='cpu')
                spectrogram = data['spectrogram'].float()
                
                # Enhanced preprocessing
                if self.use_enhanced_preprocessing:
                    spectrogram = self._enhanced_preprocessing(spectrogram)
                else:
                    # Basic preprocessing fallback
                    spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-8)
                
                # Ensure proper dimensions [C, H, W]
                if spectrogram.dim() == 2:
                    spectrogram = spectrogram.unsqueeze(0)
                
                # Apply augmentation
                if self.transform:
                    spectrogram = self.transform(spectrogram)
                
                # Final validation
                if torch.isnan(spectrogram).any() or torch.isinf(spectrogram).any():
                    raise ValueError("NaN/Inf in processed spectrogram")
                
                # Return with label if conditional
                if self.conditional and 'label' in data and self.label_encoder:
                    label = self.label_encoder.transform([data['label']])[0]
                    return spectrogram, torch.tensor(label, dtype=torch.long)
                else:
                    return spectrogram
                    
            except Exception as e:
                if attempt == 2:  # Last attempt
                    print(f"Failed to load sample {idx} after 3 attempts: {e}")
                    # Return a dummy sample
                    dummy_shape = (1, 1025, 469)  # Adjust as needed
                    dummy_tensor = torch.rand(dummy_shape) * 0.5 + 0.25
                    
                    if self.conditional:
                        return dummy_tensor, torch.tensor(0, dtype=torch.long)
                    else:
                        return dummy_tensor
                continue

# Utility function for creating improved datasets
def create_improved_datasets(data_dir, label_encoder=None, conditional=False,
                           train_ratio=0.7, val_ratio=0.15, enhanced_aug=True):
    """Create datasets with enhanced preprocessing and augmentation"""
    
    from data_loading import load_file_paths
    
    file_paths = load_file_paths(data_dir)
    
    # Split paths for different dataset parts
    total_size = len(file_paths)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Random shuffle with fixed seed for reproducibility
    torch.manual_seed(42)
    indices = torch.randperm(total_size)
    
    train_paths = [file_paths[i] for i in indices[:train_size]]
    val_paths = [file_paths[i] for i in indices[train_size:train_size + val_size]]
    test_paths = [file_paths[i] for i in indices[train_size + val_size:]]
    
    # Create datasets with different augmentation levels
    train_dataset = ImprovedSpectrogramDataset(
        train_paths,
        label_encoder=label_encoder,
        conditional=conditional,
        transform=enhanced_aug,  # Full augmentation for training
        use_enhanced_preprocessing=True
    )
    
    val_dataset = ImprovedSpectrogramDataset(
        val_paths,
        label_encoder=label_encoder,
        conditional=conditional,
        transform=False,  # No augmentation for validation
        use_enhanced_preprocessing=True
    )
    
    test_dataset = ImprovedSpectrogramDataset(
        test_paths,
        label_encoder=label_encoder,
        conditional=conditional,
        transform=False,  # No augmentation for testing
        use_enhanced_preprocessing=True
    )
    
    return train_dataset, val_dataset, test_dataset