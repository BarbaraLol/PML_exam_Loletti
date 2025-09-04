import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from scipy.signal import get_window, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

from model import ImprovedVariationalAutoEncoder, ConditionalVariationalAutoEncoder


def load_improved_vae_model(model_path, device='cpu'):
    """Load the trained VAE model with improved error handling"""
    print(f"Loading VAE model from: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None, None
    
    # Extract configuration
    config = None
    for config_key in ['config', 'model_config', 'args']:
        if config_key in checkpoint:
            config = checkpoint[config_key]
            break
    
    if config is None:
        print("No config found in checkpoint, using defaults")
        config = {
            'input_shape': (1, 128, 938),
            'latent_dim': 64,
            'conditional': False,
            'num_classes': 0,
            'embed_dim': 64
        }
    elif hasattr(config, '__dict__'):
        config = vars(config)
    
    print(f"Model config: {config}")
    
    # Initialize model
    try:
        if config.get('conditional', False) and config.get('num_classes', 0) > 0:
            model = ConditionalVariationalAutoEncoder(
                input_shape=config.get('input_shape', (1, 128, 938)),
                latent_dim=config.get('latent_dim', 64),
                num_classes=config.get('num_classes', 3),
                embed_dim=config.get('embed_dim', 64),
                beta=config.get('beta', 1e-6)
            )
            is_conditional = True
        else:
            model = ImprovedVariationalAutoEncoder(
                input_shape=config.get('input_shape', (1, 128, 938)),
                latent_dim=config.get('latent_dim', 64),
                beta=config.get('beta', 1e-6)
            )
            is_conditional = False
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load with error handling
        try:
            model.load_state_dict(state_dict, strict=True)
            print("Model loaded successfully with strict=True")
        except RuntimeError as e:
            print(f"Strict loading failed: {e}")
            print("Attempting non-strict loading...")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
        
        model.to(device)
        model.eval()
        
        return model, is_conditional, config
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None, None, None


class ImprovedSpectrogramToAudio:
    """Enhanced spectrogram to audio conversion"""
    
    def __init__(self, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Precompute mel basis for inversion
        self.mel_basis = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, 
            fmin=80, fmax=8000
        )
        
        # Pseudo-inverse for mel inversion
        self.inv_mel_basis = np.linalg.pinv(self.mel_basis)
        
        print(f"Audio converter initialized:")
        print(f"  Sample rate: {sr} Hz")
        print(f"  FFT size: {n_fft}")
        print(f"  Hop length: {hop_length}")
        print(f"  Mel bins: {n_mels}")
    
    def denormalize_spectrogram(self, normalized_spec):
        """Convert normalized spectrogram back to reasonable scale"""
        # Ensure input is numpy
        if torch.is_tensor(normalized_spec):
            normalized_spec = normalized_spec.detach().cpu().numpy()
        
        # Clip to valid range
        normalized_spec = np.clip(normalized_spec, 0, 1)
        
        # Convert from [0,1] back to dB-like scale
        db_spec = normalized_spec * 80.0 - 60.0  # Map [0,1] to [-60, 20] dB
        
        # Convert dB to linear scale
        linear_spec = librosa.db_to_amplitude(db_spec)
        
        return linear_spec
    
    def mel_to_stft_spectrogram(self, mel_spec):
        """Convert mel spectrogram to STFT magnitude spectrogram"""
        # Invert mel transformation
        stft_spec = np.dot(self.inv_mel_basis, mel_spec)
        
        # Ensure positive values
        stft_spec = np.maximum(stft_spec, 1e-8)
        
        return stft_spec
    
    def enhanced_griffin_lim(self, spectrogram, n_iter=100, momentum=0.99):
        """Enhanced Griffin-Lim algorithm with momentum"""
        print(f"Converting spectrogram to audio using Enhanced Griffin-Lim...")
        print(f"Spectrogram shape: {spectrogram.shape}")
        
        # Denormalize spectrogram
        magnitude_spec = self.denormalize_spectrogram(spectrogram)
        
        # Convert mel to STFT if needed
        if magnitude_spec.shape[0] == self.n_mels:
            print("Converting from mel-scale to linear frequency...")
            magnitude_spec = self.mel_to_stft_spectrogram(magnitude_spec)
        
        # Ensure correct frequency dimension
        expected_freq_bins = self.n_fft // 2 + 1
        if magnitude_spec.shape[0] != expected_freq_bins:
            print(f"Resizing from {magnitude_spec.shape[0]} to {expected_freq_bins} frequency bins...")
            # Use linear interpolation to resize
            from scipy.interpolate import interp1d
            old_freqs = np.linspace(0, 1, magnitude_spec.shape[0])
            new_freqs = np.linspace(0, 1, expected_freq_bins)
            
            resized_spec = np.zeros((expected_freq_bins, magnitude_spec.shape[1]))
            for t in range(magnitude_spec.shape[1]):
                interp_func = interp1d(old_freqs, magnitude_spec[:, t], 
                                     kind='linear', bounds_error=False, fill_value=0)
                resized_spec[:, t] = interp_func(new_freqs)
            
            magnitude_spec = resized_spec
        
        print(f"Final magnitude shape: {magnitude_spec.shape}")
        
        # Initialize with random phase
        angles = np.random.uniform(0, 2*np.pi, magnitude_spec.shape)
        complex_spec = magnitude_spec * np.exp(1j * angles)
        
        # Enhanced Griffin-Lim iteration with momentum
        prev_complex = complex_spec.copy()
        
        for i in range(n_iter):
            # ISTFT to time domain
            audio = librosa.istft(
                complex_spec,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                length=None
            )
            
            # STFT back to frequency domain
            new_complex = librosa.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft
            )
            
            # Apply magnitude constraint
            new_magnitude = np.abs(new_complex)
            new_magnitude = np.maximum(new_magnitude, 1e-10)
            new_phase = np.angle(new_complex)
            
            # Reconstruct with original magnitude
            constrained_complex = magnitude_spec * np.exp(1j * new_phase)
            
            # Apply momentum
            if i > 0 and momentum > 0:
                complex_spec = momentum * complex_spec + (1 - momentum) * constrained_complex
            else:
                complex_spec = constrained_complex
            
            # Progress update
            if i % 20 == 0:
                print(f"  Griffin-Lim iteration {i}/{n_iter}")
        
        # Final conversion
        audio = librosa.istft(
            complex_spec,
            hop_length=self.hop_length,
            win_length=self.n_fft
        )
        
        print(f"Generated audio length: {len(audio)} samples ({len(audio)/self.sr:.2f} seconds)")
        
        return audio
    
    def vocode_synthesis(self, spectrogram, formant_shift=1.0):
        """Vocoder-like synthesis as alternative method"""
        print("Converting spectrogram using vocoder synthesis...")
        
        # Denormalize spectrogram
        magnitude_spec = self.denormalize_spectrogram(spectrogram)
        
        # Convert mel to STFT if needed
        if magnitude_spec.shape[0] == self.n_mels:
            magnitude_spec = self.mel_to_stft_spectrogram(magnitude_spec)
        
        # Get frequency and time parameters
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        n_frames = magnitude_spec.shape[1]
        frame_times = librosa.frames_to_time(np.arange(n_frames), 
                                           sr=self.sr, hop_length=self.hop_length)
        
        # Initialize output audio
        audio_length = int((n_frames - 1) * self.hop_length + self.n_fft)
        audio = np.zeros(audio_length)
        
        # Generate audio frame by frame
        window = get_window('hann', self.n_fft)
        
        for frame_idx in range(n_frames):
            frame_mag = magnitude_spec[:, frame_idx]
            
            # Time vector for this frame
            frame_start = frame_idx * self.hop_length
            t_frame = np.arange(self.n_fft) / self.sr
            
            # Generate frame audio using additive synthesis
            frame_audio = np.zeros(self.n_fft)
            
            # Add harmonics for prominent frequency bins
            for freq_idx in range(0, len(freqs), 2):  # Every 2nd bin to reduce computation
                amplitude = frame_mag[freq_idx]
                frequency = freqs[freq_idx] * formant_shift
                
                if amplitude > 0.01 and 80 <= frequency <= 8000:
                    # Add sinusoid with envelope
                    phase = np.random.uniform(0, 2*np.pi)
                    envelope = np.exp(-t_frame * 5)  # Decay envelope
                    frame_audio += amplitude * envelope * np.sin(2 * np.pi * frequency * t_frame + phase)
            
            # Apply window and overlap-add
            frame_audio *= window * 0.1  # Scale down
            
            # Overlap-add to main audio buffer
            end_idx = min(frame_start + self.n_fft, len(audio))
            actual_length = end_idx - frame_start
            audio[frame_start:end_idx] += frame_audio[:actual_length]
        
        return audio
    
    def post_process_audio(self, audio, enhance_chicken_sounds=True):
        """Post-process generated audio for better quality"""
        print("Post-processing audio...")
        
        # Remove DC component
        audio = audio - np.mean(audio)
        
        # Apply bandpass filter for chicken frequency range
        if enhance_chicken_sounds:
            nyquist = self.sr // 2
            low_cut = 200 / nyquist    # Remove very low frequencies
            high_cut = 6000 / nyquist  # Focus on chicken vocal range
            
            try:
                b, a = butter(4, [low_cut, high_cut], btype='band')
                audio = filtfilt(b, a, audio)
            except:
                print("Bandpass filtering failed, skipping...")
        
        # Gentle compression using tanh
        audio = np.tanh(audio * 2.0) * 0.8
        
        # Smooth out harsh transients
        audio = gaussian_filter1d(audio, sigma=0.5)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.7  # Leave headroom
        
        # Add subtle envelope to avoid clicks
        fade_samples = int(0.01 * self.sr)  # 10ms fade
        if len(audio) > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out
        
        return audio


def generate_audio_from_vae(model, device, converter, method='enhanced_griffin_lim', 
                           class_label=None, num_samples=1):
    """Generate audio samples from trained VAE"""
    model.eval()
    
    generated_audio = []
    spectrograms = []
    
    with torch.no_grad():
        for i in range(num_samples):
            print(f"\nGenerating sample {i+1}/{num_samples}...")
            
            # Generate spectrogram
            if hasattr(model, 'sample_class') and class_label is not None:
                print(f"Generating for class {class_label}")
                spectrogram = model.sample_class(class_label, 1, device=device)
            else:
                spectrogram = model.sample(1, device=device)
            
            # Extract spectrogram
            spec_np = spectrogram.cpu().numpy()[0, 0]  # Remove batch and channel dims
            spectrograms.append(spec_np)
            
            print(f"Generated spectrogram shape: {spec_np.shape}")
            print(f"Spectrogram range: [{spec_np.min():.4f}, {spec_np.max():.4f}]")
            
            # Convert to audio
            try:
                if method == 'enhanced_griffin_lim':
                    audio = converter.enhanced_griffin_lim(spec_np, n_iter=100, momentum=0.99)
                elif method == 'vocode':
                    audio = converter.vocode_synthesis(spec_np, formant_shift=1.2)
                else:
                    # Fallback to basic Griffin-Lim
                    magnitude_spec = converter.denormalize_spectrogram(spec_np)
                    if magnitude_spec.shape[0] == converter.n_mels:
                        magnitude_spec = converter.mel_to_stft_spectrogram(magnitude_spec)
                    audio = librosa.griffinlim(
                        magnitude_spec,
                        n_iter=60,
                        hop_length=converter.hop_length
                    )
                
                # Post-process
                audio = converter.post_process_audio(audio, enhance_chicken_sounds=True)
                generated_audio.append(audio)
                
                print(f"Audio generated: {len(audio)} samples ({len(audio)/converter.sr:.2f}s)")
                
            except Exception as e:
                print(f"Error converting spectrogram to audio: {e}")
                # Generate silence as fallback
                audio = np.zeros(int(10 * converter.sr))
                generated_audio.append(audio)
    
    return generated_audio, spectrograms


def create_comprehensive_analysis(spectrograms, audio_list, sr, save_path=None):
    """Create comprehensive analysis plots"""
    n_samples = len(spectrograms)
    
    fig, axes = plt.subplots(3, n_samples, figsize=(5*n_samples, 12))
    if n_samples == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(n_samples):
        spec = spectrograms[i]
        audio = audio_list[i]
        
        # Original generated spectrogram
        im1 = axes[0, i].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        axes[0, i].set_title(f'Generated Spectrogram {i+1}')
        axes[0, i].set_xlabel('Time Frames')
        axes[0, i].set_ylabel('Frequency Bins')
        plt.colorbar(im1, ax=axes[0, i])
        
        # Generated audio waveform
        time_axis = np.linspace(0, len(audio) / sr, len(audio))
        axes[1, i].plot(time_axis, audio, 'b-', alpha=0.8)
        axes[1, i].set_title(f'Generated Audio {i+1}')
        axes[1, i].set_xlabel('Time (s)')
        axes[1, i].set_ylabel('Amplitude')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_ylim([-1, 1])
        
        # Reconstructed spectrogram for comparison
        if len(audio) > 0:
            D = librosa.stft(audio, n