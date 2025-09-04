import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from model import VariationalAutoEncoder, ConditionalVariationalAutoEncoder


def load_vae_model(model_path, device='cpu'):
    """Load the trained VAE model with flexible config handling"""
    print(f"Loading VAE model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint
    config = None
    for config_key in ['model_config', 'config', 'args']:
        if config_key in checkpoint:
            config = checkpoint[config_key]
            break
    
    if hasattr(config, '__dict__'):
        config = vars(config)
    
    # Set defaults if config missing
    if config is None or not isinstance(config, dict):
        config = {
            'latent_dim': 256,
            'input_shape': (1, 1025, 938),
            'beta': 0.001,
            'conditional': False,
            'num_classes': 0
        }
    
    print(f"Model config: latent_dim={config.get('latent_dim')}, conditional={config.get('conditional')}")
    
    # Initialize model
    if config.get('conditional', False) and config.get('num_classes', 0) > 0:
        model = ConditionalVariationalAutoEncoder(
            input_shape=config.get('input_shape', (1, 1025, 938)),
            num_classes=config.get('num_classes', 3),
            latent_dim=config.get('latent_dim', 256),
            embed_dim=config.get('embed_dim', 50)
        )
        is_conditional = True
    else:
        model = VariationalAutoEncoder(
            input_shape=config.get('input_shape', (1, 1025, 938)),
            latent_dim=config.get('latent_dim', 256)
        )
        is_conditional = False
    
    # Load with strict=False to handle architecture mismatches
    state_dict = checkpoint['model_state_dict']
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: {len(missing_keys)} missing keys in state dict")
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected keys in state dict")
    
    model.to(device)
    model.eval()
    
    return model, is_conditional, config


def generate_spectrogram_from_vae(model, is_conditional=False, class_label=None, device='cpu'):
    """Generate a spectrogram using the trained VAE"""
    model.eval()
    
    with torch.no_grad():
        if is_conditional and class_label is not None:
            print(f"Generating spectrogram for class {class_label}")
            spectrogram = model.sample_class(class_label, 1, device=device)
        else:
            print("Generating unconditional spectrogram")
            spectrogram = model.sample(1, device=device)
    
    # Convert to numpy and remove batch dimension
    spec_np = spectrogram.cpu().numpy()[0, 0]  # Remove batch and channel dims
    
    return spec_np


def spectrogram_to_audio_griffin_lim(spectrogram, sr=22050, n_fft=2048, hop_length=512, n_iter=60):
    """Convert spectrogram to audio using Griffin-Lim algorithm"""
    print(f"Converting spectrogram to audio...")
    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Sample rate: {sr}, n_fft: {n_fft}, hop_length: {hop_length}")
    
    # Ensure spectrogram is in the right format
    # Your spectrograms are normalized to [0, 1], so we need to convert back
    
    # Method 1: Treat as magnitude spectrogram
    magnitude_spec = spectrogram
    
    # Optionally apply some processing to make it more audio-like
    # Add some emphasis to higher frequencies (chickens have high-frequency content)
    freq_weights = np.linspace(0.5, 1.5, magnitude_spec.shape[0])
    magnitude_spec = magnitude_spec * freq_weights[:, np.newaxis]
    
    # Convert to linear scale if it was in dB (adjust based on your preprocessing)
    # magnitude_spec = np.power(10, magnitude_spec)  # Uncomment if your spectrograms are in dB
    
    # Use Griffin-Lim to reconstruct audio
    audio = librosa.griffinlim(
        magnitude_spec,
        n_iter=n_iter,
        hop_length=hop_length,
        n_fft=n_fft,
        length=None
    )
    
    return audio, sr


def spectrogram_to_audio_mel(spectrogram, sr=22050, n_fft=2048, hop_length=512, n_mels=128, n_iter=60):
    """Convert mel-spectrogram to audio using mel inversion + Griffin-Lim"""
    print(f"Converting mel-spectrogram to audio...")
    print(f"Spectrogram shape: {spectrogram.shape}")
    
    # If your spectrograms are mel-scaled, we need to invert the mel transformation
    try:
        # Create mel filter bank
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=spectrogram.shape[0])
        
        # Invert mel transformation (approximate)
        magnitude_spec = np.dot(mel_basis.T, spectrogram)
        
        # Apply Griffin-Lim
        audio = librosa.griffinlim(
            magnitude_spec,
            n_iter=n_iter,
            hop_length=hop_length,
            n_fft=n_fft
        )
        
        return audio, sr
        
    except Exception as e:
        print(f"Mel inversion failed: {e}")
        print("Falling back to direct Griffin-Lim...")
        return spectrogram_to_audio_griffin_lim(spectrogram, sr, n_fft, hop_length, n_iter)


def enhance_audio_for_chicken_sounds(audio, sr):
    """Apply audio processing to enhance chicken-like characteristics"""
    print("Enhancing audio for chicken sounds...")
    
    # Apply some filtering to emphasize frequencies typical of chicken sounds
    # Chickens make sounds mainly in 1-4 kHz range
    
    # High-pass filter to remove low rumble
    audio_filtered = librosa.effects.preemphasis(audio, coef=0.97)
    
    # Apply some compression to make it more audible
    audio_compressed = np.tanh(audio_filtered * 2.0) * 0.8
    
    # Add slight pitch variation to make it more natural
    # This is a simple approach - you could use more sophisticated methods
    audio_enhanced = audio_compressed
    
    return audio_enhanced


def save_audio_file(audio, sr, output_path, format='wav'):
    """Save audio to file"""
    print(f"Saving audio to: {output_path}")
    
    # Normalize audio to prevent clipping
    audio_normalized = audio / (np.max(np.abs(audio)) + 1e-8)
    audio_normalized = audio_normalized * 0.8  # Leave some headroom
    
    # Save the audio file
    sf.write(output_path, audio_normalized, sr, format=format.upper())
    
    print(f"Audio saved successfully!")
    print(f"Duration: {len(audio_normalized) / sr:.2f} seconds")


def plot_audio_analysis(spectrogram, audio, sr, save_path=None):
    """Plot analysis of the generated audio"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot original spectrogram
    axes[0].imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Generated Spectrogram')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Frequency')
    
    # Plot waveform
    time = np.linspace(0, len(audio) / sr, len(audio))
    axes[1].plot(time, audio)
    axes[1].set_title('Generated Audio Waveform')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    # Plot audio spectrogram for comparison
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[2], cmap='viridis')
    axes[2].set_title('Reconstructed Audio Spectrogram')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Analysis plot saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate Chicken Sounds from VAE')
    parser.add_argument('--model_path', required=True, help='Path to trained VAE model (.pth)')
    parser.add_argument('--output_dir', default='generated_audio', help='Output directory for audio files')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of audio samples to generate')
    parser.add_argument('--class_label', type=int, default=None, help='Class label for conditional generation')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Audio sample rate')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT size')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length for STFT')
    parser.add_argument('--griffin_lim_iters', type=int, default=60, help='Griffin-Lim iterations')
    parser.add_argument('--enhance_audio', action='store_true', help='Apply audio enhancement for chicken sounds')
    parser.add_argument('--method', choices=['griffin_lim', 'mel_inversion'], default='griffin_lim',
                       help='Method for spectrogram to audio conversion')
    parser.add_argument('--device', default='auto', help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--save_analysis', action='store_true', help='Save analysis plots')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load the VAE model
        model, is_conditional, config = load_vae_model(args.model_path, device)
        
        print(f"\nGenerating {args.num_samples} chicken sounds...")
        
        for i in range(args.num_samples):
            print(f"\n--- Generating Sample {i+1}/{args.num_samples} ---")
            
            # Generate spectrogram from VAE
            spectrogram = generate_spectrogram_from_vae(
                model, is_conditional, args.class_label, device
            )
            
            # Convert spectrogram to audio
            if args.method == 'mel_inversion':
                audio, sr = spectrogram_to_audio_mel(
                    spectrogram, args.sample_rate, args.n_fft, 
                    args.hop_length, n_iter=args.griffin_lim_iters
                )
            else:
                audio, sr = spectrogram_to_audio_griffin_lim(
                    spectrogram, args.sample_rate, args.n_fft,
                    args.hop_length, args.griffin_lim_iters
                )
            
            # Enhance audio if requested
            if args.enhance_audio:
                audio = enhance_audio_for_chicken_sounds(audio, sr)
            
            # Save audio file
            if args.class_label is not None:
                filename = f"generated_chicken_class{args.class_label}_sample{i+1:02d}.wav"
            else:
                filename = f"generated_chicken_sample{i+1:02d}.wav"
            
            output_path = output_dir / filename
            save_audio_file(audio, sr, output_path)
            
            # Save analysis plot if requested
            if args.save_analysis:
                analysis_path = output_dir / f"analysis_sample{i+1:02d}.png"
                plot_audio_analysis(spectrogram, audio, sr, analysis_path)
        
        print(f"\n🐔 Successfully generated {args.num_samples} chicken sounds!")
        print(f"Audio files saved in: {output_dir}")
        print(f"\nTo listen to your generated chicken sounds:")
        print(f"  cd {output_dir}")
        print(f"  # Use any audio player to play the .wav files")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()