import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from audio_generation_model import SpectrogramVAE, ConditionalSpectrogramVAE


def load_trained_vae(model_path, device='cpu'):
    """Load a trained VAE model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']
    
    # Initialize model based on config
    if config.get('conditional', False) and config.get('num_classes', 0) > 0:
        model = ConditionalSpectrogramVAE(
            input_shape=config['input_shape'],
            num_classes=config['num_classes'],
            latent_dim=config['latent_dim'],
            beta=config['beta']
        )
    else:
        model = SpectrogramVAE(
            input_shape=config['input_shape'],
            latent_dim=config['latent_dim'],
            beta=config['beta']
        )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def generate_samples(model, num_samples=10, class_label=None, device='cpu'):
    """Generate new spectrograms using the trained VAE"""
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'num_classes') and class_label is not None:
            # Conditional generation
            samples = model.sample_class(class_label, num_samples, device=device)
        else:
            # Unconditional generation
            samples = model.sample(num_samples, device=device)
    
    return samples.cpu().numpy()


def interpolate_samples(model, spec1, spec2, num_steps=10, class_labels=None, device='cpu'):
    """Interpolate between two spectrograms in latent space"""
    model.eval()
    
    # Add batch dimension if needed
    if spec1.dim() == 3:
        spec1 = spec1.unsqueeze(0)
    if spec2.dim() == 3:
        spec2 = spec2.unsqueeze(0)
    
    spec1, spec2 = spec1.to(device), spec2.to(device)
    
    with torch.no_grad():
        if hasattr(model, 'num_classes') and class_labels is not None:
            # Conditional interpolation
            mu1, _ = model.encode(spec1, class_labels[0:1])
            mu2, _ = model.encode(spec2, class_labels[1:2])
            
            interpolations = []
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                # Use the first class label for all interpolations
                x_interp = model.decode(z_interp, class_labels[0:1])
                interpolations.append(x_interp)
            
            result = torch.cat(interpolations, dim=0)
        else:
            # Standard interpolation
            result = model.interpolate(spec1, spec2, num_steps)
    
    return result.cpu().numpy()


def visualize_spectrograms(spectrograms, titles=None, save_path=None, figsize=None):
    """Visualize multiple spectrograms in a grid"""
    num_specs = len(spectrograms)
    
    # Determine grid size
    if num_specs <= 4:
        rows, cols = 1, num_specs
    elif num_specs <= 8:
        rows, cols = 2, 4
    elif num_specs <= 12:
        rows, cols = 3, 4
    else:
        rows = int(np.ceil(num_specs / 4))
        cols = 4
    
    if figsize is None:
        figsize = (cols * 4, rows * 3)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, spec in enumerate(spectrograms):
        if i >= len(axes):
            break
            
        # Remove channel dimension for visualization
        if spec.ndim == 3:
            spec_vis = spec[0]  # Remove channel dimension
        else:
            spec_vis = spec
            
        axes[i].imshow(spec_vis, aspect='auto', origin='lower', cmap='viridis')
        
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
        else:
            axes[i].set_title(f'Spectrogram {i+1}')
        
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(spectrograms), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def explore_latent_space(model, device='cpu', grid_size=5, range_scale=2.0):
    """Explore the latent space by sampling from a grid"""
    model.eval()
    
    # Create a grid in the first two dimensions of latent space
    x = np.linspace(-range_scale, range_scale, grid_size)
    y = np.linspace(-range_scale, range_scale, grid_size)
    
    samples = []
    titles = []
    
    with torch.no_grad():
        for i, x_val in enumerate(x):
            for j, y_val in enumerate(y):
                # Create latent vector with zeros except first two dims
                z = torch.zeros(1, model.latent_dim, device=device)
                z[0, 0] = x_val
                z[0, 1] = y_val
                
                # Generate sample
                sample = model.decode(z)
                samples.append(sample[0].cpu().numpy())
                titles.append(f'({x_val:.1f}, {y_val:.1f})')
    
    return samples, titles


def main():
    parser = argparse.ArgumentParser(description='VAE Inference and Generation')
    parser.add_argument('--model_path', required=True, help='Path to trained VAE model')
    parser.add_argument('--output_dir', default='vae_outputs', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--class_label', type=int, default=None, help='Class label for conditional generation')
    parser.add_argument('--explore_latent', action='store_true', help='Explore latent space')
    parser.add_argument('--device', default='auto', help='Device to use (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model, config = load_trained_vae(args.model_path, device)
    
    print("Model Configuration:")
    print(f"  Type: {'Conditional' if config.get('conditional', False) else 'Standard'} VAE")
    print(f"  Input Shape: {config['input_shape']}")
    print(f"  Latent Dim: {config['latent_dim']}")
    print(f"  Beta: {config['beta']}")
    if config.get('num_classes'):
        print(f"  Classes: {config['num_classes']}")
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    samples = generate_samples(model, args.num_samples, args.class_label, device)
    
    # Visualize samples
    sample_titles = [f"Generated {i+1}" for i in range(len(samples))]
    if args.class_label is not None:
        sample_titles = [f"Class {args.class_label} - Sample {i+1}" for i in range(len(samples))]
    
    visualize_spectrograms(
        samples, 
        titles=sample_titles,
        save_path=os.path.join(args.output_dir, 'generated_samples.png')
    )
    
    # Save samples as numpy arrays
    np.save(os.path.join(args.output_dir, 'generated_samples.npy'), samples)
    print(f"Samples saved to: {args.output_dir}/generated_samples.npy")
    
    # Explore latent space if requested
    if args.explore_latent:
        print("\nExploring latent space...")
        latent_samples, latent_titles = explore_latent_space(model, device)
        
        visualize_spectrograms(
            latent_samples,
            titles=latent_titles, 
            save_path=os.path.join(args.output_dir, 'latent_space_exploration.png')
        )
    
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()