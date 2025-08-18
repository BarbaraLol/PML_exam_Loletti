import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from audio_generation_model import SpectrogramVAE, ConditionalSpectrogramVAE


def load_trained_vae(model_path, device='cpu'):
    """Load a trained VAE model from checkpoint"""
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']
    
    print("Model configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize model based on config
    if config.get('conditional', False) and config.get('num_classes', 0) > 0:
        model = ConditionalSpectrogramVAE(
            input_shape=config['input_shape'],
            num_classes=config['num_classes'],
            latent_dim=config['latent_dim'],
            beta=config['beta']
        )
        print("Loaded Conditional VAE")
    else:
        model = SpectrogramVAE(
            input_shape=config['input_shape'],
            latent_dim=config['latent_dim'],
            beta=config['beta']
        )
        print("Loaded Standard VAE")
    
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
            print(f"Generating {num_samples} samples for class {class_label}")
            samples = model.sample_class(class_label, num_samples, device=device)
        else:
            # Unconditional generation
            print(f"Generating {num_samples} unconditional samples")
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
            
        im = axes[i].imshow(spec_vis, aspect='auto', origin='lower', cmap='viridis')
        
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
        else:
            axes[i].set_title(f'Spectrogram {i+1}')
        
        axes[i].axis('off')
        
        # Add colorbar to first subplot
        if i == 0:
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
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
    
    print(f"Exploring latent space with {grid_size}x{grid_size} grid...")
    
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


def analyze_latent_representations(model, dataloader, device='cpu', num_samples=100):
    """Analyze the learned latent representations"""
    model.eval()
    
    latent_vectors = []
    labels = []
    
    print(f"Analyzing latent representations from {num_samples} samples...")
    
    with torch.no_grad():
        samples_collected = 0
        for batch in dataloader:
            if samples_collected >= num_samples:
                break
                
            if len(batch) == 2:
                data, batch_labels = batch
                data = data.to(device)
                
                if hasattr(model, 'num_classes') and batch_labels.shape != data.shape:
                    # Conditional VAE
                    batch_labels = batch_labels.to(device)
                    mu, _ = model.encode(data, batch_labels)
                    labels.extend(batch_labels.cpu().numpy())
                else:
                    # Standard VAE
                    mu, _ = model.encode(data)
                    labels.extend([0] * data.size(0))  # Dummy labels
                
                latent_vectors.append(mu.cpu().numpy())
                samples_collected += data.size(0)
    
    if latent_vectors:
        latent_vectors = np.concatenate(latent_vectors, axis=0)
        labels = np.array(labels)
        
        # Plot 2D projection of latent space
        plt.figure(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(latent_vectors[mask, 0], latent_vectors[mask, 1], 
                       c=[colors[i]], label=f'Class {label}', alpha=0.6)
        
        plt.xlabel('Latent Dimension 0')
        plt.ylabel('Latent Dimension 1')
        plt.title('Latent Space Visualization (First 2 Dimensions)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Statistics
        print(f"\nLatent Space Statistics:")
        print(f"  Shape: {latent_vectors.shape}")
        print(f"  Mean: {latent_vectors.mean(axis=0)[:5]}... (first 5 dims)")
        print(f"  Std: {latent_vectors.std(axis=0)[:5]}... (first 5 dims)")
    
    return latent_vectors, labels


def main():
    parser = argparse.ArgumentParser(description='VAE Inference and Generation')
    parser.add_argument('--model_path', required=True, help='Path to trained VAE model')
    parser.add_argument('--output_dir', default='vae_outputs', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--class_label', type=int, default=None, help='Class label for conditional generation')
    parser.add_argument('--explore_latent', action='store_true', help='Explore latent space')
    parser.add_argument('--grid_size', type=int, default=5, help='Grid size for latent exploration')
    parser.add_argument('--analyze_latent', action='store_true', help='Analyze latent representations')
    parser.add_argument('--data_dir', default=None, help='Data directory for latent analysis')
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
    model, config = load_trained_vae(args.model_path, device)
    
    print("\nModel Configuration:")
    print(f"  Type: {'Conditional' if config.get('conditional', False) else 'Standard'} VAE")
    print(f"  Input Shape: {config['input_shape']}")
    print(f"  Latent Dim: {config['latent_dim']}")
    print(f"  Beta: {config['beta']}")
    if config.get('num_classes'):
        print(f"  Classes: {config['num_classes']}")
    
    # Generate samples
    print(f"\n{'='*50}")
    print("GENERATING SAMPLES")
    print('='*50)
    
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
        print(f"\n{'='*50}")
        print("EXPLORING LATENT SPACE")
        print('='*50)
        
        latent_samples, latent_titles = explore_latent_space(
            model, device, grid_size=args.grid_size
        )
        
        visualize_spectrograms(
            latent_samples,
            titles=latent_titles, 
            save_path=os.path.join(args.output_dir, 'latent_space_exploration.png')
        )
    
    # Analyze latent representations if requested
    if args.analyze_latent and args.data_dir:
        print(f"\n{'='*50}")
        print("ANALYZING LATENT REPRESENTATIONS")
        print('='*50)
        
        # Load data for analysis
        from data_loading import create_vae_datasets, load_file_paths, encode_labels
        from sklearn.preprocessing import LabelEncoder
        from torch.utils.data import DataLoader
        
        try:
            file_paths = load_file_paths(args.data_dir)
            
            # Setup label encoder if conditional
            label_encoder = None
            if config.get('conditional', False):
                labels = encode_labels(file_paths)
                if labels:
                    label_encoder = LabelEncoder()
                    label_encoder.fit(labels)
            
            # Create dataset for analysis
            _, _, test_dataset, _, _ = create_vae_datasets(
                args.data_dir,
                label_encoder=label_encoder,
                conditional=config.get('conditional', False),
                augment=False  # No augmentation for analysis
            )
            
            # Create dataloader
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Analyze latent space
            latent_vectors, labels = analyze_latent_representations(
                model, test_loader, device, num_samples=200
            )
            
            # Save latent analysis
            np.save(os.path.join(args.output_dir, 'latent_vectors.npy'), latent_vectors)
            np.save(os.path.join(args.output_dir, 'latent_labels.npy'), labels)
            print(f"Latent analysis saved to: {args.output_dir}")
            
        except Exception as e:
            print(f"Error in latent analysis: {e}")
    
    print(f"\n{'='*50}")
    print("INFERENCE COMPLETE")
    print('='*50)
    print(f"All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()