import torch
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
import os
from torch.utils.data import DataLoader
from data_loading import create_vae_datasets, encode_labels, load_file_paths
from sklearn.preprocessing import LabelEncoder
from model import ConditionalVariationalAutoEncoder, VariationalAutoEncoder

def detect_and_load_model(model_path, spectrogram_shape, device='cpu'):
    """Auto-detect if model is conditional or standard VAE and load appropriately"""
    
    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Check if model has conditional-specific layers
    state_dict_keys = list(checkpoint['model_state_dict'].keys())
    
    is_conditional = any('label_embedding' in key for key in state_dict_keys)
    has_expanded_encoder = any(key.endswith('encoder.0.weight') and 
                              checkpoint['model_state_dict'][key].shape[1] > 1 
                              for key in state_dict_keys)
    
    # Extract model parameters from checkpoint
    if 'args' in checkpoint:
        args = checkpoint['args']
        latent_dim = args.get('latent_dim', 1024)
        embed_dim = args.get('embed_dim', 50)
        num_classes = args.get('num_classes', 3)
    else:
        # Fallback defaults
        latent_dim = 1024
        embed_dim = 50
        num_classes = 3
    
    print(f"Model type detection:")
    print(f"  Has label_embedding: {is_conditional}")
    print(f"  Has expanded encoder: {has_expanded_encoder}")
    print(f"  Latent dim: {latent_dim}")
    
    # Initialize appropriate model
    if is_conditional or has_expanded_encoder:
        print("Loading as Conditional VAE")
        model = ConditionalVariationalAutoEncoder(
            input_shape=spectrogram_shape,
            latent_dim=latent_dim,
            num_classes=num_classes,
            embed_dim=embed_dim
        ).to(device)
        conditional = True
    else:
        print("Loading as Standard VAE")
        model = VariationalAutoEncoder(
            input_shape=spectrogram_shape,
            latent_dim=latent_dim
        ).to(device)
        conditional = False
    
    # Load state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Available keys in checkpoint:")
        for key in list(checkpoint['model_state_dict'].keys())[:10]:
            print(f"  {key}: {checkpoint['model_state_dict'][key].shape}")
        raise
    
    return model, conditional, checkpoint

def extract_latent_representations_fixed(model, dataloader, device='cpu', max_samples=1000, conditional=False):
    """Fixed version that handles both conditional and standard VAEs"""
    model.eval()
    
    latent_vectors = []
    labels = []
    
    samples_collected = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if samples_collected >= max_samples:
                break
                
            try:
                if conditional and isinstance(batch, tuple) and len(batch) == 2:
                    # Conditional case
                    data, batch_labels = batch
                    data = data.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    # Get latent representations from conditional VAE
                    mu, logvar = model.encoder(data)  # Standard VAE encoder doesn't take labels
                    
                    latent_vectors.append(mu.cpu().numpy())
                    labels.extend(batch_labels.cpu().numpy())
                    
                else:
                    # Non-conditional case OR standard VAE
                    if isinstance(batch, tuple):
                        data, batch_labels = batch
                        batch_labels = batch_labels.cpu().numpy()
                    else:
                        data = batch
                        batch_labels = [0] * data.size(0)  # Dummy labels
                    
                    data = data.to(device)
                    
                    # Get latent representations from standard VAE
                    mu, logvar = model.encoder(data)
                    
                    latent_vectors.append(mu.cpu().numpy())
                    labels.extend(batch_labels)
                
                samples_collected += data.size(0)
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
    
    if latent_vectors:
        latent_vectors = np.concatenate(latent_vectors, axis=0)
        labels = np.array(labels)
        
        print(f"Extracted {len(latent_vectors)} samples")
        print(f"Unique labels: {np.unique(labels)}")
        return latent_vectors, labels
    else:
        raise ValueError("No latent representations extracted")

# Updated main function
def main():
    parser = argparse.ArgumentParser(description='Analyze latent distributions - works with both VAE types')
    parser.add_argument('--model_path', required=True, help='Path to trained VAE model')
    parser.add_argument('--data_dir', required=True, help='Directory containing spectrogram data')
    parser.add_argument('--output_dir', default='latent_analysis', help='Output directory for plots')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum samples to analyze')
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
    
    # Load data and create label encoder
    print("Setting up data...")
    file_paths = load_file_paths(args.data_dir)
    labels = encode_labels(file_paths)
    
    if labels:
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        class_names = list(label_encoder.classes_)
        print(f"Found {len(class_names)} classes: {class_names}")
        data_conditional = True
    else:
        label_encoder = None
        class_names = ['Unlabeled']
        print("No labels found - will analyze as single group")
        data_conditional = False
    
    # Get spectrogram shape
    _, _, _, spectrogram_shape, _ = create_vae_datasets(
        args.data_dir,
        label_encoder=label_encoder,
        conditional=data_conditional,
        augment=False
    )
    
    # Auto-detect and load model
    print("Auto-detecting model type...")
    model, model_conditional, checkpoint = detect_and_load_model(
        args.model_path, spectrogram_shape, device
    )
    
    # Create dataset with appropriate conditional setting
    # Use model's conditional nature, not data's
    _, _, test_dataset, _, _ = create_vae_datasets(
        args.data_dir,
        label_encoder=label_encoder if model_conditional else None,
        conditional=model_conditional,
        augment=False
    )
    
    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Extract latent representations
    print("Extracting latent representations...")
    latent_vectors, extracted_labels = extract_latent_representations_fixed(
        model, test_loader, device, args.max_samples, model_conditional
    )
    
    # If we have meaningful labels, use them; otherwise create dummy ones
    if data_conditional and len(np.unique(extracted_labels)) > 1:
        final_class_names = class_names
    else:
        # Create artificial groupings based on latent clustering if no labels
        from sklearn.cluster import KMeans
        
        print("No meaningful labels found - creating clusters from latent space...")
        n_clusters = min(3, len(latent_vectors) // 50)  # At least 50 samples per cluster
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            extracted_labels = kmeans.fit_predict(latent_vectors)
            final_class_names = [f'Cluster {i}' for i in range(n_clusters)]
        else:
            extracted_labels = np.zeros(len(latent_vectors))
            final_class_names = ['All Data']
    
    # Create analysis plots
    print("Creating latent space analysis...")
    plot_class_distributions(
        latent_vectors, 
        extracted_labels, 
        class_names=final_class_names,
        save_path=os.path.join(args.output_dir, 'latent_analysis.png')
    )
    
    # Save data for further analysis
    np.save(os.path.join(args.output_dir, 'latent_vectors.npy'), latent_vectors)
    np.save(os.path.join(args.output_dir, 'labels.npy'), extracted_labels)
    
    # Save analysis summary
    with open(os.path.join(args.output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write(f"Latent Space Analysis Summary\n")
        f.write(f"============================\n\n")
        f.write(f"Model Type: {'Conditional' if model_conditional else 'Standard'} VAE\n")
        f.write(f"Data has labels: {data_conditional}\n")
        f.write(f"Latent dimensions: {latent_vectors.shape[1]}\n")
        f.write(f"Samples analyzed: {len(latent_vectors)}\n")
        f.write(f"Classes/clusters: {len(final_class_names)}\n")
        f.write(f"Class names: {final_class_names}\n")
    
    print(f"Analysis complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()