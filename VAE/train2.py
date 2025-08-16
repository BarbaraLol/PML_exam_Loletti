import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
import csv
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Import your modules - UPDATED IMPORTS
from model import VariationalAutoEncoder, ConditionalVariationalAutoEncoder
from data_loading import create_vae_datasets, encode_labels, load_file_paths
from train_utils import save_checkpoint


def save_sample_outputs(model, device, output_dir, epoch, num_samples=8, 
                       conditional=False, num_classes=None):
    """Save sample generated spectrograms during training"""
    model.eval()
    with torch.no_grad():
        if conditional and num_classes:
            # Generate samples from each class
            samples_per_class = max(1, num_samples // num_classes)
            all_samples = []
            
            for class_idx in range(min(num_classes, num_samples)):
                class_samples = model.sample_class(
                    class_idx, samples_per_class, device=device
                )
                all_samples.append(class_samples)
            
            samples = torch.cat(all_samples, dim=0)[:num_samples]
        else:
            samples = model.sample(num_samples, device=device)
        
        # Convert to numpy and save as images
        samples_np = samples.cpu().numpy()
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(min(num_samples, len(axes))):
            ax = axes[i]
            # Remove channel dimension for visualization
            spec_vis = samples_np[i, 0] if samples_np[i].ndim == 3 else samples_np[i]
            ax.imshow(spec_vis, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f'Generated Sample {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'generated_samples_epoch_{epoch}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()


def plot_reconstruction(model, dataloader, device, output_dir, epoch, num_samples=4, conditional=False):
    """Plot original vs reconstructed spectrograms"""
    model.eval()
    with torch.no_grad():
        # Get a batch from dataloader
        for batch in dataloader:
            if conditional:
                originals = batch[0].to(device)
                labels = batch[1].to(device)
                # Get reconstruction - UPDATED CALL
                reconstructed, _, _ = model(originals, labels)
            else:  # Non-conditional
                originals = batch.to(device)
                # Get reconstruction - UPDATED CALL
                reconstructed, _, _ = model(originals)
            break
    
    # Adjust number of samples to actual batch size
    num_samples = min(num_samples, originals.size(0))
    
    # Convert to numpy
    originals_np = originals[:num_samples].cpu().numpy()
    reconstructed_np = reconstructed[:num_samples].cpu().numpy()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_samples):
        # Original
        orig_vis = originals_np[i, 0] if originals_np[i].ndim == 3 else originals_np[i]
        axes[0, i].imshow(orig_vis, aspect='auto', origin='lower', cmap='viridis')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstructed
        recon_vis = reconstructed_np[i, 0] if reconstructed_np[i].ndim == 3 else reconstructed_np[i]
        axes[1, i].imshow(recon_vis, aspect='auto', origin='lower', cmap='viridis')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'reconstruction_epoch_{epoch}.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()


def train_vae(model, train_loader, val_loader, device, args, output_dir, conditional=False):
    """Main training loop for VAE - SIMPLIFIED"""
    
    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-6
    )
    
    # Training log
    log_file = os.path.join(output_dir, "vae_training_log.csv")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_recon_loss', 'train_kl_loss', 
                        'val_loss', 'val_recon_loss', 'val_kl_loss', 'lr', 'time_elapsed'])
    
    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    print("Starting VAE training...")
    print(f"Model: {'Conditional' if conditional else 'Standard'} VAE")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Beta: {args.beta}")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_total_loss = 0
        train_recon_loss = 0  
        train_kl_loss = 0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                if conditional:
                    data, labels = batch
                    data, labels = data.to(device), labels.to(device)
                else:
                    data = batch.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass - UPDATED CALLS
                if conditional:
                    recon_x, mu, logvar = model(data, labels)
                else:
                    recon_x, mu, logvar = model(data)
                
                # Compute loss - USING MODEL'S BUILT-IN LOSS FUNCTION
                total_loss, recon_loss, kl_loss = model.loss_function(
                    recon_x, data, mu, logvar, args.beta
                )
                
                # Check for NaN losses
                if torch.isnan(total_loss):
                    print(f"NaN loss detected at epoch {epoch}, batch {batch_idx}")
                    continue
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                
                optimizer.step()
                
                # Accumulate losses
                train_total_loss += total_loss.item()
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()
                train_batches += 1
                
                # Progress updates
                if batch_idx % 20 == 0:
                    print(f"Epoch {epoch+1}/{args.epochs} [{batch_idx}/{len(train_loader)}] "
                          f"Loss: {total_loss.item():.4f} | "
                          f"Recon: {recon_loss.item():.4f} | "
                          f"KL: {kl_loss.item():.4f}")
                          
            except Exception as e:
                print(f"Training batch error: {e}")
                continue
        
        # Average training losses
        if train_batches > 0:
            train_total_loss /= train_batches
            train_recon_loss /= train_batches
            train_kl_loss /= train_batches
        else:
            print(f"No valid batches in epoch {epoch+1}")
            continue
        
        # Validation phase
        model.eval()
        val_total_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    if conditional:
                        data, labels = batch
                        data, labels = data.to(device), labels.to(device)
                        recon_x, mu, logvar = model(data, labels)
                    else:
                        data = batch.to(device)
                        recon_x, mu, logvar = model(data)
                    
                    # Compute validation loss
                    total_loss, recon_loss, kl_loss = model.loss_function(
                        recon_x, data, mu, logvar, beta=args.beta
                    )
                    
                    if not torch.isnan(total_loss):
                        val_total_loss += total_loss.item()
                        val_recon_loss += recon_loss.item()
                        val_kl_loss += kl_loss.item()
                        val_batches += 1
                        
                except Exception as e:
                    print(f"Validation batch error: {e}")
                    continue
        
        # Average validation losses
        if val_batches > 0:
            val_total_loss /= val_batches
            val_recon_loss /= val_batches
            val_kl_loss /= val_batches
        else:
            val_total_loss = val_recon_loss = val_kl_loss = float('nan')
        
        # Update learning rate
        if not np.isnan(val_total_loss):
            scheduler.step(val_total_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Time tracking
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Time: {epoch_time:.2f}s | Total: {total_time//60:.0f}m {total_time%60:.0f}s")
        print(f"LR: {current_lr:.2e} | Beta: {args.beta:.4f}")
        print(f"Train - Total: {train_total_loss:.4f} | Recon: {train_recon_loss:.4f} | KL: {train_kl_loss:.4f}")
        print(f"Val   - Total: {val_total_loss:.4f} | Recon: {val_recon_loss:.4f} | KL: {val_kl_loss:.4f}")
        
        # Log to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1, train_total_loss, train_recon_loss, train_kl_loss,
                val_total_loss, val_recon_loss, val_kl_loss, current_lr, total_time
            ])
        
        # Save visualizations periodically
        if (epoch + 1) % 10 == 0 or epoch == 0:
            try:
                save_sample_outputs(model, device, output_dir, epoch+1, conditional=conditional, num_classes=getattr(model, 'num_classes', None))
                plot_reconstruction(model, val_loader, device, output_dir, epoch+1, conditional=conditional)
            except Exception as e:
                print(f"Error saving visualizations: {e}")
        
        # Model saving and early stopping
        if not np.isnan(val_total_loss) and val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            patience_counter = 0
            
            # # Save best model
            # torch.save({
            #     'epoch': epoch+1,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'val_loss': val_total_loss,
            #     'best_val_loss': best_val_loss,
            #     'model_config': {
            #         'input_shape': model.input_shape,
            #         'latent_dim': model.latent_dim,
            #         'beta': args.beta,
            #         'conditional': conditional,
            #         'num_classes': getattr(model, 'num_classes', None)
            #     }
            # }, os.path.join(output_dir, 'best_vae_model.pth'))
            
            # print(f"✓ Saved best model at epoch {epoch+1} with val loss: {val_total_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1} after {args.patience} epochs without improvement")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
        
        # Stop if learning rate becomes too small
        if current_lr < 1e-7:
            print(f"Learning rate too small ({current_lr:.2e}), stopping training")
            break
    
    # Save final model
    # torch.save({
    #     'epoch': epoch+1,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'val_loss': val_total_loss if not np.isnan(val_total_loss) else float('inf'),
    #     'model_config': {
    #         'input_shape': model.input_shape,
    #         'latent_dim': model.latent_dim,
    #         'beta': args.beta,
    #         'conditional': conditional,
    #         'num_classes': getattr(model, 'num_classes', None)
    #     }
    # }, os.path.join(output_dir, 'final_vae_model.pth'))
    
    print(f"\nTraining completed in {total_time//60:.0f}m {total_time%60:.0f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Simple Spectrogram VAE')
    parser.add_argument('--data_dir', required=True, help="Path to spectrogram directory")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")  # Reduced default
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--latent_dim', type=int, default=256, help="Latent dimension")  # Reduced default
    parser.add_argument('--beta', type=float, default=0.001, help="Beta parameter for β-VAE")
    parser.add_argument('--conditional', action='store_true', help="Use conditional VAE")
    parser.add_argument('--embed_dim', type=int, default=50, help="Label embedding dimension")
    parser.add_argument('--output_dir', default='simple_vae_results', help="Directory to save outputs")
    parser.add_argument('--patience', type=int, default=15, help="Patience for early stopping")
    parser.add_argument('--augment', action='store_true', help="Apply data augmentation")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument('--grad_clip', type=float, default=1.0, help="Gradient clipping max norm")
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"simple_vae_experiment_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Data loading
    print("Loading data...")
    file_paths = load_file_paths(args.data_dir)
    print(f"Found {len(file_paths)} spectrogram files")
    
    if len(file_paths) == 0:
        raise ValueError(f"No .pt files found in {args.data_dir}")
    
    # Setup label encoder for conditional VAE
    label_encoder = None
    if args.conditional:
        labels = encode_labels(file_paths)
        if labels:
            label_encoder = LabelEncoder()
            label_encoder.fit(labels)
            print(f"Found {len(label_encoder.classes_)} classes: {label_encoder.classes_}")
        else:
            print("Warning: No labels found, switching to standard VAE")
            args.conditional = False
    
    # Create datasets
    try:
        train_dataset, val_dataset, test_dataset, spectrogram_shape, num_classes = create_vae_datasets(
            args.data_dir, 
            label_encoder=label_encoder,
            conditional=args.conditional,
            augment=args.augment
        )
    except Exception as e:
        print(f"Error creating datasets: {e}")
        return
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Spectrogram shape: {spectrogram_shape}")
    
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,  # Reduced for stability
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=2,  # Reduced for stability
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False
    )
    
    # Initialize model - UPDATED MODEL INITIALIZATION
    print(f"Initializing {'Conditional' if args.conditional else 'Standard'} VAE...")
    print(f"Input shape for model: {spectrogram_shape} (type: {type(spectrogram_shape)})")
    
    try:
        if args.conditional and num_classes > 0:
            model = ConditionalVariationalAutoEncoder(
                input_shape=spectrogram_shape,
                latent_dim=args.latent_dim,
                num_classes=num_classes,
                embed_dim=args.embed_dim
            ).to(device)
        else:
            spectrogram_shape = (1, 1025, 469)
            model = VariationalAutoEncoder(
                input_shape=spectrogram_shape,
                latent_dim=args.latent_dim
            ).to(device)
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save model configuration
    with open(os.path.join(output_dir, 'model_config.txt'), 'w') as f:
        f.write(f"Simple VAE Configuration\n")
        f.write(f"========================\n")
        f.write(f"Model Type: {'Conditional' if args.conditional else 'Standard'} VAE\n")
        f.write(f"Input Shape: {spectrogram_shape}\n")
        f.write(f"Latent Dimension: {args.latent_dim}\n")
        f.write(f"Beta: {args.beta}\n")
        f.write(f"Number of Classes: {num_classes}\n")
        if args.conditional:
            f.write(f"Embedding Dimension: {args.embed_dim}\n")
        f.write(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        
        if label_encoder:
            f.write(f"\nClass Labels: {list(label_encoder.classes_)}\n")
    
    # Start training
    try:
        train_vae(model, train_loader, val_loader, device, args, output_dir, args.conditional)
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate final samples
    print("Generating final sample outputs...")
    try:
        save_sample_outputs(model, device, output_dir, "final", num_samples=16, 
                           conditional=args.conditional, num_classes=num_classes)
        plot_reconstruction(model, val_loader, device, output_dir, "final", conditional=args.conditional)
    except Exception as e:
        print(f"Error generating final samples: {e}")
    
    print(f"Training complete! Results saved in: {output_dir}")


if __name__ == "__main__":
    main()