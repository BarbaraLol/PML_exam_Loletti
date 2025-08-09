import torch
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
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

# Import your modules (assuming they're in the same directory or properly installed)
from audio_generation_model import SpectrogramVAE, ConditionalSpectrogramVAE
from data_loading import create_vae_datasets, encode_labels, load_file_paths
from train_utils import save_checkpoint


def save_sample_outputs(model, device, output_dir, epoch, num_samples=8, conditional=False, num_classes=None):
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
            ax.imshow(samples_np[i, 0], aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f'Generated Sample {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'generated_samples_epoch_{epoch}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()


def plot_reconstruction(model, dataloader, device, output_dir, epoch, num_samples=4):
    """Plot original vs reconstructed spectrograms"""
    model.eval()
    with torch.no_grad():
        # Get a batch
        for batch in dataloader:
            if len(batch) == 2 and isinstance(batch[1], torch.Tensor) and batch[1].shape == batch[0].shape:
                originals = batch[0].to(device)
                reconstructed, _, _, _ = model(originals)
                break
            elif len(batch) == 2:
                originals = batch[0].to(device)
                labels = batch[1].to(device)
                reconstructed, _, _, _ = model(originals, labels)
                break
    
    # Adjust number of samples to actual batch size
    num_samples = min(num_samples, originals.size(0))
    
    # Convert to numpy
    originals_np = originals[:num_samples].cpu().numpy()
    reconstructed_np = reconstructed[:num_samples].cpu().numpy()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(originals_np[i, 0], aspect='auto', origin='lower', cmap='viridis')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(reconstructed_np[i, 0], aspect='auto', origin='lower', cmap='viridis')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'reconstruction_epoch_{epoch}.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()


def check_data_health(dataloader, device, max_batches=5):
    """Check data for NaN/Inf values and extreme ranges"""
    print("Checking data health...")
    batch_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_count >= max_batches:
            break
            
        data = batch[0] if isinstance(batch, (list, tuple)) else batch
        data = data.to(device)
        
        # Check for NaN/Inf
        has_nan = torch.isnan(data).any()
        has_inf = torch.isinf(data).any()
        
        # Check ranges
        data_min = data.min().item()
        data_max = data.max().item()
        data_mean = data.mean().item()
        data_std = data.std().item()
        
        print(f"Batch {batch_idx}: NaN={has_nan}, Inf={has_inf}, "
              f"Range=[{data_min:.4f}, {data_max:.4f}], "
              f"Mean={data_mean:.4f}, Std={data_std:.4f}")
        
        if has_nan or has_inf:
            print(f"WARNING: Invalid values found in batch {batch_idx}")
            
        batch_count += 1


def train_vae(model, train_loader, val_loader, device, args, output_dir, conditional=False):
    """FIXED Main training loop for VAE with better stability"""
    
    # Check data health first
    print("Checking training data...")
    check_data_health(train_loader, device, max_batches=3)
    print("Checking validation data...")
    check_data_health(val_loader, device, max_batches=2)
    
    # FIXED: More conservative optimizer settings
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-5, 
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=15, min_lr=1e-7, verbose=True
    )
    
    # FIXED: Gradient scaling for mixed precision (optional)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
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
    
    # FIXED: Beta annealing schedule
    initial_beta = args.beta * 0.01  # Start with 1% of target beta
    beta_increment = (args.beta - initial_beta) / (args.epochs * 0.5)  # Reach target at 50% of epochs
    
    print("Starting VAE training with stability fixes...")
    print(f"Model: {'Conditional' if conditional else 'Standard'} VAE")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Initial beta: {initial_beta:.6f}, Target beta: {args.beta:.6f}")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()  # FIXED: Moved inside the loop
        
        # FIXED: Gradual beta annealing
        current_beta = min(initial_beta + epoch * beta_increment, args.beta)
        model.beta = current_beta
        
        
        # Training phase
        model.train()
        train_total_loss = 0
        train_recon_loss = 0  
        train_kl_loss = 0
        num_batches = 0
        nan_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if conditional:
                data, labels = batch
                data, labels = data.to(device), labels.to(device)
            else:
                data, _ = batch
                data = data.to(device)
            
            # FIXED: Check for NaN in input data
            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f"Invalid values in input batch {batch_idx}, skipping...")
                nan_batches += 1
                continue
            
            optimizer.zero_grad()
            
            # FIXED: Use automatic mixed precision if available
            try:
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        if conditional:
                            recon_x, mu, logvar, z = model(data, labels)
                        else:
                            recon_x, mu, logvar, z = model(data)
                        total_loss, recon_loss, kl_loss = model.loss_function(recon_x, data, mu, logvar)
                    
                    # FIXED: Check for NaN before backward pass
                    if torch.isnan(total_loss) or torch.isnan(recon_loss) or torch.isnan(kl_loss):
                        print(f"NaN loss at batch {batch_idx}, skipping...")
                        nan_batches += 1
                        continue
                    
                    scaler.scale(total_loss).backward()
                    
                    # FIXED: Check gradients before clipping
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    
                    # Skip if gradients are too large
                    if grad_norm > 10.0:
                        print(f"Large gradient norm {grad_norm:.2f} at batch {batch_idx}, skipping...")
                        optimizer.zero_grad()
                        continue
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                else:
                    # Regular training without mixed precision
                    if conditional:
                        recon_x, mu, logvar, z = model(data, labels)
                    else:
                        recon_x, mu, logvar, z = model(data)
                    
                    total_loss, recon_loss, kl_loss = model.loss_function(recon_x, data, mu, logvar)
                    
                    # FIXED: Check for NaN before backward pass
                    if torch.isnan(total_loss) or torch.isnan(recon_loss) or torch.isnan(kl_loss):
                        print(f"NaN loss at batch {batch_idx}, skipping...")
                        nan_batches += 1
                        continue
                    
                    total_loss.backward()
                    
                    # FIXED: Check gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    
                    if grad_norm > 10.0:
                        print(f"Large gradient norm {grad_norm:.2f} at batch {batch_idx}, skipping...")
                        optimizer.zero_grad()
                        continue
                    
                    optimizer.step()
                
                # Track losses
                train_total_loss += total_loss.item()
                train_recon_loss += recon_loss.item() 
                train_kl_loss += kl_loss.item()
                num_batches += 1
                
            except RuntimeError as e:
                print(f"RuntimeError at batch {batch_idx}: {e}")
                optimizer.zero_grad()
                nan_batches += 1
                continue
            
            # FIXED: More frequent progress updates
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {total_loss.item():.6f} | "
                      f"Recon: {recon_loss.item():.6f} | "
                      f"KL: {kl_loss.item():.6f} | "
                      f"Beta: {current_beta:.6f} | "
                      f"Grad: {grad_norm:.3f}")
        
        if num_batches == 0:
            print(f"No valid batches in epoch {epoch+1}, stopping training")
            break
            
        if nan_batches > len(train_loader) * 0.5:
            print(f"Too many invalid batches ({nan_batches}/{len(train_loader)}), stopping training")
            break
        
        # Average training losses
        train_total_loss /= num_batches
        train_recon_loss /= num_batches
        train_kl_loss /= num_batches
        
        # Validation phase
        model.eval()
        val_total_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        val_batches = 0
        val_nan_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    if conditional:
                        data, labels = batch
                        data, labels = data.to(device), labels.to(device)
                    else:
                        data, _ = batch
                        data = data.to(device)
                    
                    if torch.isnan(data).any() or torch.isinf(data).any():
                        val_nan_batches += 1
                        continue
                    
                    # Forward pass
                    if conditional:
                        recon_x, mu, logvar, z = model(data, labels)
                    else:
                        recon_x, mu, logvar, z = model(data)
                    
                    # Compute loss
                    total_loss, recon_loss, kl_loss = model.loss_function(recon_x, data, mu, logvar)
                    
                    if torch.isnan(total_loss) or torch.isnan(recon_loss) or torch.isnan(kl_loss):
                        val_nan_batches += 1
                        continue
                    
                    val_total_loss += total_loss.item()
                    val_recon_loss += recon_loss.item()
                    val_kl_loss += kl_loss.item()
                    val_batches += 1
                    
                except RuntimeError as e:
                    print(f"Validation RuntimeError: {e}")
                    val_nan_batches += 1
                    continue
        
        if val_batches == 0:
            print(f"No valid validation batches in epoch {epoch+1}")
            val_total_loss = val_recon_loss = val_kl_loss = float('nan')
        else:
            # Average validation losses
            val_total_loss /= val_batches
            val_recon_loss /= val_batches  
            val_kl_loss /= val_batches
        
        # Update learning rate
        if not np.isnan(val_total_loss):
            scheduler.step(val_total_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # FIXED: Time tracking
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Time: {epoch_time:.2f}s | Total: {total_time//60:.0f}m {total_time%60:.0f}s")
        print(f"LR: {current_lr:.2e} | Beta: {current_beta:.6f}")
        print(f"Train - Total: {train_total_loss:.6f} | Recon: {train_recon_loss:.6f} | KL: {train_kl_loss:.6f}")
        print(f"Val   - Total: {val_total_loss:.6f} | Recon: {val_recon_loss:.6f} | KL: {val_kl_loss:.6f}")
        print(f"Invalid batches - Train: {nan_batches}, Val: {val_nan_batches}")
        
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
                save_sample_outputs(model, device, output_dir, epoch+1, conditional=conditional)
                plot_reconstruction(model, val_loader, device, output_dir, epoch+1)
            except Exception as e:
                print(f"Error saving visualizations: {e}")
        
        # Early stopping and model saving
        if not np.isnan(val_total_loss) and val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total_loss,
                'best_val_loss': best_val_loss,
                'model_config': {
                    'input_shape': model.input_shape,
                    'latent_dim': model.latent_dim,
                    'beta': args.beta,
                    'conditional': conditional,
                    'num_classes': getattr(model, 'num_classes', None)
                }
            }, os.path.join(output_dir, 'best_vae_model.pth'))
            
            print(f"Saved best model at epoch {epoch+1} with val loss: {val_total_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1} after {args.patience} epochs without improvement")
                print(f"Best validation loss: {best_val_loss:.6f}")
                break
        
        # Stop if learning rate is too small
        if current_lr < 1e-7:
            print(f"Learning rate too small ({current_lr:.2e}), stopping training")
            break
    
    # Save final model
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_total_loss if not np.isnan(val_total_loss) else float('inf'),
        'model_config': {
            'input_shape': model.input_shape,
            'latent_dim': model.latent_dim,
            'beta': args.beta,
            'conditional': conditional,
            'num_classes': getattr(model, 'num_classes', None)
        }
    }, os.path.join(output_dir, 'final_vae_model.pth'))
    
    print(f"\nTraining completed in {total_time//60:.0f}m {total_time%60:.0f}s")
    print(f"Best validation loss: {best_val_loss:.6f}")


def main():
    # Setup device
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Arguments with FIXED defaults
    parser = argparse.ArgumentParser(description='Train Spectrogram VAE for Chick Call Generation')
    parser.add_argument('--data_dir', required=True, help="Path to spectrogram directory")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training (reduced)")
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate (reduced)")
    parser.add_argument('--latent_dim', type=int, default=64, help="Latent dimension (reduced)")
    parser.add_argument('--beta', type=float, default=0.1, help="Beta parameter for β-VAE (reduced)")
    parser.add_argument('--conditional', action='store_true', help="Use conditional VAE")
    parser.add_argument('--output_dir', default='vae_results', help="Directory to save outputs")
    parser.add_argument('--patience', type=int, default=25, help="Patience for early stopping")
    parser.add_argument('--augment', action='store_true', default=False, help="Apply data augmentation")
    args = parser.parse_args()

    # FIXED: Force conservative hyperparameters if they're too aggressive
    if args.lr > 1e-4:
        print(f"Reducing learning rate from {args.lr} to 1e-4 for stability")
        args.lr = 1e-4
    
    if args.beta > 0.1:
        print(f"Reducing beta from {args.beta} to 0.1 for stability")
        args.beta = 0.1
    
    if args.batch_size > 16:
        print(f"Reducing batch size from {args.batch_size} to 16 for stability")
        args.batch_size = 16

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"vae_experiment_{timestamp}")
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
    
    # Create data loaders with FIXED settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,  # Reduced from 4
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=2,  # Reduced from 4
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False,
        persistent_workers=True
    )
    
    # Initialize model
    print(f"Initializing {'Conditional' if args.conditional else 'Standard'} VAE...")
    
    try:
        if args.conditional and num_classes > 0:
            model = ConditionalSpectrogramVAE(
                input_shape=spectrogram_shape,
                num_classes=num_classes,
                latent_dim=args.latent_dim,
                beta=args.beta
            ).to(device)
        else:
            model = SpectrogramVAE(
                input_shape=spectrogram_shape,
                latent_dim=args.latent_dim,
                beta=args.beta
            ).to(device)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # FIXED: Initialize weights properly
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight, gain=0.1)  # Small gain for stability
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    print("Applied Xavier initialization with small gain")
    
    # Save model architecture and arguments
    with open(os.path.join(output_dir, 'model_config.txt'), 'w') as f:
        f.write(f"Model Configuration\n")
        f.write(f"==================\n")
        f.write(f"Model Type: {'Conditional' if args.conditional else 'Standard'} VAE\n")
        f.write(f"Input Shape: {spectrogram_shape}\n")
        f.write(f"Latent Dimension: {args.latent_dim}\n")
        f.write(f"Beta: {args.beta}\n")
        f.write(f"Number of Classes: {num_classes}\n")
        f.write(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"\nTraining Arguments:\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Data Augmentation: {args.augment}\n")
        f.write(f"Early Stopping Patience: {args.patience}\n")
        
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
        plot_reconstruction(model, val_loader, device, output_dir, "final")
    except Exception as e:
        print(f"Error generating final samples: {e}")
    
    print(f"Training complete! Results saved in: {output_dir}")


if __name__ == "__main__":
    main()