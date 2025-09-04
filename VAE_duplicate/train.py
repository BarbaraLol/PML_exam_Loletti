import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
import csv
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

from model import VariationalAutoEncoder, ConditionalVariationalAutoEncoder
from data_loading import create_vae_datasets, encode_labels, load_file_paths
from train_utils import save_checkpoint, calculate_conditional_vae_accuracy, validate_training_args, create_training_summary


def save_sample_outputs(model, device, output_dir, epoch, num_samples=8, 
                       conditional=False, num_classes=None):
    """Save sample generated spectrograms during training"""
    model.eval()
    with torch.no_grad():
        if conditional and num_classes:
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

        actual_num_samples = min(samples_np.shape[0], len(axes))
        
        for i in range(actual_num_samples):
            ax = axes[i]
            spec_vis = samples_np[i, 0] if samples_np[i].ndim == 3 else samples_np[i]
            ax.imshow(spec_vis, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f'Generated Sample {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'generated_samples_epoch_{epoch}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()


def train_vae(model, train_loader, val_loader, device, args, output_dir, conditional=False):
    """Fixed training loop with stability improvements"""
    
    # FIXED: Much lower learning rate for stability
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    ) 

    # FIXED: Conservative learning rate scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.8, 
        patience=5, 
        min_lr=1e-7,
        verbose=True
    )

    # Training log setup
    if conditional:
        log_file = os.path.join(output_dir, "fixed_conditional_vae_log.csv")
        with open(log_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_recon_loss', 'train_kl_loss', 'train_acc',
                'val_loss', 'val_recon_loss', 'val_kl_loss', 'val_acc', 'lr', 'beta', 
                'mu_mean', 'mu_std', 'logvar_mean', 'logvar_std', 'time_elapsed'
            ])
    else:
        log_file = os.path.join(output_dir, "fixed_vae_log.csv")
        with open(log_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_recon_loss', 'train_kl_loss',
                'val_loss', 'val_recon_loss', 'val_kl_loss', 'lr', 'beta', 
                'mu_mean', 'mu_std', 'logvar_mean', 'logvar_std', 'time_elapsed'
            ])
    
    # Training state
    best_val_loss = float('inf')
    start_time = time.time()
    consecutive_bad_epochs = 0
    
    print("Starting FIXED VAE training with:")
    print(f"- Learning rate: {args.lr}")  
    print(f"- Beta: {args.beta}")
    print(f"- Gradient clipping: {args.grad_clip}")
    print(f"- Latent dimension: {model.latent_dim}")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # FIXED: Use constant small beta throughout training
        current_beta = args.beta
        
        # Training phase
        model.train()
        train_total_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        train_accuracy = 0
        train_batches = 0
        
        # Track latent statistics
        mu_values = []
        logvar_values = []
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Prepare batch
                if conditional:
                    data, labels = batch
                    data, labels = data.to(device), labels.to(device)
                else:
                    data = batch.to(device)
                
                # FIXED: Check for NaN/Inf in input
                if torch.isnan(data).any() or torch.isinf(data).any():
                    print(f"Warning: NaN/Inf in input data at batch {batch_idx}")
                    continue
                
                optimizer.zero_grad()
                
                # Forward pass
                if conditional:
                    recon_x, mu, logvar = model(data, labels)
                else:
                    recon_x, mu, logvar = model(data)
                
                # FIXED: Check for NaN/Inf in outputs
                if (torch.isnan(recon_x).any() or torch.isinf(recon_x).any() or
                    torch.isnan(mu).any() or torch.isinf(mu).any() or
                    torch.isnan(logvar).any() or torch.isinf(logvar).any()):
                    print(f"Warning: NaN/Inf in model outputs at batch {batch_idx}")
                    continue
                
                # Compute loss
                total_loss, recon_loss, kl_loss = model.loss_function(
                    recon_x, data, mu, logvar, beta=current_beta
                )
                
                # FIXED: Check loss for NaN/Inf
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"Warning: NaN/Inf in loss at batch {batch_idx}")
                    continue
                
                # Calculate accuracy for conditional VAE
                if conditional:
                    batch_accuracy = calculate_conditional_vae_accuracy(model, data, labels, device)
                    train_accuracy += batch_accuracy
                
                # Backward pass
                total_loss.backward()
                
                # FIXED: Aggressive gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=args.grad_clip
                )
                
                # FIXED: Check gradients for NaN/Inf
                valid_gradients = True
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"Warning: NaN/Inf gradients in {name}")
                            valid_gradients = False
                            break
                
                if not valid_gradients:
                    optimizer.zero_grad()
                    continue
                
                optimizer.step()
                
                # Accumulate losses
                train_total_loss += total_loss.item()
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()
                train_batches += 1
                
                # Track latent statistics
                mu_values.append(mu.detach().cpu())
                logvar_values.append(logvar.detach().cpu())
                
                # Print progress
                if batch_idx % 100 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch+1} Batch {batch_idx}:")
                    print(f"  LR: {current_lr:.2e} | β: {current_beta:.6f}")
                    print(f"  Loss: {total_loss.item():.4f} | Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f}")
                    print(f"  Grad Norm: {grad_norm:.4f}")
                    if conditional:
                        print(f"  Accuracy: {batch_accuracy:.4f}")
                
            except Exception as e:
                print(f"Training batch error: {e}")
                continue
        
        # Calculate latent statistics
        if mu_values and logvar_values:
            mu_tensor = torch.cat(mu_values, dim=0)
            logvar_tensor = torch.cat(logvar_values, dim=0)
            mu_mean = mu_tensor.mean().item()
            mu_std = mu_tensor.std().item()
            logvar_mean = logvar_tensor.mean().item()
            logvar_std = logvar_tensor.std().item()
        else:
            mu_mean = mu_std = logvar_mean = logvar_std = 0.0
        
        # Validation phase
        model.eval()
        val_total_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        val_accuracy = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    if conditional:
                        data, labels = batch
                        data, labels = data.to(device), labels.to(device)
                        recon_x, mu, logvar = model(data, labels)
                        
                        batch_val_accuracy = calculate_conditional_vae_accuracy(model, data, labels, device)
                        val_accuracy += batch_val_accuracy
                    else:
                        data = batch.to(device)
                        recon_x, mu, logvar = model(data)
                    
                    total_loss, recon_loss, kl_loss = model.loss_function(
                        recon_x, data, mu, logvar, beta=current_beta
                    )
                    
                    # Check for valid losses
                    if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                        val_total_loss += total_loss.item()
                        val_recon_loss += recon_loss.item()
                        val_kl_loss += kl_loss.item()
                        val_batches += 1
                    
                except Exception as e:
                    print(f"Validation batch error: {e}")
                    continue
        
        # Calculate averages
        if train_batches > 0:
            train_total_loss /= train_batches
            train_recon_loss /= train_batches
            train_kl_loss /= train_batches
            if conditional:
                train_accuracy /= train_batches
        
        if val_batches > 0:
            val_total_loss /= val_batches
            val_recon_loss /= val_batches
            val_kl_loss /= val_batches
            if conditional:
                val_accuracy /= val_batches
        
        # Time tracking
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate
        scheduler.step(val_total_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Time: {epoch_time:.2f}s | Total: {total_time//60:.0f}m {total_time%60:.0f}s")
        print(f"LR: {current_lr:.2e} | β: {current_beta:.6f}")
        print(f"Train - Total: {train_total_loss:.4f} | Recon: {train_recon_loss:.4f} | KL: {train_kl_loss:.4f}")
        print(f"Val   - Total: {val_total_loss:.4f} | Recon: {val_recon_loss:.4f} | KL: {val_kl_loss:.4f}")
        print(f"Latent - μ: {mu_mean:.4f}±{mu_std:.4f} | logvar: {logvar_mean:.4f}±{logvar_std:.4f}")
        if conditional:
            print(f"Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}")
        
        # Save to CSV
        with open(log_file, 'a') as f:
            writer = csv.writer(f)
            if conditional:
                writer.writerow([
                    epoch+1, train_total_loss, train_recon_loss, train_kl_loss, train_accuracy,
                    val_total_loss, val_recon_loss, val_kl_loss, val_accuracy, current_lr, current_beta,
                    mu_mean, mu_std, logvar_mean, logvar_std, total_time
                ])
            else:
                writer.writerow([
                    epoch+1, train_total_loss, train_recon_loss, train_kl_loss,
                    val_total_loss, val_recon_loss, val_kl_loss, current_lr, current_beta,
                    mu_mean, mu_std, logvar_mean, logvar_std, total_time
                ])
        
        # Save checkpoints
        if val_total_loss < best_val_loss and val_batches > 0:
            best_val_loss = val_total_loss
            consecutive_bad_epochs = 0
            checkpoint_data = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total_loss,
                'args': vars(args)
            }
            if conditional:
                checkpoint_data['val_accuracy'] = val_accuracy
            
            torch.save(checkpoint_data, os.path.join(output_dir, 'best_fixed_model.pth'))
            print(f"  → New best model saved! (Val Loss: {val_total_loss:.4f})")
        else:
            consecutive_bad_epochs += 1
        
        # Save samples periodically
        if (epoch + 1) % 10 == 0 or epoch == 0:
            try:
                save_sample_outputs(model, device, output_dir, epoch+1, 
                                conditional=conditional, 
                                num_classes=getattr(model, 'num_classes', None))
            except Exception as e:
                print(f"Error saving samples: {e}")
        
        # Early stopping checks
        if val_total_loss > best_val_loss * 3 and epoch > 10:
            print("Early stopping due to severely diverging validation loss")
            break
            
        if consecutive_bad_epochs >= 15:
            print("Early stopping due to no improvement for 15 epochs")
            break
            
        # Check for training failure
        if train_kl_loss > 1000 or val_kl_loss > 1000:
            print("Training failed: KL loss explosion detected")
            break
    
    print(f"\nTraining completed in {total_time//60:.0f}m {total_time%60:.0f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Create training summary
    create_training_summary(output_dir, log_file)


def main():
    parser = argparse.ArgumentParser(description='Train FIXED Spectrogram VAE')
    parser.add_argument('--data_dir', required=True, help="Path to spectrogram directory")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training")  
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")  # FIXED: Much lower
    parser.add_argument('--latent_dim', type=int, default=128, help="Latent dimension")  # FIXED: Smaller
    parser.add_argument('--beta', type=float, default=1e-4, help="Beta parameter for β-VAE")  # FIXED: Much smaller
    parser.add_argument('--conditional', action='store_true', help="Use conditional VAE")
    parser.add_argument('--embed_dim', type=int, default=32, help="Label embedding dimension")  # FIXED: Smaller
    parser.add_argument('--output_dir', default='fixed_vae_results', help="Directory to save outputs")
    parser.add_argument('--grad_clip', type=float, default=0.5, help="Gradient clipping max norm")  # FIXED: Smaller
    args = parser.parse_args()

    # Validate arguments
    warnings = validate_training_args(args)
    if warnings:
        print("TRAINING ARGUMENT WARNINGS:")
        for warning in warnings:
            print(f"  ⚠ {warning}")
        print()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"fixed_vae_experiment_{timestamp}")
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
            augment=False  # FIXED: Disable augmentation for stability
        )
    except Exception as e:
        print(f"Error creating datasets: {e}")
        return
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Spectrogram shape: {spectrogram_shape}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False
    )
    
    # Initialize model
    print(f"Initializing FIXED {'Conditional' if args.conditional else 'Standard'} VAE...")
    
    try:
        if args.conditional and num_classes > 0:
            model = ConditionalVariationalAutoEncoder(
                input_shape=spectrogram_shape,
                latent_dim=args.latent_dim,
                num_classes=num_classes,
                embed_dim=args.embed_dim
            ).to(device)
        else:
            model = VariationalAutoEncoder(
                input_shape=spectrogram_shape,
                latent_dim=args.latent_dim,
                beta=args.beta
            ).to(device)
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save model configuration
    with open(os.path.join(output_dir, 'fixed_model_config.txt'), 'w') as f:
        f.write(f"FIXED VAE Configuration\n")
        f.write(f"=======================\n")
        f.write(f"Model Type: {'Conditional' if args.conditional else 'Standard'} VAE\n")
        f.write(f"Input Shape: {spectrogram_shape}\n")
        f.write(f"Latent Dimension: {args.latent_dim}\n")
        f.write(f"Beta: {args.beta}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Gradient Clipping: {args.grad_clip}\n")
        f.write(f"Number of Classes: {num_classes}\n")
        if args.conditional:
            f.write(f"Embedding Dimension: {args.embed_dim}\n")
        f.write(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        
        if label_encoder:
            f.write(f"\nClass Labels: {list(label_encoder.classes_)}\n")
        
        f.write(f"\nFIXES APPLIED:\n")
        f.write(f"- Reduced architecture complexity (3 conv layers)\n")
        f.write(f"- Lower learning rate ({args.lr})\n")
        f.write(f"- Smaller latent dimension ({args.latent_dim})\n")
        f.write(f"- Much smaller beta ({args.beta})\n")
        f.write(f"- Aggressive gradient clipping ({args.grad_clip})\n")
        f.write(f"- Logvar initialization and clamping\n")
        f.write(f"- KL loss clamping\n")
        f.write(f"- NaN/Inf detection and handling\n")
        f.write(f"- Early stopping conditions\n")
    
    # Start training
    try:
        train_vae(model, train_loader, val_loader, device, args, output_dir, args.conditional)
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"Training complete! Results saved in: {output_dir}")


if __name__ == "__main__":
    main()