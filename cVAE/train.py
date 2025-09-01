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

# from model import VariationalAutoEncoder, ConditionalVariationalAutoEncoder
from model import SimpleVariationalAutoEncoder, SimpleConditionalVAE
from data_loading import create_vae_datasets, encode_labels, load_file_paths
from train_utils import save_checkpoint, calculate_conditional_vae_accuracy

def get_beta(epoch, total_epochs, min_beta=0.0001, max_beta=0.001):
    """Simple linear increase - much more conservative"""
    progress = epoch / total_epochs
    return min_beta + (max_beta - min_beta) * progress


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

        actual_num_samples = min(samples_np.shape[0], len(axes))
        
        for i in range(actual_num_samples):
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

##################
# For simple VAE #
##################

# def train_vae(model, train_loader, val_loader, device, args, output_dir, conditional=False):
#     """Enhanced training loop with all requested features"""
    
#     # Setup optimizer with weight decay
#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr=args.lr,
#         weight_decay=1e-5,
#         betas=(0.9, 0.999)
#     ) 

#     # Learning rate scheduling with warmup
#     total_steps = args.epochs * len(train_loader)
#     warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
#     def lr_lambda(current_step):
#         if current_step < warmup_steps:
#             return float(current_step) / float(max(1, warmup_steps))
#         # Cosine decay after warmup
#         progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
#         return 0.5 * (1.0 + math.cos(math.pi * progress))
    
#     scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
#     # Training log with additional metrics
#     log_file = os.path.join(output_dir, "vae_training_log.csv")
#     latent_stats_file = os.path.join(output_dir, "latent_stats.csv")
    
#     # Initialize logs
#     with open(log_file, 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             'epoch', 'batch', 'train_loss', 'train_recon_loss', 'train_kl_loss',
#             'val_loss', 'val_recon_loss', 'val_kl_loss', 'lr', 'beta', 
#             'grad_norm', 'time_elapsed'
#         ])
    
#     with open(latent_stats_file, 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             'epoch', 'batch', 'mu_mean', 'mu_std', 'logvar_mean',
#             'logvar_std', 'actual_var'
#         ])
    
#     # Training state
#     best_val_loss = float('inf')
#     start_time = time.time()
#     global_step = 0
    
#     print("Starting training with:")
#     print(f"- LR warmup ({warmup_steps} steps)")
#     print(f"- Beta warmup (target β={args.beta})")
#     print(f"- Gradient clipping (max_norm={args.grad_clip})")
    
#     for epoch in range(args.epochs):
#         epoch_start = time.time()
        
#         # Training phase
#         model.train()
#         train_total_loss = 0
#         train_recon_loss = 0
#         train_kl_loss = 0
#         train_batches = 0
        
#         for batch_idx, batch in enumerate(train_loader):
#             global_step += 1
            
#             try:
#                 # Prepare batch
#                 if conditional:
#                     data, labels = batch
#                     data, labels = data.to(device), labels.to(device)
#                 else:
#                     data = batch.to(device)
                
#                 # Beta warmup (linear schedule)
#                 current_beta = min(args.beta * (global_step / warmup_steps), args.beta)
                
#                 optimizer.zero_grad()
                
#                 # Forward pass
#                 if conditional:
#                     recon_x, mu, logvar = model(data, labels)
#                 else:
#                     recon_x, mu, logvar = model(data)
                
#                 # Compute loss
#                 total_loss, recon_loss, kl_loss = model.loss_function(
#                     recon_x, data, mu, logvar, beta=current_beta
#                 )
                
#                 # Backward pass
#                 total_loss.backward()
                
#                 # Gradient clipping
#                 grad_norm = torch.nn.utils.clip_grad_norm_(
#                     model.parameters(), 
#                     max_norm=args.grad_clip
#                 )
                
#                 optimizer.step()
#                 scheduler.step()
                
#                 # Accumulate losses
#                 train_total_loss += total_loss.item()
#                 train_recon_loss += recon_loss.item()
#                 train_kl_loss += kl_loss.item()
#                 train_batches += 1
                
#                 # Log latent space statistics
#                 if batch_idx % 100 == 0:
#                     # Calculate latent stats
#                     mu_mean = mu.mean().item()
#                     mu_std = mu.std().item()
#                     logvar_mean = logvar.mean().item()
#                     logvar_std = logvar.std().item()
#                     actual_var = torch.exp(logvar).mean().item()
                    
#                     # Save to CSV
#                     with open(latent_stats_file, 'a') as f:
#                         writer = csv.writer(f)
#                         writer.writerow([
#                             epoch+1, batch_idx, mu_mean, mu_std,
#                             logvar_mean, logvar_std, actual_var
#                         ])
                    
#                     # Print summary
#                     current_lr = optimizer.param_groups[0]['lr']
#                     print(f"\nEpoch {epoch+1} Batch {batch_idx}:")
#                     print(f"LR: {current_lr:.2e} | β: {current_beta:.4f}")
#                     print(f"Train Loss: {total_loss.item():.4f}")
#                     print(f"  Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f}")
#                     print(f"Grad Norm: {grad_norm:.4f}")
#                     print(f"Latent μ: {mu_mean:.4f} ± {mu_std:.4f}")
#                     print(f"Latent σ²: {actual_var:.4f} (logvar: {logvar_mean:.4f})")
                
#             except Exception as e:
#                 print(f"Training batch error: {e}")
#                 continue
        
#         # Validation phase
#         model.eval()
#         val_total_loss = 0
#         val_recon_loss = 0
#         val_kl_loss = 0
#         val_batches = 0
        
#         with torch.no_grad():
#             for batch in val_loader:
#                 try:
#                     if conditional:
#                         data, labels = batch
#                         data, labels = data.to(device), labels.to(device)
#                         recon_x, mu, logvar = model(data, labels)
#                     else:
#                         data = batch.to(device)
#                         recon_x, mu, logvar = model(data)
                    
#                     # Use final beta for validation
#                     total_loss, recon_loss, kl_loss = model.loss_function(
#                         recon_x, data, mu, logvar, beta=args.beta
#                     )
                    
#                     val_total_loss += total_loss.item()
#                     val_recon_loss += recon_loss.item()
#                     val_kl_loss += kl_loss.item()
#                     val_batches += 1
                    
#                 except Exception as e:
#                     print(f"Validation batch error: {e}")
#                     continue
        
#         # Calculate averages
#         train_total_loss /= train_batches
#         train_recon_loss /= train_batches
#         train_kl_loss /= train_batches
        
#         val_total_loss /= val_batches
#         val_recon_loss /= val_batches
#         val_kl_loss /= val_batches
        
#         # Time tracking
#         epoch_time = time.time() - epoch_start
#         total_time = time.time() - start_time
#         current_lr = optimizer.param_groups[0]['lr']
        
#         # Save to log
#         with open(log_file, 'a') as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 epoch+1, batch_idx, train_total_loss, train_recon_loss, train_kl_loss,
#                 val_total_loss, val_recon_loss, val_kl_loss, current_lr, current_beta,
#                 grad_norm.item() if batch_idx % 100 == 0 else float('nan'), total_time
#             ])
        
#         # Print epoch summary
#         print(f"\nEpoch {epoch+1} Summary:")
#         print(f"Time: {epoch_time:.2f}s | Total: {total_time//60:.0f}m {total_time%60:.0f}s")
#         print(f"LR: {current_lr:.2e} | β: {current_beta:.4f}")
#         print(f"Train - Total: {train_total_loss:.4f} | Recon: {train_recon_loss:.4f} | KL: {train_kl_loss:.4f}")
#         print(f"Val   - Total: {val_total_loss:.4f} | Recon: {val_recon_loss:.4f} | KL: {val_kl_loss:.4f}")
        
#         # Save checkpoints
#         if val_total_loss < best_val_loss:
#             best_val_loss = val_total_loss
#             torch.save({
#                 'epoch': epoch+1,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_loss': val_total_loss,
#                 'args': vars(args)
#             }, os.path.join(output_dir, 'best_model.pth'))
        
#         # Save samples periodically
#         if (epoch + 1) % 10 == 0 or epoch == 0:
#             save_sample_outputs(model, device, output_dir, epoch+1, 
#                               conditional=conditional, 
#                               num_classes=getattr(model, 'num_classes', None))
    
#     print(f"\nTraining completed in {total_time//60:.0f}m {total_time%60:.0f}s")
#     print(f"Best validation loss: {best_val_loss:.4f}")

#######################
# For conditional VAE #
#######################

def train_vae(model, train_loader, val_loader, device, args, output_dir, conditional=False):
    """Enhanced training loop with proper beta scheduling"""
    
    # Setup optimizer with higher weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-3,  # Higher weight decay
        betas=(0.9, 0.999)
    ) 

    # Learning rate scheduling with shorter warmup
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)  # 5% warmup instead of 10%
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        # Cosine decay with minimum LR
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training log setup
    if conditional:
        log_file = os.path.join(output_dir, "conditional_vae_training_log.csv")
        with open(log_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'batch', 'train_loss', 'train_recon_loss', 'train_kl_loss', 'train_acc',
                'val_loss', 'val_recon_loss', 'val_kl_loss', 'val_acc', 'lr', 'beta', 
                'grad_norm', 'time_elapsed'
            ])
    else:
        log_file = os.path.join(output_dir, "vae_training_log.csv")
        with open(log_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'batch', 'train_loss', 'train_recon_loss', 'train_kl_loss',
                'val_loss', 'val_recon_loss', 'val_kl_loss', 'lr', 'beta', 
                'grad_norm', 'time_elapsed'
            ])
    
    latent_stats_file = os.path.join(output_dir, "latent_stats.csv")
    with open(latent_stats_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'batch', 'mu_mean', 'mu_std', 'logvar_mean',
            'logvar_std', 'actual_var'
        ])
    
    # Training state
    best_val_loss = float('inf')
    start_time = time.time()
    global_step = 0
    
    print("Starting training with:")
    print(f"- Model type: {'Conditional' if conditional else 'Standard'} VAE")
    print(f"- LR: {args.lr} with {warmup_steps} warmup steps")  
    print(f"- Beta scheduling: cyclical between 0.001-0.01")
    print(f"- Gradient clipping (max_norm={args.grad_clip})")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # current_beta = get_beta(epoch, args.epochs)
        current_beta = args.beta
        print(f"Epoch {epoch+1}: Using beta = {current_beta}")
        
        # Training phase
        model.train()
        train_total_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        train_accuracy = 0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            global_step += 1
            
            try:
                # Prepare batch
                if conditional:
                    data, labels = batch
                    data, labels = data.to(device), labels.to(device)
                else:
                    data = batch.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if conditional:
                    recon_x, mu, logvar = model(data, labels)
                else:
                    recon_x, mu, logvar = model(data)
                
                # *** KEY FIX: Use current_beta from scheduling, not warmup ***
                total_loss, recon_loss, kl_loss = model.loss_function(
                    recon_x, data, mu, logvar, beta=current_beta
                )
                
                # Calculate accuracy for conditional VAE
                if conditional:
                    batch_accuracy = calculate_conditional_vae_accuracy(model, data, labels, device)
                    train_accuracy += batch_accuracy
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=args.grad_clip
                )
                
                optimizer.step()
                scheduler.step()
                
                # Accumulate losses
                train_total_loss += total_loss.item()
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()
                train_batches += 1
                
                # Log latent space statistics
                if batch_idx % 100 == 0:
                    # Calculate latent stats
                    mu_mean = mu.mean().item()
                    mu_std = mu.std().item()
                    logvar_mean = logvar.mean().item()
                    logvar_std = logvar.std().item()
                    actual_var = torch.exp(logvar).mean().item()
                    
                    # Save to CSV
                    with open(latent_stats_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            epoch+1, batch_idx, mu_mean, mu_std,
                            logvar_mean, logvar_std, actual_var
                        ])
                    
                    # Print summary
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"\nEpoch {epoch+1} Batch {batch_idx}:")
                    print(f"LR: {current_lr:.2e} | β: {current_beta:.4f}")
                    print(f"Train Loss: {total_loss.item():.4f}")
                    print(f"  Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f}")
                    if conditional:
                        print(f"  Accuracy: {batch_accuracy:.4f}")
                    print(f"Grad Norm: {grad_norm:.4f}")
                    print(f"Latent μ: {mu_mean:.4f} ± {mu_std:.4f}")
                    print(f"Latent σ²: {actual_var:.4f} (logvar: {logvar_mean:.4f})")
                
            except Exception as e:
                print(f"Training batch error: {e}")
                continue
        
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
                    
                    # *** KEY FIX: Use current_beta for validation too ***
                    total_loss, recon_loss, kl_loss = model.loss_function(
                        recon_x, data, mu, logvar, beta=current_beta
                    )
                    
                    val_total_loss += total_loss.item()
                    val_recon_loss += recon_loss.item()
                    val_kl_loss += kl_loss.item()
                    val_batches += 1
                    
                except Exception as e:
                    print(f"Validation batch error: {e}")
                    continue
        
        # Calculate averages
        train_total_loss /= train_batches
        train_recon_loss /= train_batches
        train_kl_loss /= train_batches
        if conditional:
            train_accuracy /= train_batches
        
        val_total_loss /= val_batches
        val_recon_loss /= val_batches
        val_kl_loss /= val_batches
        if conditional:
            val_accuracy /= val_batches
        
        # Time tracking
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save to CSV
        with open(log_file, 'a') as f:
            writer = csv.writer(f)
            if conditional:
                writer.writerow([
                    epoch+1, batch_idx, train_total_loss, train_recon_loss, train_kl_loss, train_accuracy,
                    val_total_loss, val_recon_loss, val_kl_loss, val_accuracy, current_lr, current_beta,
                    grad_norm.item() if 'grad_norm' in locals() else float('nan'), total_time
                ])
            else:
                writer.writerow([
                    epoch+1, batch_idx, train_total_loss, train_recon_loss, train_kl_loss,
                    val_total_loss, val_recon_loss, val_kl_loss, current_lr, current_beta,
                    grad_norm.item() if 'grad_norm' in locals() else float('nan'), total_time
                ])
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Time: {epoch_time:.2f}s | Total: {total_time//60:.0f}m {total_time%60:.0f}s")
        print(f"LR: {current_lr:.2e} | β: {current_beta:.4f}")
        print(f"Train - Total: {train_total_loss:.4f} | Recon: {train_recon_loss:.4f} | KL: {train_kl_loss:.4f}")
        if conditional:
            print(f"      - Accuracy: {train_accuracy:.4f}")
        print(f"Val   - Total: {val_total_loss:.4f} | Recon: {val_recon_loss:.4f} | KL: {val_kl_loss:.4f}")
        if conditional:
            print(f"      - Accuracy: {val_accuracy:.4f}")
        
        # Save checkpoints
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            checkpoint_data = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total_loss,
                'args': vars(args)
            }
            if conditional:
                checkpoint_data['val_accuracy'] = val_accuracy
            
            torch.save(checkpoint_data, os.path.join(output_dir, 'best_model.pth'))
        
        # Save samples periodically
        if (epoch + 1) % 10 == 0 or epoch == 0:
            save_sample_outputs(model, device, output_dir, epoch+1, 
                            conditional=conditional, 
                            num_classes=getattr(model, 'num_classes', None))
    
    print(f"\nTraining completed in {total_time//60:.0f}m {total_time%60:.0f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Simple Spectrogram VAE')
    parser.add_argument('--data_dir', required=True, help="Path to spectrogram directory")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")  
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")  # Reduced
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")  # Reduced
    parser.add_argument('--latent_dim', type=int, default=256, help="Latent dimension")  # Reduced
    parser.add_argument('--beta', type=float, default=0.01, help="Beta parameter for β-VAE") 
    parser.add_argument('--conditional', action='store_true', help="Use conditional VAE")
    parser.add_argument('--embed_dim', type=int, default=128, help="Label embedding dimension")  # Reduced
    parser.add_argument('--output_dir', default='simple_vae_results', help="Directory to save outputs")
    parser.add_argument('--patience', type=int, default=15, help="Patience for early stopping")
    parser.add_argument('--augment', action='store_true', help="Apply data augmentation")
    parser.add_argument('--weight_decay', type=float, default=1e-3, help="Weight decay for optimizer")
    parser.add_argument('--grad_clip', type=float, default=1.0, help="Gradient clipping max norm")  # Increased
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
    
    # Setup label encoder for conditional VAE
    label_encoder = None
    if args.conditional and num_classes > 0:
        model = SimpleConditionalVAE(
            input_shape=spectrogram_shape,
            latent_dim=args.latent_dim,
            num_classes=num_classes,
            embed_dim=args.embed_dim
        ).to(device)
    else:
        model = SimpleVariationalAutoEncoder(
            input_shape=spectrogram_shape,
            latent_dim=args.latent_dim
        ).to(device)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Spectrogram shape: {spectrogram_shape}")
    
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty")
    
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
    
    print(f"Training complete! Results saved in: {output_dir}")


if __name__ == "__main__":
    main()