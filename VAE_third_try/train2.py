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

# Import your modules - UPDATED IMPORTS
from model import VariationalAutoEncoder, ConditionalVariationalAutoEncoder
from data_loading2 import create_vae_datasets, encode_labels, load_file_paths
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

def save_checkpoint_enhanced(model, optimizer, scheduler, epoch, val_loss, best_val_loss, 
                           args, global_step, output_dir, is_best=False):
    """Enhanced checkpoint saving with all training state"""
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'args': vars(args),
        'model_config': {
            'input_shape': model.input_shape,
            'latent_dim': model.latent_dim,
            'beta': model.beta,
            'conditional': hasattr(model, 'num_classes'),
            'num_classes': getattr(model, 'num_classes', None)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model separately
    if is_best:
        best_path = os.path.join(output_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"✅ Saved best model at epoch {epoch} with val loss: {val_loss:.4f}")
    
    # Save latest checkpoint (for easy resuming)
    latest_path = os.path.join(output_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    print(f"💾 Checkpoint saved: epoch {epoch}")
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load checkpoint and return training state"""
    print(f"🔄 Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Get training state
    start_epoch = checkpoint['epoch']
    global_step = checkpoint.get('global_step', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    print(f"✅ Resumed from epoch {start_epoch}")
    print(f"   Global step: {global_step}")
    print(f"   Best val loss: {best_val_loss:.4f}")
    
    return start_epoch, global_step, best_val_loss


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in output directory"""
    checkpoint_pattern = os.path.join(output_dir, 'checkpoint_epoch_*.pth')
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        latest_path = os.path.join(output_dir, 'latest_checkpoint.pth')
        if os.path.exists(latest_path):
            return latest_path
        return None
    
    # Sort by epoch number
    def extract_epoch(path):
        try:
            return int(path.split('_epoch_')[1].split('.pth')[0])
        except:
            return 0
    
    latest = max(checkpoints, key=extract_epoch)
    return latest


def train_vae_with_checkpoints(model, train_loader, val_loader, device, args, output_dir, conditional=False):
    """Enhanced training loop with checkpointing and resume functionality"""
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    ) 

    # Learning rate scheduling with warmup
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training state
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    # Resume from checkpoint if specified or auto-detect
    if args.resume:
        if args.resume == 'auto':
            # Auto-find latest checkpoint
            checkpoint_path = find_latest_checkpoint(output_dir)
            if checkpoint_path:
                start_epoch, global_step, best_val_loss = load_checkpoint(
                    checkpoint_path, model, optimizer, scheduler, device
                )
            else:
                print("🔍 No checkpoint found, starting fresh")
        else:
            # Use specified checkpoint
            if os.path.exists(args.resume):
                start_epoch, global_step, best_val_loss = load_checkpoint(
                    args.resume, model, optimizer, scheduler, device
                )
            else:
                print(f"❌ Checkpoint not found: {args.resume}")
                return
    
    # Training logs
    log_file = os.path.join(output_dir, "vae_training_log.csv")
    latent_stats_file = os.path.join(output_dir, "latent_stats.csv")
    
    # Initialize logs (append mode if resuming)
    log_mode = 'a' if start_epoch > 0 else 'w'
    if log_mode == 'w':
        with open(log_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'batch', 'train_loss', 'train_recon_loss', 'train_kl_loss',
                'val_loss', 'val_recon_loss', 'val_kl_loss', 'lr', 'beta', 
                'grad_norm', 'time_elapsed'
            ])
        
        with open(latent_stats_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'batch', 'mu_mean', 'mu_std', 'logvar_mean',
                'logvar_std', 'actual_var'
            ])
    
    start_time = time.time()
    
    print(f"\n🚀 {'Resuming' if start_epoch > 0 else 'Starting'} training:")
    print(f"📊 Epochs: {start_epoch + 1} to {args.epochs}")
    print(f"🎯 Target beta: {args.beta}")
    print(f"📈 Best val loss so far: {best_val_loss:.4f}")
    print(f"💾 Checkpoints every {args.checkpoint_freq} epochs")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_losses = {'total': 0, 'recon': 0, 'kl': 0}
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
                
                # Beta warmup
                current_beta = min(args.beta * (global_step / warmup_steps), args.beta)
                
                optimizer.zero_grad()
                
                # Forward pass
                if conditional:
                    recon_x, mu, logvar = model(data, labels)
                else:
                    recon_x, mu, logvar = model(data)
                
                # Compute loss
                total_loss, recon_loss, kl_loss = model.loss_function(
                    recon_x, data, mu, logvar, beta=current_beta
                )
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip
                )
                
                optimizer.step()
                scheduler.step()
                
                # Accumulate losses
                train_losses['total'] += total_loss.item()
                train_losses['recon'] += recon_loss.item()
                train_losses['kl'] += kl_loss.item()
                train_batches += 1
                
                # Detailed logging
                if batch_idx % 50 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    elapsed = time.time() - start_time
                    
                    print(f"📊 Epoch {epoch+1} Batch {batch_idx}:")
                    print(f"   LR: {current_lr:.2e} | β: {current_beta:.4f}")
                    print(f"   Loss: {total_loss.item():.4f} (R: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f})")
                    print(f"   Grad: {grad_norm:.3f} | Step: {global_step}")
                    
                    # Log latent stats
                    if batch_idx % 100 == 0:
                        mu_mean = mu.mean().item()
                        mu_std = mu.std().item()
                        logvar_mean = logvar.mean().item()
                        logvar_std = logvar.std().item()
                        actual_var = torch.exp(logvar).mean().item()
                        
                        with open(latent_stats_file, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                epoch+1, batch_idx, mu_mean, mu_std,
                                logvar_mean, logvar_std, actual_var
                            ])
                
            except Exception as e:
                print(f"❌ Training batch error: {e}")
                continue
        
        # Validation phase
        model.eval()
        val_losses = {'total': 0, 'recon': 0, 'kl': 0}
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
                    
                    total_loss, recon_loss, kl_loss = model.loss_function(
                        recon_x, data, mu, logvar, beta=args.beta
                    )
                    
                    val_losses['total'] += total_loss.item()
                    val_losses['recon'] += recon_loss.item()
                    val_losses['kl'] += kl_loss.item()
                    val_batches += 1
                    
                except Exception as e:
                    print(f"❌ Validation batch error: {e}")
                    continue
        
        # Calculate averages
        for key in train_losses:
            train_losses[key] /= max(train_batches, 1)
            val_losses[key] /= max(val_batches, 1)
        
        # Time tracking
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Epoch summary
        print(f"\n{'='*60}")
        print(f"📈 EPOCH {epoch+1}/{args.epochs} SUMMARY")
        print(f"{'='*60}")
        print(f"⏱️  Time: {epoch_time:.1f}s | Total: {total_time//60:.0f}m {total_time%60:.0f}s")
        print(f"📊 Train Loss: {train_losses['total']:.4f} (R: {train_losses['recon']:.4f}, KL: {train_losses['kl']:.4f})")
        print(f"📊 Val Loss:   {val_losses['total']:.4f} (R: {val_losses['recon']:.4f}, KL: {val_losses['kl']:.4f})")
        print(f"⚙️  LR: {current_lr:.2e} | β: {current_beta:.4f}")
        
        # Log to CSV
        with open(log_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1, len(train_loader), train_losses['total'], train_losses['recon'], train_losses['kl'],
                val_losses['total'], val_losses['recon'], val_losses['kl'], current_lr, current_beta,
                float('nan'), total_time
            ])
        
        # Save checkpoints
        is_best = val_losses['total'] < best_val_loss
        if is_best:
            best_val_loss = val_losses['total']
        
        # Save checkpoint based on frequency
        if (epoch + 1) % args.checkpoint_freq == 0 or is_best or epoch == args.epochs - 1:
            save_checkpoint_enhanced(
                model, optimizer, scheduler, epoch + 1, val_losses['total'], 
                best_val_loss, args, global_step, output_dir, is_best
            )
        
        # Generate samples periodically
        if (epoch + 1) % args.sample_freq == 0 or epoch == 0:
            try:
                save_sample_outputs(model, device, output_dir, epoch+1, 
                                  conditional=conditional, 
                                  num_classes=getattr(model, 'num_classes', None))
                plot_reconstruction(model, val_loader, device, output_dir, epoch+1, 
                                  conditional=conditional)
                print("🎨 Generated sample visualizations")
            except Exception as e:
                print(f"❌ Error generating samples: {e}")
        
        # Early stopping check
        patience_epochs = getattr(args, 'patience', 50)
        if epoch - (best_val_loss_epoch if 'best_val_loss_epoch' in locals() else 0) > patience_epochs:
            print(f"🛑 Early stopping: no improvement for {patience_epochs} epochs")
            break
        
        if is_best:
            best_val_loss_epoch = epoch
    
    print(f"\n🎉 Training completed!")
    print(f"⏱️  Total time: {total_time//60:.0f}m {total_time%60:.0f}s")
    print(f"🏆 Best validation loss: {best_val_loss:.4f}")
    print(f"💾 Final checkpoint saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced VAE Training with Checkpointing')
    parser.add_argument('--data_dir', required=True, help="Path to spectrogram directory")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=150, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--latent_dim', type=int, default=1024, help="Latent dimension")
    parser.add_argument('--beta', type=float, default=0.005, help="Beta parameter for β-VAE")
    parser.add_argument('--conditional', action='store_true', help="Use conditional VAE")
    parser.add_argument('--embed_dim', type=int, default=100, help="Label embedding dimension")
    parser.add_argument('--output_dir', default='enhanced_vae_results', help="Directory to save outputs")
    parser.add_argument('--augment', action='store_true', help="Apply data augmentation")
    parser.add_argument('--grad_clip', type=float, default=1.0, help="Gradient clipping max norm")
    
    # Checkpointing arguments
    parser.add_argument('--resume', type=str, default=None, 
                        help="Resume from checkpoint (path or 'auto' for latest)")
    parser.add_argument('--checkpoint_freq', type=int, default=10, 
                        help="Save checkpoint every N epochs")
    parser.add_argument('--sample_freq', type=int, default=10, 
                        help="Generate samples every N epochs")
    
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume and args.resume != 'auto':
        # Use existing directory if resuming from specific checkpoint
        output_dir = os.path.dirname(args.resume)
    else:
        output_dir = os.path.join(args.output_dir, f"vae_experiment_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Data loading
    print("📊 Loading data...")
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
            print(f"🏷️ Found {len(label_encoder.classes_)} classes: {label_encoder.classes_}")
        else:
            print("⚠️ No labels found, switching to standard VAE")
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
        print(f"❌ Error creating datasets: {e}")
        return
    
    print(f"📈 Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"🎼 Spectrogram shape: {spectrogram_shape}")
    
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
    print(f"🧠 Initializing {'Conditional' if args.conditional else 'Standard'} VAE...")
    
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
        print(f"❌ Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Start training
    try:
        train_vae_with_checkpoints(model, train_loader, val_loader, device, args, output_dir, args.conditional)
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"✅ Training complete! Results saved in: {output_dir}")


if __name__ == "__main__":
    main()