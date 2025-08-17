#!/usr/bin/env python3
"""
Enhanced VAE Training Script
Complete implementation with all improvements
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
import csv
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Updated imports to use the enhanced modules
# Make sure to save the artifacts as separate .py files:
# - model.py (use the ImprovedVariationalAutoEncoder from the first artifact)
# - data_loading.py (use the EnhancedSpectrogramDataset from the third artifact)
# - train_utils.py (your existing file)

try:
    from model import VariationalAutoEncoder, ConditionalVariationalAutoEncoder
    from data_loading import create_enhanced_vae_datasets, encode_labels, load_file_paths, inspect_enhanced_files
    from train_utils import save_checkpoint
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure to save the enhanced model.py and data_loading.py files from the artifacts")
    exit(1)


def get_device_info():
    """Get detailed device information"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = {"device": device, "name": "CPU"}
    
    if torch.cuda.is_available():
        info["name"] = torch.cuda.get_device_name()
        info["memory"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["cuda_version"] = torch.version.cuda
    
    return info


def create_training_config(args):
    """Create and save training configuration"""
    config = {
        "model_config": {
            "type": "ImprovedConditionalVAE" if args.conditional else "ImprovedVariationalAutoEncoder",
            "latent_dim": args.latent_dim,
            "beta": args.beta,
            "embed_dim": args.embed_dim if args.conditional else None,
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "warmup_epochs": args.warmup_epochs,
            "patience": args.patience,
            "grad_clip": args.grad_clip,
        },
        "data_config": {
            "augment": args.augment,
            "strong_augment": args.strong_augment,
            "target_shape": args.target_shape,
            "normalize_per_sample": args.normalize_per_sample,
        },
        "system_config": {
            "device": str(get_device_info()["device"]),
            "timestamp": datetime.now().isoformat(),
        }
    }
    return config


class TrainingLogger:
    """Enhanced logging system"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_files = {
            'training': os.path.join(output_dir, 'training_log.csv'),
            'metrics': os.path.join(output_dir, 'metrics_log.csv'),
            'system': os.path.join(output_dir, 'system_log.txt')
        }
        self._init_logs()
    
    def _init_logs(self):
        """Initialize log files with headers"""
        # Training log
        with open(self.log_files['training'], 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'batch', 'phase', 'total_loss', 'recon_loss', 'kl_loss',
                'lr', 'beta', 'grad_norm', 'time_elapsed'
            ])
        
        # Metrics log
        with open(self.log_files['metrics'], 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'val_mse', 'val_mae', 'val_ssim', 'mu_mean', 'mu_std',
                'logvar_mean', 'logvar_std', 'kl_div_per_dim'
            ])
    
    def log_training(self, epoch, batch, phase, losses, lr, beta, grad_norm, elapsed_time):
        """Log training step"""
        with open(self.log_files['training'], 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, batch, phase, losses['total'], losses['recon'], losses['kl'],
                lr, beta, grad_norm, elapsed_time
            ])
    
    def log_metrics(self, epoch, metrics, latent_stats):
        """Log validation metrics"""
        with open(self.log_files['metrics'], 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, metrics.get('mse', 0), metrics.get('mae', 0), metrics.get('ssim', 0),
                latent_stats['mu_mean'], latent_stats['mu_std'],
                latent_stats['logvar_mean'], latent_stats['logvar_std'],
                latent_stats.get('kl_per_dim', 0)
            ])
    
    def log_system(self, message):
        """Log system messages"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_files['system'], 'a') as f:
            f.write(f"[{timestamp}] {message}\n")


def compute_latent_stats(mu, logvar):
    """Compute comprehensive latent space statistics"""
    with torch.no_grad():
        mu_mean = mu.mean().item()
        mu_std = mu.std().item()
        logvar_mean = logvar.mean().item()
        logvar_std = logvar.std().item()
        
        # KL divergence per dimension
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)
        kl_per_dim_mean = kl_per_dim.mean().item()
        
        return {
            'mu_mean': mu_mean,
            'mu_std': mu_std,
            'logvar_mean': logvar_mean,
            'logvar_std': logvar_std,
            'kl_per_dim': kl_per_dim_mean
        }


def enhanced_train_step(model, batch, optimizer, device, beta, conditional):
    """Single training step with enhanced error handling"""
    try:
        # Prepare batch
        if conditional:
            data, labels = batch
            data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        else:
            data = batch.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass
        if conditional:
            recon_x, mu, logvar = model(data, labels)
        else:
            recon_x, mu, logvar = model(data)
        
        # Compute loss
        total_loss, recon_loss, kl_loss = model.loss_function(
            recon_x, data, mu, logvar, beta=beta
        )
        
        # Check for NaN
        if torch.isnan(total_loss):
            raise ValueError("NaN loss detected")
        
        return {
            'total': total_loss,
            'recon': recon_loss,
            'kl': kl_loss,
            'mu': mu,
            'logvar': logvar,
            'recon_x': recon_x
        }
        
    except Exception as e:
        print(f"⚠️ Training step error: {e}")
        return None


def enhanced_val_step(model, batch, device, beta, conditional):
    """Single validation step"""
    try:
        with torch.no_grad():
            if conditional:
                data, labels = batch
                data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                recon_x, mu, logvar = model(data, labels)
            else:
                data = batch.to(device, non_blocking=True)
                recon_x, mu, logvar = model(data)
            
            total_loss, recon_loss, kl_loss = model.loss_function(
                recon_x, data, mu, logvar, beta=beta
            )
            
            # Compute reconstruction metrics
            mse = F.mse_loss(recon_x, data).item()
            mae = F.l1_loss(recon_x, data).item()
            
            # Simple SSIM approximation
            mu1, mu2 = data.mean(), recon_x.mean()
            sigma1, sigma2 = data.std(), recon_x.std()
            ssim_approx = (2 * mu1 * mu2) / (mu1**2 + mu2**2 + 1e-8)
            
            return {
                'losses': {'total': total_loss, 'recon': recon_loss, 'kl': kl_loss},
                'metrics': {'mse': mse, 'mae': mae, 'ssim': ssim_approx.item()},
                'latent': {'mu': mu, 'logvar': logvar}
            }
            
    except Exception as e:
        print(f"⚠️ Validation step error: {e}")
        return None


def run_enhanced_training():
    """Main training function with all enhancements"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Enhanced Spectrogram VAE Training')
    
    # Data arguments
    parser.add_argument('--data_dir', required=True, help="Path to spectrogram directory")
    parser.add_argument('--target_shape', nargs=2, type=int, default=[1025, 469], 
                        help="Target spectrogram shape [freq, time]")
    parser.add_argument('--normalize_per_sample', action='store_true', 
                        help="Use per-sample normalization instead of global")
    
    # Model arguments
    parser.add_argument('--latent_dim', type=int, default=512, help="Latent dimension")
    parser.add_argument('--beta', type=float, default=0.1, help="Beta parameter for β-VAE")
    parser.add_argument('--conditional', action='store_true', help="Use conditional VAE")
    parser.add_argument('--embed_dim', type=int, default=128, help="Label embedding dimension")
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--epochs', type=int, default=150, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    parser.add_argument('--warmup_epochs', type=int, default=15, help="Warmup epochs")
    parser.add_argument('--patience', type=int, default=25, help="Early stopping patience")
    parser.add_argument('--grad_clip', type=float, default=1.0, help="Gradient clipping")
    
    # Augmentation arguments
    parser.add_argument('--augment', action='store_true', help="Apply data augmentation")
    parser.add_argument('--strong_augment', action='store_true', help="Use strong augmentation")
    
    # Output arguments
    parser.add_argument('--output_dir', default='enhanced_vae_results', help="Output directory")
    parser.add_argument('--resume', type=str, default=None, help="Resume from checkpoint")
    parser.add_argument('--inspect_data', action='store_true', help="Inspect data before training")
    
    args = parser.parse_args()
    
    # Convert target_shape to tuple
    args.target_shape = tuple(args.target_shape)
    
    # Setup device and system info
    device_info = get_device_info()
    device = device_info["device"]
    
    print("🚀 ENHANCED VAE TRAINING")
    print("=" * 60)
    print(f"Device: {device_info['name']} ({device})")
    if device.type == 'cuda':
        print(f"GPU Memory: {device_info['memory']:.1f} GB")
        print(f"CUDA Version: {device_info.get('cuda_version', 'Unknown')}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"enhanced_vae_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Setup logging
    logger = TrainingLogger(output_dir)
    logger.log_system(f"Training started with args: {vars(args)}")
    
    # Save configuration
    config = create_training_config(args)
    import json
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    # Data inspection (optional)
    if args.inspect_data:
        print("\n🔍 INSPECTING DATA")
        print("=" * 60)
        inspect_enhanced_files(args.data_dir, num_samples=5)
    
    # Load and prepare data
    print("\n📊 LOADING DATA")
    print("=" * 60)
    
    file_paths = load_file_paths(args.data_dir)
    print(f"Found {len(file_paths)} spectrogram files")
    
    if len(file_paths) == 0:
        raise ValueError(f"No .pt files found in {args.data_dir}")
    
    # Setup label encoder for conditional VAE
    label_encoder = None
    num_classes = 0
    
    if args.conditional:
        print("🏷️ Setting up conditional VAE...")
        labels = encode_labels(file_paths)
        if labels:
            label_encoder = LabelEncoder()
            label_encoder.fit(labels)
            num_classes = len(label_encoder.classes_)
            print(f"Found {num_classes} classes: {label_encoder.classes_}")
            
            # Save label encoder
            import pickle
            with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
                pickle.dump(label_encoder, f)
        else:
            print("⚠️ No labels found, switching to standard VAE")
            args.conditional = False
    
    # Create datasets
    print(f"📈 Creating enhanced datasets...")
    try:
        train_dataset, val_dataset, test_dataset, target_shape, num_classes = create_enhanced_vae_datasets(
            args.data_dir,
            label_encoder=label_encoder,
            conditional=args.conditional,
            augment=args.augment,
            strong_augment=args.strong_augment,
            target_shape=args.target_shape,
            normalize_per_sample=args.normalize_per_sample
        )
    except Exception as e:
        logger.log_system(f"Dataset creation failed: {e}")
        raise
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    logger.log_system(f"Datasets created - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == 'cuda',
        drop_last=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == 'cuda',
        drop_last=False
    )
    
    # Initialize model
    print(f"\n🧠 INITIALIZING MODEL")
    print("=" * 60)
    
    try:
        if args.conditional and num_classes > 0:
            print(f"Creating Conditional VAE with {num_classes} classes")
            model = ImprovedConditionalVAE(
                input_shape=target_shape,
                latent_dim=args.latent_dim,
                num_classes=num_classes,
                embed_dim=args.embed_dim
            ).to(device)
        else:
            print("Creating Standard VAE")
            model = VariationalAutoEncoder(
                input_shape=target_shape,
                latent_dim=args.latent_dim,
                beta=args.beta
            ).to(device)
    except Exception as e:
        logger.log_system(f"Model initialization failed: {e}")
        raise
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1e6:.1f} MB (float32)")
    
    logger.log_system(f"Model created - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Setup optimizer with different learning rates for different components
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())
    other_params = [p for p in model.parameters() if p not in encoder_params and p not in decoder_params]
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': args.lr * 0.8, 'name': 'encoder'},
        {'params': decoder_params, 'lr': args.lr, 'name': 'decoder'},
        {'params': other_params, 'lr': args.lr * 1.2, 'name': 'other'}
    ], weight_decay=1e-4, betas=(0.9, 0.999))
    
    # Advanced learning rate scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return epoch / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        print(f"🔄 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
        logger.log_system(f"Resumed from checkpoint at epoch {start_epoch}")
    
    # Training loop
    print(f"\n🎯 STARTING TRAINING")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Beta: {args.beta}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    
    start_time = time.time()
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Beta annealing - more gradual
        beta_progress = min(1.0, (epoch + 1) / (args.epochs * 0.4))  # 40% of training for full beta
        effective_beta = args.beta * (beta_progress ** 1.5)  # Smoother annealing
        
        print(f"\n📅 EPOCH {epoch + 1}/{args.epochs}")
        print(f"Beta: {effective_beta:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Training phase
        model.train()
        train_losses = {'total': 0, 'recon': 0, 'kl': 0}
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            step_result = enhanced_train_step(model, batch, optimizer, device, effective_beta, args.conditional)
            
            if step_result is None:
                continue
            
            # Backward pass
            step_result['total'].backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            
            optimizer.step()
            
            # Accumulate losses
            train_losses['total'] += step_result['total'].item()
            train_losses['recon'] += step_result['recon'].item()
            train_losses['kl'] += step_result['kl'].item()
            train_batches += 1
            
            # Log every 50 batches
            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - start_time
                
                print(f"  Batch {batch_idx:3d}/{len(train_loader)} | "
                      f"Loss: {step_result['total'].item():.4f} | "
                      f"Recon: {step_result['recon'].item():.4f} | "
                      f"KL: {step_result['kl'].item():.4f} | "
                      f"Grad: {grad_norm:.3f}")
                
                # Log to file
                logger.log_training(
                    epoch + 1, batch_idx, 'train',
                    {'total': step_result['total'].item(), 'recon': step_result['recon'].item(), 'kl': step_result['kl'].item()},
                    current_lr, effective_beta, grad_norm.item(), elapsed
                )
        
        # Update learning rate
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_losses = {'total': 0, 'recon': 0, 'kl': 0}
        val_metrics = {'mse': 0, 'mae': 0, 'ssim': 0}
        val_batches = 0
        all_mu, all_logvar = [], []
        
        for batch in val_loader:
            val_result = enhanced_val_step(model, batch, device, effective_beta, args.conditional)
            
            if val_result is None:
                continue
            
            # Accumulate losses and metrics
            for key in val_losses:
                val_losses[key] += val_result['losses'][key].item()
            for key in val_metrics:
                val_metrics[key] += val_result['metrics'][key]
            
            all_mu.append(val_result['latent']['mu'])
            all_logvar.append(val_result['latent']['logvar'])
            val_batches += 1
        
        # Average losses and metrics
        for key in val_losses:
            val_losses[key] /= max(val_batches, 1)
        for key in val_metrics:
            val_metrics[key] /= max(val_batches, 1)
        for key in train_losses:
            train_losses[key] /= max(train_batches, 1)
        
        # Compute latent statistics
        if all_mu and all_logvar:
            combined_mu = torch.cat(all_mu, dim=0)
            combined_logvar = torch.cat(all_logvar, dim=0)
            latent_stats = compute_latent_stats(combined_mu, combined_logvar)
        else:
            latent_stats = {'mu_mean': 0, 'mu_std': 0, 'logvar_mean': 0, 'logvar_std': 0, 'kl_per_dim': 0}
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        print(f"\n📊 EPOCH {epoch + 1} SUMMARY")
        print("-" * 50)
        print(f"Time: {epoch_time:.1f}s | Total: {total_time//60:.0f}m {total_time%60:.0f}s")
        print(f"Train Loss: {train_losses['total']:.4f} (Recon: {train_losses['recon']:.4f}, KL: {train_losses['kl']:.4f})")
        print(f"Val Loss:   {val_losses['total']:.4f} (Recon: {val_losses['recon']:.4f}, KL: {val_losses['kl']:.4f})")
        print(f"Metrics: MSE: {val_metrics['mse']:.4f}, MAE: {val_metrics['mae']:.4f}, SSIM: {val_metrics['ssim']:.4f}")
        print(f"Latent: μ={latent_stats['mu_mean']:.3f}±{latent_stats['mu_std']:.3f}, "
              f"logvar={latent_stats['logvar_mean']:.3f}±{latent_stats['logvar_std']:.3f}")
        
        # Log metrics
        logger.log_metrics(epoch + 1, val_metrics, latent_stats)
        logger.log_training(
            epoch + 1, len(train_loader), 'val', val_losses,
            scheduler.get_last_lr()[0], effective_beta, 0, total_time
        )
        
        # Model checkpointing
        is_best = val_losses['total'] < best_val_loss
        if is_best:
            best_val_loss = val_losses['total']
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_losses['total'],
                'best_val_loss': best_val_loss,
                'metrics': val_metrics,
                'latent_stats': latent_stats,
                'args': vars(args),
                'config': config
            }, os.path.join(output_dir, 'best_model.pth'))
            
            print(f"✅ New best model saved! Val loss: {val_losses['total']:.4f}")
            logger.log_system(f"New best model saved at epoch {epoch + 1}, val_loss: {val_losses['total']:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n🛑 Early stopping triggered after {args.patience} epochs without improvement")
            print(f"Best validation loss: {best_val_loss:.4f}")
            logger.log_system(f"Early stopping at epoch {epoch + 1}, best_val_loss: {best_val_loss:.4f}")
            break
        
        # Generate samples periodically
        if (epoch + 1) % 10 == 0 or epoch == 0:
            try:
                save_enhanced_sample_outputs(model, device, output_dir, epoch + 1, 
                                           conditional=args.conditional, num_classes=num_classes)
            except Exception as e:
                print(f"⚠️ Error generating samples: {e}")
                logger.log_system(f"Sample generation error at epoch {epoch + 1}: {e}")
        
        # Periodic checkpoint
        if (epoch + 1) % 25 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_losses['total'],
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}.pth'))
    
    # Training completion
    final_time = time.time() - start_time
    print(f"\n🎉 TRAINING COMPLETED")
    print("=" * 60)
    print(f"Total time: {final_time//60:.0f}m {final_time%60:.0f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation loss: {val_losses['total']:.4f}")
    print(f"Results saved in: {output_dir}")
    
    logger.log_system(f"Training completed in {final_time//60:.0f}m {final_time%60:.0f}s, best_val_loss: {best_val_loss:.4f}")
    
    # Save final model
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_val_loss': val_losses['total'],
        'best_val_loss': best_val_loss,
        'training_time': final_time,
        'args': vars(args),
        'config': config
    }, os.path.join(output_dir, 'final_model.pth'))
    
    return output_dir, best_val_loss


# Additional utility functions
def save_enhanced_sample_outputs(model, device, output_dir, epoch, num_samples=8, 
                                conditional=False, num_classes=None):
    """Generate and save sample outputs with enhanced visualization"""
    model.eval()
    with torch.no_grad():
        if conditional and num_classes:
            samples_per_class = max(1, num_samples // num_classes)
            all_samples = []
            class_labels = []
            
            for class_idx in range(min(num_classes, num_samples)):
                class_samples = model.sample_class(class_idx, samples_per_class, device=device)
                all_samples.append(class_samples)
                class_labels.extend([f'Class {class_idx}'] * samples_per_class)
            
            samples = torch.cat(all_samples, dim=0)[:num_samples]
        else:
            samples = model.sample(num_samples, device=device)
            class_labels = [f'Sample {i+1}' for i in range(num_samples)]
        
        # Create visualization
        samples_np = samples.cpu().numpy()
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i in range(min(num_samples, len(axes))):
            ax = axes[i]
            spec_vis = samples_np[i, 0] if samples_np[i].ndim == 3 else samples_np[i]
            
            im = ax.imshow(spec_vis, aspect='auto', origin='lower', 
                          cmap='viridis', interpolation='bilinear')
            ax.set_title(f'{class_labels[i]}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency')
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        plt.suptitle(f'Generated Samples - Epoch {epoch}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'samples_epoch_{epoch}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    try:
        output_dir, best_loss = run_enhanced_training()
        print(f"\n✅ Training successful!")
        print(f"📁 Results: {output_dir}")
        print(f"🎯 Best loss: {best_loss:.4f}")
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()