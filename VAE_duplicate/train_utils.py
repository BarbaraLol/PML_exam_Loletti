import os
import torch
import csv
import numpy as np
from datetime import datetime

def save_checkpoint(model, optimizer, epoch, loss, accuracy=None, filename='checkpoint.pth'):
    """Save training checkpoint with error handling"""
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        
        if accuracy is not None:
            checkpoint['accuracy'] = accuracy
        
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def calculate_conditional_vae_accuracy(model, data, labels, device):
    """Calculate reconstruction accuracy for conditional VAE"""
    try:
        model.eval()
        with torch.no_grad():
            # Forward pass
            recon_x, mu, logvar = model(data, labels)
            
            # Check for NaN/Inf in reconstruction
            if torch.isnan(recon_x).any() or torch.isinf(recon_x).any():
                return 0.0
            
            # Simple reconstruction accuracy based on MSE threshold
            mse_per_sample = torch.mean((recon_x - data) ** 2, dim=[1, 2, 3])
            
            # Robust threshold calculation
            median_mse = torch.median(mse_per_sample)
            std_mse = torch.std(mse_per_sample)
            
            # Use a more conservative threshold
            threshold = median_mse + 0.5 * std_mse
            accurate_reconstructions = (mse_per_sample < threshold).sum().item()
            
            return accurate_reconstructions / data.size(0)
    except Exception as e:
        print(f"Error calculating VAE accuracy: {e}")
        return 0.0

def log_vae_epoch_data(epoch, train_loss, train_recon, train_kl, 
                      val_loss, val_recon, val_kl, lr, beta, filename):
    """Log VAE-specific epoch data to CSV file with error handling"""
    try:
        file_exists = os.path.isfile(filename)
        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Epoch', 'Train_Total', 'Train_Recon', 'Train_KL', 
                               'Val_Total', 'Val_Recon', 'Val_KL', 'LR', 'Beta'])
            writer.writerow([epoch+1, train_loss, train_recon, train_kl, 
                            val_loss, val_recon, val_kl, lr, beta])
    except Exception as e:
        print(f"Error writing VAE log: {e}")

def log_conditional_vae_epoch_data(epoch, train_loss, train_recon, train_kl, train_acc,
                                  val_loss, val_recon, val_kl, val_acc, lr, beta, filename):
    """Log Conditional VAE-specific epoch data with accuracy to CSV file"""
    try:
        file_exists = os.path.isfile(filename)
        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Epoch', 'Train_Total', 'Train_Recon', 'Train_KL', 'Train_Acc',
                               'Val_Total', 'Val_Recon', 'Val_KL', 'Val_Acc', 'LR', 'Beta'])
            writer.writerow([epoch+1, train_loss, train_recon, train_kl, train_acc,
                            val_loss, val_recon, val_kl, val_acc, lr, beta])
    except Exception as e:
        print(f"Error writing Conditional VAE log: {e}")

def monitor_latent_health(mu, logvar, epoch, max_mu=5.0, max_logvar=2.0, min_logvar=-10.0):
    """Monitor latent space health and return warnings"""
    warnings = []
    
    try:
        # Calculate statistics
        mu_mean = mu.mean().item()
        mu_std = mu.std().item()
        logvar_mean = logvar.mean().item()
        logvar_std = logvar.std().item()
        actual_var = torch.exp(logvar).mean().item()
        
        # Check for unhealthy latent space
        if abs(mu_mean) > max_mu:
            warnings.append(f"Large mu mean: {mu_mean:.4f}")
        
        if mu_std > max_mu:
            warnings.append(f"Large mu std: {mu_std:.4f}")
        
        if logvar_mean > max_logvar:
            warnings.append(f"Large logvar: {logvar_mean:.4f}")
        
        if logvar_mean < min_logvar:
            warnings.append(f"Very small logvar: {logvar_mean:.4f}")
        
        if actual_var > 50:
            warnings.append(f"Very large variance: {actual_var:.4f}")
        
        if actual_var < 1e-6:
            warnings.append(f"Collapsed variance: {actual_var:.6f}")
        
        # Check for NaN/Inf
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            warnings.append("NaN/Inf in mu")
        
        if torch.isnan(logvar).any() or torch.isinf(logvar).any():
            warnings.append("NaN/Inf in logvar")
        
        return warnings, {
            'mu_mean': mu_mean,
            'mu_std': mu_std, 
            'logvar_mean': logvar_mean,
            'logvar_std': logvar_std,
            'actual_var': actual_var
        }
        
    except Exception as e:
        return [f"Error monitoring latent space: {e}"], {}

def check_model_health(model, data_sample, device):
    """Check overall model health"""
    try:
        model.eval()
        health_report = []
        
        with torch.no_grad():
            # Forward pass
            if hasattr(model, 'num_classes'):  # Conditional VAE
                dummy_labels = torch.zeros(data_sample.size(0), dtype=torch.long).to(device)
                recon_x, mu, logvar = model(data_sample, dummy_labels)
            else:
                recon_x, mu, logvar = model(data_sample)
            
            # Check outputs
            if torch.isnan(recon_x).any():
                health_report.append("NaN in reconstruction")
            if torch.isinf(recon_x).any():
                health_report.append("Inf in reconstruction")
            
            # Check reconstruction range
            recon_min, recon_max = recon_x.min().item(), recon_x.max().item()
            if recon_min < -1 or recon_max > 2:
                health_report.append(f"Reconstruction out of range: [{recon_min:.3f}, {recon_max:.3f}]")
            
            # Check latent health
            warnings, stats = monitor_latent_health(mu, logvar, 0)
            health_report.extend(warnings)
            
            # Check parameter health
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    health_report.append(f"NaN in parameter: {name}")
                if torch.isinf(param).any():
                    health_report.append(f"Inf in parameter: {name}")
        
        return health_report
        
    except Exception as e:
        return [f"Error in model health check: {e}"]

def create_training_summary(output_dir, training_log_file):
    """Create a summary of training results"""
    try:
        if not os.path.exists(training_log_file):
            print(f"Training log not found: {training_log_file}")
            return
        
        # Read training log
        import pandas as pd
        df = pd.read_csv(training_log_file)
        
        # Create summary
        summary_file = os.path.join(output_dir, 'training_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("FIXED VAE TRAINING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Epochs: {len(df)}\n")
            f.write(f"Training Time: {df['time_elapsed'].iloc[-1]/60:.1f} minutes\n\n")
            
            f.write("LOSS PROGRESSION:\n")
            f.write(f"Initial Train Loss: {df['train_loss'].iloc[0]:.4f}\n")
            f.write(f"Final Train Loss: {df['train_loss'].iloc[-1]:.4f}\n")
            f.write(f"Best Val Loss: {df['val_loss'].min():.4f} (Epoch {df.loc[df['val_loss'].idxmin(), 'epoch']})\n")
            f.write(f"Final Val Loss: {df['val_loss'].iloc[-1]:.4f}\n\n")
            
            f.write("LOSS COMPONENTS:\n")
            f.write(f"Final Reconstruction Loss: {df['train_recon_loss'].iloc[-1]:.4f}\n")
            f.write(f"Final KL Loss: {df['train_kl_loss'].iloc[-1]:.4f}\n\n")
            
            if 'mu_mean' in df.columns:
                f.write("LATENT SPACE HEALTH:\n")
                f.write(f"Final μ mean: {df['mu_mean'].iloc[-1]:.4f}\n")
                f.write(f"Final μ std: {df['mu_std'].iloc[-1]:.4f}\n")
                f.write(f"Final logvar mean: {df['logvar_mean'].iloc[-1]:.4f}\n")
                f.write(f"Final logvar std: {df['logvar_std'].iloc[-1]:.4f}\n\n")
            
            f.write("TRAINING STABILITY:\n")
            train_loss_std = df['train_loss'].std()
            val_loss_std = df['val_loss'].std()
            f.write(f"Train Loss Stability (std): {train_loss_std:.4f}\n")
            f.write(f"Val Loss Stability (std): {val_loss_std:.4f}\n")
            
            if train_loss_std < 1.0 and val_loss_std < 1.0:
                f.write("✓ Training appears stable\n")
            else:
                f.write("⚠ Training shows instability\n")
        
        print(f"Training summary saved: {summary_file}")
        
    except Exception as e:
        print(f"Error creating training summary: {e}")

def validate_training_args(args):
    """Validate training arguments for stability"""
    warnings = []
    
    if args.lr > 1e-4:
        warnings.append(f"Learning rate {args.lr} may be too high for VAE stability")
    
    if args.beta > 1e-3:
        warnings.append(f"Beta {args.beta} may be too high, risk of posterior collapse")
    
    if args.latent_dim > 256:
        warnings.append(f"Latent dimension {args.latent_dim} may be too large for dataset")
    
    if args.grad_clip > 1.0:
        warnings.append(f"Gradient clipping {args.grad_clip} may be too loose")
    
    if args.batch_size < 4:
        warnings.append(f"Batch size {args.batch_size} may be too small for stable training")
    
    return warnings