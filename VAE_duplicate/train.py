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

# Fixed Model Architecture
class FixedEncoder(nn.Module):
    """Fixed encoder with proper shape handling"""
    def __init__(self, input_shape, latent_dim=128):
        super(FixedEncoder, self).__init__()
        
        if isinstance(input_shape, torch.Size):
            input_shape = tuple(input_shape)
        if len(input_shape) == 2:
            self.input_shape = (1, *input_shape)
        else:
            self.input_shape = input_shape

        # Fixed architecture with correct padding
        self.encoder = nn.Sequential(
            # First conv: 1025x938 -> 513x469
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Second conv: 513x469 -> 257x235  
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Third conv: 257x235 -> 129x118
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Fourth conv: 129x118 -> 65x59
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        # Calculate exact output size
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            encoder_output = self.encoder(dummy)
            self.encoder_shape = encoder_output.shape[1:]  # [C, H, W]
            self.encoder_flatten = encoder_output.numel() // encoder_output.shape[0]
            print(f"Fixed Encoder output shape: {self.encoder_shape}")
            print(f"Fixed Flatten size: {self.encoder_flatten}")

        # Latent layers with proper initialization
        self.fc_mu = nn.Linear(self.encoder_flatten, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_flatten, latent_dim)
        
        # Critical: Initialize logvar properly
        nn.init.normal_(self.fc_logvar.weight, 0, 0.001)
        nn.init.constant_(self.fc_logvar.bias, -3.0)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # Critical: Clamp logvar to prevent explosion
        logvar = torch.clamp(logvar, min=-15, max=5)
        
        return mu, logvar


class FixedDecoder(nn.Module):
    """Fixed decoder that matches encoder exactly"""
    def __init__(self, output_shape, encoder_shape, latent_dim=128):
        super(FixedDecoder, self).__init__()
        self.output_shape = output_shape
        self.latent_dim = latent_dim
        
        if len(encoder_shape) == 3:
            self.channels, self.height, self.width = encoder_shape
        else:
            raise ValueError(f"Invalid encoder_shape: {encoder_shape}")
        
        self.encoder_flatten = self.channels * self.height * self.width

        # FC layer to expand latent vector
        self.fc = nn.Linear(latent_dim, self.encoder_flatten)

        # Decoder layers - EXACT reverse of encoder
        self.decoder = nn.Sequential(
            # 65x59 -> 129x118
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 129x118 -> 257x235
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 257x235 -> 513x469
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            # 513x469 -> 1025x938 (target size)
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.channels, self.height, self.width)
        x = self.decoder(x)
        
        # Critical: Ensure exact output size
        target_h, target_w = self.output_shape[-2:]
        if x.shape[-2] != target_h or x.shape[-1] != target_w:
            x = torch.nn.functional.interpolate(
                x, size=(target_h, target_w), 
                mode='bilinear', align_corners=False
            )
        
        return x


class FixedVAE(nn.Module):
    """Fixed VAE with robust architecture"""
    def __init__(self, input_shape, latent_dim=128, beta=1e-4):
        super(FixedVAE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = FixedEncoder(input_shape, latent_dim)
        self.decoder = FixedDecoder(
            output_shape=input_shape,
            encoder_shape=self.encoder.encoder_shape,
            latent_dim=latent_dim
        )

    def reparameterize(self, mu, logvar):
        """Robust reparameterization"""
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, min=1e-6, max=10)  # Prevent extreme values
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss_function(self, recon_x, x, mu, logvar, beta=None):
        """Robust loss calculation"""
        if beta is None:
            beta = self.beta
        
        # Reconstruction loss
        recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')
        
        # KL divergence with stability
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = torch.clamp(kl_loss, min=0, max=100)  # Prevent explosion
        
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss

    def sample(self, num_samples, device='cpu'):
        """Generate samples"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decoder(z)
            return samples

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


# Import your existing data loading functions
from data_loading import create_vae_datasets, encode_labels, load_file_paths


def robust_train_epoch(model, train_loader, optimizer, device, beta, grad_clip):
    """Robust training epoch with error handling"""
    model.train()
    total_loss = 0
    recon_loss = 0
    kl_loss = 0
    valid_batches = 0
    
    mu_values = []
    logvar_values = []
    
    for batch_idx, batch in enumerate(train_loader):
        try:
            # Handle both conditional and non-conditional data
            if isinstance(batch, tuple):
                data = batch[0]  # Take only the spectrogram data
            else:
                data = batch
            
            data = data.to(device)
            
            # Validate input
            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f"Skipping batch {batch_idx}: NaN/Inf in input")
                continue
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_x, mu, logvar = model(data)
            
            # Validate outputs
            if (torch.isnan(recon_x).any() or torch.isinf(recon_x).any() or
                torch.isnan(mu).any() or torch.isinf(mu).any() or
                torch.isnan(logvar).any() or torch.isinf(logvar).any()):
                print(f"Skipping batch {batch_idx}: NaN/Inf in outputs")
                continue
            
            # Compute loss
            batch_total_loss, batch_recon_loss, batch_kl_loss = model.loss_function(
                recon_x, data, mu, logvar, beta=beta
            )
            
            # Validate loss
            if torch.isnan(batch_total_loss) or torch.isinf(batch_total_loss):
                print(f"Skipping batch {batch_idx}: NaN/Inf in loss")
                continue
            
            # Backward pass
            batch_total_loss.backward()
            
            # Check gradients
            valid_gradients = True
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        valid_gradients = False
                        break
            
            if not valid_gradients:
                print(f"Skipping batch {batch_idx}: Invalid gradients")
                optimizer.zero_grad()
                continue
            
            # Clip gradients
            grad_norm = clip_grad_norm_(model.parameters(), grad_clip)
            
            # Update parameters
            optimizer.step()
            
            # Accumulate statistics
            total_loss += batch_total_loss.item()
            recon_loss += batch_recon_loss.item()
            kl_loss += batch_kl_loss.item()
            valid_batches += 1
            
            # Store latent statistics
            mu_values.append(mu.detach().cpu())
            logvar_values.append(logvar.detach().cpu())
            
            # Progress reporting
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}: Loss={batch_total_loss.item():.4f}, "
                      f"Recon={batch_recon_loss.item():.4f}, KL={batch_kl_loss.item():.4f}, "
                      f"GradNorm={grad_norm:.4f}")
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    # Calculate averages
    if valid_batches > 0:
        total_loss /= valid_batches
        recon_loss /= valid_batches
        kl_loss /= valid_batches
    
    # Calculate latent statistics
    latent_stats = {}
    if mu_values and logvar_values:
        mu_tensor = torch.cat(mu_values, dim=0)
        logvar_tensor = torch.cat(logvar_values, dim=0)
        latent_stats = {
            'mu_mean': mu_tensor.mean().item(),
            'mu_std': mu_tensor.std().item(),
            'logvar_mean': logvar_tensor.mean().item(),
            'logvar_std': logvar_tensor.std().item(),
            'actual_var': torch.exp(logvar_tensor).mean().item()
        }
    
    return total_loss, recon_loss, kl_loss, latent_stats, valid_batches


def robust_validate_epoch(model, val_loader, device, beta):
    """Robust validation epoch"""
    model.eval()
    total_loss = 0
    recon_loss = 0
    kl_loss = 0
    valid_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                # Handle both conditional and non-conditional data
                if isinstance(batch, tuple):
                    data = batch[0]
                else:
                    data = batch
                
                data = data.to(device)
                
                # Validate input
                if torch.isnan(data).any() or torch.isinf(data).any():
                    continue
                
                # Forward pass
                recon_x, mu, logvar = model(data)
                
                # Validate outputs
                if (torch.isnan(recon_x).any() or torch.isinf(recon_x).any() or
                    torch.isnan(mu).any() or torch.isinf(mu).any() or
                    torch.isnan(logvar).any() or torch.isinf(logvar).any()):
                    continue
                
                # Compute loss
                batch_total_loss, batch_recon_loss, batch_kl_loss = model.loss_function(
                    recon_x, data, mu, logvar, beta=beta
                )
                
                # Validate loss
                if torch.isnan(batch_total_loss) or torch.isinf(batch_total_loss):
                    continue
                
                total_loss += batch_total_loss.item()
                recon_loss += batch_recon_loss.item()
                kl_loss += batch_kl_loss.item()
                valid_batches += 1
                
            except Exception as e:
                print(f"Validation error in batch {batch_idx}: {e}")
                continue
    
    # Calculate averages
    if valid_batches > 0:
        total_loss /= valid_batches
        recon_loss /= valid_batches
        kl_loss /= valid_batches
    
    return total_loss, recon_loss, kl_loss


def main():
    parser = argparse.ArgumentParser(description='Fixed Robust VAE Training')
    parser.add_argument('--data_dir', required=True, help="Path to spectrogram directory")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--latent_dim', type=int, default=128, help="Latent dimension")
    parser.add_argument('--beta', type=float, default=1e-4, help="Beta parameter")
    parser.add_argument('--grad_clip', type=float, default=0.5, help="Gradient clipping")
    parser.add_argument('--output_dir', default='robust_vae_results', help="Output directory")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"robust_vae_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("Loading data...")
    file_paths = load_file_paths(args.data_dir)
    print(f"Found {len(file_paths)} spectrogram files")
    
    # Create datasets (non-conditional for simplicity)
    train_dataset, val_dataset, test_dataset, spectrogram_shape, _ = create_vae_datasets(
        args.data_dir, 
        conditional=False,  # Start with non-conditional
        augment=False
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Spectrogram shape: {spectrogram_shape}")
    
    # Create data loaders with error handling
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,  # Avoid multiprocessing issues
        drop_last=True,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False
    )
    
    # Initialize model
    print("Initializing Fixed VAE...")
    model = FixedVAE(
        input_shape=spectrogram_shape,
        latent_dim=args.latent_dim,
        beta=args.beta
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5, 
        min_lr=1e-7, verbose=True
    )
    
    # Training log
    log_file = os.path.join(output_dir, "robust_vae_log.csv")
    with open(log_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'train_recon_loss', 'train_kl_loss',
            'val_loss', 'val_recon_loss', 'val_kl_loss', 'lr', 'beta',
            'mu_mean', 'mu_std', 'logvar_mean', 'logvar_std', 'actual_var',
            'train_batches', 'time_elapsed'
        ])
    
    # Training loop
    best_val_loss = float('inf')
    start_time = time.time()
    
    print(f"\nStarting Robust VAE training:")
    print(f"- Learning rate: {args.lr}")
    print(f"- Beta: {args.beta}")
    print(f"- Latent dim: {args.latent_dim}")
    print(f"- Gradient clipping: {args.grad_clip}")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{args.epochs} (LR: {current_lr:.2e})")
        
        # Training
        train_loss, train_recon, train_kl, latent_stats, train_batches = robust_train_epoch(
            model, train_loader, optimizer, device, args.beta, args.grad_clip
        )
        
        # Validation
        val_loss, val_recon, val_kl = robust_validate_epoch(
            model, val_loader, device, args.beta
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Time tracking
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        # Print summary
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Time: {epoch_time:.2f}s | Total: {total_time//60:.0f}m {total_time%60:.0f}s")
        print(f"  Train: Loss={train_loss:.4f}, Recon={train_recon:.4f}, KL={train_kl:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Recon={val_recon:.4f}, KL={val_kl:.4f}")
        print(f"  Valid batches: {train_batches}/{len(train_loader)}")
        
        if latent_stats:
            print(f"  Latent: μ={latent_stats['mu_mean']:.4f}±{latent_stats['mu_std']:.4f}, "
                  f"σ²={latent_stats['actual_var']:.4f}")
        
        # Save log
        with open(log_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1, train_loss, train_recon, train_kl,
                val_loss, val_recon, val_kl, current_lr, args.beta,
                latent_stats.get('mu_mean', 0), latent_stats.get('mu_std', 0),
                latent_stats.get('logvar_mean', 0), latent_stats.get('logvar_std', 0),
                latent_stats.get('actual_var', 0), train_batches, total_time
            ])
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, os.path.join(output_dir, 'best_robust_model.pth'))
            print(f"  → New best model saved! (Val Loss: {val_loss:.4f})")
        
        # Early stopping
        if train_batches < len(train_loader) * 0.5:
            print("Early stopping: Too many failed batches")
            break
        
        if train_kl > 1000:
            print("Early stopping: KL divergence explosion")
            break
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()