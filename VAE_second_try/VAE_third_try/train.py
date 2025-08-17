import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
import math

class AdversarialLoss(nn.Module):
    """Simple discriminator to improve reconstruction quality"""
    def __init__(self, input_shape):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.discriminator(x)

def improved_train_vae(model, train_loader, val_loader, device, args, output_dir, conditional=False):
    """Enhanced training with multiple improvements"""
    
    # Multi-optimizer setup
    vae_optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Optional discriminator for adversarial training
    discriminator = AdversarialLoss(model.input_shape).to(device) if args.adversarial else None
    disc_optimizer = optim.AdamW(discriminator.parameters(), lr=args.lr * 0.1) if discriminator else None
    
    # Advanced scheduling
    total_steps = args.epochs * len(train_loader)
    scheduler = OneCycleLR(
        vae_optimizer,
        max_lr=args.lr * 2,
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0
    )
    
    # Progressive beta schedule
    def get_beta(epoch, total_epochs, target_beta):
        """Progressive beta annealing"""
        if epoch < total_epochs * 0.3:  # First 30% - linear warmup
            return target_beta * (epoch / (total_epochs * 0.3))
        elif epoch < total_epochs * 0.7:  # Middle 40% - constant
            return target_beta
        else:  # Last 30% - gradual increase for better disentanglement
            progress = (epoch - total_epochs * 0.7) / (total_epochs * 0.3)
            return target_beta * (1 + 0.5 * progress)
    
    # Training metrics tracking
    metrics = {
        'train_loss': [], 'val_loss': [], 'recon_loss': [], 'kl_loss': [],
        'spectral_loss': [], 'beta_values': [], 'lr_values': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Starting enhanced VAE training...")
    
    for epoch in range(args.epochs):
        # Dynamic beta scheduling
        current_beta = get_beta(epoch, args.epochs, args.beta)
        
        # Training phase
        model.train()
        if discriminator:
            discriminator.train()
            
        train_losses = {'total': 0, 'recon': 0, 'kl': 0, 'spectral': 0, 'adv': 0}
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Prepare data
                if conditional:
                    data, labels = batch
                    data, labels = data.to(device), labels.to(device)
                else:
                    data = batch.to(device)
                
                # VAE forward pass
                if conditional:
                    recon_x, mu, logvar = model(data, labels)
                else:
                    recon_x, mu, logvar = model(data)
                
                # VAE loss
                vae_loss, recon_loss, kl_loss = model.loss_function(
                    recon_x, data, mu, logvar, beta=current_beta
                )
                
                # Adversarial loss (optional)
                adv_loss = 0
                if discriminator and epoch > 10:  # Start adversarial training after initial epochs
                    # Train discriminator
                    real_score = discriminator(data)
                    fake_score = discriminator(recon_x.detach())
                    
                    disc_loss = -torch.log(real_score + 1e-8).mean() - torch.log(1 - fake_score + 1e-8).mean()
                    
                    disc_optimizer.zero_grad()
                    disc_loss.backward()
                    disc_optimizer.step()
                    
                    # Adversarial loss for VAE
                    fake_score = discriminator(recon_x)
                    adv_loss = -torch.log(fake_score + 1e-8).mean() * 0.1  # Weak adversarial signal
                
                total_loss = vae_loss + adv_loss
                
                # VAE backward pass
                vae_optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping with adaptive scaling
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                vae_optimizer.step()
                scheduler.step()
                
                # Track metrics
                train_losses['total'] += total_loss.item()
                train_losses['recon'] += recon_loss.item()
                train_losses['kl'] += kl_loss.item()
                train_losses['adv'] += adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss
                train_batches += 1
                
                # Progress reporting
                if batch_idx % 50 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch+1}/{args.epochs} [{batch_idx}/{len(train_loader)}]")
                    print(f"  LR: {current_lr:.2e} | β: {current_beta:.4f}")
                    print(f"  Loss: {total_loss.item():.4f} | Recon: {recon_loss.item():.4f}")
                    print(f"  KL: {kl_loss.item():.4f} | Grad: {grad_norm:.3f}")
                    if discriminator and epoch > 10:
                        print(f"  Adv: {adv_loss:.4f}")
                        
            except Exception as e:
                print(f"Training batch error: {e}")
                continue
        
        # Validation phase with enhanced metrics
        model.eval()
        val_losses = {'total': 0, 'recon': 0, 'kl': 0}
        val_batches = 0
        
        # Additional validation metrics
        reconstruction_quality = 0
        latent_regularization = 0
        
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
                        recon_x, data, mu, logvar, beta=current_beta
                    )
                    
                    val_losses['total'] += total_loss.item()
                    val_losses['recon'] += recon_loss.item()
                    val_losses['kl'] += kl_loss.item()
                    
                    # Additional metrics
                    reconstruction_quality += torch.mean((recon_x - data) ** 2).item()
                    latent_regularization += torch.mean(mu ** 2).item()
                    
                    val_batches += 1
                    
                except Exception as e:
                    print(f"Validation batch error: {e}")
                    continue
        
        # Average losses
        for key in train_losses:
            train_losses[key] /= train_batches
        for key in val_losses:
            val_losses[key] /= val_batches
            
        reconstruction_quality /= val_batches
        latent_regularization /= val_batches
        
        # Store metrics
        metrics['train_loss'].append(train_losses['total'])
        metrics['val_loss'].append(val_losses['total'])
        metrics['recon_loss'].append(val_losses['recon'])
        metrics['kl_loss'].append(val_losses['kl'])
        metrics['beta_values'].append(current_beta)
        metrics['lr_values'].append(scheduler.get_last_lr()[0])
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_losses['total']:.4f} | Val Loss: {val_losses['total']:.4f}")
        print(f"Recon Quality: {reconstruction_quality:.4f} | Latent Reg: {latent_regularization:.4f}")
        print(f"KL Div: {val_losses['kl']:.4f} | β: {current_beta:.4f}")
        
        # Model checkpointing with multiple criteria
        is_best = False
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1
        
        # Save checkpoint
        if is_best or epoch % 20 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': vae_optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_losses['total'],
                'metrics': metrics,
                'args': vars(args)
            }
            
            if discriminator:
                checkpoint['discriminator_state_dict'] = discriminator.state_dict()
                checkpoint['disc_optimizer_state_dict'] = disc_optimizer.state_dict()
            
            filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, f"{output_dir}/{filename}")
            
            if is_best:
                print(f"✓ New best model saved! Val loss: {val_losses['total']:.4f}")
        
        # Enhanced visualizations
        if (epoch + 1) % 10 == 0:
            save_enhanced_samples(model, device, output_dir, epoch+1, conditional, metrics)
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement")
            break
    
    return metrics

def save_enhanced_samples(model, device, output_dir, epoch, conditional, metrics):
    """Save enhanced visualizations including training curves"""
    model.eval()
    
    with torch.no_grad():
        # Generate samples
        if conditional and hasattr(model, 'num_classes'):
            samples = []
            for i in range(min(8, model.num_classes)):
                sample = model.sample_class(i, 1, device=device)
                samples.append(sample)
            samples = torch.cat(samples, dim=0)
        else:
            samples = model.sample(8, device=device)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Generated samples (top 2 rows)
    for i in range(min(8, samples.shape[0])):
        ax = plt.subplot(4, 4, i+1)
        sample_vis = samples[i, 0].cpu().numpy() if samples[i].ndim == 3 else samples[i].cpu().numpy()
        plt.imshow(sample_vis, aspect='auto', origin='lower', cmap='viridis')
        plt.title(f'Generated {i+1}')
        plt.axis('off')
    
    # Training curves (bottom 2 rows)
    ax1 = plt.subplot(4, 2, 5)
    plt.plot(metrics['train_loss'], label='Train Loss', alpha=0.7)
    plt.plot(metrics['val_loss'], label='Val Loss', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(4, 2, 6)
    plt.plot(metrics['recon_loss'], label='Reconstruction', alpha=0.7)
    plt.plot(metrics['kl_loss'], label='KL Divergence', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Components')
    plt.title('Loss Breakdown')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(4, 2, 7)
    plt.plot(metrics['beta_values'], label='β Parameter', alpha=0.7)
    plt.plot(metrics['lr_values'], label='Learning Rate', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Parameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/enhanced_training_epoch_{epoch}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

# Additional helper functions for training
def compute_fid_score(real_samples, generated_samples):
    """Compute FID score for evaluation"""
    # Simplified FID computation for spectrograms
    # In practice, you'd want to use a pre-trained feature extractor
    
    def compute_statistics(samples):
        # Flatten samples and compute mean, covariance
        flat_samples = samples.view(samples.shape[0], -1)
        mu = torch.mean(flat_samples, dim=0)
        sigma = torch.cov(flat_samples.T)
        return mu, sigma
    
    mu1, sigma1 = compute_statistics(real_samples)
    mu2, sigma2 = compute_statistics(generated_samples)
    
    # Simplified FID calculation
    diff = mu1 - mu2
    fid = torch.sum(diff ** 2) + torch.trace(sigma1 + sigma2 - 2 * torch.sqrt(sigma1 @ sigma2))
    
    return fid.item()

def adaptive_beta_schedule(epoch, total_epochs, base_beta, reconstruction_quality):
    """Adaptive beta scheduling based on reconstruction quality"""
    if reconstruction_quality > 0.1:  # Poor reconstruction
        return base_beta * 0.5  # Reduce beta to focus on reconstruction
    elif reconstruction_quality < 0.01:  # Very good reconstruction
        return base_beta * 2.0  # Increase beta for better disentanglement
    else:
        return base_beta  # Keep current beta

def main():
    parser = argparse.ArgumentParser(description='Train Simple Spectrogram VAE')
    parser.add_argument('--data_dir', required=True, help="Path to spectrogram directory")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")  # Reduced default
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--latent_dim', type=int, default=256, help="Latent dimension")  # Reduced default
    parser.add_argument('--beta', type=float, default=0.01, help="Beta parameter for β-VAE") # 0.001 previously
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