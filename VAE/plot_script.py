import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np
from pathlib import Path

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_training_log(log_path):
    """Load the training log CSV file"""
    try:
        df = pd.read_csv(log_path)
        print(f"Loaded training log with {len(df)} epochs")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading log file: {e}")
        return None

def plot_losses(df, save_path=None):
    """Plot training and validation losses"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total Loss
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Total VAE Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction Loss
    axes[0, 1].plot(df['epoch'], df['train_recon_loss'], label='Train Recon Loss', linewidth=2)
    axes[0, 1].plot(df['epoch'], df['val_recon_loss'], label='Val Recon Loss', linewidth=2)
    axes[0, 1].set_title('Reconstruction Loss (MSE)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL Divergence Loss
    axes[1, 0].plot(df['epoch'], df['train_kl_loss'], label='Train KL Loss', linewidth=2)
    axes[1, 0].plot(df['epoch'], df['val_kl_loss'], label='Val KL Loss', linewidth=2)
    axes[1, 0].set_title('KL Divergence Loss', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].semilogy(df['epoch'], df['lr'], linewidth=2, color='orange')
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate (log scale)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss plots saved to: {save_path}")
    
    plt.show()

def plot_loss_components(df, save_path=None):
    """Plot the components of the VAE loss to understand the trade-off"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss ratio plot
    recon_ratio = df['train_recon_loss'] / df['train_loss']
    kl_ratio = df['train_kl_loss'] / df['train_loss']
    
    axes[0].plot(df['epoch'], recon_ratio, label='Reconstruction Ratio', linewidth=2)
    axes[0].plot(df['epoch'], kl_ratio, label='KL Ratio', linewidth=2)
    axes[0].set_title('Loss Component Ratios', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Ratio of Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Stacked area plot
    axes[1].fill_between(df['epoch'], 0, df['train_recon_loss'], 
                        alpha=0.7, label='Reconstruction Loss')
    axes[1].fill_between(df['epoch'], df['train_recon_loss'], 
                        df['train_recon_loss'] + df['train_kl_loss'],
                        alpha=0.7, label='KL Divergence Loss')
    axes[1].set_title('Loss Components (Stacked)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss component plots saved to: {save_path}")
    
    plt.show()

def plot_training_dynamics(df, save_path=None):
    """Plot training dynamics and convergence"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss difference (overfitting indicator)
    loss_diff = df['val_loss'] - df['train_loss']
    axes[0, 0].plot(df['epoch'], loss_diff, linewidth=2, color='red')
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Validation - Training Loss\n(Overfitting Indicator)', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss Difference')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss smoothness (gradient of loss)
    train_loss_smooth = df['train_loss'].rolling(window=5, center=True).mean()
    val_loss_smooth = df['val_loss'].rolling(window=5, center=True).mean()
    
    axes[0, 1].plot(df['epoch'], train_loss_smooth, label='Train Loss (Smoothed)', linewidth=2)
    axes[0, 1].plot(df['epoch'], val_loss_smooth, label='Val Loss (Smoothed)', linewidth=2)
    axes[0, 1].set_title('Smoothed Loss Curves', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training time per epoch
    if 'time_elapsed' in df.columns:
        # Calculate time per epoch
        time_per_epoch = df['time_elapsed'].diff().fillna(df['time_elapsed'].iloc[0])
        axes[1, 0].plot(df['epoch'], time_per_epoch / 60, linewidth=2, color='green')
        axes[1, 0].set_title('Time per Epoch', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (minutes)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Loss improvement rate
    train_loss_improvement = -df['train_loss'].diff()
    val_loss_improvement = -df['val_loss'].diff()
    
    # Smooth the improvement rates
    train_smooth_improvement = train_loss_improvement.rolling(window=10).mean()
    val_smooth_improvement = val_loss_improvement.rolling(window=10).mean()
    
    axes[1, 1].plot(df['epoch'], train_smooth_improvement, 
                   label='Train Improvement', linewidth=2)
    axes[1, 1].plot(df['epoch'], val_smooth_improvement, 
                   label='Val Improvement', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Loss Improvement Rate\n(Smoothed)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Decrease per Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training dynamics plots saved to: {save_path}")
    
    plt.show()

def plot_summary_stats(df, save_path=None):
    """Plot summary statistics and final results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Best epoch identification
    best_epoch = df.loc[df['val_loss'].idxmin(), 'epoch']
    best_val_loss = df['val_loss'].min()
    
    axes[0, 0].plot(df['epoch'], df['val_loss'], linewidth=2)
    axes[0, 0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].axhline(y=best_val_loss, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].scatter([best_epoch], [best_val_loss], color='red', s=100, zorder=5)
    axes[0, 0].set_title(f'Best Model at Epoch {best_epoch}\nVal Loss: {best_val_loss:.4f}', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Validation Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss distribution (histogram)
    axes[0, 1].hist(df['train_loss'], bins=20, alpha=0.6, label='Train Loss', density=True)
    axes[0, 1].hist(df['val_loss'], bins=20, alpha=0.6, label='Val Loss', density=True)
    axes[0, 1].set_title('Loss Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Loss Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Correlation between components
    correlation = np.corrcoef(df['train_recon_loss'], df['train_kl_loss'])[0, 1]
    axes[1, 0].scatter(df['train_recon_loss'], df['train_kl_loss'], alpha=0.6)
    axes[1, 0].set_title(f'Reconstruction vs KL Loss\nCorrelation: {correlation:.3f}', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Reconstruction Loss')
    axes[1, 0].set_ylabel('KL Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training progress (percentage of best loss achieved)
    final_train_loss = df['train_loss'].iloc[-1]
    final_val_loss = df['val_loss'].iloc[-1]
    train_improvement = ((df['train_loss'].iloc[0] - final_train_loss) / 
                        df['train_loss'].iloc[0]) * 100
    val_improvement = ((df['val_loss'].iloc[0] - final_val_loss) / 
                      df['val_loss'].iloc[0]) * 100
    
    # Text summary
    axes[1, 1].axis('off')
    summary_text = f"""
Training Summary:

Total Epochs: {len(df)}
Best Epoch: {best_epoch}

Final Losses:
  Train: {final_train_loss:.4f}
  Validation: {final_val_loss:.4f}

Improvement:
  Train: {train_improvement:.1f}%
  Validation: {val_improvement:.1f}%

Best Validation Loss: {best_val_loss:.4f}

Final Learning Rate: {df['lr'].iloc[-1]:.2e}
"""
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary statistics saved to: {save_path}")
    
    plt.show()

def create_training_report(log_path, output_dir=None):
    """Create a comprehensive training report with all plots"""
    df = load_training_log(log_path)
    if df is None:
        return
    
    if output_dir is None:
        output_dir = Path(log_path).parent / "training_plots"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Creating training report in: {output_dir}")
    
    # Generate all plots
    plot_losses(df, output_dir / "losses.png")
    plot_loss_components(df, output_dir / "loss_components.png")
    plot_training_dynamics(df, output_dir / "training_dynamics.png")
    plot_summary_stats(df, output_dir / "summary_stats.png")
    
    # Save processed data
    df.to_csv(output_dir / "processed_training_log.csv", index=False)
    
    print(f"Training report complete! Files saved in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Plot VAE training metrics')
    parser.add_argument('--log_path', required=True, 
                       help='Path to vae_training_log.csv file')
    parser.add_argument('--output_dir', default=None,
                       help='Directory to save plots (default: same as log file)')
    parser.add_argument('--plot_type', default='all',
                       choices=['losses', 'components', 'dynamics', 'summary', 'all'],
                       help='Type of plot to generate')
    
    args = parser.parse_args()
    
    # Load data
    df = load_training_log(args.log_path)
    if df is None:
        return
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = Path(args.log_path).parent / "training_plots"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots based on selection
    if args.plot_type == 'all':
        create_training_report(args.log_path, args.output_dir)
    elif args.plot_type == 'losses':
        plot_losses(df, output_dir / "losses.png")
    elif args.plot_type == 'components':
        plot_loss_components(df, output_dir / "from audio_generation_model importloss_components.png")
    elif args.plot_type == 'dynamics':
        plot_training_dynamics(df, output_dir / "training_dynamics.png")
    elif args.plot_type == 'summary':
        plot_summary_stats(df, output_dir / "summary_stats.png")

if __name__ == "__main__":
    main()