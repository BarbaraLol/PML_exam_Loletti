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
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2, color='blue')
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2, color='red')
    axes[0, 0].set_title('Total VAE Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction Loss
    axes[0, 1].plot(df['epoch'], df['train_recon_loss'], label='Train Recon Loss', 
                   linewidth=2, color='blue')
    axes[0, 1].plot(df['epoch'], df['val_recon_loss'], label='Val Recon Loss', 
                   linewidth=2, color='red')
    axes[0, 1].set_title('Reconstruction Loss (BCE/MSE)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL Divergence Loss
    axes[1, 0].plot(df['epoch'], df['train_kl_loss'], label='Train KL Loss', 
                   linewidth=2, color='blue')
    axes[1, 0].plot(df['epoch'], df['val_kl_loss'], label='Val KL Loss', 
                   linewidth=2, color='red')
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss ratio plot
    recon_ratio = df['train_recon_loss'] / df['train_loss']
    kl_ratio = df['train_kl_loss'] / df['train_loss']
    
    axes[0].plot(df['epoch'], recon_ratio, label='Reconstruction Ratio', linewidth=2, color='green')
    axes[0].plot(df['epoch'], kl_ratio, label='KL Ratio', linewidth=2, color='purple')
    axes[0].set_title('Loss Component Ratios\n(What % of total loss)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Ratio of Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # Absolute loss components
    axes[1].plot(df['epoch'], df['train_recon_loss'], label='Reconstruction Loss', 
                linewidth=2, color='green')
    axes[1].plot(df['epoch'], df['train_kl_loss'], label='KL Divergence Loss', 
                linewidth=2, color='purple')
    axes[1].set_title('Absolute Loss Components', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KL vs Reconstruction scatter (to see relationship)
    axes[2].scatter(df['train_recon_loss'], df['train_kl_loss'], 
                   alpha=0.6, c=df['epoch'], cmap='viridis', s=30)
    axes[2].set_title('KL vs Reconstruction\n(Colored by Epoch)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Reconstruction Loss')
    axes[2].set_ylabel('KL Loss')
    axes[2].grid(True, alpha=0.3)
    
    # Add colorbar for epoch
    cbar = plt.colorbar(axes[2].collections[0], ax=axes[2])
    cbar.set_label('Epoch')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss component plots saved to: {save_path}")
    
    plt.show()

def plot_training_dynamics(df, save_path=None):
    """Plot training dynamics and convergence indicators"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Overfitting indicator
    loss_diff = df['val_loss'] - df['train_loss']
    axes[0, 0].plot(df['epoch'], loss_diff, linewidth=2, color='red')
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].fill_between(df['epoch'], loss_diff, 0, where=(loss_diff > 0), 
                           color='red', alpha=0.3, label='Overfitting')
    axes[0, 0].fill_between(df['epoch'], loss_diff, 0, where=(loss_diff <= 0), 
                           color='green', alpha=0.3, label='Good fit')
    axes[0, 0].set_title('Overfitting Indicator\n(Val Loss - Train Loss)', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss Difference')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss smoothness (rolling average)
    window_size = max(1, len(df) // 20)
    train_loss_smooth = df['train_loss'].rolling(window=window_size, center=True).mean()
    val_loss_smooth = df['val_loss'].rolling(window=window_size, center=True).mean()
    
    axes[0, 1].plot(df['epoch'], df['train_loss'], alpha=0.3, color='blue', label='Train (raw)')
    axes[0, 1].plot(df['epoch'], df['val_loss'], alpha=0.3, color='red', label='Val (raw)')
    axes[0, 1].plot(df['epoch'], train_loss_smooth, linewidth=2, color='blue', label='Train (smooth)')
    axes[0, 1].plot(df['epoch'], val_loss_smooth, linewidth=2, color='red', label='Val (smooth)')
    axes[0, 1].set_title('Smoothed Loss Curves', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training time analysis
    if 'time_elapsed' in df.columns:
        # Calculate time per epoch
        time_per_epoch = df['time_elapsed'].diff().fillna(df['time_elapsed'].iloc[0]) / 60
        axes[1, 0].plot(df['epoch'], time_per_epoch, linewidth=2, color='green')
        axes[1, 0].set_title('Time per Epoch', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (minutes)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add average line
        avg_time = time_per_epoch.mean()
        axes[1, 0].axhline(y=avg_time, color='red', linestyle='--', 
                          label=f'Avg: {avg_time:.2f}min')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No timing data\navailable', 
                       transform=axes[1, 0].transAxes, ha='center', va='center',
                       fontsize=12)
        axes[1, 0].set_title('Time per Epoch', fontsize=14, fontweight='bold')
    
    # Loss improvement rate (derivative)
    train_loss_improvement = -df['train_loss'].diff().rolling(window=5).mean()
    val_loss_improvement = -df['val_loss'].diff().rolling(window=5).mean()
    
    axes[1, 1].plot(df['epoch'], train_loss_improvement, 
                   label='Train Improvement Rate', linewidth=2, color='blue')
    axes[1, 1].plot(df['epoch'], val_loss_improvement, 
                   label='Val Improvement Rate', linewidth=2, color='red')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Loss Improvement Rate\n(Negative of derivative, smoothed)', 
                        fontsize=14, fontweight='bold')
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
    final_val_loss = df['val_loss'].iloc[-1]
    
    axes[0, 0].plot(df['epoch'], df['val_loss'], linewidth=2, color='red')
    axes[0, 0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, 
                      label=f'Best epoch: {best_epoch}')
    axes[0, 0].axhline(y=best_val_loss, color='green', linestyle='--', alpha=0.7)
    axes[0, 0].scatter([best_epoch], [best_val_loss], color='green', s=100, zorder=5)
    axes[0, 0].scatter([df['epoch'].iloc[-1]], [final_val_loss], color='red', s=100, zorder=5,
                      label=f'Final: {final_val_loss:.4f}')
    axes[0, 0].set_title(f'Validation Loss Progress\nBest: {best_val_loss:.4f} at epoch {best_epoch}', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss distribution comparison
    axes[0, 1].hist(df['train_loss'], bins=20, alpha=0.6, label='Train Loss', 
                   density=True, color='blue')
    axes[0, 1].hist(df['val_loss'], bins=20, alpha=0.6, label='Val Loss', 
                   density=True, color='red')
    axes[0, 1].axvline(x=df['train_loss'].mean(), color='blue', linestyle='--', 
                      label=f'Train mean: {df["train_loss"].mean():.4f}')
    axes[0, 1].axvline(x=df['val_loss'].mean(), color='red', linestyle='--',
                      label=f'Val mean: {df["val_loss"].mean():.4f}')
    axes[0, 1].set_title('Loss Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Loss Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Component correlation analysis
    correlation = np.corrcoef(df['train_recon_loss'], df['train_kl_loss'])[0, 1]
    scatter = axes[1, 0].scatter(df['train_recon_loss'], df['train_kl_loss'], 
                                alpha=0.6, c=df['epoch'], cmap='viridis', s=30)
    axes[1, 0].set_title(f'Reconstruction vs KL Loss\nCorrelation: {correlation:.3f}', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Reconstruction Loss')
    axes[1, 0].set_ylabel('KL Loss')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0], label='Epoch')
    
    # Training summary text
    axes[1, 1].axis('off')
    
    # Calculate improvements
    initial_train_loss = df['train_loss'].iloc[0]
    final_train_loss = df['train_loss'].iloc[-1]
    initial_val_loss = df['val_loss'].iloc[0]
    
    train_improvement = ((initial_train_loss - final_train_loss) / initial_train_loss) * 100
    val_improvement = ((initial_val_loss - best_val_loss) / initial_val_loss) * 100
    
    summary_text = f"""
TRAINING SUMMARY
================

Total Epochs: {len(df)}
Best Epoch: {best_epoch}

FINAL LOSSES:
Train: {final_train_loss:.4f}
Validation: {final_val_loss:.4f}
Best Val: {best_val_loss:.4f}

IMPROVEMENTS:
Train: {train_improvement:.1f}%
Val: {val_improvement:.1f}%

COMPONENTS (Final):
Recon: {df['train_recon_loss'].iloc[-1]:.4f}
KL: {df['train_kl_loss'].iloc[-1]:.4f}

Final LR: {df['lr'].iloc[-1]:.2e}

STATUS: {'✓ CONVERGED' if patience_check(df) else '⚠ MAY NEED MORE TRAINING'}
"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary statistics saved to: {save_path}")
    
    plt.show()

def patience_check(df, window=10):
    """Check if model has converged based on recent loss trends"""
    if len(df) < window:
        return False
    
    recent_val_loss = df['val_loss'].iloc[-window:]
    trend = np.polyfit(range(len(recent_val_loss)), recent_val_loss, 1)[0]
    
    # If slope is very small, consider converged
    return abs(trend) < 0.001

def create_training_report(log_path, output_dir=None):
    """Create a comprehensive training report with all plots"""
    df = load_training_log(log_path)
    if df is None:
        return
    
    if output_dir is None:
        output_dir = Path(log_path).parent / "training_plots"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Creating comprehensive training report in: {output_dir}")
    print(f"Report will include {len(df)} epochs of training data")
    
    # Generate all plots
    print("📊 Generating loss plots...")
    plot_losses(df, output_dir / "01_losses.png")
    
    print("📊 Generating loss component analysis...")
    plot_loss_components(df, output_dir / "02_loss_components.png")
    
    print("📊 Generating training dynamics...")
    plot_training_dynamics(df, output_dir / "03_training_dynamics.png")
    
    print("📊 Generating summary statistics...")
    plot_summary_stats(df, output_dir / "04_summary_stats.png")
    
    # Save processed data
    df.to_csv(output_dir / "processed_training_log.csv", index=False)
    
    # Create a simple HTML report
    create_html_report(df, output_dir)
    
    print(f"✅ Training report complete! Files saved in: {output_dir}")
    print(f"🌐 Open {output_dir / 'report.html'} in your browser for a summary")

def create_html_report(df, output_dir):
    """Create a simple HTML report"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>VAE Training Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric {{ background: #f0f0f0; padding: 10px; margin: 5px 0; border-radius: 5px; }}
            .good {{ background: #d4edda; }}
            .warning {{ background: #fff3cd; }}
            .bad {{ background: #f8d7da; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>🤖 VAE Training Report</h1>
        
        <h2>📈 Key Metrics</h2>
        <div class="metric {'good' if df['val_loss'].iloc[-1] < df['val_loss'].iloc[0] else 'bad'}">
            <strong>Final Validation Loss:</strong> {df['val_loss'].iloc[-1]:.4f}
        </div>
        <div class="metric">
            <strong>Best Validation Loss:</strong> {df['val_loss'].min():.4f} (Epoch {df.loc[df['val_loss'].idxmin(), 'epoch']})
        </div>
        <div class="metric">
            <strong>Total Epochs:</strong> {len(df)}
        </div>
        <div class="metric {'good' if patience_check(df) else 'warning'}">
            <strong>Convergence Status:</strong> {'Converged ✅' if patience_check(df) else 'May need more training ⚠️'}
        </div>
        
        <h2>📊 Visualizations</h2>
        <h3>Loss Curves</h3>
        <img src="01_losses.png" alt="Loss curves">
        
        <h3>Loss Components</h3>
        <img src="02_loss_components.png" alt="Loss components">
        
        <h3>Training Dynamics</h3>
        <img src="03_training_dynamics.png" alt="Training dynamics">
        
        <h3>Summary Statistics</h3>
        <img src="04_summary_stats.png" alt="Summary statistics">
        
        <h2>🔍 Analysis</h2>
        <p><strong>Reconstruction vs KL Balance:</strong> 
        The final reconstruction loss is {df['train_recon_loss'].iloc[-1]:.4f} 
        and KL loss is {df['train_kl_loss'].iloc[-1]:.4f}. 
        {'This suggests good balance.' if abs(df['train_recon_loss'].iloc[-1] / df['train_kl_loss'].iloc[-1] - 1) < 2 else 'Consider adjusting beta parameter.'}
        </p>
        
        <p><strong>Overfitting Check:</strong> 
        {'No significant overfitting detected.' if (df['val_loss'] - df['train_loss']).iloc[-1] < 0.1 else 'Some overfitting may be present.'}
        </p>
        
        <hr>
        <small>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
    </body>
    </html>
    """
    
    with open(output_dir / "report.html", "w") as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description='Plot VAE Training Metrics')
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
        plot_loss_components(df, output_dir / "loss_components.png")
    elif args.plot_type == 'dynamics':
        plot_training_dynamics(df, output_dir / "training_dynamics.png")
    elif args.plot_type == 'summary':
        plot_summary_stats(df, output_dir / "summary_stats.png")

if __name__ == "__main__":
    main()