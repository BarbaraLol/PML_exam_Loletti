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
        
        # Check for extreme values that might indicate training instability
        numeric_cols = ['train_loss', 'val_loss', 'train_recon_loss', 'train_kl_loss']
        for col in numeric_cols:
            if col in df.columns:
                max_val = df[col].max()
                min_val = df[col].min()
                if max_val > 1e6 or min_val < 0:
                    print(f"⚠️  Warning: {col} has extreme values (min: {min_val:.2e}, max: {max_val:.2e})")
        
        return df
    except Exception as e:
        print(f"Error loading log file: {e}")
        return None

def plot_losses_robust(df, save_path=None):
    """Plot training and validation losses with robust handling of extreme values"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Helper function to clip extreme values for better visualization
    def clip_outliers(series, percentile=99):
        upper_bound = np.percentile(series.dropna(), percentile)
        return np.clip(series, 0, upper_bound)
    
    # Total Loss (with clipping for visualization)
    train_loss_clipped = clip_outliers(df['train_loss'])
    val_loss_clipped = clip_outliers(df['val_loss'])
    
    axes[0, 0].plot(df['epoch'], train_loss_clipped, label='Train Loss (clipped)', 
                   linewidth=2, color='blue', alpha=0.7)
    axes[0, 0].plot(df['epoch'], val_loss_clipped, label='Val Loss (clipped)', 
                   linewidth=2, color='red', alpha=0.7)
    axes[0, 0].set_title('Total VAE Loss (Clipped for Visualization)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')  # Log scale often helps with extreme values
    
    # Reconstruction Loss
    train_recon_clipped = clip_outliers(df['train_recon_loss'])
    val_recon_clipped = clip_outliers(df['val_recon_loss'])
    
    axes[0, 1].plot(df['epoch'], train_recon_clipped, label='Train Recon Loss', 
                   linewidth=2, color='blue')
    axes[0, 1].plot(df['epoch'], val_recon_clipped, label='Val Recon Loss', 
                   linewidth=2, color='red')
    axes[0, 1].set_title('Reconstruction Loss (BCE/MSE)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # KL Divergence Loss (often the problematic one)
    train_kl_clipped = clip_outliers(df['train_kl_loss'])
    val_kl_clipped = clip_outliers(df['val_kl_loss'])
    
    axes[0, 2].plot(df['epoch'], train_kl_clipped, label='Train KL Loss', 
                   linewidth=2, color='blue')
    axes[0, 2].plot(df['epoch'], val_kl_clipped, label='Val KL Loss', 
                   linewidth=2, color='red')
    axes[0, 2].set_title('KL Divergence Loss', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('KL Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale('log')
    
    # Learning Rate
    axes[1, 0].semilogy(df['epoch'], df['lr'], linewidth=2, color='orange')
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate (log scale)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Beta parameter (KL weight)
    axes[1, 1].plot(df['epoch'], df['beta'], linewidth=2, color='purple')
    axes[1, 1].set_title('Beta Schedule (KL Weight)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Beta Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Training time analysis
    if 'time_elapsed' in df.columns:
        time_per_epoch = df['time_elapsed'].diff().fillna(df['time_elapsed'].iloc[0]) / 60
        axes[1, 2].plot(df['epoch'], time_per_epoch, linewidth=2, color='green')
        axes[1, 2].set_title('Time per Epoch', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Time (minutes)')
        axes[1, 2].grid(True, alpha=0.3)
        
        avg_time = time_per_epoch.mean()
        axes[1, 2].axhline(y=avg_time, color='red', linestyle='--', 
                          label=f'Avg: {avg_time:.2f}min')
        axes[1, 2].legend()
    else:
        axes[1, 2].text(0.5, 0.5, 'No timing data\navailable', 
                       transform=axes[1, 2].transAxes, ha='center', va='center',
                       fontsize=12)
        axes[1, 2].set_title('Time per Epoch', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss plots saved to: {save_path}")
    
    plt.show()

def plot_training_stability(df, save_path=None):
    """Analyze training stability and identify problematic epochs"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Identify epochs with extreme losses
    extreme_threshold = np.percentile(df['train_loss'].dropna(), 95)
    extreme_epochs = df[df['train_loss'] > extreme_threshold]['epoch']
    
    # Loss stability over time
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', alpha=0.7, color='blue')
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', alpha=0.7, color='red')
    
    # Highlight problematic epochs
    for epoch in extreme_epochs:
        axes[0, 0].axvline(x=epoch, color='orange', alpha=0.5, linestyle='--')
    
    axes[0, 0].set_title('Loss Over Time (Extreme Epochs Highlighted)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # KL vs Reconstruction balance
    # Calculate ratios where total loss is reasonable
    reasonable_mask = df['train_loss'] < extreme_threshold
    df_reasonable = df[reasonable_mask]
    
    if len(df_reasonable) > 0:
        recon_ratio = df_reasonable['train_recon_loss'] / df_reasonable['train_loss']
        kl_ratio = df_reasonable['train_kl_loss'] / df_reasonable['train_loss']
        
        axes[0, 1].plot(df_reasonable['epoch'], recon_ratio, label='Reconstruction Ratio', 
                       linewidth=2, color='green')
        axes[0, 1].plot(df_reasonable['epoch'], kl_ratio, label='KL Ratio', 
                       linewidth=2, color='purple')
        axes[0, 1].set_title('Loss Component Balance\n(Only "reasonable" epochs)', 
                           fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Ratio of Total Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
    
    # Beta vs KL loss relationship
    axes[1, 0].scatter(df['beta'], df['train_kl_loss'], alpha=0.6, c=df['epoch'], 
                      cmap='viridis', s=30)
    axes[1, 0].set_title('Beta vs KL Loss\n(Colored by Epoch)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Beta (KL Weight)')
    axes[1, 0].set_ylabel('KL Loss')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gradient norm (if available)
    if 'grad_norm' in df.columns and not df['grad_norm'].isna().all():
        grad_norm_clean = df['grad_norm'].dropna()
        if len(grad_norm_clean) > 0:
            axes[1, 1].plot(df['epoch'], df['grad_norm'], linewidth=2, color='red', alpha=0.7)
            axes[1, 1].set_title('Gradient Norm', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
    else:
        axes[1, 1].text(0.5, 0.5, 'No gradient norm\ndata available', 
                       transform=axes[1, 1].transAxes, ha='center', va='center',
                       fontsize=12)
        axes[1, 1].set_title('Gradient Norm', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training stability plots saved to: {save_path}")
    
    plt.show()

def diagnose_training_issues(df):
    """Diagnose potential training issues and provide suggestions"""
    print("\n🔍 TRAINING DIAGNOSTICS")
    print("=" * 50)
    
    # Check for exploding losses
    max_train_loss = df['train_loss'].max()
    max_kl_loss = df['train_kl_loss'].max()
    
    if max_train_loss > 1e6:
        print(f"❌ EXPLODING LOSSES detected!")
        print(f"   Max train loss: {max_train_loss:.2e}")
        print(f"   This usually indicates:")
        print(f"   - Learning rate too high")
        print(f"   - Beta schedule too aggressive")
        print(f"   - Numerical instability")
    
    if max_kl_loss > 1e6:
        print(f"❌ EXPLODING KL LOSS detected!")
        print(f"   Max KL loss: {max_kl_loss:.2e}")
        print(f"   This usually indicates:")
        print(f"   - Posterior collapse")
        print(f"   - Beta warmup too fast")
        print(f"   - Need KL annealing")
    
    # Check final performance
    final_train_loss = df['train_loss'].iloc[-1]
    final_val_loss = df['val_loss'].iloc[-1]
    
    print(f"\n📊 FINAL PERFORMANCE:")
    print(f"   Final train loss: {final_train_loss:.4f}")
    print(f"   Final val loss: {final_val_loss:.4f}")
    
    if final_val_loss > final_train_loss * 2:
        print(f"⚠️  Possible overfitting detected")
    elif final_val_loss < final_train_loss * 0.5:
        print(f"⚠️  Unusual validation behavior - check data")
    else:
        print(f"✅ Reasonable train/val loss balance")
    
    # Check for convergence
    recent_epochs = min(5, len(df))
    recent_train_loss = df['train_loss'].tail(recent_epochs)
    train_trend = recent_train_loss.diff().mean()
    
    if train_trend < -0.1:
        print(f"📈 Training still improving (trend: {train_trend:.4f})")
    elif abs(train_trend) < 0.01:
        print(f"✅ Training appears to have converged")
    else:
        print(f"⚠️  Training may be diverging (trend: {train_trend:.4f})")
    
    # Beta schedule analysis
    final_beta = df['beta'].iloc[-1]
    print(f"\n🎛️  HYPERPARAMETER ANALYSIS:")
    print(f"   Final learning rate: {df['lr'].iloc[-1]:.6f}")
    print(f"   Final beta: {final_beta:.6f}")
    
    if final_beta > 0.01:
        print(f"   ℹ️  High beta value - strong KL regularization")
    elif final_beta < 0.0001:
        print(f"   ℹ️  Low beta value - weak KL regularization")
    
    print("\n💡 RECOMMENDATIONS:")
    if max_train_loss > 1e6:
        print("   - Reduce learning rate by 10x")
        print("   - Use slower beta warmup")
        print("   - Add gradient clipping")
        print("   - Check for NaN values in data")
    
    if max_kl_loss > max_train_loss * 100:
        print("   - Implement beta annealing (start from 0)")
        print("   - Reduce KL weight (beta) significantly")
        print("   - Check encoder output ranges")
    
    print("=" * 50)

def create_training_analysis_report(log_path, output_dir=None):
    """Create a comprehensive training analysis report"""
    df = load_training_log(log_path)
    if df is None:
        return
    
    if output_dir is None:
        output_dir = Path(log_path).parent / "training_analysis"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Creating comprehensive training analysis in: {output_dir}")
    
    # Run diagnostics
    diagnose_training_issues(df)
    
    # Generate plots
    print("\n📊 Generating robust loss plots...")
    plot_losses_robust(df, output_dir / "01_losses_robust.png")
    
    print("📊 Generating training stability analysis...")
    plot_training_stability(df, output_dir / "02_training_stability.png")
    
    # Save processed data
    df.to_csv(output_dir / "processed_training_log.csv", index=False)
    
    # Create analysis summary
    create_analysis_html_report(df, output_dir)
    
    print(f"✅ Training analysis complete! Files saved in: {output_dir}")
    print(f"🌐 Open {output_dir / 'analysis_report.html'} in your browser")

def create_analysis_html_report(df, output_dir):
    """Create HTML report with training analysis"""
    max_loss = df['train_loss'].max()
    final_loss = df['train_loss'].iloc[-1]
    max_kl = df['train_kl_loss'].max()
    
    # Determine status
    if max_loss > 1e6:
        status = "🚨 TRAINING ISSUES DETECTED"
        status_class = "bad"
    elif final_loss < df['train_loss'].iloc[0] * 0.1:
        status = "✅ TRAINING SUCCESSFUL"
        status_class = "good"
    else:
        status = "⚠️ TRAINING NEEDS ATTENTION"
        status_class = "warning"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>VAE Training Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric {{ background: #f0f0f0; padding: 10px; margin: 5px 0; border-radius: 5px; }}
            .good {{ background: #d4edda; }}
            .warning {{ background: #fff3cd; }}
            .bad {{ background: #f8d7da; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            .code {{ background: #f8f9fa; padding: 10px; border-radius: 5px; font-family: monospace; }}
        </style>
    </head>
    <body>
        <h1>🤖 VAE Training Analysis Report</h1>
        
        <div class="metric {status_class}">
            <strong>Overall Status:</strong> {status}
        </div>
        
        <h2>📈 Key Metrics</h2>
        <div class="metric">
            <strong>Total Epochs:</strong> {len(df)}
        </div>
        <div class="metric {'bad' if max_loss > 1e6 else 'good'}">
            <strong>Max Training Loss:</strong> {max_loss:.2e}
        </div>
        <div class="metric {'bad' if max_kl > 1e6 else 'good'}">
            <strong>Max KL Loss:</strong> {max_kl:.2e}
        </div>
        <div class="metric">
            <strong>Final Training Loss:</strong> {final_loss:.4f}
        </div>
        <div class="metric">
            <strong>Final Validation Loss:</strong> {df['val_loss'].iloc[-1]:.4f}
        </div>
        
        <h2>📊 Visualizations</h2>
        <h3>Robust Loss Analysis</h3>
        <img src="01_losses_robust.png" alt="Robust loss curves">
        
        <h3>Training Stability</h3>
        <img src="02_training_stability.png" alt="Training stability analysis">
        
        <h2>💡 Recommendations</h2>
        {'<div class="metric bad">Consider restarting training with lower learning rate and slower beta warmup.</div>' if max_loss > 1e6 else ''}
        {'<div class="metric warning">Monitor KL loss - consider beta annealing strategy.</div>' if max_kl > 1e6 else ''}
        {'<div class="metric good">Training appears stable. Consider fine-tuning hyperparameters for better performance.</div>' if max_loss < 1e6 else ''}
        
        <hr>
        <small>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
    </body>
    </html>
    """
    
    with open(output_dir / "analysis_report.html", "w") as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description='Analyze VAE Training with Robust Handling')
    parser.add_argument('--log_path', required=True, 
                       help='Path to training log CSV file')
    parser.add_argument('--output_dir', default=None,
                       help='Directory to save analysis (default: same as log file)')
    parser.add_argument('--analysis_type', default='full',
                       choices=['losses', 'stability', 'full'],
                       help='Type of analysis to perform')
    
    args = parser.parse_args()
    
    # Load data
    df = load_training_log(args.log_path)
    if df is None:
        return
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = Path(args.log_path).parent / "training_analysis"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Run analysis
    if args.analysis_type == 'full':
        create_training_analysis_report(args.log_path, args.output_dir)
    elif args.analysis_type == 'losses':
        plot_losses_robust(df, output_dir / "losses_robust.png")
    elif args.analysis_type == 'stability':
        plot_training_stability(df, output_dir / "training_stability.png")

if __name__ == "__main__":
    main()