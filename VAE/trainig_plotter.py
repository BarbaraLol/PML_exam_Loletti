import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from pathlib import Path
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11

def load_training_data(log_path):
    """Load and clean training log data"""
    try:
        df = pd.read_csv(log_path)
        print(f"Loaded training data with {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        
        # Handle any infinite or extremely large values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Replace infinite values with NaN, then fill with reasonable values
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if df[col].isna().any():
                print(f"Warning: Found NaN/inf values in {col}, filling with median")
                df[col] = df[col].fillna(df[col].median())
        
        return df
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None

def load_latent_data(log_path):
    """Load and clean latent statistics data"""
    try:
        df = pd.read_csv(log_path)
        print(f"Loaded latent data with {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        
        # Handle any infinite or extremely large values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if df[col].isna().any():
                print(f"Warning: Found NaN/inf values in {col}, filling with median")
                df[col] = df[col].fillna(df[col].median())
        
        return df
    except Exception as e:
        print(f"Error loading latent data: {e}")
        return None

def plot_training_metrics(training_df, save_dir):
    """Create comprehensive training metrics plots"""
    print("📊 Creating training metrics plots...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Loss curves (log scale)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.semilogy(training_df['epoch'], training_df['train_loss'], 
                 label='Training Loss', linewidth=2.5, alpha=0.8)
    ax1.semilogy(training_df['epoch'], training_df['val_loss'], 
                 label='Validation Loss', linewidth=2.5, alpha=0.8)
    ax1.set_title('📈 Training & Validation Loss (Log Scale)', fontweight='bold', fontsize=16)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. KL Divergence
    ax2 = fig.add_subplot(gs[0, 2])
    # Use log scale for KL if values are very large
    max_kl = max(training_df['train_kl_loss'].max(), training_df['val_kl_loss'].max())
    if max_kl > 1000:
        ax2.semilogy(training_df['epoch'], training_df['train_kl_loss'], 
                     label='Train KL', linewidth=2, alpha=0.8)
        ax2.semilogy(training_df['epoch'], training_df['val_kl_loss'], 
                     label='Val KL', linewidth=2, alpha=0.8)
        ax2.set_ylabel('KL Loss (log scale)')
    else:
        ax2.plot(training_df['epoch'], training_df['train_kl_loss'], 
                 label='Train KL', linewidth=2, alpha=0.8)
        ax2.plot(training_df['epoch'], training_df['val_kl_loss'], 
                 label='Val KL', linewidth=2, alpha=0.8)
        ax2.set_ylabel('KL Loss')
    
    ax2.set_title('🔄 KL Divergence', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Reconstruction Loss
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(training_df['epoch'], training_df['train_recon_loss'], 
             label='Training Reconstruction', linewidth=2.5, alpha=0.8)
    ax3.plot(training_df['epoch'], training_df['val_recon_loss'], 
             label='Validation Reconstruction', linewidth=2.5, alpha=0.8)
    ax3.set_title('🔧 Reconstruction Loss Evolution', fontweight='bold', fontsize=16)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Reconstruction Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Learning Rate and Beta
    ax4 = fig.add_subplot(gs[1, 2])
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(training_df['epoch'], training_df['lr'], 
                     'b-', label='Learning Rate', linewidth=2.5)
    line2 = ax4_twin.plot(training_df['epoch'], training_df['beta'], 
                          'r-', label='Beta (β)', linewidth=2.5)
    
    ax4.set_title('⚙️ LR & β Schedule', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate', color='b')
    ax4_twin.set_ylabel('Beta (β)', color='r')
    ax4.tick_params(axis='y', labelcolor='b')
    ax4_twin.tick_params(axis='y', labelcolor='r')
    ax4.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    
    # 5. Loss components breakdown
    ax5 = fig.add_subplot(gs[2, :])
    epochs = training_df['epoch']
    
    # Stack plot of loss components
    recon_loss = training_df['train_recon_loss']
    kl_loss = training_df['train_kl_loss'] * training_df['beta']  # Weighted KL
    
    ax5.fill_between(epochs, 0, recon_loss, alpha=0.7, label='Reconstruction Loss')
    ax5.fill_between(epochs, recon_loss, recon_loss + kl_loss, alpha=0.7, label='Weighted KL Loss')
    ax5.plot(epochs, training_df['train_loss'], 'k-', linewidth=3, label='Total Loss')
    
    ax5.set_title('📊 Training Loss Components Breakdown', fontweight='bold', fontsize=16)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Training summary statistics
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    # Calculate statistics
    final_train_loss = training_df['train_loss'].iloc[-1]
    final_val_loss = training_df['val_loss'].iloc[-1]
    final_recon_loss = training_df['val_recon_loss'].iloc[-1]
    final_kl_loss = training_df['val_kl_loss'].iloc[-1]
    final_lr = training_df['lr'].iloc[-1]
    final_beta = training_df['beta'].iloc[-1]
    total_epochs = training_df['epoch'].max()
    
    # Best validation loss
    best_val_epoch = training_df.loc[training_df['val_loss'].idxmin(), 'epoch']
    best_val_loss = training_df['val_loss'].min()
    
    summary_text = f"""
TRAINING SUMMARY
{'='*50}

FINAL METRICS (Epoch {total_epochs}):
• Training Loss: {final_train_loss:.4f}
• Validation Loss: {final_val_loss:.4f}
• Reconstruction Loss: {final_recon_loss:.4f}
• KL Divergence: {final_kl_loss:.4f}
• Learning Rate: {final_lr:.2e}
• Beta (β): {final_beta:.4f}

BEST PERFORMANCE:
• Best Validation Loss: {best_val_loss:.4f} (Epoch {best_val_epoch})
• Improvement: {((training_df['val_loss'].iloc[0] - best_val_loss) / training_df['val_loss'].iloc[0] * 100):.1f}%

TRAINING STABILITY:
• Final 10 epochs val loss std: {training_df['val_loss'].tail(10).std():.4f}
• Training converged: {'✅ Yes' if training_df['val_loss'].tail(10).std() < training_df['val_loss'].std() * 0.1 else '⚠️ Still learning'}
"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('🧠 VAE Training Metrics Analysis', fontsize=20, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Training metrics saved to: {save_dir}/training_metrics.png")

def plot_latent_analysis(latent_df, save_dir):
    """Create comprehensive latent space analysis plots"""
    print("🎯 Creating latent space analysis plots...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Calculate epoch progression (handle multiple batches per epoch)
    latent_df['epoch_progress'] = latent_df['epoch'] + latent_df['batch'] / latent_df['batch'].max()
    
    # 1. Latent space convergence to N(0,1)
    ax1 = fig.add_subplot(gs[0, :])
    
    ax1.plot(latent_df['epoch_progress'], latent_df['mu_mean'], 
             label='μ Mean', linewidth=2.5, alpha=0.8)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Target μ = 0')
    ax1.fill_between(latent_df['epoch_progress'], -0.1, 0.1, alpha=0.2, color='green', label='Good range')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(latent_df['epoch_progress'], latent_df['actual_var'], 
                  'orange', label='σ² (Variance)', linewidth=2.5, alpha=0.8)
    ax1_twin.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Target σ² = 1')
    ax1_twin.fill_between(latent_df['epoch_progress'], 0.8, 1.2, alpha=0.2, color='green')
    
    ax1.set_title('🎯 Latent Space Convergence to Standard Normal N(0,1)', fontweight='bold', fontsize=16)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('μ Mean', color='blue')
    ax1_twin.set_ylabel('σ² (Variance)', color='orange')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_twin.tick_params(axis='y', labelcolor='orange')
    ax1.grid(True, alpha=0.3)
    
    # Combined legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # 2. Latent statistics evolution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(latent_df['epoch_progress'], latent_df['mu_mean'], linewidth=2, alpha=0.8)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_title('μ Mean Evolution', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('μ Mean')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(latent_df['epoch_progress'], latent_df['mu_std'], 'green', linewidth=2, alpha=0.8)
    ax3.set_title('μ Standard Deviation', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('μ Std')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(latent_df['epoch_progress'], latent_df['logvar_mean'], 'purple', linewidth=2, alpha=0.8)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Target = 0')
    ax4.set_title('log(σ²) Mean', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('log(σ²) Mean')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 3. Distance from ideal N(0,1)
    ax5 = fig.add_subplot(gs[2, :2])
    
    # Calculate distance metric: sqrt(μ² + (σ² - 1)²)
    distance_from_ideal = np.sqrt(latent_df['mu_mean']**2 + (latent_df['actual_var'] - 1)**2)
    
    ax5.plot(latent_df['epoch_progress'], distance_from_ideal, 
             linewidth=3, alpha=0.8, color='darkred')
    ax5.set_title('📏 Distance from Ideal N(0,1) Distribution', fontweight='bold', fontsize=14)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('√(μ² + (σ² - 1)²)')
    ax5.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(latent_df['epoch_progress'], distance_from_ideal, 1)
    p = np.poly1d(z)
    ax5.plot(latent_df['epoch_progress'], p(latent_df['epoch_progress']), 
             'r--', alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.4f})')
    ax5.legend()
    
    # 4. Quality score over time
    ax6 = fig.add_subplot(gs[2, 2])
    quality_score = 1 / (1 + distance_from_ideal)
    ax6.plot(latent_df['epoch_progress'], quality_score, 
             linewidth=3, alpha=0.8, color='darkgreen')
    ax6.set_title('🏆 Quality Score', fontweight='bold')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Score (0-1)')
    ax6.set_ylim(0, 1)
    ax6.grid(True, alpha=0.3)
    
    # 5. Latent space analysis summary
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis('off')
    
    # Calculate final statistics
    final_mu_mean = latent_df['mu_mean'].iloc[-1]
    final_var = latent_df['actual_var'].iloc[-1]
    final_distance = distance_from_ideal.iloc[-1]
    final_quality = quality_score.iloc[-1]
    
    # Improvement metrics
    initial_distance = distance_from_ideal.iloc[0]
    improvement = ((initial_distance - final_distance) / initial_distance) * 100
    
    # Convergence assessment
    mu_status = "✅ EXCELLENT" if abs(final_mu_mean) < 0.1 else ("🟡 GOOD" if abs(final_mu_mean) < 0.3 else "🔴 NEEDS WORK")
    var_status = "✅ EXCELLENT" if 0.8 < final_var < 1.2 else ("🟡 GOOD" if 0.5 < final_var < 1.5 else "🔴 NEEDS WORK")
    
    # Check if converged (stable in recent measurements)
    recent_stability = latent_df['mu_mean'].tail(20).std() if len(latent_df) >= 20 else latent_df['mu_mean'].std()
    converged = "✅ CONVERGED" if recent_stability < 0.05 else "🟡 STILL LEARNING"
    
    summary_text = f"""
LATENT SPACE ANALYSIS SUMMARY
{'='*60}

FINAL VALUES:
• μ Mean: {final_mu_mean:.6f}  {mu_status}
• σ² (Variance): {final_var:.6f}  {var_status}
• Distance from N(0,1): {final_distance:.6f}
• Quality Score: {final_quality:.3f}/1.000

TRAINING PROGRESS:
• Total Training Points: {len(latent_df)}
• Improvement: {improvement:.1f}%
• Convergence Status: {converged}
• Recent Stability (μ std): {recent_stability:.6f}

TARGETS vs ACTUAL:
• μ ≈ 0 (actual: {abs(final_mu_mean):.6f})
• σ² ≈ 1 (actual: {final_var:.6f})

OVERALL ASSESSMENT:
{'🎯 EXCELLENT! Your VAE has learned a high-quality latent representation.' if final_quality > 0.9 else 
 ('👍 GOOD! The latent space is well-structured.' if final_quality > 0.7 else 
  '⚠️ NEEDS IMPROVEMENT. Consider adjusting β or training longer.')}
"""
    
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('🎯 VAE Latent Space Analysis', fontsize=20, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(save_dir, 'latent_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Latent analysis saved to: {save_dir}/latent_analysis.png")

def plot_combined_overview(training_df, latent_df, save_dir):
    """Create a combined overview plot"""
    print("📊 Creating combined overview plot...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Loss overview
    ax1.semilogy(training_df['epoch'], training_df['train_loss'], 
                 label='Training', linewidth=2.5, alpha=0.8)
    ax1.semilogy(training_df['epoch'], training_df['val_loss'], 
                 label='Validation', linewidth=2.5, alpha=0.8)
    ax1.set_title('📈 Loss Overview', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Latent convergence
    latent_df['epoch_progress'] = latent_df['epoch'] + latent_df['batch'] / latent_df['batch'].max()
    ax2.plot(latent_df['epoch_progress'], latent_df['mu_mean'], 
             label='μ Mean', linewidth=2.5)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(latent_df['epoch_progress'], latent_df['actual_var'], 
                  'orange', label='σ² Variance', linewidth=2.5)
    ax2_twin.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
    ax2.set_title('🎯 Latent Convergence', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('μ Mean', color='blue')
    ax2_twin.set_ylabel('σ² Variance', color='orange')
    ax2.grid(True, alpha=0.3)
    
    # 3. KL vs Reconstruction trade-off
    ax3.plot(training_df['epoch'], training_df['train_recon_loss'], 
             label='Reconstruction', linewidth=2.5)
    ax3_twin = ax3.twinx()
    # Scale KL by beta for visualization
    scaled_kl = training_df['train_kl_loss'] * training_df['beta']
    ax3_twin.plot(training_df['epoch'], scaled_kl, 
                  'red', label='Weighted KL', linewidth=2.5)
    
    ax3.set_title('⚖️ Reconstruction vs KL Trade-off', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Reconstruction Loss', color='blue')
    ax3_twin.set_ylabel('Weighted KL Loss', color='red')
    ax3.grid(True, alpha=0.3)
    
    # 4. Training progress indicators
    distance_from_ideal = np.sqrt(latent_df['mu_mean']**2 + (latent_df['actual_var'] - 1)**2)
    ax4.plot(latent_df['epoch_progress'], distance_from_ideal, 
             linewidth=3, alpha=0.8, color='purple')
    ax4.set_title('📏 Overall Quality Progress', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Distance from N(0,1)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('🧠 VAE Training Overview', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(save_dir, 'training_overview.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Overview saved to: {save_dir}/training_overview.png")

def create_html_report(training_df, latent_df, save_dir):
    """Create an HTML report summarizing the training"""
    print("📄 Creating HTML report...")
    
    # Calculate key metrics
    final_train_loss = training_df['train_loss'].iloc[-1]
    final_val_loss = training_df['val_loss'].iloc[-1]
    final_mu_mean = latent_df['mu_mean'].iloc[-1]
    final_var = latent_df['actual_var'].iloc[-1]
    total_epochs = training_df['epoch'].max()
    
    # Quality assessment
    mu_good = abs(final_mu_mean) < 0.1
    var_good = 0.8 < final_var < 1.2
    overall_good = mu_good and var_good
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>VAE Training Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; color: #333; border-bottom: 3px solid #667eea; padding-bottom: 20px; margin-bottom: 30px; }}
            .metric {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #667eea; }}
            .good {{ border-left-color: #28a745; background: #d4f4e4; }}
            .warning {{ border-left-color: #ffc107; background: #fff3cd; }}
            .bad {{ border-left-color: #dc3545; background: #f8d7da; }}
            .images {{ display: grid; grid-template-columns: 1fr; gap: 20px; margin: 20px 0; }}
            .images img {{ width: 100%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #667eea; }}
            h2 {{ color: #555; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .summary {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 VAE Training Analysis Report</h1>
                <p>Comprehensive analysis of your Variational Autoencoder training results</p>
                <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <h2 style="color: white; border: none;">📊 Executive Summary</h2>
                <p><strong>Training completed with {total_epochs} epochs</strong></p>
                <p><strong>Overall Quality:</strong> {'🎯 Excellent - VAE successfully learned a high-quality latent representation!' if overall_good else '⚠️ Good progress - consider additional tuning for optimal performance.'}</p>
            </div>
            
            <h2>🔑 Key Metrics</h2>
            <div class="metric {'good' if final_val_loss < 1000 else 'warning' if final_val_loss < 10000 else 'bad'}">
                <strong>Final Validation Loss:</strong> {final_val_loss:.4f}
            </div>
            <div class="metric {'good' if mu_good else 'warning' if abs(final_mu_mean) < 0.3 else 'bad'}">
                <strong>Final μ Mean:</strong> {final_mu_mean:.6f} (Target: ~0.000)
                <br><small>{'✅ Excellent convergence to zero mean' if mu_good else '⚠️ Good progress, but could converge closer to zero' if abs(final_mu_mean) < 0.3 else '🔴 Needs improvement - mean should be closer to zero'}</small>
            </div>
            <div class="metric {'good' if var_good else 'warning' if 0.5 < final_var < 1.5 else 'bad'}">
                <strong>Final Variance (σ²):</strong> {final_var:.6f} (Target: ~1.000)
                <br><small>{'✅ Excellent convergence to unit variance' if var_good else '⚠️ Reasonable variance, but could be closer to 1.0' if 0.5 < final_var < 1.5 else '🔴 Variance needs adjustment'}</small>
            </div>
            <div class="metric">
                <strong>Distance from N(0,1):</strong> {np.sqrt(final_mu_mean**2 + (final_var - 1)**2):.6f}
                <br><small>Lower values indicate better convergence to standard normal distribution</small>
            </div>
            
            <h2>📈 Training Progress Visualizations</h2>
            <div class="images">
                <div>
                    <h3>Training Metrics Overview</h3>
                    <img src="training_overview.png" alt="Training Overview">
                </div>
                <div>
                    <h3>Detailed Training Metrics</h3>
                    <img src="training_metrics.png" alt="Training Metrics">
                </div>
                <div>
                    <h3>Latent Space Analysis</h3>
                    <img src="latent_analysis.png" alt="Latent Analysis">
                </div>
            </div>
            
            <h2>🔍 Detailed Analysis</h2>
            
            <h3>Training Performance</h3>
            <p><strong>Loss Evolution:</strong> {'Training shows good convergence with validation loss stabilizing.' if training_df['val_loss'].tail(10).std() < training_df['val_loss'].std() * 0.2 else 'Training may benefit from additional epochs or learning rate adjustment.'}</p>
            
            <p><strong>Best Performance:</strong> Best validation loss of {training_df['val_loss'].min():.4f} achieved at epoch {training_df.loc[training_df['val_loss'].idxmin(), 'epoch']}</p>
            
            <h3>Latent Space Quality</h3>
            <p><strong>Convergence Assessment:</strong></p>
            <ul>
                <li><strong>Mean Convergence:</strong> {'✅ Excellent' if mu_good else '⚠️ Needs improvement'} - The latent space means are {'very close to zero' if mu_good else 'approaching zero but could be better'}</li>
                <li><strong>Variance Convergence:</strong> {'✅ Excellent' if var_good else '⚠️ Needs improvement'} - The latent space variance is {'very close to 1.0' if var_good else 'reasonable but could be closer to 1.0'}</li>
            </ul>
            
            <h3>Recommendations</h3>
            <ul>
                {'<li>✅ Your VAE has achieved excellent latent space quality! The model is ready for generation and downstream tasks.</li>' if overall_good else ''}
                {'<li>Consider training for more epochs to improve μ convergence to zero.</li>' if not mu_good else ''}
                {'<li>Adjust the β parameter to better balance reconstruction and KL losses for variance convergence.</li>' if not var_good else ''}
                {'<li>Monitor the reconstruction quality to ensure the model preserves important features.</li>' if final_val_loss > 1000 else ''}
                <li>Consider evaluating generation quality with sample outputs and reconstruction comparisons.</li>
            </ul>
            
            <hr style="margin: 30px 0;">
            <p style="text-align: center; color: #666; font-size: 0.9em;">
                Report generated by VAE Analysis Script • 
                Training Data: {len(training_df)} epochs • 
                Latent Data: {len(latent_df)} measurements
            </p>
        </div>
    </body>
    </html>
    """
    
    report_path = os.path.join(save_dir, 'vae_training_report.html')
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ HTML report saved to: {report_path}")
    print(f"🌐 Open the report in your browser: file://{os.path.abspath(report_path)}")

def main():
    """Main function to run the analysis"""
    parser = argparse.ArgumentParser(description='Analyze VAE Training Results')
    parser.add_argument('--training_log', default='vae_results/10sec_chunks/simple_vae_experiment_20250817_160711/vae_training_log.csv', 
                       help='Path to training log CSV file')
    parser.add_argument('--latent_stats', default='vae_results/10sec_chunks/simple_vae_experiment_20250817_160711/latent_stats.csv',
                       help='Path to latent statistics CSV file')
    parser.add_argument('--output_dir', default='vae_analysis_results',
                       help='Directory to save plots and reports')
    parser.add_argument('--plot_type', default='all',
                       choices=['training', 'latent', 'overview', 'all'],
                       help='Type of plots to generate')
    parser.add_argument('--create_report', action='store_true', default=True,
                       help='Create HTML report')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Load data
    print("📊 Loading training data...")
    training_df = load_training_data(args.training_log)
    if training_df is None:
        print("❌ Failed to load training data")
        return
    
    print("🎯 Loading latent statistics...")
    latent_df = load_latent_data(args.latent_stats)
    if latent_df is None:
        print("❌ Failed to load latent data")
        return
    
    print(f"✅ Successfully loaded data:")
    print(f"   • Training: {len(training_df)} epochs")
    print(f"   • Latent: {len(latent_df)} measurements")
    
    # Generate plots based on selection
    if args.plot_type in ['training', 'all']:
        plot_training_metrics(training_df, output_dir)
    
    if args.plot_type in ['latent', 'all']:
        plot_latent_analysis(latent_df, output_dir)
    
    if args.plot_type in ['overview', 'all']:
        plot_combined_overview(training_df, latent_df, output_dir)
    
    # Create HTML report
    if args.create_report:
        create_html_report(training_df, latent_df, output_dir)
    
    print(f"\n{'='*60}")
    print("🎉 ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"📁 All results saved in: {output_dir.absolute()}")
    print(f"📊 Generated plots:")
    if args.plot_type in ['training', 'all']:
        print(f"   • training_metrics.png")
    if args.plot_type in ['latent', 'all']:
        print(f"   • latent_analysis.png")
    if args.plot_type in ['overview', 'all']:
        print(f"   • training_overview.png")
    if args.create_report:
        print(f"📄 HTML Report: vae_training_report.html")
    print(f"\n💡 Quick start:")
    print(f"   python vae_analysis.py --training_log vae_training_log.csv --latent_stats latent_stats.csv")

if __name__ == "__main__":
    main()


# Additional utility functions for advanced analysis

def analyze_training_stability(training_df, window_size=10):
    """Analyze training stability over time"""
    print("🔍 Analyzing training stability...")
    
    # Calculate rolling statistics
    training_df['val_loss_rolling_mean'] = training_df['val_loss'].rolling(window=window_size).mean()
    training_df['val_loss_rolling_std'] = training_df['val_loss'].rolling(window=window_size).std()
    
    # Stability metrics
    final_stability = training_df['val_loss_rolling_std'].iloc[-window_size:].mean()
    overall_std = training_df['val_loss'].std()
    stability_ratio = final_stability / overall_std
    
    print(f"Training Stability Analysis:")
    print(f"  • Final {window_size}-epoch stability: {final_stability:.4f}")
    print(f"  • Overall loss std: {overall_std:.4f}")
    print(f"  • Stability ratio: {stability_ratio:.4f}")
    print(f"  • Assessment: {'✅ Stable' if stability_ratio < 0.1 else '⚠️ Still learning' if stability_ratio < 0.3 else '🔴 Unstable'}")
    
    return stability_ratio

def detect_training_issues(training_df, latent_df):
    """Detect common training issues"""
    print("🔍 Detecting potential training issues...")
    
    issues = []
    
    # Check for exploding gradients
    if training_df['train_loss'].max() > training_df['train_loss'].iloc[0] * 10:
        issues.append("⚠️ Possible exploding gradients detected")
    
    # Check for mode collapse in latent space
    if latent_df['mu_std'].min() < 0.01:
        issues.append("⚠️ Very low latent diversity - possible mode collapse")
    
    # Check for KL vanishing
    if (latent_df['actual_var'] > 10).any():
        issues.append("⚠️ Very high variance - KL divergence might be vanishing")
    
    # Check for poor reconstruction
    final_recon = training_df['val_recon_loss'].iloc[-1]
    if final_recon > training_df['val_recon_loss'].iloc[0] * 0.8:
        issues.append("⚠️ Reconstruction loss not improving well")
    
    # Check for beta too high/low
    final_beta = training_df['beta'].iloc[-1]
    final_kl = training_df['val_kl_loss'].iloc[-1]
    if final_kl < 0.1 and final_beta > 0.001:
        issues.append("⚠️ KL loss very low - consider reducing β")
    elif final_kl > 1000 and final_beta < 0.01:
        issues.append("⚠️ KL loss very high - consider increasing β")
    
    if issues:
        print("Potential issues detected:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ No major training issues detected")
    
    return issues

def create_training_comparison(results_dir):
    """Compare multiple training runs if available"""
    results_path = Path(results_dir)
    
    # Look for multiple experiment directories
    experiment_dirs = [d for d in results_path.iterdir() if d.is_dir() and 'experiment' in d.name]
    
    if len(experiment_dirs) > 1:
        print(f"🔄 Found {len(experiment_dirs)} experiments for comparison")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for exp_dir in experiment_dirs:
            training_file = exp_dir / 'vae_training_log.csv'
            if training_file.exists():
                df = pd.read_csv(training_file)
                label = exp_dir.name.split('_')[-1]  # Extract timestamp or identifier
                
                axes[0, 0].semilogy(df['epoch'], df['val_loss'], label=label, alpha=0.8)
                axes[0, 1].plot(df['epoch'], df['beta'], label=label, alpha=0.8)
                axes[1, 0].plot(df['epoch'], df['val_recon_loss'], label=label, alpha=0.8)
                axes[1, 1].semilogy(df['epoch'], df['val_kl_loss'], label=label, alpha=0.8)
        
        axes[0, 0].set_title('Validation Loss Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Beta Schedule Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Reconstruction Loss Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('KL Loss Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('🔄 Training Runs Comparison', fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(results_path / 'experiments_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Comparison plot saved to: {results_path}/experiments_comparison.png")
    else:
        print("ℹ️ Only one experiment found, skipping comparison")


# Example usage and testing
if __name__ == "__main__":
    # You can also run specific analysis functions directly
    
    # Example: Quick analysis without command line args
    """
    training_df = load_training_data('vae_training_log.csv')
    latent_df = load_latent_data('latent_stats.csv')
    
    if training_df is not None and latent_df is not None:
        output_dir = Path('quick_analysis')
        output_dir.mkdir(exist_ok=True)
        
        plot_combined_overview(training_df, latent_df, output_dir)
        analyze_training_stability(training_df)
        detect_training_issues(training_df, latent_df)
    """
    
    # Run main analysis
    main()