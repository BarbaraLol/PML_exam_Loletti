import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import os

# Create output directory for saved plots
output_dir = 'vae_analysis_plots'
os.makedirs(output_dir, exist_ok=True)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the data
training_df = pd.read_csv('results/vae_experiment_20250817_130717/vae_training_log.csv')
latent_df = pd.read_csv('results/vae_experiment_20250817_130717/latent_stats.csv')

# Create a comprehensive figure with multiple subplots
fig = plt.figure(figsize=(20, 24))
gs = GridSpec(6, 3, figure=fig, hspace=0.3, wspace=0.3)

# 1. Training Loss Evolution
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(training_df['epoch'], training_df['train_loss'], label='Training Loss', linewidth=2)
ax1.plot(training_df['epoch'], training_df['val_loss'], label='Validation Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Total Loss')
ax1.set_title('VAE Training and Validation Loss Over Time', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Reconstruction Loss vs KL Loss
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(training_df['epoch'], training_df['train_recon_loss'], label='Train Recon', linewidth=2)
ax2.plot(training_df['epoch'], training_df['val_recon_loss'], label='Val Recon', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Reconstruction Loss')
ax2.set_title('Reconstruction Loss', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(training_df['epoch'], training_df['train_kl_loss'], label='Train KL', linewidth=2, color='orange')
ax3.plot(training_df['epoch'], training_df['val_kl_loss'], label='Val KL', linewidth=2, color='red')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('KL Divergence Loss')
ax3.set_title('KL Divergence Loss', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 3. Beta Schedule and Learning Rate
ax4 = fig.add_subplot(gs[1, 2])
ax4_twin = ax4.twinx()
line1 = ax4.plot(training_df['epoch'], training_df['lr'], label='Learning Rate', linewidth=2, color='green')
line2 = ax4_twin.plot(training_df['epoch'], training_df['beta'], label='Beta (KL weight)', linewidth=2, color='purple')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Learning Rate', color='green')
ax4_twin.set_ylabel('Beta', color='purple')
ax4.set_title('Learning Rate & Beta Schedule', fontweight='bold')
# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax4.legend(lines, labels, loc='upper right')
ax4.grid(True, alpha=0.3)

# 4. Gradient Norm and Training Time
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(training_df['epoch'], training_df['grad_norm'], linewidth=2, color='brown')
ax5.set_xlabel('Epoch')
ax5.set_ylabel('Gradient Norm')
ax5.set_title('Gradient Norm Evolution', fontweight='bold')
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(training_df['epoch'], training_df['time_elapsed'], linewidth=2, color='teal')
ax6.set_xlabel('Epoch')
ax6.set_ylabel('Time Elapsed (seconds)')
ax6.set_title('Training Time per Epoch', fontweight='bold')
ax6.grid(True, alpha=0.3)

# 5. Loss Components Stacked Area Chart
ax7 = fig.add_subplot(gs[2, 2])
ax7.fill_between(training_df['epoch'], 0, training_df['train_recon_loss'], 
                alpha=0.7, label='Reconstruction Loss')
ax7.fill_between(training_df['epoch'], training_df['train_recon_loss'], 
                training_df['train_recon_loss'] + training_df['train_kl_loss'], 
                alpha=0.7, label='KL Loss')
ax7.set_xlabel('Epoch')
ax7.set_ylabel('Loss Components')
ax7.set_title('Training Loss Decomposition', fontweight='bold')
ax7.legend()

# 6. Latent Space Statistics - Mean and Std of Mu
ax8 = fig.add_subplot(gs[3, 0])
ax8.plot(latent_df['epoch'], latent_df['mu_mean'], linewidth=2, label='μ Mean')
ax8.fill_between(latent_df['epoch'], 
                latent_df['mu_mean'] - latent_df['mu_std'],
                latent_df['mu_mean'] + latent_df['mu_std'],
                alpha=0.3, label='μ ± Std')
ax8.set_xlabel('Epoch')
ax8.set_ylabel('Latent Mean (μ)')
ax8.set_title('Latent Space Mean Statistics', fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)
ax8.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Target (0)')

# 7. Latent Space Statistics - LogVar
ax9 = fig.add_subplot(gs[3, 1])
ax9.plot(latent_df['epoch'], latent_df['logvar_mean'], linewidth=2, label='log(σ²) Mean', color='orange')
ax9.fill_between(latent_df['epoch'], 
                latent_df['logvar_mean'] - latent_df['logvar_std'],
                latent_df['logvar_mean'] + latent_df['logvar_std'],
                alpha=0.3, color='orange', label='log(σ²) ± Std')
ax9.set_xlabel('Epoch')
ax9.set_ylabel('Log Variance')
ax9.set_title('Latent Space Log-Variance Statistics', fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)
ax9.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Target (0)')

# 8. Actual Variance
ax10 = fig.add_subplot(gs[3, 2])
ax10.plot(latent_df['epoch'], latent_df['actual_var'], linewidth=2, color='purple')
ax10.set_xlabel('Epoch')
ax10.set_ylabel('Actual Variance')
ax10.set_title('Actual Latent Variance', fontweight='bold')
ax10.grid(True, alpha=0.3)
ax10.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Target (1)')
ax10.legend()

# 9. Latent Space Health - Deviation from Standard Normal
ax11 = fig.add_subplot(gs[4, :2])
mu_deviation = np.abs(latent_df['mu_mean'])
var_deviation = np.abs(latent_df['actual_var'] - 1)

ax11.plot(latent_df['epoch'], mu_deviation, label='|μ| deviation from 0', linewidth=2)
ax11.plot(latent_df['epoch'], var_deviation, label='|σ²| deviation from 1', linewidth=2)
ax11.set_xlabel('Epoch')
ax11.set_ylabel('Absolute Deviation')
ax11.set_title('Latent Space Health: Deviation from Standard Normal', fontweight='bold')
ax11.legend()
ax11.grid(True, alpha=0.3)
ax11.set_yscale('log')

# 10. Training Efficiency Metrics
ax12 = fig.add_subplot(gs[4, 2])
# Calculate loss reduction rate
loss_improvement = (training_df['train_loss'].iloc[0] - training_df['train_loss']) / training_df['train_loss'].iloc[0] * 100
ax12.plot(training_df['epoch'], loss_improvement, linewidth=2, color='green')
ax12.set_xlabel('Epoch')
ax12.set_ylabel('Loss Improvement (%)')
ax12.set_title('Training Progress (%)', fontweight='bold')
ax12.grid(True, alpha=0.3)

# 11. Correlation Matrix of Training Metrics
ax13 = fig.add_subplot(gs[5, 0])
training_corr = training_df[['train_loss', 'train_recon_loss', 'train_kl_loss', 
                           'val_loss', 'grad_norm', 'lr', 'beta']].corr()
sns.heatmap(training_corr, annot=True, cmap='coolwarm', center=0, ax=ax13, 
            square=True, fmt='.2f')
ax13.set_title('Training Metrics Correlation', fontweight='bold')

# 12. Latent Statistics Distribution (if multiple batches per epoch)
ax14 = fig.add_subplot(gs[5, 1])
# Show distribution of mu_mean across all measurements
ax14.hist(latent_df['mu_mean'], bins=30, alpha=0.7, label='μ mean', density=True)
ax14.hist(latent_df['actual_var'], bins=30, alpha=0.7, label='σ² actual', density=True)
ax14.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Target μ')
ax14.axvline(x=1, color='orange', linestyle='--', alpha=0.7, label='Target σ²')
ax14.set_xlabel('Value')
ax14.set_ylabel('Density')
ax14.set_title('Latent Statistics Distribution', fontweight='bold')
ax14.legend()

# 13. Training Summary Statistics
ax15 = fig.add_subplot(gs[5, 2])
ax15.axis('off')
summary_text = f"""
Training Summary:

Final Training Loss: {training_df['train_loss'].iloc[-1]:.4f}
Final Validation Loss: {training_df['val_loss'].iloc[-1]:.4f}
Best Validation Loss: {training_df['val_loss'].min():.4f}

Loss Improvement: {((training_df['train_loss'].iloc[0] - training_df['train_loss'].iloc[-1]) / training_df['train_loss'].iloc[0] * 100):.1f}%

Final μ mean: {latent_df['mu_mean'].iloc[-1]:.4f}
Final σ² actual: {latent_df['actual_var'].iloc[-1]:.4f}

Total Training Time: {training_df['time_elapsed'].sum():.1f}s
Avg Time/Epoch: {training_df['time_elapsed'].mean():.2f}s
"""

ax15.text(0.1, 0.9, summary_text, transform=ax15.transAxes, fontsize=12, 
         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

plt.suptitle('VAE Training Analysis Dashboard', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{output_dir}/vae_training_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional specific plots for detailed analysis
print("\n" + "="*60)
print("ADDITIONAL ANALYSIS PLOTS")
print("="*60)

# Plot 1: Detailed Loss Evolution with Smoothing
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Smoothed loss curves
window = min(10, len(training_df) // 5)  # Adaptive window size
train_loss_smooth = training_df['train_loss'].rolling(window=window, center=True).mean()
val_loss_smooth = training_df['val_loss'].rolling(window=window, center=True).mean()

ax1.plot(training_df['epoch'], training_df['train_loss'], alpha=0.3, color='blue')
ax1.plot(training_df['epoch'], train_loss_smooth, linewidth=2, color='blue', label='Training (smoothed)')
ax1.plot(training_df['epoch'], training_df['val_loss'], alpha=0.3, color='orange')  
ax1.plot(training_df['epoch'], val_loss_smooth, linewidth=2, color='orange', label='Validation (smoothed)')
ax1.set_title('Smoothed Loss Curves', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Beta-KL Loss relationship
ax2.scatter(training_df['beta'], training_df['train_kl_loss'], alpha=0.6, color='red')
ax2.set_xlabel('Beta (KL Weight)')
ax2.set_ylabel('KL Loss')
ax2.set_title('KL Loss vs Beta Weight', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Latent space convergence
epochs_latent = latent_df['epoch'].unique()
mu_mean_per_epoch = latent_df.groupby('epoch')['mu_mean'].mean()
var_mean_per_epoch = latent_df.groupby('epoch')['actual_var'].mean()

ax3.plot(epochs_latent, mu_mean_per_epoch, 'o-', label='Mean μ', linewidth=2)
ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Target (0)')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Mean of μ')
ax3.set_title('Latent Mean Convergence', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4.plot(epochs_latent, var_mean_per_epoch, 'o-', color='purple', label='Mean σ²', linewidth=2)
ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Target (1)')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Mean Variance')
ax4.set_title('Latent Variance Convergence', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/detailed_loss_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: 3D Latent Space Evolution
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Color by epoch for evolution
epochs = latent_df['epoch'].values
scatter = ax.scatter(latent_df['mu_mean'], latent_df['actual_var'], latent_df['logvar_mean'], 
                    c=epochs, cmap='viridis', alpha=0.6)

ax.set_xlabel('μ Mean')
ax.set_ylabel('Actual Variance')
ax.set_zlabel('Log Variance Mean')
ax.set_title('Latent Space Statistics Evolution', fontweight='bold')

# Add target point
ax.scatter([0], [1], [0], color='red', s=100, marker='*', label='Target (0, 1, 0)')

plt.colorbar(scatter, label='Epoch')
ax.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/latent_space_3d_evolution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAll plots saved to '{output_dir}/' directory:")
print("1. vae_training_dashboard.png - Main comprehensive dashboard")
print("2. detailed_loss_analysis.png - Detailed loss analysis")
print("3. latent_space_3d_evolution.png - 3D latent space evolution")

print("\nAnalysis complete! The plots show:")
print("1. Training and validation loss evolution")
print("2. Reconstruction vs KL loss components")
print("3. Learning rate and beta scheduling")
print("4. Gradient norm stability")
print("5. Latent space statistics convergence")
print("6. Overall training health metrics")

