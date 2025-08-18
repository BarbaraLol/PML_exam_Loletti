import pandas as pd
import matplotlib.pyplot as plt
import os

def save_individual_plots(log_file='vae_training_log.csv', latent_stats_file='latent_stats.csv', output_dir='vae_plots'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    log_df = pd.read_csv(log_file)
    latent_df = pd.read_csv(latent_stats_file)

    # 1. Training and Validation Losses
    plt.figure(figsize=(10, 6))
    plt.plot(log_df['epoch'], log_df['train_loss'], label='Train Loss')
    plt.plot(log_df['epoch'], log_df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/01_total_loss.png', bbox_inches='tight', dpi=150)
    plt.close()

    # 2. Reconstruction and KL losses
    plt.figure(figsize=(10, 6))
    plt.plot(log_df['epoch'], log_df['train_recon_loss'], label='Train Recon Loss')
    plt.plot(log_df['epoch'], log_df['train_kl_loss'], label='Train KL Loss')
    plt.plot(log_df['epoch'], log_df['val_recon_loss'], '--', label='Val Recon Loss')
    plt.plot(log_df['epoch'], log_df['val_kl_loss'], '--', label='Val KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Reconstruction vs KL Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/02_recon_kl_loss.png', bbox_inches='tight', dpi=150)
    plt.close()

    # 3. Learning Rate Schedule
    plt.figure(figsize=(10, 6))
    plt.plot(log_df['epoch'], log_df['lr'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.savefig(f'{output_dir}/03_learning_rate.png', bbox_inches='tight', dpi=150)
    plt.close()

    # 4. Beta Schedule
    plt.figure(figsize=(10, 6))
    plt.plot(log_df['epoch'], log_df['beta'])
    plt.xlabel('Epoch')
    plt.ylabel('Beta')
    plt.title('Beta Schedule')
    plt.grid(True)
    plt.savefig(f'{output_dir}/04_beta_schedule.png', bbox_inches='tight', dpi=150)
    plt.close()

    # 5. Latent Mean Statistics
    plt.figure(figsize=(10, 6))
    plt.plot(latent_df['epoch'], latent_df['mu_mean'], label='Mean')
    plt.plot(latent_df['epoch'], latent_df['mu_std'], label='Std')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Latent Space Mean (μ) Statistics')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/05_latent_mean_stats.png', bbox_inches='tight', dpi=150)
    plt.close()

    # 6. Latent Variance
    plt.figure(figsize=(10, 6))
    plt.plot(latent_df['epoch'], latent_df['actual_var'])
    plt.xlabel('Epoch')
    plt.ylabel('Variance')
    plt.title('Latent Space Variance (exp(logvar))')
    plt.grid(True)
    plt.savefig(f'{output_dir}/06_latent_variance.png', bbox_inches='tight', dpi=150)
    plt.close()

    # 7. KL to Reconstruction Loss Ratio
    plt.figure(figsize=(10, 6))
    kl_recon_ratio = log_df['train_kl_loss'] / log_df['train_recon_loss']
    plt.plot(log_df['epoch'], kl_recon_ratio)
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')
    plt.title('KL Loss / Reconstruction Loss Ratio')
    plt.grid(True)
    plt.savefig(f'{output_dir}/07_kl_recon_ratio.png', bbox_inches='tight', dpi=150)
    plt.close()

    # 8. Gradient Norms
    plt.figure(figsize=(10, 6))
    plt.plot(log_df['epoch'], log_df['grad_norm'])
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norms During Training')
    plt.grid(True)
    plt.savefig(f'{output_dir}/08_gradient_norms.png', bbox_inches='tight', dpi=150)
    plt.close()

    # 9. Combined Loss Overview
    plt.figure(figsize=(10, 6))
    plt.plot(log_df['epoch'], log_df['train_loss'], label='Train')
    plt.plot(log_df['epoch'], log_df['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Combined Loss Overview')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/09_combined_loss.png', bbox_inches='tight', dpi=150)
    plt.close()

    # 10. Latent Space Mean and Variance
    plt.figure(figsize=(10, 6))
    plt.plot(latent_df['epoch'], latent_df['mu_mean'], label='Mean (μ)')
    plt.plot(latent_df['epoch'], latent_df['actual_var'], label='Variance (σ²)')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Latent Space Mean and Variance')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/10_latent_mean_variance.png', bbox_inches='tight', dpi=150)
    plt.close()

    print(f"All plots saved to directory: {output_dir}")

if __name__ == "__main__":
    save_individual_plots()