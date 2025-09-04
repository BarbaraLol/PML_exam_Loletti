import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the CSV file
csv_file = 'results/20sec_chunks/training_log.csv'  # Update path as needed
df = pd.read_csv(csv_file)

# Create output directory for plots
output_dir = 'results/20sec_chunks/plots'
os.makedirs(output_dir, exist_ok=True)

# Set style for better looking plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# 1. Training vs Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train_acc'], 'b-', linewidth=2, marker='o', markersize=4, label='Training Accuracy')
plt.plot(df['epoch'], df['val_acc'], 'r-', linewidth=2, marker='s', markersize=4, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('BCNN: Training vs Validation Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'bcnn_accuracy.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'bcnn_accuracy.pdf'), bbox_inches='tight')
plt.close()

# 2. Training vs Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, marker='o', markersize=4, label='Training Loss')
plt.plot(df['epoch'], df['val_loss'], 'r-', linewidth=2, marker='s', markersize=4, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('BCNN: Training vs Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'bcnn_loss.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'bcnn_loss.pdf'), bbox_inches='tight')
plt.close()

# 3. Learning Rate Schedule
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['lr'], 'g-', linewidth=2, marker='D', markersize=4)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('BCNN: Learning Rate Schedule')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'bcnn_learning_rate.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'bcnn_learning_rate.pdf'), bbox_inches='tight')
plt.close()

# 4. KL Divergence Loss
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['kl_loss'], 'purple', linewidth=2, marker='^', markersize=4)
plt.xlabel('Epoch')
plt.ylabel('KL Divergence Loss')
plt.title('BCNN: KL Divergence Evolution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'bcnn_kl_loss.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'bcnn_kl_loss.pdf'), bbox_inches='tight')
plt.close()

# 5. Overfitting Analysis (Train - Val Accuracy Gap)
accuracy_gap = df['train_acc'] - df['val_acc']
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], accuracy_gap, 'red', linewidth=2, marker='v', markersize=4)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Epoch')
plt.ylabel('Training - Validation Accuracy (%)')
plt.title('BCNN: Overfitting Analysis')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'bcnn_overfitting.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'bcnn_overfitting.pdf'), bbox_inches='tight')
plt.close()

# 6. Training Time Analysis
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['time_elapsed']/60, 'orange', linewidth=2, marker='h', markersize=4)
plt.xlabel('Epoch')
plt.ylabel('Cumulative Training Time (minutes)')
plt.title('BCNN: Training Time Progress')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'bcnn_training_time.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'bcnn_training_time.pdf'), bbox_inches='tight')
plt.close()

# 7. Combined Loss Components
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Classification loss
ax1.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, marker='o', markersize=4, label='Train Loss')
ax1.plot(df['epoch'], df['val_loss'], 'r-', linewidth=2, marker='s', markersize=4, label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Classification Loss')
ax1.set_title('Classification Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# KL loss
ax2.plot(df['epoch'], df['kl_loss'], 'purple', linewidth=2, marker='^', markersize=4)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('KL Divergence Loss')
ax2.set_title('KL Divergence Loss')
ax2.grid(True, alpha=0.3)

plt.suptitle('BCNN: Loss Components Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'bcnn_loss_components.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'bcnn_loss_components.pdf'), bbox_inches='tight')
plt.close()

# Print summary statistics
print("BCNN TRAINING RESULTS SUMMARY")
print("=" * 50)
print(f"Best Validation Accuracy: {df['val_acc'].max():.2f}% (Epoch {df.loc[df['val_acc'].idxmax(), 'epoch']})")
print(f"Final Validation Accuracy: {df['val_acc'].iloc[-1]:.2f}%")
print(f"Best Validation Loss: {df['val_loss'].min():.4f} (Epoch {df.loc[df['val_loss'].idxmin(), 'epoch']})")
print(f"Final Training Accuracy: {df['train_acc'].iloc[-1]:.2f}%")
print(f"Total Training Time: {df['time_elapsed'].iloc[-1]/60:.1f} minutes")
print(f"Final KL Loss: {df['kl_loss'].iloc[-1]:.4f}")
print(f"Epochs Completed: {len(df)}")

# Overfitting analysis
final_gap = df['train_acc'].iloc[-1] - df['val_acc'].iloc[-1]
print(f"Final Accuracy Gap: {final_gap:.2f}%")
if final_gap > 8:
    print("Status: Significant overfitting detected")
elif final_gap > 5:
    print("Status: Moderate overfitting observed")
else:
    print("Status: Minimal overfitting")

print(f"\nAll plots saved to: {output_dir}/")
print("Available files:")
for filename in sorted(os.listdir(output_dir)):
    if filename.endswith(('.png', '.pdf')):
        print(f"  - {filename}")