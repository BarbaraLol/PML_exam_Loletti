import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

def load_training_data(csv_path):
    """Load training data from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found!")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def plot_training_curves(df, save_dir='results/5sec_chunks'):
    """Plot training and validation curves."""
    
    # Create plots directory
    Path(save_dir).mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CNN Training Results', fontsize=16, fontweight='bold')
    
    epochs = df['epoch']
    
    # Plot 1: Loss curves
    axes[0, 0].plot(epochs, df['train_loss'], label='Training Loss', linewidth=2, alpha=0.8)
    axes[0, 0].plot(epochs, df['val_loss'], label='Validation Loss', linewidth=2, alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training vs Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Highlight best validation loss
    best_val_idx = df['val_loss'].idxmin()
    best_val_loss = df.loc[best_val_idx, 'val_loss']
    best_epoch = df.loc[best_val_idx, 'epoch']
    axes[0, 0].scatter(best_epoch, best_val_loss, color='red', s=100, zorder=5)
    axes[0, 0].annotate(f'Best: {best_val_loss:.4f}\nEpoch {best_epoch}', 
                       xy=(best_epoch, best_val_loss), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 2: Accuracy curves
    axes[0, 1].plot(epochs, df['train_acc'], label='Training Accuracy', linewidth=2, alpha=0.8)
    axes[0, 1].plot(epochs, df['val_acc'], label='Validation Accuracy', linewidth=2, alpha=0.8)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training vs Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Highlight best validation accuracy
    best_acc_idx = df['val_acc'].idxmax()
    best_val_acc = df.loc[best_acc_idx, 'val_acc']
    best_acc_epoch = df.loc[best_acc_idx, 'epoch']
    axes[0, 1].scatter(best_acc_epoch, best_val_acc, color='red', s=100, zorder=5)
    axes[0, 1].annotate(f'Best: {best_val_acc:.4f}\nEpoch {best_acc_epoch}', 
                       xy=(best_acc_epoch, best_val_acc), 
                       xytext=(10, -20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 3: Learning rate schedule
    axes[1, 0].plot(epochs, df['learning_rate'], linewidth=2, alpha=0.8, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Early stopping counter
    axes[1, 1].plot(epochs, df['early_stop_counter'], linewidth=2, alpha=0.8, color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Early Stop Counter')
    axes[1, 1].set_title('Early Stopping Counter')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add horizontal line for patience threshold if available
    if 'early_stop_counter' in df.columns:
        max_counter = df['early_stop_counter'].max()
        if max_counter > 0:
            axes[1, 1].axhline(y=max_counter, color='red', linestyle='--', alpha=0.7, 
                              label=f'Max Patience: {max_counter}')
            axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(save_dir) / 'training_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {plot_path}")
    
    return fig

def plot_loss_vs_accuracy(df, save_dir='results/5sec_chunks'):
    """Plot loss vs accuracy correlation."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Training correlation
    axes[0].scatter(df['train_loss'], df['train_acc'], alpha=0.6, s=50)
    axes[0].set_xlabel('Training Loss')
    axes[0].set_ylabel('Training Accuracy')
    axes[0].set_title('Training: Loss vs Accuracy')
    axes[0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['train_loss'], df['train_acc'], 1)
    p = np.poly1d(z)
    axes[0].plot(df['train_loss'], p(df['train_loss']), "r--", alpha=0.8, linewidth=2)
    
    # Validation correlation
    axes[1].scatter(df['val_loss'], df['val_acc'], alpha=0.6, s=50, color='orange')
    axes[1].set_xlabel('Validation Loss')
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].set_title('Validation: Loss vs Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['val_loss'], df['val_acc'], 1)
    p = np.poly1d(z)
    axes[1].plot(df['val_loss'], p(df['val_loss']), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(save_dir) / 'loss_vs_accuracy.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Loss vs Accuracy plot saved to: {plot_path}")
    
    return fig

def generate_training_summary(df):
    """Generate and print training summary statistics."""
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    
    # Basic info
    total_epochs = len(df)
    print(f"Total epochs trained: {total_epochs}")
    
    # Best metrics
    best_val_loss_idx = df['val_loss'].idxmin()
    best_val_acc_idx = df['val_acc'].idxmax()
    
    print(f"\nBest Validation Loss: {df.loc[best_val_loss_idx, 'val_loss']:.6f} (Epoch {df.loc[best_val_loss_idx, 'epoch']})")
    print(f"Best Validation Accuracy: {df.loc[best_val_acc_idx, 'val_acc']:.6f} (Epoch {df.loc[best_val_acc_idx, 'epoch']})")
    
    # Final metrics
    final_idx = df.index[-1]
    print(f"\nFinal Metrics:")
    print(f"  Train Loss: {df.loc[final_idx, 'train_loss']:.6f}")
    print(f"  Train Accuracy: {df.loc[final_idx, 'train_acc']:.6f}")
    print(f"  Validation Loss: {df.loc[final_idx, 'val_loss']:.6f}")
    print(f"  Validation Accuracy: {df.loc[final_idx, 'val_acc']:.6f}")
    
    # Learning rate info
    initial_lr = df['learning_rate'].iloc[0]
    final_lr = df['learning_rate'].iloc[-1]
    print(f"\nLearning Rate:")
    print(f"  Initial: {initial_lr:.8f}")
    print(f"  Final: {final_lr:.8f}")
    print(f"  Reduction Factor: {initial_lr / final_lr:.2f}x")
    
    # Early stopping info
    max_early_stop_counter = df['early_stop_counter'].max()
    if max_early_stop_counter > 0:
        print(f"\nEarly Stopping:")
        print(f"  Max counter reached: {max_early_stop_counter}")
        if df['early_stop_counter'].iloc[-1] == max_early_stop_counter:
            print(f"  Training stopped early due to no improvement")
        else:
            print(f"  Training completed without early stopping")
    
    # Overfitting analysis
    train_val_loss_diff = df['train_loss'].iloc[-1] - df['val_loss'].iloc[-1]
    train_val_acc_diff = df['val_acc'].iloc[-1] - df['train_acc'].iloc[-1]
    
    print(f"\nOverfitting Analysis:")
    print(f"  Final Loss Difference (Train - Val): {train_val_loss_diff:.6f}")
    print(f"  Final Accuracy Difference (Val - Train): {train_val_acc_diff:.6f}")
    
    if train_val_loss_diff < -0.1:
        print("  → Potential underfitting (train loss much higher than val loss)")
    elif train_val_loss_diff > 0.1:
        print("  → Potential overfitting (train loss much lower than val loss)")
    else:
        print("  → Good balance between training and validation performance")

def main():
    parser = argparse.ArgumentParser(description='Plot CNN training results')
    parser.add_argument('csv_path', help='Path to the training results CSV file')
    parser.add_argument('--save_dir', default='plresults/5sec_chunks', help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Load data
    df = load_training_data(args.csv_path)
    if df is None:
        return
    
    print(f"Loaded training data: {len(df)} epochs")
    
    # Generate plots
    print("Generating training curves...")
    plot_training_curves(df, args.save_dir)
    
    print("Generating loss vs accuracy plots...")
    plot_loss_vs_accuracy(df, args.save_dir)
    
    # Generate summary
    generate_training_summary(df)
    
    print(f"\nAll plots saved to: {args.save_dir}/")
    print("Done!")

def plot_csv_file(csv_path, save_dir='results/5sec_chunks'):
    """Convenience function to plot results from a CSV file."""
    df = load_training_data(csv_path)
    if df is None:
        return
    
    plot_training_curves(df, save_dir)
    plot_loss_vs_accuracy(df, save_dir)
    generate_training_summary(df)
    
    plt.show()  # Display plots

if __name__ == "__main__":
    main()