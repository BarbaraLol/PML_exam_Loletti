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

def load_latent_stats(log_path):
    """Load the latent statistics CSV file"""
    try:
        df = pd.read_csv(log_path)
        print(f"Loaded latent statistics with {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading log file: {e}")
        return None

def plot_latent_evolution(df, save_path=None):
    """Plot how latent statistics evolve during training"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Mean of mu (should converge towards 0)
    axes[0, 0].plot(df['epoch'], df['mu_mean'], linewidth=2, color='blue')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Target (0)')
    axes[0, 0].set_title('Mean of μ (Latent Means)\nShould converge to ~0', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('μ Mean')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Std of mu (diversity of latent means)
    axes[0, 1].plot(df['epoch'], df['mu_std'], linewidth=2, color='green')
    axes[0, 1].set_title('Std of μ (Diversity of Means)\nHigher = more diverse latent codes', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('μ Std')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean of logvar (log variance evolution)
    axes[0, 2].plot(df['epoch'], df['logvar_mean'], linewidth=2, color='orange')
    axes[0, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Target (0)')
    axes[0, 2].set_title('Mean of log(σ²)\nShould converge to ~0', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('log(σ²) Mean')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Std of logvar
    axes[1, 0].plot(df['epoch'], df['logvar_std'], linewidth=2, color='purple')
    axes[1, 0].set_title('Std of log(σ²)\nVariability in uncertainties', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('log(σ²) Std')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Actual variance (exp of logvar)
    axes[1, 1].plot(df['epoch'], df['actual_var'], linewidth=2, color='red')
    axes[1, 1].axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Target (1)')
    axes[1, 1].set_title('Actual Variance σ²\nShould converge to ~1', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('σ² (Actual Variance)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # KL divergence approximation (how close to standard normal)
    # KL ≈ 0.5 * (μ² + σ² - log(σ²) - 1)
    kl_approx = 0.5 * (df['mu_mean']**2 + df['actual_var'] - df['logvar_mean'] - 1)
    axes[1, 2].plot(df['epoch'], kl_approx, linewidth=2, color='darkred')
    axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Target (0)')
    axes[1, 2].set_title('Approximate KL Divergence\nLower = closer to N(0,1)', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('KL ≈ 0.5(μ²+σ²-log(σ²)-1)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Latent evolution plots saved to: {save_path}")
    
    plt.show()

def plot_latent_distributions(df, save_path=None):
    """Plot distributions of latent statistics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Distribution of mu means over time
    recent_epochs = max(1, len(df) // 4)  # Last 25% of training
    early_data = df.head(recent_epochs)
    late_data = df.tail(recent_epochs)
    
    axes[0, 0].hist(early_data['mu_mean'], bins=30, alpha=0.6, label='Early Training', 
                   density=True, color='lightblue')
    axes[0, 0].hist(late_data['mu_mean'], bins=30, alpha=0.6, label='Late Training', 
                   density=True, color='darkblue')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', label='Target (0)')
    axes[0, 0].set_title('Distribution of μ Means', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('μ Mean Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Distribution of mu stds
    axes[0, 1].hist(early_data['mu_std'], bins=30, alpha=0.6, label='Early Training', 
                   density=True, color='lightgreen')
    axes[0, 1].hist(late_data['mu_std'], bins=30, alpha=0.6, label='Late Training', 
                   density=True, color='darkgreen')
    axes[0, 1].set_title('Distribution of μ Stds', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('μ Std Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribution of logvar means
    axes[0, 2].hist(early_data['logvar_mean'], bins=30, alpha=0.6, label='Early Training', 
                   density=True, color='lightyellow')
    axes[0, 2].hist(late_data['logvar_mean'], bins=30, alpha=0.6, label='Late Training', 
                   density=True, color='orange')
    axes[0, 2].axvline(x=0, color='red', linestyle='--', label='Target (0)')
    axes[0, 2].set_title('Distribution of log(σ²) Means', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('log(σ²) Mean Value')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Distribution of actual variances
    axes[1, 0].hist(early_data['actual_var'], bins=30, alpha=0.6, label='Early Training', 
                   density=True, color='lightcoral')
    axes[1, 0].hist(late_data['actual_var'], bins=30, alpha=0.6, label='Late Training', 
                   density=True, color='darkred')
    axes[1, 0].axvline(x=1, color='black', linestyle='--', label='Target (1)')
    axes[1, 0].set_title('Distribution of Actual Variance σ²', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('σ² Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation between mu_mean and actual_var over time
    correlation_early = np.corrcoef(early_data['mu_mean'], early_data['actual_var'])[0, 1]
    correlation_late = np.corrcoef(late_data['mu_mean'], late_data['actual_var'])[0, 1]
    
    scatter_early = axes[1, 1].scatter(early_data['mu_mean'], early_data['actual_var'], 
                                      alpha=0.6, s=20, label=f'Early (r={correlation_early:.3f})',
                                      color='lightblue')
    scatter_late = axes[1, 1].scatter(late_data['mu_mean'], late_data['actual_var'], 
                                     alpha=0.6, s=20, label=f'Late (r={correlation_late:.3f})',
                                     color='darkblue')
    axes[1, 1].axhline(y=1, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('μ Mean vs σ² Correlation', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('μ Mean')
    axes[1, 1].set_ylabel('σ² (Actual Variance)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Training progress visualization
    progress_metric = np.sqrt(df['mu_mean']**2 + (df['actual_var'] - 1)**2)
    axes[1, 2].plot(df['epoch'], progress_metric, linewidth=2, color='purple')
    axes[1, 2].set_title('Distance from Ideal N(0,1)\n√(μ² + (σ²-1)²)', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Distance from N(0,1)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Latent distribution plots saved to: {save_path}")
    
    plt.show()

def plot_convergence_analysis(df, save_path=None):
    """Analyze convergence of latent space to standard normal"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rolling averages to smooth out noise
    window_size = max(1, len(df) // 20)
    
    mu_mean_smooth = df['mu_mean'].rolling(window=window_size, center=True).mean()
    actual_var_smooth = df['actual_var'].rolling(window=window_size, center=True).mean()
    
    # Convergence to zero mean
    axes[0, 0].plot(df['epoch'], df['mu_mean'], alpha=0.3, color='blue', label='Raw')
    axes[0, 0].plot(df['epoch'], mu_mean_smooth, linewidth=2, color='darkblue', label='Smoothed')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Target')
    axes[0, 0].fill_between(df['epoch'], -0.1, 0.1, alpha=0.2, color='green', label='Good range')
    axes[0, 0].set_title('Convergence of μ Mean to 0', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('μ Mean')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Convergence to unit variance
    axes[0, 1].plot(df['epoch'], df['actual_var'], alpha=0.3, color='red', label='Raw')
    axes[0, 1].plot(df['epoch'], actual_var_smooth, linewidth=2, color='darkred', label='Smoothed')
    axes[0, 1].axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Target')
    axes[0, 1].fill_between(df['epoch'], 0.8, 1.2, alpha=0.2, color='green', label='Good range')
    axes[0, 1].set_title('Convergence of σ² to 1', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('σ² (Actual Variance)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Overall convergence metric
    distance_from_ideal = np.sqrt((df['mu_mean'] - 0)**2 + (df['actual_var'] - 1)**2)
    distance_smooth = distance_from_ideal.rolling(window=window_size, center=True).mean()
    
    axes[1, 0].plot(df['epoch'], distance_from_ideal, alpha=0.3, color='purple', label='Raw')
    axes[1, 0].plot(df['epoch'], distance_smooth, linewidth=2, color='darkmagenta', label='Smoothed')
    axes[1, 0].set_title('Overall Distance from N(0,1)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('√((μ-0)² + (σ²-1)²)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Stability analysis (variance of recent measurements)
    stability_window = max(10, len(df) // 10)
    mu_stability = df['mu_mean'].rolling(window=stability_window).std()
    var_stability = df['actual_var'].rolling(window=stability_window).std()
    
    axes[1, 1].plot(df['epoch'], mu_stability, label='μ Mean Stability', linewidth=2, color='blue')
    axes[1, 1].plot(df['epoch'], var_stability, label='σ² Stability', linewidth=2, color='red')
    axes[1, 1].set_title(f'Training Stability\n(Rolling Std, window={stability_window})', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Rolling Standard Deviation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence analysis plots saved to: {save_path}")
    
    plt.show()

def plot_summary_stats(df, save_path=None):
    """Plot summary statistics and final assessment"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Final values assessment
    final_mu_mean = df['mu_mean'].iloc[-1]
    final_var = df['actual_var'].iloc[-1]
    final_distance = np.sqrt(final_mu_mean**2 + (final_var - 1)**2)
    
    # Performance over time
    epochs = df['epoch'].values
    performance = 1 / (1 + np.sqrt(df['mu_mean']**2 + (df['actual_var'] - 1)**2))
    
    axes[0, 0].plot(epochs, performance, linewidth=2, color='darkgreen')
    axes[0, 0].set_title('Latent Space Quality Score\n1/(1+distance from N(0,1))', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Quality Score (0-1)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Box plots of early vs late training
    split_point = len(df) // 2
    early_data = df.iloc[:split_point]
    late_data = df.iloc[split_point:]
    
    box_data = [
        early_data['mu_mean'], late_data['mu_mean'],
        early_data['actual_var'], late_data['actual_var']
    ]
    box_labels = ['μ Early', 'μ Late', 'σ² Early', 'σ² Late']
    
    bp = axes[0, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
    colors = ['lightblue', 'darkblue', 'lightcoral', 'darkred']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(y=1, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Early vs Late Training Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning curve derivatives (rate of change)
    mu_rate = -np.gradient(np.abs(df['mu_mean']))  # Negative because we want decreasing absolute value
    var_rate = -np.gradient(np.abs(df['actual_var'] - 1))  # Rate of approaching 1
    
    axes[1, 0].plot(epochs, mu_rate, label='μ Improvement Rate', linewidth=2, color='blue')
    axes[1, 0].plot(epochs, var_rate, label='σ² Improvement Rate', linewidth=2, color='red')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Improvement Rates\n(Negative gradient of |error|)', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Rate of Improvement')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary text box
    axes[1, 1].axis('off')
    
    # Calculate final assessment
    mu_status = "✅ GOOD" if abs(final_mu_mean) < 0.1 else ("⚠️ OK" if abs(final_mu_mean) < 0.3 else "❌ POOR")
    var_status = "✅ GOOD" if 0.8 < final_var < 1.2 else ("⚠️ OK" if 0.5 < final_var < 1.5 else "❌ POOR")
    
    # Check if converged (low variance in recent epochs)
    recent_stability = df['mu_mean'].tail(20).std() if len(df) >= 20 else df['mu_mean'].std()
    converged = "✅ CONVERGED" if recent_stability < 0.01 else "⚠️ STILL LEARNING"
    
    # Calculate improvement
    initial_distance = np.sqrt(df['mu_mean'].iloc[0]**2 + (df['actual_var'].iloc[0] - 1)**2)
    improvement = ((initial_distance - final_distance) / initial_distance) * 100
    
    summary_text = f"""
LATENT SPACE ANALYSIS
====================

FINAL VALUES:
μ Mean: {final_mu_mean:.4f} {mu_status}
σ² (Var): {final_var:.4f} {var_status}
Distance from N(0,1): {final_distance:.4f}

TRAINING PROGRESS:
Total Epochs: {len(df)}
Improvement: {improvement:.1f}%
Status: {converged}

TARGETS:
μ Mean ≈ 0 (current: {abs(final_mu_mean):.4f})
σ² ≈ 1 (current: {final_var:.4f})

ASSESSMENT:
Quality Score: {performance.iloc[-1]:.3f}/1.000
Overall: {'🎯 EXCELLENT' if performance.iloc[-1] > 0.9 else ('👍 GOOD' if performance.iloc[-1] > 0.8 else '⚠️ NEEDS WORK')}
"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary statistics saved to: {save_path}")
    
    plt.show()

def create_latent_report(log_path, output_dir=None):
    """Create a comprehensive latent space analysis report"""
    df = load_latent_stats(log_path)
    if df is None:
        return
    
    if output_dir is None:
        output_dir = Path(log_path).parent / "latent_analysis"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Creating comprehensive latent space report in: {output_dir}")
    print(f"Report will analyze {len(df)} epochs of latent statistics")
    
    # Generate all plots
    print("📊 Generating latent evolution plots...")
    plot_latent_evolution(df, output_dir / "01_latent_evolution.png")
    
    print("📊 Generating latent distributions...")
    plot_latent_distributions(df, output_dir / "02_latent_distributions.png")
    
    print("📊 Generating convergence analysis...")
    plot_convergence_analysis(df, output_dir / "03_convergence_analysis.png")
    
    print("📊 Generating summary statistics...")
    plot_summary_stats(df, output_dir / "04_summary_stats.png")
    
    # Save processed data
    df.to_csv(output_dir / "processed_latent_stats.csv", index=False)
    
    # Create HTML report
    create_html_report(df, output_dir)
    
    print(f"✅ Latent space analysis complete! Files saved in: {output_dir}")
    print(f"🌐 Open {output_dir / 'report.html'} in your browser for a summary")

def create_html_report(df, output_dir):
    """Create a simple HTML report for latent space analysis"""
    final_mu_mean = df['mu_mean'].iloc[-1]
    final_var = df['actual_var'].iloc[-1]
    final_distance = np.sqrt(final_mu_mean**2 + (final_var - 1)**2)
    
    mu_good = abs(final_mu_mean) < 0.1
    var_good = 0.8 < final_var < 1.2
    overall_good = mu_good and var_good
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>VAE Latent Space Analysis Report</title>
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
        <h1>🧠 VAE Latent Space Analysis Report</h1>
        
        <h2>📈 Key Metrics</h2>
        <div class="metric {'good' if mu_good else 'warning' if abs(final_mu_mean) < 0.3 else 'bad'}">
            <strong>Final μ Mean:</strong> {final_mu_mean:.4f} (Target: ~0.000)
        </div>
        <div class="metric {'good' if var_good else 'warning' if 0.5 < final_var < 1.5 else 'bad'}">
            <strong>Final σ² (Variance):</strong> {final_var:.4f} (Target: ~1.000)
        </div>
        <div class="metric">
            <strong>Distance from N(0,1):</strong> {final_distance:.4f}
        </div>
        <div class="metric">
            <strong>Total Training Epochs:</strong> {len(df)}
        </div>
        <div class="metric {'good' if overall_good else 'warning'}">
            <strong>Overall Assessment:</strong> {'🎯 Excellent latent space!' if overall_good else '⚠️ Needs improvement'}
        </div>
        
        <h2>📊 Visualizations</h2>
        <h3>Latent Space Evolution</h3>
        <img src="01_latent_evolution.png" alt="Latent space evolution over training">
        
        <h3>Distribution Analysis</h3>
        <img src="02_latent_distributions.png" alt="Latent space distributions">
        
        <h3>Convergence Analysis</h3>
        <img src="03_convergence_analysis.png" alt="Convergence to standard normal">
        
        <h3>Summary Statistics</h3>
        <img src="04_summary_stats.png" alt="Summary statistics">
        
        <h2>🔍 Analysis</h2>
        <p><strong>Mean Convergence:</strong> 
        {'✅ Excellent! The latent means are very close to 0.' if mu_good else 
         ('⚠️ Acceptable, but could be closer to 0.' if abs(final_mu_mean) < 0.3 else 
          '❌ Poor convergence - latent means are far from 0.')}
        </p>
        
        <p><strong>Variance Convergence:</strong> 
        {'✅ Excellent! The latent variance is very close to 1.' if var_good else
         ('⚠️ Acceptable, but variance could be closer to 1.' if 0.5 < final_var < 1.5 else
          '❌ Poor convergence - variance is far from 1.')}
        </p>
        
        <p><strong>Overall Latent Space Quality:</strong> 
        {'🎯 Your VAE has learned an excellent latent representation that closely follows a standard normal distribution.' if overall_good else
         '⚠️ Consider adjusting the β parameter in your loss function or training for more epochs.'}
        </p>
        
        <hr>
        <small>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
    </body>
    </html>
    """
    
    with open(output_dir / "report.html", "w") as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description='Analyze VAE Latent Space Statistics')
    parser.add_argument('--log_path', required=True, 
                       help='Path to latent_stats.csv file')
    parser.add_argument('--output_dir', default=None,
                       help='Directory to save plots (default: same as log file)')
    parser.add_argument('--plot_type', default='all',
                       choices=['evolution', 'distributions', 'convergence', 'summary', 'all'],
                       help='Type of plot to generate')
    
    args = parser.parse_args()
    
    # Load data
    df = load_latent_stats(args.log_path)
    if df is None:
        return
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = Path(args.log_path).parent / "latent_analysis"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots based on selection
    if args.plot_type == 'all':
        create_latent_report(args.log_path, args.output_dir)
    elif args.plot_type == 'evolution':
        plot_latent_evolution(df, output_dir / "latent_evolution.png")
    elif args.plot_type == 'distributions':
        plot_latent_distributions(df, output_dir / "latent_distributions.png")
    elif args.plot_type == 'convergence':
        plot_convergence_analysis(df, output_dir / "convergence_analysis.png")
    elif args.plot_type == 'summary':
        plot_summary_stats(df, output_dir / "summary_stats.png")

if __name__ == "__main__":
    main()