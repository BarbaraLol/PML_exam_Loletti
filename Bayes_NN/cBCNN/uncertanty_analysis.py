import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import argparse
import os
from model import BayesianChickCallDetector
from data_loading import SpectrogramDataset, load_file_paths, encode_labels
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import json

def fix_dims(batch):
    """Fix input dimensions"""
    data, target = batch
    while data.dim() > 4:
        data = data.squeeze(1)
    return data, target

def analyze_prediction_uncertainty(model, dataloader, device, num_samples=50, class_names=None):
    """
    Analyze prediction uncertainty using Monte Carlo sampling
    """
    model.eval()  # Set to eval but we'll enable training mode for sampling
    
    all_mean_predictions = []
    all_uncertainties = []
    all_targets = []
    all_predicted_classes = []
    all_epistemic_uncertainties = []
    all_aleatoric_uncertainties = []
    
    print(f"Analyzing prediction uncertainty with {num_samples} samples...")
    print(f"Processing {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            data, targets = fix_dims(batch)
            data = data.to(device)
            batch_predictions = []
            
            # Sample multiple predictions for each input
            model.train()  # Enable sampling in Bayesian layers
            for sample_idx in range(num_samples):
                pred = model(data, sample=True)
                pred_probs = F.softmax(pred, dim=1)
                batch_predictions.append(pred_probs.cpu())
            
            # Stack predictions: [num_samples, batch_size, num_classes]
            batch_predictions = torch.stack(batch_predictions)
            
            # Calculate mean predictions
            mean_pred = batch_predictions.mean(dim=0)
            
            # Calculate epistemic uncertainty (predictive entropy)
            epistemic_uncertainty = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=1)
            
            # Calculate aleatoric uncertainty (expected entropy)
            individual_entropies = -torch.sum(batch_predictions * torch.log(batch_predictions + 1e-8), dim=2)
            aleatoric_uncertainty = individual_entropies.mean(dim=0)
            
            # Total uncertainty
            total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
            
            # Get predicted classes
            predicted_classes = torch.argmax(mean_pred, dim=1)
            
            # Store results
            all_mean_predictions.append(mean_pred)
            all_uncertainties.append(total_uncertainty)
            all_epistemic_uncertainties.append(epistemic_uncertainty)
            all_aleatoric_uncertainties.append(aleatoric_uncertainty)
            all_targets.append(targets)
            all_predicted_classes.append(predicted_classes)
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx+1}/{len(dataloader)}")
    
    # Concatenate all results
    all_mean_predictions = torch.cat(all_mean_predictions, dim=0)
    all_uncertainties = torch.cat(all_uncertainties, dim=0)
    all_epistemic_uncertainties = torch.cat(all_epistemic_uncertainties, dim=0)
    all_aleatoric_uncertainties = torch.cat(all_aleatoric_uncertainties, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_predicted_classes = torch.cat(all_predicted_classes, dim=0)
    
    # Calculate accuracy
    accuracy = (all_predicted_classes == all_targets).float().mean().item()
    
    print(f"Uncertainty analysis completed!")
    print(f"Overall accuracy: {accuracy*100:.2f}%")
    
    return {
        'predictions': all_mean_predictions.numpy(),
        'uncertainties': all_uncertainties.numpy(),
        'epistemic_uncertainties': all_epistemic_uncertainties.numpy(),
        'aleatoric_uncertainties': all_aleatoric_uncertainties.numpy(),
        'targets': all_targets.numpy(),
        'predicted_classes': all_predicted_classes.numpy(),
        'accuracy': accuracy,
        'num_samples': num_samples
    }

def plot_uncertainty_analysis(results, class_names=None, save_dir=None):
    """
    Create comprehensive uncertainty analysis plots
    """
    predictions = results['predictions']
    uncertainties = results['uncertainties']
    epistemic_uncertainties = results['epistemic_uncertainties']
    aleatoric_uncertainties = results['aleatoric_uncertainties']
    targets = results['targets']
    predicted_classes = results['predicted_classes']
    accuracy = results['accuracy']
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(predictions.shape[1])]
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle(f'Bayesian CNN Uncertainty Analysis (Accuracy: {accuracy*100:.2f}%)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Total uncertainty distribution
    axes[0, 0].hist(uncertainties, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Total Uncertainty')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Total Uncertainties')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(np.mean(uncertainties), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(uncertainties):.3f}')
    axes[0, 0].legend()
    
    # 2. Epistemic vs Aleatoric uncertainty
    axes[0, 1].scatter(epistemic_uncertainties, aleatoric_uncertainties, alpha=0.6, s=20)
    axes[0, 1].set_xlabel('Epistemic Uncertainty (Model uncertainty)')
    axes[0, 1].set_ylabel('Aleatoric Uncertainty (Data uncertainty)')
    axes[0, 1].set_title('Epistemic vs Aleatoric Uncertainty')
    axes[0, 1].grid(True, alpha=0.3)
    # Add diagonal line
    max_val = max(epistemic_uncertainties.max(), aleatoric_uncertainties.max())
    axes[0, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal uncertainty')
    axes[0, 1].legend()
    
    # 3. Uncertainty vs Accuracy
    correct_predictions = (predicted_classes == targets)
    correct_uncertainties = uncertainties[correct_predictions]
    incorrect_uncertainties = uncertainties[~correct_predictions]
    
    axes[0, 2].hist(correct_uncertainties, bins=30, alpha=0.7, label='Correct', color='green')
    axes[0, 2].hist(incorrect_uncertainties, bins=30, alpha=0.7, label='Incorrect', color='red')
    axes[0, 2].set_xlabel('Total Uncertainty')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Uncertainty: Correct vs Incorrect Predictions')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Uncertainty by true class
    class_uncertainties = []
    for class_idx in range(len(class_names)):
        class_mask = targets == class_idx
        if class_mask.sum() > 0:
            class_uncertainties.append(uncertainties[class_mask])
        else:
            class_uncertainties.append([])
    
    bp = axes[1, 0].boxplot([unc for unc in class_uncertainties if len(unc) > 0], 
                           labels=[class_names[i] for i in range(len(class_names)) if len(class_uncertainties[i]) > 0])
    axes[1, 0].set_ylabel('Total Uncertainty')
    axes[1, 0].set_title('Uncertainty Distribution by True Class')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Confidence vs Uncertainty scatter
    max_probs = np.max(predictions, axis=1)  # Confidence = max probability
    axes[1, 1].scatter(max_probs, uncertainties, alpha=0.6, s=20, c=correct_predictions, 
                      cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 1].set_xlabel('Prediction Confidence (Max Probability)')
    axes[1, 1].set_ylabel('Total Uncertainty')
    axes[1, 1].set_title('Confidence vs Uncertainty (Green=Correct, Red=Wrong)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Confusion Matrix
    cm = confusion_matrix(targets, predicted_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 2])
    axes[1, 2].set_title('Confusion Matrix')
    axes[1, 2].set_ylabel('True Class')
    axes[1, 2].set_xlabel('Predicted Class')
    
    # 7. High uncertainty samples analysis
    uncertainty_thresholds = [50, 75, 90, 95]
    threshold_accuracies = []
    threshold_counts = []
    
    for threshold in uncertainty_thresholds:
        high_unc_threshold = np.percentile(uncertainties, threshold)
        high_unc_mask = uncertainties > high_unc_threshold
        if high_unc_mask.sum() > 0:
            high_unc_acc = (predicted_classes[high_unc_mask] == targets[high_unc_mask]).mean()
            threshold_accuracies.append(high_unc_acc * 100)
            threshold_counts.append(high_unc_mask.sum())
        else:
            threshold_accuracies.append(0)
            threshold_counts.append(0)
    
    ax2_1 = axes[2, 0]
    ax2_2 = ax2_1.twinx()
    
    bars = ax2_1.bar([f'Top {100-t}%' for t in uncertainty_thresholds], threshold_accuracies, 
                     alpha=0.7, color='orange', label='Accuracy')
    line = ax2_2.plot([f'Top {100-t}%' for t in uncertainty_thresholds], threshold_counts, 
                      'ro-', label='Sample Count')
    
    ax2_1.set_ylabel('Accuracy (%)', color='orange')
    ax2_2.set_ylabel('Number of Samples', color='red')
    ax2_1.set_title('Accuracy vs Uncertainty Percentiles')
    ax2_1.tick_params(axis='x', rotation=45)
    ax2_1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, threshold_accuracies):
        ax2_1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 8. Calibration plot (reliability diagram)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        bin_counts.append(in_bin.sum())
        
        if prop_in_bin > 0:
            accuracy_in_bin = correct_predictions[in_bin].mean()
            avg_confidence_in_bin = max_probs[in_bin].mean()
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
    
    axes[2, 1].bar(range(n_bins), bin_accuracies, alpha=0.7, color='lightblue', 
                   edgecolor='black', label='Accuracy')
    axes[2, 1].plot(range(n_bins), [bin_confidences[i] for i in range(n_bins)], 
                    'ro-', markersize=6, label='Confidence')
    axes[2, 1].plot([0, n_bins-1], [bin_confidences[0], bin_confidences[-1]], 
                    'k--', alpha=0.5, label='Perfect calibration')
    axes[2, 1].set_xlabel('Confidence Bin')
    axes[2, 1].set_ylabel('Accuracy / Confidence')
    axes[2, 1].set_title('Model Calibration (Reliability Diagram)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # 9. Summary statistics text
    axes[2, 2].axis('off')
    summary_text = f"""
UNCERTAINTY ANALYSIS SUMMARY
{'='*40}

Dataset Statistics:
• Total samples: {len(uncertainties):,}
• Number of classes: {len(class_names)}
• Overall accuracy: {accuracy*100:.2f}%

Uncertainty Statistics:
• Mean total uncertainty: {np.mean(uncertainties):.4f}
• Std total uncertainty: {np.std(uncertainties):.4f}
• Mean epistemic uncertainty: {np.mean(epistemic_uncertainties):.4f}
• Mean aleatoric uncertainty: {np.mean(aleatoric_uncertainties):.4f}

High Uncertainty Analysis:
• Top 10% uncertain samples: {np.sum(uncertainties > np.percentile(uncertainties, 90)):,}
• Accuracy on high uncertainty: {(predicted_classes[uncertainties > np.percentile(uncertainties, 90)] == targets[uncertainties > np.percentile(uncertainties, 90)]).mean()*100:.2f}%
• Accuracy on low uncertainty: {(predicted_classes[uncertainties <= np.percentile(uncertainties, 90)] == targets[uncertainties <= np.percentile(uncertainties, 90)]).mean()*100:.2f}%

Model Behavior:
• Correctly classified samples have {'lower' if np.mean(correct_uncertainties) < np.mean(incorrect_uncertainties) else 'higher'} uncertainty
• Mean uncertainty (correct): {np.mean(correct_uncertainties):.4f}
• Mean uncertainty (incorrect): {np.mean(incorrect_uncertainties):.4f}
    """
    
    axes[2, 2].text(0.05, 0.95, summary_text, transform=axes[2, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'uncertainty_analysis.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(os.path.join(save_dir, 'uncertainty_analysis.pdf'), 
                   bbox_inches='tight', facecolor='white')
        print(f"Uncertainty analysis plots saved to {save_dir}")
    
    plt.show()
    
    # Print detailed summary
    print("\n" + "="*70)
    print("DETAILED UNCERTAINTY ANALYSIS SUMMARY")
    print("="*70)
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Total Samples: {len(uncertainties):,}")
    
    print(f"\nUNCERTAINTY STATISTICS:")
    print(f"Total Uncertainty - Mean: {np.mean(uncertainties):.4f}, Std: {np.std(uncertainties):.4f}")
    print(f"Epistemic Uncertainty - Mean: {np.mean(epistemic_uncertainties):.4f}, Std: {np.std(epistemic_uncertainties):.4f}")
    print(f"Aleatoric Uncertainty - Mean: {np.mean(aleatoric_uncertainties):.4f}, Std: {np.std(aleatoric_uncertainties):.4f}")
    
    print(f"\nCORRECT vs INCORRECT PREDICTIONS:")
    print(f"Mean uncertainty (correct predictions): {np.mean(correct_uncertainties):.4f}")
    print(f"Mean uncertainty (incorrect predictions): {np.mean(incorrect_uncertainties):.4f}")
    print(f"Uncertainty separation: {np.mean(incorrect_uncertainties) - np.mean(correct_uncertainties):.4f}")
    
    # High uncertainty analysis
    high_unc_threshold = np.percentile(uncertainties, 90)
    high_unc_mask = uncertainties > high_unc_threshold
    high_unc_accuracy = (predicted_classes[high_unc_mask] == targets[high_unc_mask]).mean()
    low_unc_accuracy = (predicted_classes[~high_unc_mask] == targets[~high_unc_mask]).mean()
    
    print(f"\nHIGH UNCERTAINTY ANALYSIS (Top 10%):")
    print(f"High uncertainty samples: {high_unc_mask.sum():,} ({high_unc_mask.mean()*100:.1f}%)")
    print(f"Accuracy on high uncertainty samples: {high_unc_accuracy*100:.2f}%")
    print(f"Accuracy on low uncertainty samples: {low_unc_accuracy*100:.2f}%")
    print(f"Accuracy difference: {(low_unc_accuracy - high_unc_accuracy)*100:.2f} percentage points")
    
    # Per-class analysis
    print(f"\nPER-CLASS UNCERTAINTY ANALYSIS:")
    for class_idx in range(len(class_names)):
        class_mask = targets == class_idx
        if class_mask.sum() > 0:
            class_accuracy = (predicted_classes[class_mask] == targets[class_mask]).mean()
            class_uncertainty = np.mean(uncertainties[class_mask])
            print(f"{class_names[class_idx]}: Accuracy={class_accuracy*100:.2f}%, "
                  f"Mean Uncertainty={class_uncertainty:.4f}, Samples={class_mask.sum():,}")
    
    # Classification report
    print(f"\nCLASSIFICATION REPORT:")
    print(classification_report(targets, predicted_classes, target_names=class_names, digits=3))
    
    return {
        'high_uncertainty_threshold': high_unc_threshold,
        'high_uncertainty_accuracy': high_unc_accuracy,
        'low_uncertainty_accuracy': low_unc_accuracy,
        'mean_correct_uncertainty': np.mean(correct_uncertainties),
        'mean_incorrect_uncertainty': np.mean(incorrect_uncertainties)
    }

def identify_challenging_samples(results, save_dir=None, top_n=20):
    """
    Identify and save the most challenging samples (highest uncertainty)
    """
    uncertainties = results['uncertainties']
    targets = results['targets']
    predicted_classes = results['predicted_classes']
    predictions = results['predictions']
    
    # Get indices of highest uncertainty samples
    high_uncertainty_indices = np.argsort(uncertainties)[-top_n:]
    
    challenging_samples_data = []
    
    print(f"\nTOP {top_n} MOST UNCERTAIN PREDICTIONS:")
    print("-" * 90)
    print(f"{'Rank':<4} {'Sample':<8} {'Uncertainty':<12} {'True':<8} {'Pred':<8} {'Correct':<8} {'Confidence':<12}")
    print("-" * 90)
    
    for i, idx in enumerate(high_uncertainty_indices):
        uncertainty = uncertainties[idx]
        true_class = targets[idx]
        pred_class = predicted_classes[idx]
        is_correct = true_class == pred_class
        confidence = np.max(predictions[idx])
        
        challenging_samples_data.append({
            'rank': i + 1,
            'sample_idx': int(idx),
            'uncertainty': float(uncertainty),
            'true_class': int(true_class),
            'predicted_class': int(pred_class),
            'correct': bool(is_correct),
            'confidence': float(confidence)
        })
        
        print(f"{i+1:<4} {idx:<8} {uncertainty:<12.4f} {true_class:<8} {pred_class:<8} "
              f"{str(is_correct):<8} {confidence:<12.4f}")
    
    # Save challenging samples data
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        challenging_file = os.path.join(save_dir, 'challenging_samples.json')
        with open(challenging_file, 'w') as f:
            json.dump(challenging_samples_data, f, indent=2)
        print(f"\nChallenging samples data saved to {challenging_file}")
    
    return high_uncertainty_indices, challenging_samples_data

def save_uncertainty_results(results, analysis_summary, save_dir):
    """
    Save all uncertainty analysis results to files
    """
    if save_dir is None:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save raw results
    results_file = os.path.join(save_dir, 'uncertainty_results.npz')
    np.savez(results_file,
             predictions=results['predictions'],
             uncertainties=results['uncertainties'],
             epistemic_uncertainties=results['epistemic_uncertainties'],
             aleatoric_uncertainties=results['aleatoric_uncertainties'],
             targets=results['targets'],
             predicted_classes=results['predicted_classes'],
             accuracy=results['accuracy'],
             num_samples=results['num_samples'])
    
    # Save analysis summary
    summary_file = os.path.join(save_dir, 'uncertainty_summary.json')
    summary_data = {
        'overall_accuracy': float(results['accuracy']),
        'total_samples': int(len(results['uncertainties'])),
        'num_monte_carlo_samples': int(results['num_samples']),
        'uncertainty_statistics': {
            'total_mean': float(np.mean(results['uncertainties'])),
            'total_std': float(np.std(results['uncertainties'])),
            'epistemic_mean': float(np.mean(results['epistemic_uncertainties'])),
            'epistemic_std': float(np.std(results['epistemic_uncertainties'])),
            'aleatoric_mean': float(np.mean(results['aleatoric_uncertainties'])),
            'aleatoric_std': float(np.std(results['aleatoric_uncertainties']))
        },
        'performance_analysis': analysis_summary
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Uncertainty results saved to {save_dir}")

def load_model_and_data(model_path, data_dir):
    """
    Load trained model and validation data
    """
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path)
    
    # Load data
    print(f"Loading data from {data_dir}...")
    file_paths = load_file_paths(data_dir)
    label_encoder = LabelEncoder()
    label_encoder.fit(encode_labels(file_paths))
    
    # Create dataset
    dataset = SpectrogramDataset(file_paths, label_encoder)
    sample_shape = torch.load(file_paths[0])['spectrogram'].shape
    num_classes = len(label_encoder.classes_)
    
    # Initialize model
    model = BayesianChickCallDetector(sample_shape, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    return model, dataset, label_encoder

def main():
    parser = argparse.ArgumentParser(description='Bayesian CNN Uncertainty Analysis')
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', required=True, help='Path to spectrogram directory')
    parser.add_argument('--output_dir', default='uncertainty_analysis', help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of Monte Carlo samples')
    parser.add_argument('--class_names', nargs='+', default=None, 
                       help='Class names (e.g., --class_names "Call Type 1" "Call Type 2" "Call Type 3")')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and data
    model, dataset, label_encoder = load_model_and_data(args.model_path, args.data_dir)
    
    # Use provided class names or generate default ones
    if args.class_names:
        class_names = args.class_names
    else:
        class_names = [f"Class_{name}" for name in label_encoder.classes_]
    
    print(f"Class names: {class_names}")
    
    # Create data loader (use full dataset or specify validation indices if available)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    print(f"Analyzing {len(dataset)} samples...")
    
    # Perform uncertainty analysis
    results = analyze_prediction_uncertainty(
        model, dataloader, device, num_samples=args.num_samples, class_names=class_names
    )
    
    # Create plots and analysis
    analysis_summary = plot_uncertainty_analysis(results, class_names=class_names, 
                                                save_dir=args.output_dir)
    
    # Identify challenging samples
    challenging_indices, challenging_data = identify_challenging_samples(
        results, save_dir=args.output_dir, top_n=20
    )
    
    # Save all results
    save_uncertainty_results(results, analysis_summary, args.output_dir)
    
    print(f"\nUncertainty analysis completed!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

# Usage example:
"""
python uncertainty_analysis.py --model_path results/best_model.pth --data_dir path/to/spectrograms --class_names "Broiler" "Layer" "Bantam" --num_samples 100
"""