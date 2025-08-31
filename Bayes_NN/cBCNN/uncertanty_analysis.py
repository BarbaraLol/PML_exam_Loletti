import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import argparse
import os
from model import HierarchicalBayesianChickCallDetector  # CHANGED: Use hierarchical model
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

def get_call_type_targets(targets, call_type_mapping):
    """Convert fine-grained targets to call type targets"""
    return torch.tensor([call_type_mapping[target.item()] for target in targets], 
                       dtype=torch.long, device=targets.device)

def analyze_prediction_uncertainty(model, dataloader, device, call_type_mapping, num_samples=50, class_names=None):
    """
    Analyze prediction uncertainty using Monte Carlo sampling for hierarchical model
    """
    model.eval()  # Set to eval but we'll enable training mode for sampling
    
    all_mean_predictions = []
    all_uncertainties = []
    all_targets = []
    all_predicted_classes = []
    all_epistemic_uncertainties = []
    all_aleatoric_uncertainties = []
    
    # For hierarchical model, also track call type predictions
    all_call_type_predictions = []
    all_call_type_targets = []
    all_call_type_uncertainties = []
    
    print(f"Analyzing prediction uncertainty with {num_samples} samples...")
    print(f"Processing {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            data, targets = fix_dims(batch)
            data = data.to(device)
            targets = targets.to(device)
            call_type_targets = get_call_type_targets(targets, call_type_mapping)
            
            batch_predictions = []
            batch_call_type_predictions = []
            
            # Sample multiple predictions for each input
            model.train()  # Enable sampling in Bayesian layers
            for sample_idx in range(num_samples):
                # For hierarchical model during inference (no call type targets provided)
                final_pred, call_type_pred = model(data, call_type_targets=None)
                
                final_pred_probs = F.softmax(final_pred, dim=1)
                call_type_pred_probs = F.softmax(call_type_pred, dim=1)
                
                batch_predictions.append(final_pred_probs.cpu())
                batch_call_type_predictions.append(call_type_pred_probs.cpu())
            
            # Stack predictions: [num_samples, batch_size, num_classes]
            batch_predictions = torch.stack(batch_predictions)
            batch_call_type_predictions = torch.stack(batch_call_type_predictions)
            
            # Calculate mean predictions
            mean_pred = batch_predictions.mean(dim=0)
            mean_call_type_pred = batch_call_type_predictions.mean(dim=0)
            
            # Calculate epistemic uncertainty (predictive entropy)
            epistemic_uncertainty = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=1)
            call_type_epistemic_uncertainty = -torch.sum(mean_call_type_pred * torch.log(mean_call_type_pred + 1e-8), dim=1)
            
            # Calculate aleatoric uncertainty (expected entropy)
            individual_entropies = -torch.sum(batch_predictions * torch.log(batch_predictions + 1e-8), dim=2)
            aleatoric_uncertainty = individual_entropies.mean(dim=0)
            
            call_type_individual_entropies = -torch.sum(batch_call_type_predictions * torch.log(batch_call_type_predictions + 1e-8), dim=2)
            call_type_aleatoric_uncertainty = call_type_individual_entropies.mean(dim=0)
            
            # Total uncertainty
            total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
            call_type_total_uncertainty = call_type_epistemic_uncertainty + call_type_aleatoric_uncertainty
            
            # Get predicted classes
            predicted_classes = torch.argmax(mean_pred, dim=1)
            
            # Store results
            all_mean_predictions.append(mean_pred)
            all_uncertainties.append(total_uncertainty)
            all_epistemic_uncertainties.append(epistemic_uncertainty)
            all_aleatoric_uncertainties.append(aleatoric_uncertainty)
            all_targets.append(targets.cpu())
            all_predicted_classes.append(predicted_classes)
            
            # Store call type results
            all_call_type_predictions.append(mean_call_type_pred)
            all_call_type_targets.append(call_type_targets.cpu())
            all_call_type_uncertainties.append(call_type_total_uncertainty)
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx+1}/{len(dataloader)}")
    
    # Concatenate all results
    all_mean_predictions = torch.cat(all_mean_predictions, dim=0)
    all_uncertainties = torch.cat(all_uncertainties, dim=0)
    all_epistemic_uncertainties = torch.cat(all_epistemic_uncertainties, dim=0)
    all_aleatoric_uncertainties = torch.cat(all_aleatoric_uncertainties, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_predicted_classes = torch.cat(all_predicted_classes, dim=0)
    
    # Call type results
    all_call_type_predictions = torch.cat(all_call_type_predictions, dim=0)
    all_call_type_targets = torch.cat(all_call_type_targets, dim=0)
    all_call_type_uncertainties = torch.cat(all_call_type_uncertainties, dim=0)
    
    # Calculate accuracies
    accuracy = (all_predicted_classes == all_targets).float().mean().item()
    call_type_predicted_classes = torch.argmax(all_call_type_predictions, dim=1)
    call_type_accuracy = (call_type_predicted_classes == all_call_type_targets).float().mean().item()
    
    print(f"Uncertainty analysis completed!")
    print(f"Overall classification accuracy: {accuracy*100:.2f}%")
    print(f"Call type classification accuracy: {call_type_accuracy*100:.2f}%")
    
    return {
        'predictions': all_mean_predictions.numpy(),
        'uncertainties': all_uncertainties.numpy(),
        'epistemic_uncertainties': all_epistemic_uncertainties.numpy(),
        'aleatoric_uncertainties': all_aleatoric_uncertainties.numpy(),
        'targets': all_targets.numpy(),
        'predicted_classes': all_predicted_classes.numpy(),
        'accuracy': accuracy,
        'call_type_predictions': all_call_type_predictions.numpy(),
        'call_type_targets': all_call_type_targets.numpy(),
        'call_type_uncertainties': all_call_type_uncertainties.numpy(),
        'call_type_accuracy': call_type_accuracy,
        'num_samples': num_samples
    }

def plot_uncertainty_analysis(results, class_names=None, call_type_names=None, save_dir=None):
    """
    Create comprehensive uncertainty analysis plots for hierarchical model
    """
    predictions = results['predictions']
    uncertainties = results['uncertainties']
    epistemic_uncertainties = results['epistemic_uncertainties']
    aleatoric_uncertainties = results['aleatoric_uncertainties']
    targets = results['targets']
    predicted_classes = results['predicted_classes']
    accuracy = results['accuracy']
    
    # Call type results
    call_type_predictions = results['call_type_predictions']
    call_type_targets = results['call_type_targets']
    call_type_uncertainties = results['call_type_uncertainties']
    call_type_accuracy = results['call_type_accuracy']
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(predictions.shape[1])]
    
    if call_type_names is None:
        call_type_names = [f'Call Type {i}' for i in range(call_type_predictions.shape[1])]
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    fig.suptitle(f'Hierarchical Bayesian CNN Uncertainty Analysis\n' +
                 f'Classification Acc: {accuracy*100:.2f}% | Call Type Acc: {call_type_accuracy*100:.2f}%', 
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
    
    # 4. Call type uncertainty distribution
    axes[1, 0].hist(call_type_uncertainties, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Call Type Uncertainty')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Call Type Uncertainties')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(np.mean(call_type_uncertainties), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(call_type_uncertainties):.3f}')
    axes[1, 0].legend()
    
    # 5. Call Type vs Final Classification Uncertainty
    axes[1, 1].scatter(call_type_uncertainties, uncertainties, alpha=0.6, s=20, 
                      c=correct_predictions, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 1].set_xlabel('Call Type Uncertainty')
    axes[1, 1].set_ylabel('Final Classification Uncertainty')
    axes[1, 1].set_title('Call Type vs Classification Uncertainty\n(Green=Correct, Red=Wrong)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Confusion Matrix for Final Classification
    cm = confusion_matrix(targets, predicted_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 2])
    axes[1, 2].set_title('Final Classification Confusion Matrix')
    axes[1, 2].set_ylabel('True Class')
    axes[1, 2].set_xlabel('Predicted Class')
    
    # 7. Call Type Confusion Matrix
    call_type_predicted_classes = np.argmax(call_type_predictions, axis=1)
    call_type_cm = confusion_matrix(call_type_targets, call_type_predicted_classes)
    sns.heatmap(call_type_cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=call_type_names, yticklabels=call_type_names, ax=axes[2, 0])
    axes[2, 0].set_title('Call Type Confusion Matrix')
    axes[2, 0].set_ylabel('True Call Type')
    axes[2, 0].set_xlabel('Predicted Call Type')
    
    # 8. Confidence vs Uncertainty scatter
    max_probs = np.max(predictions, axis=1)  # Confidence = max probability
    axes[2, 1].scatter(max_probs, uncertainties, alpha=0.6, s=20, c=correct_predictions, 
                      cmap='RdYlGn', vmin=0, vmax=1)
    axes[2, 1].set_xlabel('Prediction Confidence (Max Probability)')
    axes[2, 1].set_ylabel('Total Uncertainty')
    axes[2, 1].set_title('Confidence vs Uncertainty (Green=Correct, Red=Wrong)')
    axes[2, 1].grid(True, alpha=0.3)
    
    # 9. High uncertainty samples analysis
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
    
    ax2_2 = axes[2, 2]
    ax2_2_twin = ax2_2.twinx()
    
    bars = ax2_2.bar([f'Top {100-t}%' for t in uncertainty_thresholds], threshold_accuracies, 
                     alpha=0.7, color='orange', label='Accuracy')
    line = ax2_2_twin.plot([f'Top {100-t}%' for t in uncertainty_thresholds], threshold_counts, 
                          'ro-', label='Sample Count')
    
    ax2_2.set_ylabel('Accuracy (%)', color='orange')
    ax2_2_twin.set_ylabel('Number of Samples', color='red')
    ax2_2.set_title('Accuracy vs Uncertainty Percentiles')
    ax2_2.tick_params(axis='x', rotation=45)
    ax2_2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, threshold_accuracies):
        ax2_2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 10. Calibration plot (reliability diagram)
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
    
    axes[3, 0].bar(range(n_bins), bin_accuracies, alpha=0.7, color='lightblue', 
                   edgecolor='black', label='Accuracy')
    axes[3, 0].plot(range(n_bins), [bin_confidences[i] for i in range(n_bins)], 
                    'ro-', markersize=6, label='Confidence')
    axes[3, 0].plot([0, n_bins-1], [bin_confidences[0], bin_confidences[-1]], 
                    'k--', alpha=0.5, label='Perfect calibration')
    axes[3, 0].set_xlabel('Confidence Bin')
    axes[3, 0].set_ylabel('Accuracy / Confidence')
    axes[3, 0].set_title('Model Calibration (Reliability Diagram)')
    axes[3, 0].legend()
    axes[3, 0].grid(True, alpha=0.3)
    
    # 11. Hierarchical Performance Analysis
    axes[3, 1].axis('off')
    hierarchical_text = f"""
HIERARCHICAL MODEL ANALYSIS
{'='*35}

Call Type Performance:
• Accuracy: {call_type_accuracy*100:.2f}%
• Mean uncertainty: {np.mean(call_type_uncertainties):.4f}
• Std uncertainty: {np.std(call_type_uncertainties):.4f}

Final Classification Performance:
• Accuracy: {accuracy*100:.2f}%
• Mean uncertainty: {np.mean(uncertainties):.4f}
• Std uncertainty: {np.std(uncertainties):.4f}

Uncertainty Correlation:
• Call type vs final: {np.corrcoef(call_type_uncertainties, uncertainties)[0,1]:.3f}

Performance by Call Type:
"""
    
    # Add per-call-type analysis
    for call_type_idx in range(len(call_type_names)):
        mask = call_type_targets == call_type_idx
        if mask.sum() > 0:
            ct_acc = (predicted_classes[mask] == targets[mask]).mean()
            hierarchical_text += f"• {call_type_names[call_type_idx]}: {ct_acc*100:.1f}% ({mask.sum()} samples)\n"
    
    axes[3, 1].text(0.05, 0.95, hierarchical_text, transform=axes[3, 1].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    # 12. Summary statistics text
    axes[3, 2].axis('off')
    summary_text = f"""
UNCERTAINTY ANALYSIS SUMMARY
{'='*40}

Dataset Statistics:
• Total samples: {len(uncertainties):,}
• Number of classes: {len(class_names)}
• Number of call types: {len(call_type_names)}
• Overall accuracy: {accuracy*100:.2f}%
• Call type accuracy: {call_type_accuracy*100:.2f}%

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
• Mean uncertainty (correct): {np.mean(correct_uncertainties):.}