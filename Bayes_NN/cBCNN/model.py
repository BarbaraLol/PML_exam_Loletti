# MEMORY-OPTIMIZED VERSION of model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator

@variational_estimator
class HierarchicalBayesianChickCallDetector(nn.Module):
    def __init__(self, in_features, out_features, num_call_types=3):
        super().__init__()
        
        self.input_shape = in_features
        self.num_call_types = num_call_types
        self.num_classes = out_features
        
        prior_sigma_1, prior_sigma_2, prior_pi = 0.1, 0.02, 0.3  # REDUCED PRIORS
        
        # SMALLER feature extractor
        self.feature_extractor = nn.Sequential(
            BayesianConv2d(1, 16, kernel_size=(3, 3), padding=1,  # REDUCED from 32 to 16
                          prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            BayesianConv2d(16, 32, kernel_size=(3, 3), padding=1,  # REDUCED from 64 to 32
                          prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)  # INCREASED dropout
        )
        
        # Calculate feature size
        self._to_linear = self._get_conv_output(in_features)
        print(f"Flattened feature size: {self._to_linear}")  # DEBUG
        
        # ADD ADDITIONAL POOLING to reduce feature size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Force to 4x4
        
        # SMALLER shared feature processing
        reduced_features = 32 * 16  # 32 channels * 4*4 spatial = 512
        self.shared_fc = BayesianLinear(reduced_features, 64,  # REDUCED from 256 to 64
                                      prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        
        # SMALLER call type classifier
        self.call_type_classifier = nn.Sequential(
            BayesianLinear(64, 32,  # REDUCED from 128 to 32
                          prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi),
            nn.ReLU(),
            nn.Dropout(0.4),
            BayesianLinear(32, num_call_types,
                          prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        )
        
        # SMALLER final classifier
        self.final_classifier = BayesianLinear(64 + num_call_types, out_features,  # REDUCED
                                             prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        
        self.dropout = nn.Dropout(0.4)
    
    def _get_conv_output(self, shape):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *shape)
            x = self.feature_extractor(dummy)
            x = self.adaptive_pool(x)  # Apply adaptive pooling
            return x.view(1, -1).size(1)
    
    def forward(self, x, call_type_targets=None, sample=True):
        # Fix dimensions
        while x.dim() > 4:
            x = x.squeeze(1)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # APPLY ADAPTIVE POOLING to control feature size
        features = self.adaptive_pool(features)
        
        features = features.view(features.size(0), -1)
        features = F.relu(self.shared_fc(features))
        features = self.dropout(features)
        
        # First stage: predict call type
        call_type_logits = self.call_type_classifier(features)
        call_type_probs = F.softmax(call_type_logits, dim=1)
        
        # Second stage: final classification
        if self.training and call_type_targets is not None:
            call_type_one_hot = F.one_hot(call_type_targets, num_classes=self.num_call_types).float()
            conditional_features = torch.cat([features, call_type_one_hot], dim=1)
        else:
            conditional_features = torch.cat([features, call_type_probs], dim=1)
        
        final_output = self.final_classifier(conditional_features)
        
        if self.training:
            return final_output, call_type_logits
        else:
            return final_output
    
    def predict_with_uncertainty(self, x, num_samples=100):
        """Get predictions with uncertainty estimates"""
        self.train()  # Enable sampling
        predictions = []
        call_type_predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                if self.training:
                    final_pred, call_type_pred = self.forward(x, sample=True)
                    call_type_predictions.append(F.softmax(call_type_pred, dim=1))
                else:
                    final_pred = self.forward(x, sample=True)
                predictions.append(F.softmax(final_pred, dim=1))
        
        predictions = torch.stack(predictions)
        
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean(dim=1)
        
        results = {'predictions': mean_pred, 'uncertainty': uncertainty}
        
        if call_type_predictions:
            call_type_predictions = torch.stack(call_type_predictions)
            results['call_type_predictions'] = call_type_predictions.mean(dim=0)
            results['call_type_uncertainty'] = call_type_predictions.var(dim=0).mean(dim=1)
        
        return results