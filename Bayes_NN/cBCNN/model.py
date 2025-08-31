import torch
import torch.nn as nn
import torch.nn.functional as F
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator

@variational_estimator
class ConditionalBayesianChickCallDetector(nn.Module):
    def __init__(self, in_features, out_features, num_conditions=3):
        super().__init__()
        
        self.input_shape = in_features
        self.num_conditions = num_conditions
        self.num_classes = out_features
        
        # Prior parameters
        prior_sigma_1 = 0.5
        prior_sigma_2 = 0.05
        prior_pi = 0.3
        
        # Condition embedding layer
        self.condition_embedding_dim = 32
        self.condition_embedding = nn.Embedding(num_conditions, self.condition_embedding_dim)
        
        # Shared feature extraction layers (condition-agnostic)
        self.conv1 = BayesianConv2d(1, 32, kernel_size=(3, 3), padding=1, 
                                   prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = BayesianConv2d(32, 64, kernel_size=(3, 3), padding=1,
                                   prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Calculate flattened size
        self._to_linear = self._get_conv_output(in_features)
        
        # Condition-specific feature processing
        self.feature_dim = 256
        self.shared_features = BayesianLinear(self._to_linear, self.feature_dim,
                                            prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        
        # Conditional layers - separate processing paths for each condition
        self.conditional_layers = nn.ModuleList([
            nn.Sequential(
                BayesianLinear(self.feature_dim + self.condition_embedding_dim, 128,
                              prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi),
                nn.ReLU(),
                nn.Dropout(0.3),
                BayesianLinear(128, out_features,
                              prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
            ) for _ in range(num_conditions)
        ])
        
        # Alternative: Condition-modulated layers (more sophisticated)
        self.use_modulation = True
        if self.use_modulation:
            self.feature_modulation = BayesianLinear(self.condition_embedding_dim, self.feature_dim,
                                                   prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
            self.final_classifier = BayesianLinear(self.feature_dim, out_features,
                                                 prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        
        self.dropout = nn.Dropout(0.3)
        
    def _get_conv_output(self, shape):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *shape)
            x = F.max_pool2d(F.relu(self.bn1(self.conv1(dummy))), (2, 2))
            x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))
            return x.view(1, -1).size(1)
    
    def forward(self, x, condition_ids, sample=True):
        """
        Forward pass with condition information
        
        Args:
            x: Input spectrograms [batch_size, channels, height, width]
            condition_ids: Condition labels [batch_size] - type of chicken call
            sample: Whether to sample from Bayesian layers
        """
        # Fix input dimensions
        while x.dim() > 4:
            x = x.squeeze(1)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Shared feature extraction
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Shared feature processing
        features = F.relu(self.shared_features(x))
        features = self.dropout(features)
        
        # Get condition embeddings
        condition_embeds = self.condition_embedding(condition_ids)
        
        if self.use_modulation:
            # Condition-modulated approach
            # Generate modulation weights from condition embedding
            modulation_weights = torch.sigmoid(self.feature_modulation(condition_embeds))
            
            # Modulate features based on condition
            modulated_features = features * modulation_weights
            
            # Final classification
            output = self.final_classifier(modulated_features)
            
        else:
            # Condition-specific pathway approach
            batch_size = x.size(0)
            outputs = torch.zeros(batch_size, self.num_classes).to(x.device)
            
            # Process each sample through its corresponding conditional pathway
            for i in range(batch_size):
                condition_id = condition_ids[i].item()
                sample_features = features[i:i+1]  # Keep batch dimension
                sample_condition = condition_embeds[i:i+1]
                
                # Concatenate features with condition embedding
                conditional_input = torch.cat([sample_features, sample_condition], dim=1)
                
                # Pass through condition-specific pathway
                sample_output = self.conditional_layers[condition_id](conditional_input)
                outputs[i] = sample_output.squeeze(0)
        
        return output if self.use_modulation else outputs
    
    def predict_with_uncertainty(self, x, condition_ids, num_samples=100):
        """
        Get predictions with uncertainty estimates for conditional model
        """
        self.train()  # Enable sampling
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x, condition_ids, sample=True)
                predictions.append(F.softmax(pred, dim=1))
        
        predictions = torch.stack(predictions)
        
        # Calculate mean and uncertainty
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean(dim=1)
        
        return mean_pred, uncertainty


# FIXED: Proper Hierarchical Bayesian CNN
@variational_estimator
class HierarchicalBayesianChickCallDetector(nn.Module):
    def __init__(self, in_features, out_features, num_call_types=3):
        super().__init__()
        
        self.input_shape = in_features
        self.num_call_types = num_call_types
        self.num_classes = out_features
        
        prior_sigma_1, prior_sigma_2, prior_pi = 0.5, 0.05, 0.3
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            BayesianConv2d(1, 32, kernel_size=(3, 3), padding=1, 
                          prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            BayesianConv2d(32, 64, kernel_size=(3, 3), padding=1,
                          prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        
        # Calculate feature size
        self._to_linear = self._get_conv_output(in_features)
        
        # Shared feature processing
        self.shared_fc = BayesianLinear(self._to_linear, 256,
                                      prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        
        # Call type classifier (first stage) - predicts broad categories
        self.call_type_classifier = nn.Sequential(
            BayesianLinear(256, 128, 
                          prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi),
            nn.ReLU(),
            nn.Dropout(0.3),
            BayesianLinear(128, num_call_types,
                          prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        )
        
        # Final classifier (second stage) - uses both features AND predicted call type
        self.final_classifier = BayesianLinear(256 + num_call_types, out_features,
                                             prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        
        self.dropout = nn.Dropout(0.3)
    
    def _get_conv_output(self, shape):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *shape)
            x = self.feature_extractor(dummy)
            return x.view(1, -1).size(1)
    
    def forward(self, x, call_type_targets=None, sample=True):
        """
        Forward pass with hierarchical structure
        
        Args:
            x: Input spectrograms
            call_type_targets: Ground truth call types (only used during training)
            sample: Whether to sample from Bayesian layers
        """
        # Fix dimensions
        while x.dim() > 4:
            x = x.squeeze(1)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Extract features
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        features = F.relu(self.shared_fc(features))
        features = self.dropout(features)
        
        # First stage: predict call type
        call_type_logits = self.call_type_classifier(features)
        call_type_probs = F.softmax(call_type_logits, dim=1)
        
        # Second stage: final classification
        # During training, we can use ground truth call types for better supervision
        if self.training and call_type_targets is not None:
            # Use one-hot encoded ground truth call types
            call_type_one_hot = F.one_hot(call_type_targets, num_classes=self.num_call_types).float()
            conditional_features = torch.cat([features, call_type_one_hot], dim=1)
        else:
            # Use predicted call type probabilities
            conditional_features = torch.cat([features, call_type_probs], dim=1)
        
        final_output = self.final_classifier(conditional_features)
        
        if self.training:
            # Return both outputs during training for multi-task loss
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
        
        # Calculate mean and uncertainty for final predictions
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean(dim=1)
        
        results = {'predictions': mean_pred, 'uncertainty': uncertainty}
        
        # If we have call type predictions, add them too
        if call_type_predictions:
            call_type_predictions = torch.stack(call_type_predictions)
            results['call_type_predictions'] = call_type_predictions.mean(dim=0)
            results['call_type_uncertainty'] = call_type_predictions.var(dim=0).mean(dim=1)
        
        return results


# Standard Bayesian CNN (fallback option)
@variational_estimator
class BayesianChickCallDetector(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.input_shape = in_features
        self.num_classes = out_features
        
        # Prior parameters
        prior_sigma_1 = 0.5
        prior_sigma_2 = 0.05
        prior_pi = 0.3
        
        # Feature extraction layers
        self.conv1 = BayesianConv2d(1, 32, kernel_size=(3, 3), padding=1, 
                                   prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = BayesianConv2d(32, 64, kernel_size=(3, 3), padding=1,
                                   prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Calculate flattened size
        self._to_linear = self._get_conv_output(in_features)
        
        # Classifier layers
        self.fc1 = BayesianLinear(self._to_linear, 256,
                                 prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        
        self.fc2 = BayesianLinear(256, 128,
                                 prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        
        self.fc3 = BayesianLinear(128, out_features,
                                 prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        
        self.dropout = nn.Dropout(0.3)
        
    def _get_conv_output(self, shape):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *shape)
            x = F.max_pool2d(F.relu(self.bn1(self.conv1(dummy))), (2, 2))
            x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))
            return x.view(1, -1).size(1)
    
    def forward(self, x, sample=True):
        """
        Forward pass
        
        Args:
            x: Input spectrograms [batch_size, channels, height, width]
            sample: Whether to sample from Bayesian layers
        """
        # Fix input dimensions
        while x.dim() > 4:
            x = x.squeeze(1)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Feature extraction
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def predict_with_uncertainty(self, x, num_samples=100):
        """Get predictions with uncertainty estimates"""
        self.train()  # Enable sampling
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x, sample=True)
                predictions.append(F.softmax(pred, dim=1))
        
        predictions = torch.stack(predictions)
        
        # Calculate mean and uncertainty
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean(dim=1)
        
        return mean_pred, uncertainty