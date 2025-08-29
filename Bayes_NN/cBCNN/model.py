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


# Modified training function for conditional BCNN
def train_conditional_model(model, train_loader, val_loader, epochs=100, device='cuda'):
    """
    Training loop for conditional Bayesian CNN
    """
    from blitz.losses import kl_divergence_from_nn
    import torch.optim as optim
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=8)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for data, targets in train_loader:
            # Fix dimensions
            while data.dim() > 4:
                data = data.squeeze(1)
            
            data, targets = data.to(device), targets.to(device)
            
            # For conditional training, we use the target labels as conditions
            # This assumes we know the call type during training
            condition_ids = targets  # Use true labels as conditions
            
            optimizer.zero_grad()
            outputs = model(data, condition_ids)
            
            # Calculate losses
            nll_loss = criterion(outputs, targets)
            kl_loss = kl_divergence_from_nn(model) / len(train_loader.dataset)
            loss = nll_loss + kl_loss
            
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += nll_loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                while data.dim() > 4:
                    data = data.squeeze(1)
                
                data, targets = data.to(device), targets.to(device)
                
                # During validation, we still use true labels as conditions
                # In real inference, you'd need to predict or know the call type
                condition_ids = targets
                
                outputs = model(data, condition_ids, sample=False)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * data.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100. * val_correct / val_total
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, Best={best_val_acc:.2f}%")
    
    return model


# Hierarchical approach: First predict call type, then classify within type
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
        
        # Call type classifier (first stage)
        self.call_type_classifier = nn.Sequential(
            BayesianLinear(self._to_linear, 128, 
                          prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi),
            nn.ReLU(),
            nn.Dropout(0.3),
            BayesianLinear(128, num_call_types,
                          prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        )
        
        # Final classifier (second stage) - conditioned on call type
        self.final_classifier = BayesianLinear(self._to_linear + num_call_types, out_features,
                                             prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
    
    def _get_conv_output(self, shape):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *shape)
            x = self.feature_extractor(dummy)
            return x.view(1, -1).size(1)
    
    def forward(self, x, sample=True):
        # Fix dimensions
        while x.dim() > 4:
            x = x.squeeze(1)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Extract features
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        # First stage: predict call type
        call_type_logits = self.call_type_classifier(features)
        call_type_probs = F.softmax(call_type_logits, dim=1)
        
        # Second stage: final classification conditioned on call type
        conditional_features = torch.cat([features, call_type_probs], dim=1)
        final_output = self.final_classifier(conditional_features)
        
        if self.training:
            # Return both outputs during training for multi-task loss
            return final_output, call_type_logits
        else:
            return final_output

# Usage examples:
"""
# 1. Conditional BCNN (requires known call type)
conditional_model = ConditionalBayesianChickCallDetector(
    in_features=spectrogram_shape, 
    out_features=num_classes, 
    num_conditions=3  # 3 types of chicken calls
)

# 2. Hierarchical BCNN (learns to predict call type first)
hierarchical_model = HierarchicalBayesianChickCallDetector(
    in_features=spectrogram_shape,
    out_features=num_classes,
    num_call_types=3
)

# Training with hierarchical loss
def train_hierarchical_model(model, train_loader, device='cuda'):
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    classification_loss = nn.CrossEntropyLoss()
    call_type_loss = nn.CrossEntropyLoss()
    
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        
        # Assume you have call_type_targets as well
        call_type_targets = get_call_type_from_targets(targets)  # You need to implement this
        
        final_output, call_type_output = model(data)
        
        # Multi-task loss
        loss1 = classification_loss(final_output, targets)
        loss2 = call_type_loss(call_type_output, call_type_targets)
        kl_loss = kl_divergence_from_nn(model) / len(train_loader.dataset)
        
        total_loss = loss1 + 0.5 * loss2 + kl_loss  # Weight the auxiliary task
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
"""