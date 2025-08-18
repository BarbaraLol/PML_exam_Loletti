import torch
import torch.nn as nn
import torch.nn.functional as F
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator

@variational_estimator
class BayesianChickCallDetector(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # Store input shape for later use
        self.input_shape = in_features

        # Bayesian convolutional layers with batch normalization
        self.conv1 = BayesianConv2d(1, 16, kernel_size=(3, 3), padding=1, prior_sigma_1=0.1, prior_sigma_2=0.01, prior_pi=0.25)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = BayesianConv2d(16, 32, kernel_size=(3, 3), padding=1, prior_sigma_1=0.1, prior_sigma_2=0.01, prior_pi=0.25)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = BayesianConv2d(32, 64, kernel_size=(3, 3), padding=1, prior_sigma_1=0.1, prior_sigma_2=0.01, prior_pi=0.25)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = BayesianConv2d(64, 128, kernel_size=(3, 3), padding=1, prior_sigma_1=0.1, prior_sigma_2=0.01, prior_pi=0.25)
        self.bn4 = nn.BatchNorm2d(128)

        # Calculate flattened size after convolutions
        self._to_linear = self._get_conv_output(in_features)

        # Bayesian fully connected layers
        self.fc1 = BayesianLinear(self._to_linear, 512, prior_sigma_1=0.1, prior_sigma_2=0.01, prior_pi=0.25)
        self.fc2 = BayesianLinear(512, 256, prior_sigma_1=0.1, prior_sigma_2=0.01, prior_pi=0.25)
        self.fc3 = BayesianLinear(256, 128, prior_sigma_1=0.1, prior_sigma_2=0.01, prior_pi=0.25)
        self.fc4 = BayesianLinear(128, out_features, prior_sigma_1=0.1, prior_sigma_2=0.01, prior_pi=0.25)

        # Dropout layers
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)

    def _get_conv_output(self, shape):
        """Calculate the flattened size after all convolutions and pooling"""
        with torch.no_grad():
            # Create dummy input with correct shape
            dummy = torch.zeros(1, 1, *shape)
            
            # Apply same operations as forward pass
            x = F.max_pool2d(F.relu(self.bn1(self.conv1(dummy))), (2, 2))
            x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))
            x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), (2, 2))
            x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), (2, 2))
            
            return x.view(1, -1).size(1)

    def forward(self, x, sample=True):
        # Ensure correct input dimensions [batch, channels, height, width]
        while x.dim() > 4:
            x = x.squeeze(1)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Bayesian feature extraction with batch normalization
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2, 2))  # /2
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))  # /4
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), (2, 2))  # /8
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), (2, 2))  # /16

        x = self.dropout1(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Bayesian classification layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)

        x = F.relu(self.fc2(x))
        x = self.dropout3(x)

        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Final output layer

        return x

    def predict_with_uncertainty(self, x, num_samples=100):
        """
        Get predictions with uncertainty estimates by sampling multiple times
        """
        self.train()  # Enable sampling
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x, sample=True)
                predictions.append(F.softmax(pred, dim=1))
        
        predictions = torch.stack(predictions)
        
        # Calculate mean and uncertainty
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean(dim=1)  # Average variance across classes
        
        return mean_pred, uncertainty