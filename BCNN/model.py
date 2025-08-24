import torch
import torch.nn as nn
import torch.nn.functional as F
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator

# Fully Bayesian CNN
@variational_estimator
class BayesianChickCallDetector(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # Store input shape for later use
        self.input_shape = in_features

        # Prior parameters definitions
        prior_sigma_1=0.5 # 0.1 previously
        prior_sigma_2=0.05 # 0.005 previously
        prior_pi=0.3 # 0.25 previously

        # Fully connected Bayesian convolutional layers
        self.conv1 = BayesianConv2d(1, 32, kernel_size=(3, 3), padding=1, prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = BayesianConv2d(32, 64, kernel_size=(3, 3), padding=1, prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = BayesianConv2d(64, 128, kernel_size=(3, 3), padding=1, prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        self.bn3 = nn.BatchNorm2d(64)
        # self.conv4 = BayesianConv2d(64, 128, kernel_size=(3, 3), padding=1, prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        # self.bn4 = nn.BatchNorm2d(128)

        # Calculating the flattened size
        self._to_linear = self._get_conv_output(in_features)

        # Bayesian classifier
        # self.fcbayesian1 = BayesianLinear(self._to_linear, 512, prior_sigma_1=0.1, prior_sigma_2=0.01, prior_pi=0.25)
        self.fcbayesian2 = BayesianLinear(self._to_linear, 256, prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        self.fcbayesian3 = BayesianLinear(256, out_features, prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        # self.fcbayesian4 = BayesianLinear(128, out_features, prior_sigma_1=0.1, prior_sigma_2=0.01, prior_pi=0.25)

        # self.dropout1 = nn.Dropout(0.2)  # Light dropout after conv layers
        self.dropout2 = nn.Dropout(0.3)  # Heavier dropout in classifier
        self.dropout3 = nn.Dropout(0.5)  # Progressive dropout

        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(256)

    def _get_conv_output(self, shape):
        """Calculate the flattened size after all convolutions"""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *shape)
            
            # Apply the same operations as in forward pass
            x = F.max_pool2d(F.relu(self.bn1(self.conv1(dummy)),) (2, 2))
            x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))
            x = F.max_pool2d(F.relu(self.bn1(self.conv3(x))), (2, 2))
            # x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), (2, 2))
            
            return x.view(1, -1).size(1)
        
    
    def forward(self, x, sample=True):
        # Input normalization to ensure a correct 4D shape ([batch, 1, freq, time])
        while x.dim() > 4:  # Remove extra dimensions if they exist
            x = x.squeeze(1)
        if x.dim() == 3:    # Add channel dimension if missing
            x = x.unsqueeze(1)

        # Bayesian feature extraction with progressive pooling
        # BayesianConv2d doesn't take sample parameter - sampling is handled internally
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2, 2))  # / 2
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))  # / 4
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), (2, 2))  # / 8
        # x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), (2, 2))  # / 16

        x = self.dropout1(x)  # Dropout after feature extraction

        # Flatten and classify
        x = x.view(x.size(0), -1)

        # Fully Bayesian classification with uncertainty at every level
        # # BayesianLinear does take sample parameter
        # x = F.relu(self.fcbayesian1(x))
        # x = self.dropout2(x)

        # x = F.relu(self.fcbayesian2(x))
        # x = self.dropout3(x)

        # x = F.relu(self.fcbayesian3(x))
        # x = self.fcbayesian4(x)  # Final classification layer
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x