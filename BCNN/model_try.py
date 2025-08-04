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

        # Fully connected Bayesian convolutional layers
        self.conv1 = BayesianConv2d(1, 16, kernel_size = 3, padding = 1, prior_sigma_1 = 0.1, prior_sigma_2 = 0.01, prior_pi = 0.25)
        self_bn1 = nn.BatchNorm2d(16)
        self.conv2 = BayesianConv2d(16, 32, kernel_size = 3, padding = 1, prior_sigma_1 = 0.15, prior_sigma_2 = 0.005, prior_pi = 0.25)
        self_bn2 = nn.BatchNorm2d(32)
        self.conv3 = BayesianConv2d(32, 64, kernel_size = 3, padding = 1, prior_sigma_1 = 0.15, prior_sigma_2 = 0.005, prior_pi = 0.25)
        self_bn3 = nn.BatchNorm2d(64)
        self.conv4 = BayesianConv2d(64, 128, kernel_size = 3, padding = 1, prior_sigma_1 = 0.15, prior_sigma_2 = 0.005, prior_pi = 0.25)
        self_bn4 = nn.BatchNorm2d(128)

        # Calculating the flattened size
        self._to_linear = self._get_conv_output(in_features)

        # Bayesian classifier
        self.fcbayesian1 = BayesianLinear(self._to_linear, 512, prior_sigma_1 = 0.1, prior_sigma_2 = 0.01, prior_pi = 0.25)
        self.fcbayesian2 = BayesianLinear(512, 256, prior_sigma_1 = 0.1, prior_sigma_2 = 0.01, prior_pi = 0.25)
        self.fcbayesian3 = BayesianLinear(256, 128, prior_sigma_1 = 0.1, prior_sigma_2 = 0.01, prior_pi = 0.25)
        self.fcbayesian4 = BayesianLinear(128, out_features, prior_sigma_1 = 0.1, prior_sigma_2 = 0.01, prior_pi = 0.25)

        self.dropout1 = nn.Dropout(0.2)  # Light dropout after conv layers
        self.dropout2 = nn.Dropout(0.3)  # Heavier dropout in classifier
        self.dropout3 = nn.Dropout(0.4)  # Progressive dropout

    def _get_conv_output(in_features):
        """Calculate the flattened size after all convolutions"""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *shape)
            x = F.MaxPool2d(F.relu(self.fcbayesian1(self.conv1(dummy))), 2)
            x = F.MaxPool2d(F.relu(self.fcbayesian2(self.conv2(x))), 2)
            x = F.MaxPool2d(F.relu(self.fcbayesian3(self.conv3(x))), 2)
            x = F.MaxPool2d(F.relu(self.fcbayesian4(self.conv4(x))), 2)

            return x.view(-1, 1).shape[1]
        
    
    def forward(self, x, sample=True):
        # Input normalization to ensure a correct 4D shape ([batch, 1, freq, time])
        while x.dim() > 4:  # Remove extra dimensions if they exist
            x = x.squeeze(1)
        if x.dim() == 3:    # Add channel dimension if missing
            x = x.unsqueeze(1)

        # Bayesian feature extraction with progressive pooling
        x = F.MaxPool2d(F.relu(self.fcbayesian1(self.conv1(x))), 2) # / 2
        x = F.MaxPool2d(F.relu(self.fcbayesian2(self.conv2(x))), 2) # / 4
        x = F.MaxPool2d(F.relu(self.fcbayesian3(self.conv3(x))), 2) # / 8
        x = F.MaxPool2d(F.relu(self.fcbayesian4(self.conv4(x))), 2) # / 16

        x = self.dropout1(x) # Dropout after feature extraction

        # Flatten and classify
        x = x.view(x.size(0), -1)

        # Fully Bayesian classification with uncertanty at every level
        x = F.relu(self.fcbayesian1(x))
        x = self.dropout2(x)

        x = F.relu(self.fcbayesian2(x))
        x = self.dropout3(x)

        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Final classification layer

        return x

