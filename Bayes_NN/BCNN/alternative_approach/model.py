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

        # Lower variances for stability, higher pi for more deterministic weights
        prior_sigma_1 = 0.1   # Main prior variance
        prior_sigma_2 = 0.01  # Secondary prior variance  
        prior_pi = 0.75       # Mixture weight (more deterministic)
        
        fc_prior_sigma_1 = 0.1     # Same for FC layers
        fc_prior_sigma_2 = 0.01    
        fc_prior_pi = 0.75

        # Fully connected Bayesian convolutional layers
        self.conv1 = BayesianConv2d(1, 16, kernel_size=(3, 3), padding=1, prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = BayesianConv2d(16, 32, kernel_size=(3, 3), padding=1, prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = BayesianConv2d(64, 128, kernel_size=(3, 3), padding=1, prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.conv4 = BayesianConv2d(64, 128, kernel_size=(3, 3), padding=1, prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)
        # self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        # Calculating the flattened size
        self._to_linear = self._get_conv_output(in_features)

        # Bayesian classifier
        # self.fcbayesian1 = BayesianLinear(self._to_linear, 512, prior_sigma_1=0.1, prior_sigma_2=0.01, prior_pi=0.25)
        self.fcbayesian1 = BayesianLinear(self._to_linear, 64, prior_sigma_1=fc_prior_sigma_1, prior_sigma_2=fc_prior_sigma_2, prior_pi=fc_prior_pi)
        self.fcbayesian2 = BayesianLinear(64, out_features, prior_sigma_1=fc_prior_sigma_1, prior_sigma_2=fc_prior_sigma_2, prior_pi=fc_prior_pi)
        # self.fcbayesian4 = BayesianLinear(128, out_features, prior_sigma_1=0.1, prior_sigma_2=0.01, prior_pi=0.25)

        # self.dropout1 = nn.Dropout(0.2)  # Light dropout after conv layers
        self.dropout1 = nn.Dropout(0.3)  # Heavier dropout in classifier
        # self.dropout2 = nn.Dropout(0.5)  # Progressive dropout

        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(64)

    def _get_conv_output(self, shape):
        """Calculate the flattened size after all convolutions"""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *shape)
            
            # Apply the same operations as in forward pass
            x = F.max_pool2d(F.relu(self.conv1(dummy)))
            x = F.max_pool2d(F.relu(self.conv2(x)))
            # x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), (2, 2))
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
        x = F.max_pool2d(F.relu(self.conv1(x))), (2, 2)  # / 2
        x = F.max_pool2d(F.relu(self.conv2(x))), (2, 2)  # / 4
        # x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), (2, 2))  # / 8
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
        x = F.relu(self.fcbayesian1(x))
        x = self.dropout1(x)
        x = self.fcbayesian2(x)

        return x

    def predict_with_uncertainty(self, x, num_samples=50):
        """
        Get predictions with uncertainty estimates
        This is the key advantage of BCNN over regular CNN
        """
        self.train()  # Enable sampling
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x, sample=True)
                predictions.append(F.softmax(pred, dim=1))
        
        predictions = torch.stack(predictions)
        
        # Calculate mean prediction and uncertainty
        mean_pred = predictions.mean(dim=0)
        # Predictive entropy as uncertainty measure
        entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=1)
        
        return mean_pred, entropy