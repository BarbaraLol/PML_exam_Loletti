import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters (smaller initialization range)
        # weight mu small to prevents extreme initial weight values (unstable gradients)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.1, 0.1))
        # it controls the log variance (σ = log(1 + exp(ρ)))
        # ρ = -6 converts to σ ≈ 0.0025
        # ρ = -5 converts to σ ≈ 0.0067
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-6, -5))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.1, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-6, -5))
        
        # Priors
        # Maybe trying reduce from0.1 to 0.01?
        self.register_buffer('weight_prior_mu', torch.zeros(out_features, in_features))
        self.register_buffer('weight_prior_sigma', 0.01 * torch.ones(out_features, in_features))
        self.register_buffer('bias_prior_mu', torch.zeros(out_features))
        self.register_buffer('bias_prior_sigma', 0.01 * torch.ones(out_features))
    
    # Posterior
    @property
    def weight_sigma(self):
        """Convert rho to sigma using softplus"""
        return torch.log1p(torch.exp(self.weight_rho))
    
    @property
    def bias_sigma(self):
        """Convert rho to sigma using softplus"""
        return torch.log1p(torch.exp(self.bias_rho))
    
    def forward(self, x, sample=True):
        if self.training or sample:
            # Sample weights with device awareness (everything must be on the same device: GPU/CPU)
            device = self.weight_mu.device 
            weight = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_mu, device=device)
            bias = self.bias_mu + self.bias_sigma * torch.randn_like(self.bias_mu, device=device)
        else:
            # Use mean for inference
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_loss(self):
        """KL divergence between posterior and prior"""
        kl_weight = (torch.log(self.weight_prior_sigma / self.weight_sigma) - 0.5 
                    + (self.weight_sigma**2 + (self.weight_mu - self.weight_prior_mu)**2) 
                    / (2 * self.weight_prior_sigma**2)).sum()
        
        kl_bias = (torch.log(self.bias_prior_sigma / self.bias_sigma) - 0.5 
                  + (self.bias_sigma**2 + (self.bias_mu - self.bias_prior_mu)**2) 
                  / (2 * self.bias_prior_sigma**2)).sum()
        
        return kl_weight + kl_bias

class ChickCallDetector(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        
        # Feature extractor (hierarchical features from spectrograms)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), # for stable training
            nn.ReLU(inplace=True), # non-linearity
            nn.MaxPool2d(2), # reduces spatial dimensions by 2x
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Calculate flattened size
        self._to_linear = self._get_conv_output(input_shape)
        
        # Bayesian classifier (Classification of features with uncertainty estimation)
        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 256), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.3), # for additional regulariztion
            BayesianLinear(256, 128),  # First Bayesian layer
            nn.ReLU(inplace=True),
            BayesianLinear(128, num_classes)  # Second Bayesian layer
        )

    # Compute flattened size (after convolution)
    def _get_conv_output(self, shape):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *shape)
            dummy = self.features(dummy)
            return dummy.view(1, -1).shape[1]

    def forward(self, x, sample=True):
        # Input normalization to ensure a correct 4D shape ([batch, 1, freq, time])
        while x.dim() > 4:  # Remove extra dimensions if they exist
            x = x.squeeze(1)
        if x.dim() == 3:    # Add channel dimension if missing
            x = x.unsqueeze(1)
        
        # Feature extraction (CNN)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Bayesian classifier
        x = self.classifier[0](x)  # Linear
        x = self.classifier[1](x)  # ReLU
        x = self.classifier[2](x)  # Dropout
        # The sample parameter used to control stochasticity
        x = self.classifier[3](x, sample=sample)  # First Bayesian
        x = self.classifier[4](x)  # ReLU
        x = self.classifier[5](x, sample=sample)  # Second Bayesian
        
        return x
    
    # Total KL divergence loss
    def kl_loss(self):
        """Sum KL losses from all Bayesian layers"""
        return sum(layer.kl_loss() for layer in self.classifier 
                 if isinstance(layer, BayesianLinear))