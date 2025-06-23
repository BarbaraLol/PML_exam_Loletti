import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.distributions import Normal, Categorical

class HybridCNN_BNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(HybridCNN_BNN, self).__init__()
        
        # CNN Feature Extractor
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Calculate the flattened size after convolutions
        self._to_linear = None
        self._get_conv_output_size(input_shape)
        
        # BNN Classifier with proper initialization
        self.bnn_fc1 = nn.Linear(self._to_linear, 256)
        self.bnn_fc2 = nn.Linear(256, 128)
        self.bnn_out = nn.Linear(128, num_classes)
        
        # CRITICAL FIX: Proper weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Proper weight initialization to prevent exploding gradients"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Small initialization for BNN layers
                nn.init.normal_(m.weight, 0, 0.01)  # Very small std
                nn.init.constant_(m.bias, 0)
        
    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            dummy = self.pool(F.relu(self.conv1(dummy)))
            dummy = self.pool(F.relu(self.conv2(dummy)))
            dummy = self.pool(F.relu(self.conv3(dummy)))
            self._to_linear = dummy.view(1, -1).shape[1]
    
    def forward(self, x):
        # CNN forward pass
        x = x.unsqueeze(1)  # Add channel dimension (batch, 1, freq, time)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # BNN forward pass
        x = F.relu(self.bnn_fc1(x))
        x = F.relu(self.bnn_fc2(x))
        x = self.bnn_out(x)
        return x
    
    def model(self, x_data, y_data=None):
        # CRITICAL FIX: Much smaller prior scale to prevent exploding gradients
        prior_scale = 0.1  # Reduced from 1.0
        
        # Priors for BNN weights with smaller scale
        bnn_fc1w_prior = Normal(loc=torch.zeros_like(self.bnn_fc1.weight), 
                              scale=prior_scale * torch.ones_like(self.bnn_fc1.weight)).to_event(2)
        bnn_fc1b_prior = Normal(loc=torch.zeros_like(self.bnn_fc1.bias), 
                              scale=prior_scale * torch.ones_like(self.bnn_fc1.bias)).to_event(1)
        
        bnn_fc2w_prior = Normal(loc=torch.zeros_like(self.bnn_fc2.weight), 
                              scale=prior_scale * torch.ones_like(self.bnn_fc2.weight)).to_event(2)
        bnn_fc2b_prior = Normal(loc=torch.zeros_like(self.bnn_fc2.bias), 
                              scale=prior_scale * torch.ones_like(self.bnn_fc2.bias)).to_event(1)
        
        outw_prior = Normal(loc=torch.zeros_like(self.bnn_out.weight), 
                          scale=prior_scale * torch.ones_like(self.bnn_out.weight)).to_event(2)
        outb_prior = Normal(loc=torch.zeros_like(self.bnn_out.bias), 
                          scale=prior_scale * torch.ones_like(self.bnn_out.bias)).to_event(1)
        
        priors = {
            'bnn_fc1.weight': bnn_fc1w_prior,
            'bnn_fc1.bias': bnn_fc1b_prior,
            'bnn_fc2.weight': bnn_fc2w_prior,
            'bnn_fc2.bias': bnn_fc2b_prior,
            'bnn_out.weight': outw_prior,
            'bnn_out.bias': outb_prior
        }
        
        lifted_module = pyro.random_module("module", self, priors)
        lifted_reg_model = lifted_module()
        
        # CRITICAL FIX: Proper forward pass
        with pyro.plate("data", x_data.size(0)):
            # Forward pass through the lifted model
            logits = lifted_reg_model(x_data)
            
            # CRITICAL FIX: Use proper observation model
            if y_data is not None:
                pyro.sample("obs", Categorical(logits=logits), obs=y_data)
        
        return logits
    
    def guide(self, x_data, y_data=None):
        softplus = torch.nn.Softplus()
        
        # CRITICAL FIX: Better initialization for variational parameters
        init_scale = 0.01  # Much smaller initial scale
        
        # Variational parameters for BNN weights
        bnn_fc1w_mu = pyro.param("bnn_fc1w_mu", 
                                 torch.randn_like(self.bnn_fc1.weight) * init_scale)
        bnn_fc1w_sigma = softplus(pyro.param("bnn_fc1w_sigma", 
                                           -3.0 + 0.1 * torch.randn_like(self.bnn_fc1.weight)))
        bnn_fc1w_prior = Normal(loc=bnn_fc1w_mu, scale=bnn_fc1w_sigma).to_event(2)
        
        bnn_fc1b_mu = pyro.param("bnn_fc1b_mu", 
                                torch.randn_like(self.bnn_fc1.bias) * init_scale)
        bnn_fc1b_sigma = softplus(pyro.param("bnn_fc1b_sigma", 
                                           -3.0 + 0.1 * torch.randn_like(self.bnn_fc1.bias)))
        bnn_fc1b_prior = Normal(loc=bnn_fc1b_mu, scale=bnn_fc1b_sigma).to_event(1)
        
        bnn_fc2w_mu = pyro.param("bnn_fc2w_mu", 
                                torch.randn_like(self.bnn_fc2.weight) * init_scale)
        bnn_fc2w_sigma = softplus(pyro.param("bnn_fc2w_sigma", 
                                           -3.0 + 0.1 * torch.randn_like(self.bnn_fc2.weight)))
        bnn_fc2w_prior = Normal(loc=bnn_fc2w_mu, scale=bnn_fc2w_sigma).to_event(2)
        
        bnn_fc2b_mu = pyro.param("bnn_fc2b_mu", 
                                torch.randn_like(self.bnn_fc2.bias) * init_scale)
        bnn_fc2b_sigma = softplus(pyro.param("bnn_fc2b_sigma", 
                                           -3.0 + 0.1 * torch.randn_like(self.bnn_fc2.bias)))
        bnn_fc2b_prior = Normal(loc=bnn_fc2b_mu, scale=bnn_fc2b_sigma).to_event(1)
        
        outw_mu = pyro.param("outw_mu", 
                            torch.randn_like(self.bnn_out.weight) * init_scale)
        outw_sigma = softplus(pyro.param("outw_sigma", 
                                       -3.0 + 0.1 * torch.randn_like(self.bnn_out.weight)))
        outw_prior = Normal(loc=outw_mu, scale=outw_sigma).to_event(2)
        
        outb_mu = pyro.param("outb_mu", 
                            torch.randn_like(self.bnn_out.bias) * init_scale)
        outb_sigma = softplus(pyro.param("outb_sigma", 
                                       -3.0 + 0.1 * torch.randn_like(self.bnn_out.bias)))
        outb_prior = Normal(loc=outb_mu, scale=outb_sigma).to_event(1)
        
        priors = {
            'bnn_fc1.weight': bnn_fc1w_prior,
            'bnn_fc1.bias': bnn_fc1b_prior,
            'bnn_fc2.weight': bnn_fc2w_prior,
            'bnn_fc2.bias': bnn_fc2b_prior,
            'bnn_out.weight': outw_prior,
            'bnn_out.bias': outb_prior
        }
        
        lifted_module = pyro.random_module("module", self, priors)
        return lifted_module()
    
    def _get_cnn_features(self, x):
        # CNN forward pass only
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        return x
    
    def _bnn_forward(self, x):
        # BNN forward pass only
        x = F.relu(self.bnn_fc1(x))
        x = F.relu(self.bnn_fc2(x))
        x = self.bnn_out(x)
        return x