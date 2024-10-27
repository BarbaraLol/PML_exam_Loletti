import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
from pyro.distributions import Normal, Categorical

# Bayesian Neural Network implementation
class BNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super (BNN, self).__init__()

    # Define fully connected layers
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    # Flattening the input spectrogram
    x = x.view(x.size(0), -1) # All dimentions are flattened but batch size

    output = F.relu(self.fc1(x))
    output = F.relu(self.fc2(output))
    output = self.out(output)
    return output

  def model(self, x_data, y_data = None):
    #Prior distributions for weightes and biases
    fc1w_prior = Normal(loc=torch.zeros_like(self.fc1.weight), scale=torch.ones_like(self.fc1.weight)).to_event(2)
    fc1b_prior = Normal(loc=torch.zeros_like(self.fc1.bias), scale=torch.ones_like(self.fc1.bias)).to_event(1)

    fc2w_prior = Normal(loc=torch.zeros_like(self.fc2.weight), scale=torch.ones_like(self.fc2.weight)).to_event(2)
    fc2b_prior = Normal(loc=torch.zeros_like(self.fc2.bias), scale=torch.ones_like(self.fc2.bias)).to_event(1)

    outputw_prior = Normal(loc=torch.zeros_like(self.out.weight), scale=torch.ones_like(self.out.weight)).to_event(2)
    outputb_prior = Normal(loc=torch.zeros_like(self.out.bias), scale=torch.ones_like(self.out.bias)).to_event(1)

    priors = {
        'fc1.weight': fc1w_prior, 
        'fc1.bias': fc1b_prior, 
        'fc2.weight': fc2w_prior, 
        'fc2.bias': fc2b_prior, 
        'output.weight': outputw_prior, 
        'output.bias': outputb_prior
    }
    
    # Lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", self, priors)
    lifted_reg_model = lifted_module()

    with pyro.plate("data", len(x_data)):
      logits = lifted_reg_model(x_data)
      obs = pyro.sample("obs", Categorical(logits=logits), obs=y_data)

    return logits

  def guide(self, x_data, y_data = None):
    # Variable for the softplus function
    softplus = torch.nn.Softplus()
    # Weight and bias variational distribution priors
    # First layer weight distribution priors
    fc1w_mu = torch.randn_like(self.fc1.weight)
    fc1w_sigma = torch.randn_like(self.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param).to_event(2)

    # First layer bias distribution priors
    fc1b_mu = torch.randn_like(self.fc1.bias)
    fc1b_sigma = torch.randn_like(self.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param).to_event(1)

    # Second layer weight distribution priors
    fc2w_mu = torch.randn_like(self.fc2.weight)
    fc2w_sigma = torch.randn_like(self.fc2.weight)
    fc2w_mu_param = pyro.param("fc2w_mu", fc2w_mu)
    fc2w_sigma_param = softplus(pyro.param("fc2w_sigma", fc2w_sigma))
    fc2w_prior = Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param).to_event(2)

    # Second layer bias distribution priors
    fc2b_mu = torch.randn_like(self.fc2.bias)
    fc2b_sigma = torch.randn_like(self.fc2.bias)
    fc2b_mu_param = pyro.param("fc2b_mu", fc2b_mu)
    fc2b_sigma_param = softplus(pyro.param("fc2b_sigma", fc2b_sigma))
    fc2b_prior = Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param).to_event(1)

    # Output layer weight distribution priors
    outputw_mu = torch.randn_like(self.out.weight)
    outputw_sigma = torch.randn_like(self.out.weight)
    outputw_mu_param = pyro.param("outputw_mu", outputw_mu)
    outputw_sigma_param = softplus(pyro.param("outputw_sigma", outputw_sigma))
    outputw_prior = Normal(loc=outputw_mu_param, scale=outputw_sigma_param).to_event(2)
    
    # Output layer bias distribution priors
    outputb_mu = torch.randn_like(self.out.bias)
    outputb_sigma = torch.randn_like(self.out.bias)
    outputb_mu_param = pyro.param("outputb_mu", outputb_mu)
    outputb_sigma_param = softplus(pyro.param("outputb_sigma", outputb_sigma))
    outputb_prior = Normal(loc=outputb_mu_param, scale=outputb_sigma_param).to_event(1)

    priors = {
        'fc1.weight': fc1w_prior, 
        'fc1.bias': fc1b_prior, 
        'fc2.weight': fc2w_prior, 
        'fc2.bias': fc2b_prior, 
        'output.weight': outputw_prior, 
        'output.bias': outputb_prior
    }

    lifted_module = pyro.random_module("module", self, priors)

    return lifted_module()