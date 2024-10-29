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
    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.fc4 = nn.Linear(hidden_size, hidden_size)
    self.fc5 = nn.Linear(hidden_size, hidden_size)
    self.fc6 = nn.Linear(hidden_size, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    # Flattening the input spectrogram
    print("Original input shape:", x.shape)
    x = x.squeeze(-1)  # Removes the last dimension if it's of size 1
    print("After flattening: ", x.shape)
    output = F.relu(self.fc1(x))
    output = F.relu(self.fc2(output))
    output = F.relu(self.fc3(output))
    output = F.relu(self.fc4(output))
    output = F.relu(self.fc5(output))
    output = F.relu(self.fc6(output))
    output = self.out(output)
    print("After output layer: ", output.shape)
    return output

  def model(self, x_data, y_data = None):
    #Prior distributions for weightes and biases
    fc1w_prior = Normal(loc=torch.zeros_like(self.fc1.weight), scale=torch.ones_like(self.fc1.weight)).to_event(2)
    fc1b_prior = Normal(loc=torch.zeros_like(self.fc1.bias), scale=torch.ones_like(self.fc1.bias)).to_event(1)

    fc2w_prior = Normal(loc=torch.zeros_like(self.fc2.weight), scale=torch.ones_like(self.fc2.weight)).to_event(2)
    fc2b_prior = Normal(loc=torch.zeros_like(self.fc2.bias), scale=torch.ones_like(self.fc2.bias)).to_event(1)

    fc3w_prior = Normal(loc=torch.zeros_like(self.fc3.weight), scale=torch.ones_like(self.fc3.weight)).to_event(2)
    fc3b_prior = Normal(loc=torch.zeros_like(self.fc3.bias), scale=torch.ones_like(self.fc3.bias)).to_event(1)

    fc4w_prior = Normal(loc=torch.zeros_like(self.fc4.weight), scale=torch.ones_like(self.fc4.weight)).to_event(2)
    fc4b_prior = Normal(loc=torch.zeros_like(self.fc4.bias), scale=torch.ones_like(self.fc4.bias)).to_event(1)

    fc5w_prior = Normal(loc=torch.zeros_like(self.fc5.weight), scale=torch.ones_like(self.fc5.weight)).to_event(2)
    fc5b_prior = Normal(loc=torch.zeros_like(self.fc5.bias), scale=torch.ones_like(self.fc5.bias)).to_event(1)

    fc6w_prior = Normal(loc=torch.zeros_like(self.fc6.weight), scale=torch.ones_like(self.fc6.weight)).to_event(2)
    fc6b_prior = Normal(loc=torch.zeros_like(self.fc6.bias), scale=torch.ones_like(self.fc6.bias)).to_event(1)

    outputw_prior = Normal(loc=torch.zeros_like(self.out.weight), scale=torch.ones_like(self.out.weight)).to_event(2)
    outputb_prior = Normal(loc=torch.zeros_like(self.out.bias), scale=torch.ones_like(self.out.bias)).to_event(1)

    priors = {
        'fc1.weight': fc1w_prior, 
        'fc1.bias': fc1b_prior, 
        'fc2.weight': fc2w_prior, 
        'fc2.bias': fc2b_prior,  
        'fc3.weight': fc3w_prior, 
        'fc3.bias': fc3b_prior,
        'fc4.weight': fc4w_prior, 
        'fc4.bias': fc4b_prior, 
        'fc5.weight': fc5w_prior, 
        'fc5.bias': fc5b_prior,  
        'fc6.weight': fc6w_prior, 
        'fc6.bias': fc6b_prior,
        'out.weight': outputw_prior, 
        'out.bias': outputb_prior
    }
    
    # Lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", self, priors)
    lifted_reg_model = lifted_module()

    # Ensure x_data is flattened
    x_data = x_data.view(x_data.size(0), -1)  # Flatten to match fc1
    print("x_data shape in model after flattening:", x_data.shape)

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

    # Third layer weight distribution priors
    fc3w_mu = torch.randn_like(self.fc3.weight)
    fc3w_sigma = torch.randn_like(self.fc3.weight)
    fc3w_mu_param = pyro.param("fc3w_mu", fc3w_mu)
    fc3w_sigma_param = softplus(pyro.param("fc3w_sigma", fc3w_sigma))
    fc3w_prior = Normal(loc=fc3w_mu_param, scale=fc3w_sigma_param).to_event(2)

    # Third layer bias distribution priors
    fc3b_mu = torch.randn_like(self.fc3.bias)
    fc3b_sigma = torch.randn_like(self.fc3.bias)
    fc3b_mu_param = pyro.param("fc3b_mu", fc3b_mu)
    fc3b_sigma_param = softplus(pyro.param("fc3b_sigma", fc3b_sigma))
    fc3b_prior = Normal(loc=fc3b_mu_param, scale=fc3b_sigma_param).to_event(1)

    # Fourth layer weight distribution priors
    fc4w_mu = torch.randn_like(self.fc4.weight)
    fc4w_sigma = torch.randn_like(self.fc4.weight)
    fc4w_mu_param = pyro.param("fc4w_mu", fc4w_mu)
    fc4w_sigma_param = softplus(pyro.param("fc4w_sigma", fc4w_sigma))
    fc4w_prior = Normal(loc=fc4w_mu_param, scale=fc4w_sigma_param).to_event(2)

    # Fourth layer bias distribution priors
    fc4b_mu = torch.randn_like(self.fc4.bias)
    fc4b_sigma = torch.randn_like(self.fc4.bias)
    fc4b_mu_param = pyro.param("fc4b_mu", fc4b_mu)
    fc4b_sigma_param = softplus(pyro.param("fc4b_sigma", fc4b_sigma))
    fc4b_prior = Normal(loc=fc4b_mu_param, scale=fc4b_sigma_param).to_event(1)

    # Fifth layer weight distribution priors
    fc5w_mu = torch.randn_like(self.fc5.weight)
    fc5w_sigma = torch.randn_like(self.fc5.weight)
    fc5w_mu_param = pyro.param("fc5w_mu", fc5w_mu)
    fc5w_sigma_param = softplus(pyro.param("fc5w_sigma", fc5w_sigma))
    fc5w_prior = Normal(loc=fc5w_mu_param, scale=fc5w_sigma_param).to_event(2)

    # Fifth layer bias distribution priors
    fc5b_mu = torch.randn_like(self.fc5.bias)
    fc5b_sigma = torch.randn_like(self.fc5.bias)
    fc5b_mu_param = pyro.param("fc5b_mu", fc5b_mu)
    fc5b_sigma_param = softplus(pyro.param("fc5b_sigma", fc5b_sigma))
    fc5b_prior = Normal(loc=fc5b_mu_param, scale=fc5b_sigma_param).to_event(1)

    # Sixth layer weight distribution priors
    fc6w_mu = torch.randn_like(self.fc6.weight)
    fc6w_sigma = torch.randn_like(self.fc6.weight)
    fc6w_mu_param = pyro.param("fc6w_mu", fc6w_mu)
    fc6w_sigma_param = softplus(pyro.param("fc6w_sigma", fc6w_sigma))
    fc6w_prior = Normal(loc=fc6w_mu_param, scale=fc6w_sigma_param).to_event(2)

    # Sixth layer bias distribution priors
    fc6b_mu = torch.randn_like(self.fc6.bias)
    fc6b_sigma = torch.randn_like(self.fc6.bias)
    fc6b_mu_param = pyro.param("fc6b_mu", fc6b_mu)
    fc6b_sigma_param = softplus(pyro.param("fc6b_sigma", fc6b_sigma))
    fc6b_prior = Normal(loc=fc6b_mu_param, scale=fc6b_sigma_param).to_event(1)

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
        'fc3.weight': fc3w_prior, 
        'fc3.bias': fc3b_prior, 
        'fc4.weight': fc4w_prior, 
        'fc4.bias': fc4b_prior, 
        'fc5.weight': fc5w_prior, 
        'fc5.bias': fc5b_prior, 
        'fc6.weight': fc6w_prior, 
        'fc6.bias': fc6b_prior, 
        'out.weight': outputw_prior, 
        'out.bias': outputb_prior
    }

    lifted_module = pyro.random_module("module", self, priors)

    return lifted_module()
