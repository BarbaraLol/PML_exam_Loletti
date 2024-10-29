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
    self.out = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    # Flattening the input spectrogram
    print("Original input shape:", x.shape)
    x = x.squeeze(-1)  # Removes the last dimension if it's of size 1
    #print("Flattened input size:", x.numel())
    #x = x.view(x.size(0), -1) # All dimentions are flattened but batch size
    print("After flattening: ", x.shape)
    output = F.relu(self.fc1(x))
    print("After fc1: ", output.shape)
    output = F.relu(self.fc2(output))
    print("After fc2: ", output.shape)
    output = F.relu(self.fc3(output))
    print("After fc3: ", output.shape)
    output = self.out(output)
    print("After output layer: ", output.shape)
    return output

  def model(self, x_data, y_data = None):
    #Prior distributions for weightes and biases
    fc1w_prior = Normal(loc=torch.zeros_like(self.fc1.weight), scale=torch.ones_like(self.fc1.weight)).to_event(2)
    fc1b_prior = Normal(loc=torch.zeros_like(self.fc1.bias), scale=torch.ones_like(self.fc1.bias)).to_event(1)

    #fc2w_prior = Normal(loc=torch.zeros_like(self.fc2.weight), scale=torch.ones_like(self.fc2.weight)).to_event(2)
    #fc2b_prior = Normal(loc=torch.zeros_like(self.fc2.bias), scale=torch.ones_like(self.fc2.bias)).to_event(1)

    #fc3w_prior = Normal(loc=torch.zeros_like(self.fc3.weight), scale=torch.ones_like(self.fc3.weight)).to_event(2)
    #fc3b_prior = Normal(loc=torch.zeros_like(self.fc3.bias), scale=torch.ones_like(self.fc3.bias)).to_event(1)

    #outputw_prior = Normal(loc=torch.zeros_like(self.out.weight), scale=torch.ones_like(self.out.weight)).to_event(2)
    #outputb_prior = Normal(loc=torch.zeros_like(self.out.bias), scale=torch.ones_like(self.out.bias)).to_event(1)

    priors = {
        'fc1.weight': fc1w_prior, 
        'fc1.bias': fc1b_prior, 
        #'fc2.weight': fc2w_prior, 
        #'fc2.bias': fc2b_prior,  
        #'fc3.weight': fc3w_prior, 
        #'fc3.bias': fc3b_prior,
        #'out.weight': outputw_prior, 
        #'out.bias': outputb_prior
    }
    
    # Lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", self, priors)
    lifted_reg_model = lifted_module()

    # Ensure x_data is flattened
    x_data = x_data.view(x_data.size(0), -1)  # Flatten to match fc1
    print("x_data shape in model after flattening:", x_data.shape)

    with pyro.plate("data", len(x_data)):
      # Passing through each layer explicitly with transformed dimensions
      #x = x_data.view(x_data.size(0), -1)  # Ensure x is flattened if needed
      #x = F.relu(lifted_reg_model.fc1(x))
      #x = F.relu(lifted_reg_model.fc2(x))
      #x = F.relu(lifted_reg_model.fc3(x)
      #logits = lifted_reg_model(x_data)
      output = F.relu(lifted_reg_model.fc1(x_data))  # Pass through fc1 only
      print("Output shape after fc1 in model:", output.shape)  # Should print [128, 256]
      obs = pyro.sample("obs", Categorical(logits=output), obs=y_data)

    return output

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
        'out.weight': outputw_prior, 
        'out.bias': outputb_prior
    }

    lifted_module = pyro.random_module("module", self, priors)

    return lifted_module()
