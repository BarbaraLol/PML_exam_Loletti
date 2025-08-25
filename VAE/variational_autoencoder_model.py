import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

class VariationalAutoencoderEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim = 128):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Flattening the input (from 2D to 1D)
        self.flatten = input_shape[0] * input_shape[1] # frequency * time

        # hidden layer for complexity
        self.hidden1 = nn.Linear(self.flatten, 1024)
        self.hidden2 = nn.Linear(1024, 512)

        # Dense layers for data's mean and variance (logarithmic)
        self.fc_mean = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # starting from x shape: [batch, 1, frequence, time] -> flatten into [batch, freq * time]
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # Hidden layer with activation function
        h = F.relu(self.hidden1(x))
        h = self.dropout(h)
        h = F.relu(self.hidden2(h))
        h = self.dropout(h)

        # Calculatind mean and log-variance
        mu = self.fc_mean(h)
        logvar = self.fc_logvar(h)

        # Clamping layers for more stability
        logvar = torch.clamp(logvar, min = -20, max = 20)

        return mu, logvar
    
class VariationalAutoencoderDecoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.output_size = output_shape[0] * output_shape[1]

        # Hidden layer for
        self.hidden1 = nn.Linear(self.latent_dim, 1024)
        self.hidden2 = nn.Linear(1024, 512)

        # Output layer
        self.output_layer = nn.Linear(512, self.output_size)

        self.dropout = nn.Dropout(0.2)

    def forward(self, z):
        # starting from the hidden layer 
        h = F.relu(self.hidden1(z))
        h = self.dropout(h)
        h = F.relu(self.hidden2(h))
        h = self.dropout(h)

        # Output obtained with sigmoid activation function
        x = torch.sigmoid(self.output_layer(h))
        
        # Reshaping into spectrogram dimentions [batch, 1, frequency, time]
        x = x.view(x.size(0), 1, self.output_shape[0], self.output_shape[1])

        return x

class SpectrogramVAE(nn.Module):
    def __init__(self, input_shape, latent_dim = 128, beta = 1.0):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = VariationalAutoencoderEncoder(input_shape, latent_dim)
        self.decoder = VariationalAutoencoderDecoder(latent_dim, input_shape)

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # Return mean during evaluation
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

    def loss_function(self, recon_x, x, mu, logvar):
        """Proper loss function for spectrograms"""
        # Using the binary cross entropy for the sigmoid output
        # Basically, we're treating the spectrograms as probability 
        binary_cross_loss = F.binary_cross_entropy(recon_x, x, reduction = 'sum')

        # Alternative: MSE
        # mse_loss = F.mse_loss(recon_x, x, reduction='sum')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = binary_cross_loss + self.beta * kl_loss

        # Return per-sample losses for monitoring
        batch_size = x.size(0)
        return (total_loss / batch_size, 
                binary_cross_loss / batch_size, 
                kl_loss / batch_size)

    def sample(self, num_samples, device = "cpu"):
        """Generation of new samples"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device = device)
            samples = self.decode(z)

            return samples

    def interpolate(self, x1, x2, num_steps = 10):
        """Interpolation between two spectrograms"""
        self.eval()
        with torch.no_grad():
            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)

            interpolation = []

            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                x_interp = self.decode(z_interp)
                interpolation.append(x_interp)

            return torch.cat(interpolation, dim = 0)

class ConditionalSpectrogramVAE(SpectrogramVAE):
    def __init__(self, input_shape, num_classes, latent_dim = 128, beta = 1.0):
        """Conditional VAE for class-specific generation"""
        super().__init__(input_shape, latent_dim, beta)
        self.num_classes = num_classes

        # Class embeddings
        self.class_embedding = nn.Embedding(num_classes, 64)

        # Modify the encoder to include the class information
        original_input_Size = self.encoder.flatten
        self.encoder.hidden1 = nn.Linear(original_input_Size + 64, 1024)

        # Modify decoder to include class information
        self.decoder.hidden1 = nn.Linear(latent_dim + 64, 512)

    def encode(self, x, class_labels):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        # Getting the class embeddings
        class_emb = self.class_embedding(class_labels)

        # Concatenating the input with the class embbedings
        x_combined = torch.cat([x_flat, class_emb], dim = 1)

        # Passing through the modified encoder
        h = F.relu(self.encoder.hidden1(x_combined))
        h = self.encoder.dropout(h)
        h = F.relu(self.encoder.hidden2(h))
        h = self.encoder.dropout(h)

        mu = self.encoder.fc_mean(h)
        logvar = self.encoder.fc_logvar(h)
        logvar = torch.clamp(logvar, min = -20, max = 20)

        return mu, logvar
    
    def decode(self, z, class_labels):
        # Getting the class embeddings
        class_emb = self.class_embedding(class_labels)

        # Concatenating latent with class embedding
        z_combined = torch.cat([z, class_emb], dim = 1)

        # Passing through the modified decoder
        h = F.relu(self.decoder.hidden1(z_combined))
        h = self.decoder.dropout(h)
        h = F.relu(self.decoder.hidden2(h))
        h = self.decoder.dropout(h)

        # Output and restoring the size
        x = torch.sigmoid(self.decoder.output_layer(h))
        x = x.view(x.size(0), 1, self.decoder.output_shape[0], self.decoder.output_shape[1])

        return x

    def forward(self, x, class_labels):
        mu, logvar = self.encode(x, class_labels)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, class_labels)

        return recon_x, mu, logvar, z

    def sample_class(self, class_label, num_samples, device = 'cpu'):
        """Generative samples for a specific class"""
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device = device)
            class_lables = torch.full((num_samples,), class_label, device = device, dtype = torch.long)
            samples = self.decode(z, class_labels)
            return samples
    