import torch
import torch.nn as nn
import torch.nn.functional as F  # Fixed import
import numpy as np 

class SpectrogramEncoder(nn.Module):
    '''Encoder for spectrogram VAE'''
    def __init__(self, input_shape, latent_dim=128):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Convolutional encoder
        # 4 2d convolutional layers with batch normalization and LeakyReLU activation function
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        # Computing the flattened size
        self.flatten_size = self._get_flatten_size()

        # Mean and Log variance 
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def _get_flatten_size(self):  # Fixed method definition
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *self.input_shape)  # Fixed torch.zeros
            dummy = self.encoder(dummy)
            return dummy.view(-1).shape[0]  # Fixed indexing

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class SpectrogramDecoder(nn.Module):
    '''Decoder for spectrogram VAE'''
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        # Calculate the size after encoder
        h_enc = output_shape[0] // 16
        w_enc = output_shape[1] // 16

        self.h_enc, self.w_enc = h_enc, w_enc

        # Dense layer to reshape
        self.fc = nn.Linear(latent_dim, 256 * h_enc * w_enc)

        # Transposed convolution (deconvoluting)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # Fixed ConvTranspose2d
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # Fixed ConvTranspose2d
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # Fixed ConvTranspose2d
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),    # Fixed ConvTranspose2d
            nn.Tanh() # Output in [-1, 1] range
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, self.h_enc, self.w_enc)
        x = self.decoder(x)  # Fixed duplicate line

        # Ensuring the correct shape
        if x.shape[2:] != self.output_shape:
            x = F.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=False)

        return x


class SpectrogramVAE(nn.Module):
    '''Variational Autoencoder for spectrograms'''
    def __init__(self, input_shape, latent_dim=128, beta=1.0):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta # β-VAE parameter for disentanglement

        self.encoder = SpectrogramEncoder(input_shape, latent_dim)
        self.decoder = SpectrogramDecoder(latent_dim, input_shape)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):  # Fixed method name
        return self.decoder(z)

    def reparameterize(self, mu, logvar):  # Fixed method name
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

    def loss_function(self, recon_x, x, mu, logvar, reduction='mean'):
        '''VAE loss function computation: reconstruction + KL divergence'''
        # Reconstruction loss (MSE for spectrograms)
        recon_loss = F.mse_loss(recon_x, x, reduction=reduction)

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # Mean instead of sum

        if reduction == 'mean':
            kl_loss = kl_loss / x.size(0)

        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

    def sample(self, num_samples, device='cpu'):
        '''Generation of new spectrograms from prior distribution'''
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)  # Fixed device parameter
            samples = self.decode(z)
            return samples
        
    def interpolate(self, x1, x2, num_steps=10):
        '''Interpolation between two spectrograms in the latent space'''
        with torch.no_grad():
            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)  # Fixed method name

            interpolations = []

            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                x_interp = self.decode(z_interp)
                interpolations.append(x_interp)  # Fixed method name

            return torch.cat(interpolations, dim=0)


class ConditionalSpectrogramVAE(SpectrogramVAE):
    '''Conditional VAE that can generate spectrograms for specific classes'''
    def __init__(self, input_shape, num_classes, latent_dim=128, beta=1.0):
        super().__init__(input_shape, latent_dim, beta)
        self.num_classes = num_classes

        # Encoder modification to include class conditioning
        self.class_embedding = nn.Embedding(num_classes, 64)

        # Modify the final layer to add class info
        original_flatten_size = self.encoder.flatten_size
        self.encoder.fc_mu = nn.Linear(original_flatten_size + 64, latent_dim)
        self.encoder.fc_logvar = nn.Linear(original_flatten_size + 64, latent_dim)

        # Modify decoder to include class conditioning
        self.decoder.fc = nn.Linear(latent_dim + 64, 256 * self.decoder.h_enc * self.decoder.w_enc)  # Fixed calculation

    def encode(self, x, class_labels):  # Fixed method name and parameter
        # Getting features from the encoder
        features = self.encoder.encoder(x)
        features = features.view(features.size(0), -1)

        # Adding class embeddings
        class_emb = self.class_embedding(class_labels)
        combined = torch.cat([features, class_emb], dim=1)

        mu = self.encoder.fc_mu(combined)
        logvar = self.encoder.fc_logvar(combined)

        return mu, logvar

    def decode(self, z, class_labels):  # Fixed parameter name
        class_emb = self.class_embedding(class_labels)
        combined = torch.cat([z, class_emb], dim=1)  # Fixed variable name

        x = self.decoder.fc(combined)
        x = x.view(x.size(0), 256, self.decoder.h_enc, self.decoder.w_enc)
        x = self.decoder.decoder(x)

        if x.shape[2:] != self.decoder.output_shape:
            x = F.interpolate(x, size=self.decoder.output_shape, mode='bilinear', align_corners=False)
        
        return x

    def forward(self, x, class_labels):  # Fixed parameter name
        mu, logvar = self.encode(x, class_labels)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, class_labels)

        return recon_x, mu, logvar, z

    def sample_class(self, class_label, num_samples, device='cpu'):  # Fixed method name
        """Generate spectrograms for a specific class"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)  # Fixed torch.randn
            class_labels = torch.full((num_samples,), class_label, device=device, dtype=torch.long)  # Fixed parameter name
            samples = self.decode(z, class_labels)
            return samples