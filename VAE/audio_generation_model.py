import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class SpectrogramEncoder(nn.Module):
    '''Encoder for spectrogram VAE'''
    def __init__(self, input_shape, latent_dim=1024):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        # Get the exact shape and size after encoding
        self.encoded_shape = self._get_encoded_shape()
        self.flatten_size = self.encoded_shape[1] * self.encoded_shape[2] * self.encoded_shape[3]  # C*H*W

        print(f"Encoder - Input shape: {input_shape}")
        print(f"Encoder - Output shape after conv: {self.encoded_shape}")
        print(f"Encoder - Flatten size: {self.flatten_size}")

        # Mean and Log variance 
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def _get_encoded_shape(self):
        """Get the complete shape after encoding (including batch and channel dims)"""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *self.input_shape)
            encoded = self.encoder(dummy)
            print(f"Encoded tensor shape: {encoded.shape}")
            return encoded.shape

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten everything except batch dimension
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class SpectrogramDecoder(nn.Module):
    '''Decoder for spectrogram VAE - CORRECTED VERSION'''
    def __init__(self, latent_dim, output_shape, encoded_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.encoded_shape = encoded_shape  # Full shape [1, C, H, W]
        
        # Extract dimensions
        self.channels = encoded_shape[1]  # Should be 512
        self.height = encoded_shape[2]
        self.width = encoded_shape[3]
        
        print(f"Decoder - Latent dim: {latent_dim}")
        print(f"Decoder - Output shape: {output_shape}")
        print(f"Decoder - Encoded shape: {encoded_shape}")
        print(f"Decoder - Will reshape to: [{self.channels}, {self.height}, {self.width}]")

        # Dense layer - output size matches encoder flatten size
        fc_output_size = self.channels * self.height * self.width
        self.fc = nn.Linear(latent_dim, fc_output_size)
        print(f"Decoder - FC output size: {fc_output_size}")

        # Transposed convolution layers
        self.decoder = nn.Sequential(
            # Input: 512 channels -> Output: 256 channels
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Input: 256 channels -> Output: 128 channels
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Input: 128 channels -> Output: 64 channels
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Input: 64 channels -> Output: 1 channel (final output)
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range for spectrograms
        )

    def forward(self, z):
        batch_size = z.size(0)
        
        # Pass through FC layer
        x = self.fc(z)
        
        # Reshape to match encoder output
        x = x.view(batch_size, self.channels, self.height, self.width)
        
        # Pass through decoder
        x = self.decoder(x)

        # Interpolate to exact output shape if needed
        if x.shape[2:] != self.output_shape:
            x = F.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=False)

        return x


class SpectrogramVAE(nn.Module):
    '''Variational Autoencoder for spectrograms - CORRECTED VERSION'''
    def __init__(self, input_shape, latent_dim=1024, beta=1.0):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta

        # Initialize encoder first to get encoded shape
        self.encoder = SpectrogramEncoder(input_shape, latent_dim)
        
        # Pass encoded shape to decoder
        self.decoder = SpectrogramDecoder(latent_dim, input_shape, self.encoder.encoded_shape)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        if self.training:
            logvar = torch.clamp(logvar, min=-20, max=20)
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
        '''VAE loss function with numerical stability'''
        
        # Check for NaN inputs
        if torch.isnan(recon_x).any() or torch.isnan(x).any():
            print("NaN detected in reconstruction or input!")
            return torch.tensor(float('nan')), torch.tensor(float('nan')), torch.tensor(float('nan'))
        
        # Reconstruction loss (MSE for spectrograms)
        mse_loss = F.mse_loss(recon_x, x, reduction='mean')
        l1_loss = F.l1_loss(recon_x, x, reduction='mean')
        recon_loss = 0.8 * mse_loss + 0.2 * l1_loss
        
        # Clamp to prevent extreme values
        logvar = torch.clamp(logvar, min=-20, max=20)
        mu = torch.clamp(mu, min=-100, max=100)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)
        
        # Check for NaN in losses
        if torch.isnan(recon_loss) or torch.isnan(kl_loss):
            print(f"NaN in losses - recon: {recon_loss}, kl: {kl_loss}")
            return torch.tensor(float('nan')), recon_loss, kl_loss
        
        # Use beta as provided
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss

    def sample(self, num_samples, device='cpu'):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.decode(z)
            return samples
        
    def interpolate(self, x1, x2, num_steps=10):
        '''Interpolation between two spectrograms in the latent space'''
        with torch.no_grad():
            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)

            interpolations = []
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                x_interp = self.decode(z_interp)
                interpolations.append(x_interp)

            return torch.cat(interpolations, dim=0)


class ConditionalSpectrogramVAE(SpectrogramVAE):
    '''Conditional VAE that can generate spectrograms for specific classes - CORRECTED VERSION'''
    def __init__(self, input_shape, num_classes, latent_dim=1024, beta=1.0):
        # Initialize the parent class first
        super().__init__(input_shape, latent_dim, beta)
        self.num_classes = num_classes

        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, 64)

        # Store original flatten size before modification
        original_flatten_size = self.encoder.flatten_size

        # Modify encoder to include class conditioning
        self.encoder.fc_mu = nn.Linear(original_flatten_size + 64, latent_dim)
        self.encoder.fc_logvar = nn.Linear(original_flatten_size + 64, latent_dim)

        # Modify decoder to include class conditioning
        fc_output_size = self.encoder.channels * self.encoder.height * self.encoder.width
        self.decoder.fc = nn.Linear(latent_dim + 64, fc_output_size)

    def encode(self, x, class_labels):
        # Get features from the encoder
        features = self.encoder.encoder(x)
        features = features.view(features.size(0), -1)

        # Add class embeddings
        class_emb = self.class_embedding(class_labels)
        combined = torch.cat([features, class_emb], dim=1)

        mu = self.encoder.fc_mu(combined)
        logvar = self.encoder.fc_logvar(combined)

        return mu, logvar

    def decode(self, z, class_labels):
        class_emb = self.class_embedding(class_labels)
        combined = torch.cat([z, class_emb], dim=1)

        batch_size = z.size(0)
        x = self.decoder.fc(combined)
        
        # Use the encoder's actual encoded dimensions
        x = x.view(batch_size, self.decoder.channels, self.decoder.height, self.decoder.width)
        x = self.decoder.decoder(x)

        # Always interpolate to exact output shape
        if x.shape[2:] != self.decoder.output_shape:
            x = F.interpolate(x, size=self.decoder.output_shape, mode='bilinear', align_corners=False)
        
        return x

    def forward(self, x, class_labels):
        mu, logvar = self.encode(x, class_labels)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, class_labels)

        return recon_x, mu, logvar, z

    def sample_class(self, class_label, num_samples, device='cpu'):
        """Generate spectrograms for a specific class"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            class_labels = torch.full((num_samples,), class_label, device=device, dtype=torch.long)
            samples = self.decode(z, class_labels)
            return samples


# Debug function to test the architecture
def test_vae_architecture(input_shape, latent_dim=128):
    """Test function to debug VAE architecture"""
    print(f"\n{'='*60}")
    print(f"TESTING VAE ARCHITECTURE")
    print(f"{'='*60}")
    print(f"Input shape: {input_shape}")
    
    # Create model
    vae = SpectrogramVAE(input_shape, latent_dim)
    
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 1, *input_shape)
    
    print(f"\nInput tensor shape: {x.shape}")
    print(f"Total model parameters: {sum(p.numel() for p in vae.parameters()):,}")
    
    # Test forward pass
    try:
        print(f"\nTesting forward pass...")
        recon_x, mu, logvar, z = vae(x)
        print(f"✅ Forward pass successful!")
        print(f"  Input shape: {x.shape}")
        print(f"  Reconstruction shape: {recon_x.shape}")
        print(f"  Latent mu shape: {mu.shape}")
        print(f"  Latent logvar shape: {logvar.shape}")
        print(f"  Latent z shape: {z.shape}")
        
        # Test loss calculation
        total_loss, recon_loss, kl_loss = vae.loss_function(recon_x, x, mu, logvar)
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Recon loss: {recon_loss.item():.4f}")
        print(f"  KL loss: {kl_loss.item():.4f}")
        
        # Test sampling
        samples = vae.sample(2)
        print(f"  Sample shape: {samples.shape}")
        
        # Test shapes match
        assert x.shape == recon_x.shape, f"Shape mismatch: input {x.shape} vs recon {recon_x.shape}"
        assert samples.shape[1:] == x.shape[1:], f"Sample shape mismatch: {samples.shape} vs {x.shape}"
        
        print(f"✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    print(f"{'='*60}")
    return vae


if __name__ == "__main__":
    # Test with your actual input shape
    test_shape = (1025, 469)
    model = test_vae_architecture(test_shape)