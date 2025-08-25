import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 

class AttentionBlock(nn.Module):
    """Self-attention for capturing frequency patterns in spectrograms"""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        
        # Attention weights
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x


class ResidualBlock(nn.Module):
    """Residual block with proper normalization"""
    def __init__(self, in_channels, use_attention=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # Use BatchNorm instead of GroupNorm for simplicity
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(in_channels)
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(0.1)
        
        self.attention = AttentionBlock(in_channels) if use_attention else None
        
    def forward(self, x):
        residual = x
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        
        if self.attention:
            out = self.attention(out)
        
        out += residual
        return self.activation(out)


class Encoder(nn.Module):
    """Clean, working encoder architecture"""
    def __init__(self, input_shape, latent_dim=1024):
        super().__init__()
        
        # Handle input shape
        if isinstance(input_shape, torch.Size):
            input_shape = tuple(input_shape)
        if len(input_shape) == 2:
            self.input_shape = (1, *input_shape)
        else:
            self.input_shape = input_shape

        # Progressive encoding layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )
        
        self.res1 = ResidualBlock(32)
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )
        
        self.res2 = ResidualBlock(64)
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )
        
        self.res3 = ResidualBlock(128)
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )
        
        self.res4 = ResidualBlock(256, use_attention=True)
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )
        
        self.res5 = ResidualBlock(512, use_attention=True)
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU()
        )

        # Calculate output shape
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            dummy = self.forward_conv_only(dummy)
            self.encoder_shape = dummy.shape[1:]
            self.encoder_flatten = dummy.numel() // dummy.shape[0]
            print(f"✅ Encoder output shape: {self.encoder_shape}")
            print(f"✅ Flatten size: {self.encoder_flatten}")

        # Latent projection layers
        self.fc_mu = nn.Sequential(
            nn.Linear(self.encoder_flatten, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.encoder_flatten, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward_conv_only(self, x):
        """Forward pass through convolutional layers only (for shape calculation)"""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.layer1(x)
        x = self.res1(x)
        x = self.layer2(x)
        x = self.res2(x)
        x = self.layer3(x)
        x = self.res3(x)
        x = self.layer4(x)
        x = self.res4(x)
        x = self.layer5(x)
        x = self.res5(x)
        x = self.layer6(x)
        
        return x

    def forward(self, x):
        x = self.forward_conv_only(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # Clamp logvar to prevent extreme values
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return mu, logvar


class Decoder(nn.Module):
    """Clean, working decoder architecture"""
    def __init__(self, output_shape, encoder_shape, latent_dim=1024):
        super().__init__()
        self.output_shape = output_shape
        self.latent_dim = latent_dim

        # Handle encoder shape
        if len(encoder_shape) == 3:
            self.channels, self.height, self.width = encoder_shape
        elif len(encoder_shape) == 4:
            _, self.channels, self.height, self.width = encoder_shape
        else:
            raise ValueError(f"Invalid encoder_shape: {encoder_shape}")
        
        self.encoder_flatten = self.channels * self.height * self.width

        # FC layers to expand latent vector
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, self.encoder_flatten),
            nn.GELU()
        )

        # Progressive decoding layers (mirror of encoder)
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )
        
        self.res1 = ResidualBlock(512, use_attention=True)
        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )
        
        self.res2 = ResidualBlock(256, use_attention=True)
        
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )
        
        self.res3 = ResidualBlock(128)
        
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )
        
        self.res4 = ResidualBlock(64)
        
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )
        
        self.res5 = ResidualBlock(32)
        
        # Final output layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        batch_size = z.size(0)
        
        # Expand latent vector
        x = self.fc(z)
        
        # Reshape for convolutional layers
        x = x.view(batch_size, self.channels, self.height, self.width)
        
        # Progressive decoding
        x = self.layer1(x)
        x = self.res1(x)
        x = self.layer2(x)
        x = self.res2(x)
        x = self.layer3(x)
        x = self.res3(x)
        x = self.layer4(x)
        x = self.res4(x)
        x = self.layer5(x)
        x = self.res5(x)
        x = self.final_layer(x)
        
        return x


class SpectralLoss(nn.Module):
    """Enhanced loss function for spectrograms"""
    def __init__(self):
        super().__init__()
        
    def forward(self, recon_x, x):
        # Base reconstruction loss
        mse_loss = F.mse_loss(recon_x, x, reduction='mean')
        l1_loss = F.l1_loss(recon_x, x, reduction='mean')
        
        # Combine MSE and L1 for better detail preservation
        reconstruction_loss = 0.8 * mse_loss + 0.2 * l1_loss
        
        # Frequency-weighted loss (emphasize important frequencies)
        if x.dim() == 4 and x.shape[-2] > 1:
            freq_size = x.shape[-2]
            # Higher weight for lower frequencies (more important for audio)
            freq_weights = torch.linspace(1.5, 0.5, freq_size).to(x.device)
            freq_weights = freq_weights.view(1, 1, -1, 1)
            
            weighted_recon = recon_x * freq_weights
            weighted_x = x * freq_weights
            freq_loss = F.mse_loss(weighted_recon, weighted_x, reduction='mean')
            
            return reconstruction_loss + 0.2 * freq_loss
        
        return reconstruction_loss


class VariationalAutoEncoder(nn.Module):
    """Enhanced VAE with better architecture and loss"""
    def __init__(self, input_shape, latent_dim=1024, beta=1.0):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta

        # Initialize encoder and decoder
        self.encoder = Encoder(input_shape, latent_dim)
        self.decoder = Decoder(
            output_shape=input_shape,
            encoder_shape=self.encoder.encoder_shape,
            latent_dim=latent_dim
        )
        
        # Enhanced loss function
        self.spectral_loss = SpectralLoss()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def loss_function(self, recon_x, x, mu, logvar, beta=None):
        """Enhanced loss with spectral awareness and free bits"""
        batch_size = x.size(0)
        
        # Use spectral loss for reconstruction
        reconstruction_loss = self.spectral_loss(recon_x, x)
        
        # KL divergence with free bits to prevent posterior collapse
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # Free bits: don't penalize KL if it's below threshold
        free_bits = 0.2  # Allow some deviation before penalizing
        kl_loss = torch.clamp(kl_loss, min=free_bits)
        
        # Use provided beta or class default
        effective_beta = self.beta if beta is None else beta
        total_loss = reconstruction_loss + (effective_beta * kl_loss)

        return total_loss, reconstruction_loss, kl_loss

    def sample(self, num_samples, device='cpu'):
        """Generate samples from the latent space"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decoder(z)
            return samples

    def forward(self, x):
        # Encode
        mu, logvar = self.encoder(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_x = self.decoder(z)

        # Ensure output matches input size exactly
        if recon_x.shape != x.shape:
            recon_x = F.interpolate(
                recon_x,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        return recon_x, mu, logvar


class ConditionalVariationalAutoEncoder(VariationalAutoEncoder):
    """Enhanced Conditional VAE"""
    def __init__(self, input_shape, latent_dim=1024, num_classes=3, embed_dim=100):
        super().__init__(input_shape, latent_dim)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Label embedding
        self.label_embedding = nn.Sequential(
            nn.Embedding(num_classes, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Modify encoder first layer to accept label channels
        original_conv = self.encoder.layer1[0]  # First conv layer
        self.encoder.layer1[0] = nn.Conv2d(
            1 + embed_dim,  # Input channels: spectrogram + label embedding
            original_conv.out_channels,  # Output channels (32)
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding
        )
        
        # Modify decoder FC layer to accept latent + label embedding
        original_fc_out = self.decoder.fc[-1].out_features
        self.decoder.fc = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, original_fc_out),
            nn.GELU()
        )

    def sample_class(self, class_label, num_samples, device='cpu'):
        """Generate samples for a specific class"""
        with torch.no_grad():
            # Create latent samples
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            # Create class labels
            labels = torch.full((num_samples,), class_label, dtype=torch.long).to(device)
            label_embed = self.label_embedding(labels)
            
            # Concatenate latent with label embedding
            z_conditioned = torch.cat([z, label_embed], dim=1)
            samples = self.decoder(z_conditioned)
            
            return samples

    def forward(self, x, labels):
        batch_size = x.size(0)
        
        # Embed labels
        label_embed = self.label_embedding(labels)  # [batch_size, embed_dim]
        
        # Expand label embedding to match spatial dimensions
        label_embed_expanded = label_embed.view(batch_size, self.embed_dim, 1, 1)
        label_embed_expanded = label_embed_expanded.expand(-1, -1, x.size(2), x.size(3))
        
        # Concatenate input with label embedding
        x_conditioned = torch.cat([x, label_embed_expanded], dim=1)
        
        # Encode
        mu, logvar = self.encoder(x_conditioned)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Concatenate latent with label embedding for decoder
        z_conditioned = torch.cat([z, label_embed], dim=1)
        
        # Decode
        recon_x = self.decoder(z_conditioned)
        
        # Ensure output matches input size
        if recon_x.shape != x.shape:
            recon_x = F.interpolate(
                recon_x,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        return recon_x, mu, logvar


# Beta annealing function for better training
def get_annealed_beta(epoch, max_epochs, target_beta, annealing_type='linear'):
    """
    Gradually increase beta during training
    
    Args:
        epoch: Current epoch
        max_epochs: Total number of epochs
        target_beta: Final beta value
        annealing_type: 'linear', 'sigmoid', or 'cyclical'
    """
    if annealing_type == 'linear':
        # Linear increase over first 60% of training
        annealing_epochs = int(max_epochs * 0.6)
        if epoch < annealing_epochs:
            return target_beta * (epoch / annealing_epochs)
        else:
            return target_beta
            
    elif annealing_type == 'sigmoid':
        # Sigmoid annealing - smooth S-curve
        progress = epoch / max_epochs
        return target_beta * (1 / (1 + np.exp(-10 * (progress - 0.5))))
        
    elif annealing_type == 'cyclical':
        # Cyclical annealing - helps avoid local minima
        cycle_length = max_epochs // 4
        cycle_progress = (epoch % cycle_length) / cycle_length
        return target_beta * cycle_progress
        
    else:
        return target_beta