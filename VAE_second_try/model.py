import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return self.relu(out)


class Encoder(nn.Module):
    """Encoder part of the VAE"""
    def __init__(self, input_shape, latent_dim=512):
        super(Encoder, self).__init__()
        # self.input_shape = input_shape
        # Ensure input_shape is (channels, height, width)
        if isinstance(input_shape, torch.Size):
            input_shape = tuple(input_shape)
        if len(input_shape) == 2:
            self.input_shape = (1, *input_shape)  # Add channel dim
        else:
            self.input_shape = input_shape

        # Conditional archietcture for the encoder 
        # 6 convolutional layers        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2), 

            # Residual blocks
            # ResidualBlock(32),
            # nn.Conv2d(32, 64, kernel_size = 2, stride=2, padding = 1),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(0.2),
            
            # ResidualBlock(64),
            # nn.Conv2d(64, 128, kernel_size = 2, stride=2, padding = 1),
            # nn.BatchNorm2d(128), 
            # nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2), 
 

            nn.Conv2d(64, 128, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),  

            nn.Conv2d(128, 256, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2), 

            nn.Conv2d(256, 512, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2), 

            nn.Conv2d(512, 512, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2), 
        )

        # flattening after convolution
        # self.encoder_shape = self._get_shape()
        # self.encoder_flatten = self.encoder_shape[1] * self.encoder_shape[2] * self.encoder_shape[3] # channels * height * width
        # print(self.encoder_shape[1])
        # print(self.encoder_shape[2])
        # print(self.encoder_shape[3])
        # self.encoder_shape = self._calculate_flatten_size()  # Now gets the returned value
        # print 
        # self.encoder_flatten = self.encoder_shape[1] * self.encoder_shape[2] * self.encoder_shape[3]
        # Calculate and store both the shape and flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            self.encoder_output = self.encoder(dummy)
            self.encoder_shape = self.encoder_output.shape[1:]  # [512, 17, 9]
            self.encoder_flatten = self.encoder_output.numel() // self.encoder_output.shape[0]
            print(f"Encoder output shape: {self.encoder_shape}")
            print(f"Flatten size: {self.encoder_flatten}")

        self.fc_mu = nn.Linear(self.encoder_flatten, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_flatten, latent_dim)

        # FC mu and sigma
        self.fc_mu = nn.Linear(self.encoder_flatten, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_flatten, latent_dim)

    
    # def _get_shape(self):
    #     """Calculate output shape after convolutions"""
    #     with torch.no_grad():
    #         # Ensure input is 4D: [batch, channels, height, width]
    #         if len(self.input_shape) == 2:
    #             # If input_shape is (height, width), add channel dim
    #             dummy = torch.zeros(1, 1, *self.input_shape)  # [1, 1, 1025, 469]
    #         else:
    #             # If input_shape is already (channels, height, width)
    #             dummy = torch.zeros(1, *self.input_shape)  # [1, 1, 1025, 469]
            
    #         return self.encoder(dummy).shape  # [1, channels, h, w]
    # def _calculate_flatten_size(self):
    #     with torch.no_grad():
    #         dummy = torch.zeros(1, *self.input_shape)
    #         self.encoder_output = self.encoder(dummy)
    #         self.encoder_shape = self.encoder_output.shape[1:]  # [512, 17, 9]
    #         self.encoder_flatten = self.encoder_output.numel() // self.encoder_output.shape[0]
    #         print(f"Encoder output shape: {self.encoder_shape}")
    #         print(f"Flatten size: {self.encoder_flatten}")

    def forward(self, x):
        # Debug input shape
        # print(f"Encoder input shape: {x.shape}")
        
        # Ensure proper 4D input
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dim
        
        x = self.encoder(x)
        # print(f"Encoder output before flatten: {x.shape}")
        
        # Verify expected flatten size
        expected_flatten = 512*17*9
        actual_flatten = x.shape[1] * x.shape[2] * x.shape[3]
        assert actual_flatten == expected_flatten, \
            f"Flatten size mismatch: {actual_flatten} vs {expected_flatten}"
        
        x = x.view(x.size(0), -1)
        # print(f"Flattened shape: {x.shape}")
        
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    """Decoder part of the VAE"""
    def __init__(self, output_shape, encoder_shape, latent_dim=512):
        super(Decoder, self).__init__()
        self.output_shape = output_shape
        self.latent_dim = latent_dim
        # self.encoder_shape = encoder_shape  # [batch, channels, height, width]

        # # Extract dimensions from encoder shape
        # self.channels = encoder_shape[1]  # Number of channels
        # self.height = encoder_shape[2]    # Height
        # self.width = encoder_shape[3]     # Width
        # Handle both 3D and 4D encoder_shape
        if len(encoder_shape) == 3:  # [C, H, W]
            self.channels, self.height, self.width = encoder_shape
        elif len(encoder_shape) == 4:  # [B, C, H, W]
            _, self.channels, self.height, self.width = encoder_shape
        else:
            raise ValueError(f"Invalid encoder_shape: {encoder_shape}")
        
        # Flattened size for FC layer
        self.encoder_flatten = self.channels * self.height * self.width

        # FC layer to expand latent vector
        self.fc = nn.Linear(latent_dim, self.encoder_flatten)

        # Conditiona architecture for the decoder
        # 6 layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # ResidualBlock(512),

            nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # ResidualBlock(256),

            nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 1, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, z):
        batch_size = z.size(0)

        # passing through the FC layer
        x = self.fc(z) # expanding the latent vector

        # Reshaping for the convolutional layers to match encoder output
        x = x.view(x.size(0), self.channels, self.height, self.width)

        # going through the decoder
        x = self.decoder(x)

        return x


class VariationalAutoEncoder(nn.Module):
    """Variationa Autoencoder form spectrograms"""
    def __init__(self, input_shape, latent_dim = 512, beta = 1.0): # beta
        super(VariationalAutoEncoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = Encoder(input_shape, latent_dim)
        self.decoder = Decoder(
            output_shape=input_shape,
            encoder_shape=self.encoder.encoder_shape,  # Now passes [512,17,9]
            latent_dim=latent_dim
        )

    # Second step possible implementations
    # Skip connections for better detail preservation
    # β-VAE capability for controlled disentanglement

    # Reparametrization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)

        return mu + eps * std

    def loss_function(self, recon_x, x, mu, logvar, beta=None):
        """VAE Loss = reconstruction + KL divergence"""
        # Reconstruction loss with MSE
        reconstruction_loss = F.mse_loss(recon_x, x, reduction = 'sum') / x.size(0) # Mean per batch

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0) # Mean per batch

        # Use beta if provided, else use class default
        effective_beta = self.beta if beta is None else beta
        total_loss = reconstruction_loss + (effective_beta * kl_loss)

        return total_loss, reconstruction_loss, kl_loss

    def sample(self, num_samples, device = 'cpu'):
        """Generation of new samples for the latent space"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decoder(z)
            return samples

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)

        # Reshape reconstruction to match input
        if recon_x.shape != x.shape:
            recon_x = F.interpolate(
                recon_x,
                size=x.shape[-2:],
                mode='bilinear'
            )

        return recon_x, mu, logvar


class ConditionalVariationalAutoEncoder(VariationalAutoEncoder):
    """VAE that is able to generate spectrograms for specific lables"""
    def __init__(self, input_shape, latent_dim = 512, num_classes = 3, embed_dim=100):
        super().__init__(input_shape, latent_dim)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Encoder modification to accept concatenated input
        # Adjusting the first convolutional layer so to accept additional informations/channels
        original_conv = self.encoder.encoder[0]
        self.encoder.encoder[0] = nn.Conv2d(
            1 + embed_dim,  # Original + embedded label channels
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding
        )
        
        # Modify decoder FC layer to accept latent + label embedding
        original_fc = self.decoder.fc
        self.decoder.fc = nn.Linear(
            latent_dim + embed_dim,
            original_fc.out_features
        )

    def sample_class(self, class_label, num_samples, device='cpu'):
        """Generate samples for a specific class"""
        with torch.no_grad():
            # Create latent samples
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            # Create class labels
            labels = torch.full((num_samples,), class_label, dtype=torch.long).to(device)
            label_embed = self.label_embedding(labels)
            
            # Concatenate and decode
            z_conditioned = torch.cat([z, label_embed], dim=1)
            samples = self.decoder(z_conditioned)
            
            return samples 

    def encode(self, x, class_labels):
        pass

    def decode(self, z, class_labels):
        pass

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
        
        return recon_x, mu, logvar
