import copy
import math
from typing import List  # NOQA

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

# import time
from torch import profiler  # NOQA
from torch import einsum, nn

import transformer_engine.pytorch as te
#from transformer_engine.common.recipe import Format, DelayedScaling
#from transformer_engine.pytorch import Linear, LayerNorm

'''

device_capability = torch.cuda.get_device_capability()
if device_capability[0]+device_capability[1]*0.1 >= 8.9:
    print("GPU compatible with fp8 computation, setting USE_FP8 to True")
    USE_FP8 = True
else:
    print("GPU not compatible with fp8 computation, setting USE_FP8 to False")
    USE_FP8 = False

fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")

'''

class TransformerToLatent_sparse2_fG(nn.Module):
    def __init__(self, latent_dim, embed_dim, dim_intermediate, num_heads, num_layers, dropout):
        super().__init__()
        self.embed_dim = embed_dim  # Embedding dimension
        
        # Linear layer to project each token to embedding space
        self.tokenizer = nn.Linear(29, embed_dim)
        # Positional embeddings
        self.positional_embedding = nn.Parameter(torch.randn(288, embed_dim))  # 12*24 tokens

        # Transformer encoder
        self.transformer_layers = nn.ModuleList([
            te.TransformerLayer(
                hidden_size=embed_dim,
                num_attention_heads=num_heads,
                ffn_hidden_size=2*embed_dim,
                apply_residual_connection_post_layernorm=False,
                layernorm_epsilon=1e-5,
                hidden_dropout=dropout,
                attention_dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.fc_mix = nn.Linear(dim_intermediate*288, 2*latent_dim)
        self.fc = nn.Linear(embed_dim, dim_intermediate)
        self.layernorm1 = nn.LayerNorm([dim_intermediate])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = rearrange(x, "b (h latlon) -> b latlon h", h=29) # (bs*step, 288, 29): 288 tokens with 29 dimensions each
        x_emb = self.tokenizer(x)  # (bs, 288, embed_dim)
        x_emb = x_emb + self.positional_embedding  # (bs, 288, embed_dim)

        for layer in self.transformer_layers:
            x_emb = layer(x_emb)
        '''
        if USE_FP8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                for layer in self.transformer_layers:
                    # TE’s layer expects (batch, seq_len, hidden_size), 
                    # which matches x_emb shape (bs, 264, embed_dim).
                    x_emb = layer(x_emb)
        else:
            for layer in self.transformer_layers:
                # TE’s layer expects (batch, seq_len, hidden_size), 
                # which matches x_emb shape (bs, 288, embed_dim).
                x_emb = layer(x_emb)
        '''
        
        x_emb = self.relu(self.layernorm1(self.fc(x_emb)))  # (bs, 288, something)
        x_out = rearrange(x_emb, "bs c t -> bs (c t)")
        x_out = self.fc_mix(x_out)  # (bs, 288, something)
        return x_out

class TransformerToLatent_sparser_fG(nn.Module):
    def __init__(self, latent_dim, embed_dim, dim_intermediate, num_heads, num_layers, dropout):
        super().__init__()
        self.embed_dim = embed_dim  # Embedding dimension
        
        # Linear layer to project each token to embedding space
        self.tokenizer = nn.Linear(29, embed_dim)
        # Positional embeddings
        self.positional_embedding = nn.Parameter(torch.randn(72, embed_dim))  # 6*12 tokens: shall we also tokenize in the height direction?

        # Transformer encoder
        self.transformer_layers = nn.ModuleList([
            te.TransformerLayer(
                hidden_size=embed_dim,
                num_attention_heads=num_heads,
                ffn_hidden_size=2*embed_dim,
                apply_residual_connection_post_layernorm=False,
                layernorm_epsilon=1e-5,
                hidden_dropout=dropout,
                attention_dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.fc_mix = Linear(dim_intermediate*72, 2*latent_dim)
        self.fc = Linear(embed_dim, dim_intermediate)
        self.layernorm1 = LayerNorm([dim_intermediate])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = rearrange(x, "b (h latlon) -> b latlon h", h=33) # (bs*step, 72, 33): 72 tokens with 33 dimensions each
        x_emb = self.tokenizer(x)  # (bs, 72, embed_dim)
        x_emb = x_emb + self.positional_embedding  # (bs, 72, embed_dim)

        if USE_FP8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                for layer in self.transformer_layers:
                    # TE’s layer expects (batch, seq_len, hidden_size), 
                    # which matches x_emb shape (bs, 72, embed_dim).
                    x_emb = layer(x_emb)
        else:
            for layer in self.transformer_layers:
                # TE’s layer expects (batch, seq_len, hidden_size), 
                # which matches x_emb shape (bs, 72, embed_dim).
                x_emb = layer(x_emb)
        
        x_emb = self.relu(self.layernorm1(self.fc(x_emb)))  # (bs, 72, something)
        x_out = rearrange(x_emb, "bs c t -> bs (c t)")
        x_out = self.fc_mix(x_out)  # (bs, 72, something)
        return x_out

class TransformerToLatent_sparsest_fG(nn.Module):
    def __init__(self, latent_dim, embed_dim, dim_intermediate, num_heads, num_layers, dropout):
        super().__init__()
        self.embed_dim = embed_dim  # Embedding dimension
        
        # Linear layer to project each token to embedding space
        self.tokenizer = nn.Linear(29, embed_dim)
        # Positional embeddings
        self.positional_embedding = nn.Parameter(torch.randn(18, embed_dim))  # 6*12 tokens: shall we also tokenize in the height direction?

        # Transformer encoder
        self.transformer_layers = nn.ModuleList([
            te.TransformerLayer(
                hidden_size=embed_dim,
                num_attention_heads=num_heads,
                ffn_hidden_size=2*embed_dim,
                apply_residual_connection_post_layernorm=False,
                layernorm_epsilon=1e-5,
                hidden_dropout=dropout,
                attention_dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.fc_mix = nn.Linear(dim_intermediate*18, 2*latent_dim)
        self.fc = nn.Linear(embed_dim, dim_intermediate)
        self.layernorm1 = nn.LayerNorm([dim_intermediate])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = rearrange(x, "b (h latlon) -> b latlon h", h=29) # (bs*step, 18, 33): 18 tokens with 33 dimensions each
        x_emb = self.tokenizer(x)  # (bs, 18, embed_dim)
        x_emb = x_emb + self.positional_embedding  # (bs, 18, embed_dim)

        for layer in self.transformer_layers:
            x_emb = layer(x_emb)
        '''
        if USE_FP8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                for layer in self.transformer_layers:
                    # TE’s layer expects (batch, seq_len, hidden_size), 
                    # which matches x_emb shape (bs, 18, embed_dim).
                    x_emb = layer(x_emb)
        else:
            for layer in self.transformer_layers:
                # TE’s layer expects (batch, seq_len, hidden_size), 
                # which matches x_emb shape (bs, 18, embed_dim).
                x_emb = layer(x_emb)
        '''
        
        x_emb = self.relu(self.layernorm1(self.fc(x_emb)))  # (bs, 18, something)
        x_out = rearrange(x_emb, "bs c t -> bs (c t)")
        x_out = self.fc_mix(x_out)  # (bs, 18, something)
        return x_out


class MLPMixerLike(nn.Module):
    def __init__(self, channel, embed_dim, channel_, embed_dim_, num_layers):
        """
        Args:
            channel: original channel dimension
            embed_dim: original embedding dimension
            channel_: target channel dimension after mixing
            embed_dim_: target embedding dimension after mixing
                      (latent_dim will be channel_ * embed_dim_)
        """
        super().__init__()
        self.num_layers = num_layers
        self.embed_mixers = nn.ModuleList()
        self.channel_mixers = nn.ModuleList()
        
        if num_layers == 1:
            # Only final layer: no ReLU.
            self.embed_mixers.append(
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim_)
                )
            )
            self.channel_mixers.append(
                nn.Sequential(
                    nn.LayerNorm(channel),
                    nn.Linear(channel, channel_)
                )
            )
        else:
            # First layer: increases dimensions and applies ReLU.
            self.embed_mixers.append(
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim_),
                    nn.ReLU()
                )
            )
            self.channel_mixers.append(
                nn.Sequential(
                    nn.LayerNorm(channel),
                    nn.Linear(channel, channel_),
                    nn.ReLU()
                )
            )
            # Intermediate layers: keep dimensions constant, with ReLU.
            for _ in range(1, num_layers - 1):
                self.embed_mixers.append(
                    nn.Sequential(
                        nn.LayerNorm(embed_dim_),
                        nn.Linear(embed_dim_, embed_dim_),
                        nn.ReLU()
                    )
                )
                self.channel_mixers.append(
                    nn.Sequential(
                        nn.LayerNorm(channel_),
                        nn.Linear(channel_, channel_),
                        nn.ReLU()
                    )
                )
            # Final layer: no ReLU.
            self.embed_mixers.append(
                nn.Sequential(
                    nn.LayerNorm(embed_dim_),
                    nn.Linear(embed_dim_, embed_dim_)
                )
            )
            self.channel_mixers.append(
                nn.Sequential(
                    nn.LayerNorm(channel_),
                    nn.Linear(channel_, channel_)
                )
            )
        '''
        # MLP for mixing along the embedding (last) dimension:
        self.embed_mixer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim_),
            nn.GELU(),
            nn.LayerNorm(embed_dim_),
            nn.Linear(embed_dim_, embed_dim_)
        )
        # MLP for mixing along the channel dimension:
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, channel_),
            nn.GELU(),
            nn.LayerNorm(channel_),
            nn.Linear(channel_, channel_)
        )
        '''
    
    def forward(self, x):
        x = x.float()
        with torch.autocast(enabled=False, device_type="cuda"):
            for embed_mixer, channel_mixer in zip(self.embed_mixers, self.channel_mixers):
                x = embed_mixer(x)
                x = x.transpose(1, 2)
                x = channel_mixer(x)
                x = x.transpose(1, 2)
        x = x.half()
        return x

class TransformerToLatent_sparse2_fG_MLPMixerLike(nn.Module):
    def __init__(self, latent_dim, embed_dim, dim_intermediate, num_heads, num_layers, num_MLPMixer_layers, dropout):
        super().__init__()
        self.embed_dim = embed_dim  # Embedding dimension
        
        # Linear layer to project each token to embedding space
        self.tokenizer = nn.Linear(33, embed_dim)
        # Positional embeddings
        self.positional_embedding = nn.Parameter(torch.randn(264, embed_dim))  # 11*24 tokens

        # Transformer encoder
        self.transformer_layers = nn.ModuleList([
            te.TransformerLayer(
                hidden_size=embed_dim,
                num_attention_heads=num_heads,
                ffn_hidden_size=2*embed_dim,
                apply_residual_connection_post_layernorm=False,
                layernorm_epsilon=1e-5,
                hidden_dropout=dropout,
                attention_dropout=dropout
            )
            for _ in range(num_layers)
        ])
        channel = 264 # tokens
        channel_ = 288 # 
        embed_dim_ = 2*int(latent_dim//channel_)
        self.mixer = MLPMixerLike(channel, embed_dim, channel_, embed_dim_, num_MLPMixer_layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = rearrange(x, "b (h latlon) -> b latlon h", h=33) # (bs*step, 264, 33): 264 tokens with 33 dimensions each
        x_emb = self.tokenizer(x)  # (bs, 264, embed_dim)
        x_emb = x_emb + self.positional_embedding  # (bs, 264, embed_dim)

        if USE_FP8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                for layer in self.transformer_layers:
                    # TE’s layer expects (batch, seq_len, hidden_size), 
                    # which matches x_emb shape (bs, 264, embed_dim).
                    x_emb = layer(x_emb)
        else:
            for layer in self.transformer_layers:
                # TE’s layer expects (batch, seq_len, hidden_size), 
                # which matches x_emb shape (bs, 264, embed_dim).
                x_emb = layer(x_emb)
        
        x_emb = self.mixer(x_emb)
        x_emb = rearrange(x_emb, "bs c t -> bs (c t)")
        #print(f"{x_emb.shape=}")
        return x_emb

class Pixelshuffle(nn.Module):
    def __init__(self, latent_dim, out_channels, height, latitude, longitude):
        super().__init__()
        self.height = height
        self.latitude = latitude
        self.longitude = longitude
        
        # Treat height as a channel
        self.num_channels = height

        # Calculate intermediate dimensions
        self.init_latitude = latitude // 16
        self.init_longitude = longitude // 16

        # Fully connected layer to expand latent dimension
        self.fc = Linear(latent_dim, latent_dim)
        self.layernorm_first1 = LayerNorm([latent_dim])
        assert latent_dim % (self.init_latitude*self.init_longitude) == 0
        self.first_channel = int(latent_dim//(self.init_latitude*self.init_longitude))

        # Transpose convolutional layers to upscale dimensions
        self.conv_transpose2d_0 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.first_channel,
                out_channels=384 * 4,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.PixelShuffle(upscale_factor=2)
        )
        self.conv_transpose2d_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=384,
                out_channels=192 * 4,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.PixelShuffle(upscale_factor=2)
        )
        self.conv_transpose2d_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=192,
                out_channels=96 * 4,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.PixelShuffle(upscale_factor=2)
        )
        self.conv_transpose2d_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=96,
                out_channels=48 * 4,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.PixelShuffle(upscale_factor=2)
        )
        # Additional convolutional layers to introduce nonlinearity
        self.conv2d_1 = nn.Conv2d(
            in_channels=48,
            out_channels=4*self.num_channels+1,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # LayerNorm for intermediate normalization
        self.layernorm0 = LayerNorm([384, 6, 12])
        self.layernorm1 = LayerNorm([192, 12, 24])
        self.layernorm2 = LayerNorm([96, 24, 48])
        self.layernorm3 = LayerNorm([48, latitude, longitude])
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, latent_dim = x.size()
        if USE_FP8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                x = self.layernorm_first1(self.fc(x))
                x = x.view(batch_size, self.first_channel, self.init_latitude, self.init_longitude)
                # Apply transpose convolutions
                x = self.conv_transpose2d_0(x)
                x = self.layernorm0(x)
                x = self.relu(x)
                x = self.conv_transpose2d_1(x)
                x = self.layernorm1(x)
                x = self.relu(x)
                x = self.conv_transpose2d_2(x)
                x = self.layernorm2(x)
                x = self.relu(x)
                x = self.conv_transpose2d_3(x)
        else:
            x = self.layernorm_first1(self.fc(x))
            x = x.view(batch_size, self.first_channel, self.init_latitude, self.init_longitude)
            # Apply transpose convolutions
            x = self.conv_transpose2d_0(x)
            x = self.layernorm0(x)
            x = self.relu(x)
            x = self.conv_transpose2d_1(x)
            x = self.layernorm1(x)
            x = self.relu(x)
            x = self.conv_transpose2d_2(x)
            x = self.layernorm2(x)
            x = self.relu(x)
            x = self.conv_transpose2d_3(x)
        x = self.conv2d_1(x)

        # Reshape back to (batch_size, height, latitude, longitude)
        x = x.view(batch_size, 4*self.height+1, self.latitude, self.longitude)
        x = rearrange(x, "bs h lat lon -> bs (h lat lon)")
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        latent_dim,
        fc_middle_dim,
        out_channels,
        token_dim,
        height, 
        latitude, 
        longitude,
        num_layers,       # number of transformer layers to stack
        ffn_hidden_factor,  # feedforward dimension = token_dim * factor (here 132*4=528)
        num_attention_heads,
        dropout,
        patch_size
    ):
        super().__init__()
        # The target output dimension: 33*48*96 = 152064.
        self.out_channels = out_channels
        self.height = height
        self.latitude = latitude
        self.longitude = longitude
        total_output_dim = out_channels * latitude * longitude
        self.patch_size = patch_size
        self.new_lat = latitude // patch_size   # e.g., 48/4 = 12
        self.new_lon = longitude // patch_size    # e.g., 96/4 = 24
        self.token_dim = token_dim

        self.fc_middle_dim = fc_middle_dim
        #self.fc_mix = nn.Linear(latent_dim, self.fc_middle_dim)
        #self.layernorm1 = nn.LayerNorm([self.fc_middle_dim])
        self.fc_mix = nn.Linear(latent_dim, self.fc_middle_dim)
        self.layernorm1 = nn.LayerNorm([self.fc_middle_dim])
        self.fc2 = nn.Linear(int(self.fc_middle_dim//(self.new_lat*self.new_lon)), out_channels * (patch_size ** 2))
        self.layernorm2 = nn.LayerNorm([out_channels * (patch_size ** 2)])
        self.fc3 = nn.Linear(out_channels * (patch_size ** 2), self.token_dim)
        self.layernorm3 = nn.LayerNorm([self.token_dim])
        
        # Learned positional embeddings for the token sequence (length = new_lat * new_lon).
        self.pos_embedding = nn.Parameter(torch.randn(1, self.new_lat * self.new_lon, self.token_dim))
        
        # Build a transformer encoder by stacking TE TransformerLayer modules.
        # TE's TransformerLayer expects input shape: [seq_len, batch, hidden_size].
        self.transformer_layers = nn.ModuleList([
            te.TransformerLayer(
                hidden_size=self.token_dim,
                ffn_hidden_size=self.token_dim * ffn_hidden_factor,
                num_attention_heads=num_attention_heads,
                attention_dropout=dropout,
                hidden_dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.final_layer = nn.Linear(self.token_dim, out_channels*(patch_size**2))

    def forward(self, x):
        '''
        if USE_FP8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                b = x.size(0)
                x = self.layernorm1(self.fc_mix(x))
                x = rearrange(x, "b (x z) -> b x z", x=self.new_lat*self.new_lon, z=int(self.fc_middle_dim//(self.new_lat*self.new_lon)))
                x = self.layernorm2(self.fc2(x))
                x = x.view(b, self.out_channels, self.latitude, self.longitude)
                x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                              p1=self.patch_size, p2=self.patch_size)
                x = self.layernorm3(self.fc3(x))
                x = x + self.pos_embedding
                # Rearrange to [seq_len, batch, token_dim] for TE TransformerLayer.
                x = x.transpose(0, 1)
                # Pass through stacked transformer layers.
                for layer in self.transformer_layers:
                    x = layer(x, attention_mask=None)
        else:
        '''
        b = x.size(0)
        x = self.layernorm1(self.fc_mix(x))
        x = rearrange(x, "b (x z) -> b x z", x=self.new_lat*self.new_lon, z=int(self.fc_middle_dim//(self.new_lat*self.new_lon)))
        x = self.layernorm2(self.fc2(x))
        x = x.view(b, self.out_channels, self.latitude, self.longitude)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                      p1=self.patch_size, p2=self.patch_size)
        x = self.layernorm3(self.fc3(x))
        x = x + self.pos_embedding
        # Rearrange to [seq_len, batch, token_dim] for TE TransformerLayer.
        x = x.transpose(0, 1)
        # Pass through stacked transformer layers.
        for layer in self.transformer_layers:
            x = layer(x, attention_mask=None)
            
        x = self.final_layer(x)
        #x = self.final_transformer(x) # the final transformer is kept fp16
        # Rearrange back to (b, new_lat*new_lon, token_dim).
        x = x.transpose(0, 1)
        # Unpatch tokens: rearrange from tokens to spatial layout.
        x = rearrange(x, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                      h=self.new_lat, p1=self.patch_size, p2=self.patch_size)
        x = rearrange(x, "bs h lat lon -> bs (h lat lon)")
        
        return x

class TransformerDecoder_MLPMixerLike(nn.Module):
    def __init__(
        self, 
        latent_dim,
        fc_middle_dim,
        out_channels,
        token_dim,
        height, 
        latitude, 
        longitude,
        num_layers,       # number of transformer layers to stack
        num_MLPMixer_layers,
        ffn_hidden_factor,  # feedforward dimension = token_dim * factor (here 132*4=528)
        num_attention_heads,
        dropout,
        patch_size
    ):
        super().__init__()
        # The target output dimension: 33*48*96 = 152064.
        self.out_channels = out_channels
        self.height = height
        self.latitude = latitude
        self.longitude = longitude
        total_output_dim = out_channels * latitude * longitude
        self.patch_size = patch_size
        self.new_lat = latitude // patch_size   # e.g., 48/4 = 12
        self.new_lon = longitude // patch_size    # e.g., 96/4 = 24
        self.token_dim = token_dim

        self.fc_middle_dim = fc_middle_dim
        #self.fc_mix = nn.Linear(latent_dim, self.fc_middle_dim)
        #self.layernorm1 = nn.LayerNorm([self.fc_middle_dim])
        #self.fc_mix = Linear(latent_dim, self.fc_middle_dim)
        self.channel = 288
        embed_dim = int(latent_dim//self.channel)
        embed_dim_ = int(fc_middle_dim//self.channel)
        self.mixer = MLPMixerLike(self.channel, embed_dim, self.channel, embed_dim_, num_MLPMixer_layers)
        self.layernorm1 = LayerNorm([self.fc_middle_dim])
        self.fc2 = Linear(int(self.fc_middle_dim//(self.new_lat*self.new_lon)), out_channels * (patch_size ** 2))
        self.layernorm2 = LayerNorm([out_channels * (patch_size ** 2)])
        self.fc3 = Linear(out_channels * (patch_size ** 2), self.token_dim)
        self.layernorm3 = LayerNorm([self.token_dim])
        
        # Learned positional embeddings for the token sequence (length = new_lat * new_lon).
        self.pos_embedding = nn.Parameter(torch.randn(1, self.new_lat * self.new_lon, self.token_dim))
        
        # Build a transformer encoder by stacking TE TransformerLayer modules.
        # TE's TransformerLayer expects input shape: [seq_len, batch, hidden_size].
        self.transformer_layers = nn.ModuleList([
            te.TransformerLayer(
                hidden_size=self.token_dim,
                ffn_hidden_size=self.token_dim * ffn_hidden_factor,
                num_attention_heads=num_attention_heads,
                attention_dropout=dropout,
                hidden_dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.final_layer = Linear(self.token_dim, out_channels*(patch_size**2))

    def forward(self, x):
        if USE_FP8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                b = x.size(0)
                x = rearrange(x, 'b (c x) -> b c x', c=self.channel)
                x = rearrange(self.mixer(x), 'b c x -> b (c x)')
                x = rearrange(x, "b (x z) -> b x z", x=self.new_lat*self.new_lon, z=int(self.fc_middle_dim//(self.new_lat*self.new_lon)))
                x = self.layernorm2(self.fc2(x))
                x = x.view(b, self.out_channels, self.latitude, self.longitude)
                x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                              p1=self.patch_size, p2=self.patch_size)
                x = self.layernorm3(self.fc3(x))
                x = x + self.pos_embedding
                # Rearrange to [seq_len, batch, token_dim] for TE TransformerLayer.
                x = x.transpose(0, 1)
                # Pass through stacked transformer layers.
                for layer in self.transformer_layers:
                    x = layer(x, attention_mask=None)
        else:
            b = x.size(0)
            x = rearrange(x, 'b (c x) -> b c x', c=self.channel)
            x = rearrange(self.mixer(x), 'b c x -> b (c x)')
            x = rearrange(x, "b (x z) -> b x z", x=self.new_lat*self.new_lon, z=int(self.fc_middle_dim//(self.new_lat*self.new_lon)))
            x = self.layernorm2(self.fc2(x))
            x = x.view(b, self.out_channels, self.latitude, self.longitude)
            x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                          p1=self.patch_size, p2=self.patch_size)
            x = self.layernorm3(self.fc3(x))
            x = x + self.pos_embedding
            # Rearrange to [seq_len, batch, token_dim] for TE TransformerLayer.
            x = x.transpose(0, 1)
            # Pass through stacked transformer layers.
            for layer in self.transformer_layers:
                x = layer(x, attention_mask=None)

        x = self.final_layer(x)
        #x = self.final_transformer(x) # the final transformer is kept fp16
        # Rearrange back to (b, new_lat*new_lon, token_dim).
        x = x.transpose(0, 1)
        # Unpatch tokens: rearrange from tokens to spatial layout.
        x = rearrange(x, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                      h=self.new_lat, p1=self.patch_size, p2=self.patch_size)
        x = rearrange(x, "bs h lat lon -> bs (h lat lon)")
        
        return x

    
class TransformerDecoderNoMixing(nn.Module):
    def __init__(
        self, 
        latent_dim, 
        out_channels,
        token_dim,
        height, 
        latitude, 
        longitude,
        num_layers,
        ffn_hidden_factor,
        num_attention_heads,
        patch_size
    ):
        super().__init__()
        # The target output dimension: 33*48*96 = 152064.
        self.out_channels = out_channels
        self.height = height
        self.latitude = latitude
        self.longitude = longitude
        self.latent_dim = latent_dim
        total_output_dim = out_channels * latitude * longitude
        self.patch_size = patch_size
        self.new_lat = latitude // patch_size   # e.g., 48/4 = 12
        self.new_lon = longitude // patch_size    # e.g., 96/4 = 24
        # Each token is formed by the entire channel dimension (33) times the patch area (4*4 = 16):
        #self.token_dim = out_channels * (patch_size ** 2)  # 33 * 16 = 528
        self.token_dim = token_dim
        self.fc1 = nn.Linear(int(self.latent_dim//(self.new_lat*self.new_lon)), out_channels * (patch_size ** 2)) # to accomodate dimensions non-divisible by 8
        self.fc2 = Linear(out_channels * (patch_size ** 2), self.token_dim)
        self.layernorm1 = LayerNorm([out_channels * (patch_size ** 2)])
        self.layernorm2 = LayerNorm([self.token_dim])
        
        # Learned positional embeddings for the token sequence (length = new_lat * new_lon).
        self.pos_embedding = nn.Parameter(torch.randn(1, self.new_lat * self.new_lon, self.token_dim))
        
        # Build a transformer encoder by stacking TE TransformerLayer modules.
        # TE's TransformerLayer expects input shape: [seq_len, batch, hidden_size].
        self.transformer_layers = nn.ModuleList([
            te.TransformerLayer(
                hidden_size=self.token_dim,
                ffn_hidden_size=self.token_dim * ffn_hidden_factor,
                num_attention_heads=num_attention_heads,
                attention_dropout=0.0,
                hidden_dropout=0.0
            )
            for _ in range(num_layers)
        ])
        self.final_layer = Linear(self.token_dim, self.height*patch_size*patch_size)

    def forward(self, x):
        b = x.size(0)
        x = rearrange(x, "b (x z) -> b x z", x=self.new_lat*self.new_lon, z=int(self.latent_dim//(self.new_lat*self.new_lon)))
        x = self.layernorm1(self.fc1(x))
        # Reshape to (b, out_channels, latitude, longitude) i.e., (b, 33, 48, 96)
        x = x.view(b, self.out_channels, self.latitude, self.longitude)
        # Tokenize: merge non-overlapping patches using einops.rearrange.
        # From (b, c, H, W) -> (b, (H/patch_size * W/patch_size), c * patch_size * patch_size)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                      p1=self.patch_size, p2=self.patch_size)
        x = self.layernorm2(self.fc2(x))
        x = x + self.pos_embedding
        # Rearrange to [seq_len, batch, token_dim] for TE TransformerLayer.
        x = x.transpose(0, 1)
        # Pass through stacked transformer layers.
        if USE_FP8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                for layer in self.transformer_layers:
                    x = layer(x, attention_mask=None)
        else:
            for layer in self.transformer_layers:
                x = layer(x, attention_mask=None)
                
        # Rearrange back to (b, new_lat*new_lon, token_dim).
        x = x.transpose(0, 1)
        x = self.final_layer(x)
        # Unpatch tokens: rearrange from tokens to spatial layout.
        x = rearrange(x, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                      h=self.new_lat, p1=self.patch_size, p2=self.patch_size)
        x = rearrange(x, "bs h lat lon -> bs (h lat lon)")
        
        return x

