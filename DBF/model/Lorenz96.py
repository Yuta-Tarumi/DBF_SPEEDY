"""Light-weight encoder/decoder modules for the Lorenz-96 example."""
from __future__ import annotations

import math
from typing import Tuple, Union

import torch
from torch import nn

from .dbf import (
    compute_K_sigma_block_diag,
    kl_divergence_gaussians,
    sample_gaussian,
)
from einops import rearrange

class ResidualConvBlock(nn.Module):
    """Residual 1D convolutional block with skip connections and LayerNorm."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        padding_mode: str = "circular",
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )
        self.norm1 = nn.LayerNorm(40)#ChannelLayerNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.conv1(self.norm1(x)))
        return self.act(out) + residual


class Conv1DEncoder(nn.Module):
    """Map observations to the posterior natural parameters with residual blocks."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_channels: int = 64,
        num_blocks: int = 8,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(
                1,
                hidden_channels,
                kernel_size=5,
                padding=2,
                padding_mode="circular",
            ),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            ResidualConvBlock(hidden_channels, kernel_size=5) for _ in range(num_blocks)
        )
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_channels * 40, 2 * latent_dim),
        )
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, received {x.shape[-1]}"
            )
        x = x.unsqueeze(1)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
            #print(f"{x.shape=}")
        x = rearrange(x, "b t h -> b (t h)")
        out = self.head(x)
        #print(f"{out.shape=}")
        return out.squeeze(-1)


class Conv1DDecoder(nn.Module):
    """Decode latent states back to the physical space with residual refinement."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_channels: int = 64,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels * output_dim),
            nn.GELU(),
            nn.Linear(hidden_channels * output_dim, hidden_channels * output_dim),
        )
        self.blocks = nn.ModuleList(
            ResidualConvBlock(hidden_channels, kernel_size=5) for _ in range(num_blocks)
        )
        self.head = nn.Sequential(
            #ChannelLayerNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, latent_dim)
        out = self.input_proj(z)
        out = out.view(z.shape[0], self.hidden_channels, self.output_dim)
        for block in self.blocks:
            out = block(out)
        out = self.head(out)
        return out.squeeze(1)


class Lorenz96DBF(nn.Module):
    """A small wrapper around :class:`DeepBayesianFilterBlockDiag` for Lorenz-96."""

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        encoder: nn.Module,
        decoder: nn.Module,
        model_seed: int = 0,
        init_cov: float = 10.0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if latent_dim % 2 != 0:
            raise ValueError("latent_dim must be even for the block-diagonal structure")
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.encoder = encoder
        self.decoder = decoder
        self.device = torch.device(device)

        torch.manual_seed(model_seed)
        self.num_blocks = latent_dim // 2
        print(f"{self.num_blocks=}")
        self.lambdas = nn.Parameter(0.01 * torch.randn(latent_dim))
        self.log_Q = torch.tensor(-2.0, device="cuda:7")
        #self.log_R = torch.tensor(0.0, device="cuda:7")
        #self.log_Q = nn.Parameter(torch.tensor(0.0))
        self.log_R = nn.Parameter(torch.full((obs_dim,), 1.5))

        init_sigma = init_cov * torch.eye(2)
        self.register_buffer("init_mu", torch.zeros(self.num_blocks, 2))
        self.register_buffer("init_sigma", init_sigma.unsqueeze(0).repeat(self.num_blocks, 1, 1))

    def to(self, *args, **kwargs):  # type: ignore[override]
        module = super().to(*args, **kwargs)
        device = next(module.parameters()).device
        module.device = device
        return module

    def forward(
        self,
        obs_seq: torch.Tensor,
        target_seq: torch.Tensor,
        return_reconstruction: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        batch_size, T, _ = obs_seq.shape
        if target_seq.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Target dimension {target_seq.shape[-1]} does not match expected {self.obs_dim}"
            )
        device = obs_seq.device

        init_mu = self.init_mu.to(device).unsqueeze(0).expand(batch_size, self.num_blocks, 2)
        init_sigma = self.init_sigma.to(device).unsqueeze(0).expand(batch_size, self.num_blocks, 2, 2)

        mu_pred_list = torch.empty(batch_size, T, self.num_blocks, 2, device=device)
        mu_filtered_list = torch.empty_like(mu_pred_list)
        sigma_pred_list = torch.empty(batch_size, T, self.num_blocks, 2, 2, device=device)
        sigma_filtered_list = torch.empty_like(sigma_pred_list)

        mu_pred = init_mu
        sigma_pred = init_sigma
        mu_pred_list[:, 0] = mu_pred
        sigma_pred_list[:, 0] = sigma_pred

        A_block = compute_K_sigma_block_diag(self.lambdas, num_real=0, num_complex=self.num_blocks).to(device)
        Q_tensor = torch.exp(self.log_Q) * torch.eye(2, device=device)

        obs_flat = rearrange(obs_seq, "b t y -> (b t) y")
        enc_out = self.encoder(obs_flat)
        enc_out = rearrange(enc_out, "(b t) y -> b t y", t=T)#.view(batch_size, T, self.latent_dim * 2)
        f_enc_all = enc_out[:, :, : self.latent_dim].view(batch_size, T, self.num_blocks, 2)
        g_raw = enc_out[:, :, self.latent_dim :].view(batch_size, T, self.num_blocks, 2)
        maximum_g = 100
        g_enc_all = maximum_g*torch.tanh(g_raw**2/maximum_g)
        #print(f"{torch.max(g_enc_all)=}, {torch.median(g_enc_all)=}, {torch.min(g_enc_all)=}")
        gf_all = g_enc_all * f_enc_all

        for t in range(T):
            f_enc = f_enc_all[:, t]
            g_enc = g_enc_all[:, t]
            sigma_pred_inv, _ = torch.linalg.inv_ex(sigma_pred)
            sigma_filtered, _ = torch.linalg.inv_ex(sigma_pred_inv + torch.diag_embed(g_enc))
            sigma_filtered = 0.5 * (sigma_filtered + sigma_filtered.transpose(-2, -1))
            mu_filtered = torch.einsum("bzij,bzjk,bzk->bzi", sigma_filtered, sigma_pred_inv, mu_pred) + gf_all[:, t]

            mu_pred = torch.einsum("zxy,bzy->bzx", A_block, mu_filtered)
            sigma_pred = torch.einsum("zij,bzjk,zlk->bzil", A_block, sigma_filtered, A_block)
            sigma_pred = 0.5 * (sigma_pred + sigma_pred.transpose(-2, -1)) + Q_tensor

            mu_filtered_list[:, t] = mu_filtered
            sigma_filtered_list[:, t] = sigma_filtered
            if t + 1 < T:
                mu_pred_list[:, t + 1] = mu_pred
                sigma_pred_list[:, t + 1] = sigma_pred

        return self._compute_loss(
            mu_filtered_list,
            sigma_filtered_list,
            mu_pred_list,
            sigma_pred_list,
            target_seq,
            return_reconstruction=return_reconstruction,
        )

    def _compute_loss(
        self,
        mu_filtered_list: torch.Tensor,
        sigma_filtered_list: torch.Tensor,
        mu_pred_list: torch.Tensor,
        sigma_pred_list: torch.Tensor,
        target: torch.Tensor,
        return_reconstruction: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        batch_size = target.shape[0]
        loss_kl = torch.sum(
            kl_divergence_gaussians(
                mu_filtered_list,
                sigma_filtered_list,
                mu_pred_list,
                sigma_pred_list,
            )
        ) / batch_size

        samples = sample_gaussian(mu_filtered_list, sigma_filtered_list, num_samples=1)[0]
        samples_flat = samples.reshape(-1, self.latent_dim)
        recon_flat = self.decoder(samples_flat)
        recon = recon_flat.view(target.shape)
        #print(f"{torch.max(recon)=}, {torch.min(recon)=}")
        log_std = self.log_R.view(1, 1, -1)
        var = torch.exp(2 * log_std)
        diff = target - recon
        log_prob = -0.5 * (
            math.log(2 * math.pi) + 2 * log_std + diff.pow(2) / var
        )
        loss_integral = -torch.sum(log_prob) / batch_size
        total_loss = loss_kl + loss_integral
        if return_reconstruction:
            return total_loss, loss_kl, loss_integral, recon
        return total_loss, loss_kl, loss_integral
