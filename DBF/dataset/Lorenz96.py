"""Lorenz-96 dataset that generates trajectories on the fly."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def _lorenz96_rhs(x: np.ndarray, forcing: float) -> np.ndarray:
    """Compute time derivative of the Lorenz-96 system."""
    xp1 = np.roll(x, -1)
    xm1 = np.roll(x, 1)
    xm2 = np.roll(x, 2)
    return (xp1 - xm2) * xm1 - x + forcing


def _rk4_step(x: np.ndarray, dt: float, forcing: float) -> np.ndarray:
    """Advance the Lorenz-96 system by one Runge--Kutta 4 step."""
    k1 = _lorenz96_rhs(x, forcing)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, forcing)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, forcing)
    k4 = _lorenz96_rhs(x + dt * k3, forcing)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@dataclass
class Lorenz96Config:
    """Configuration options for :class:`Lorenz96Dataset`."""

    state_dim: int = 40
    forcing: float = 8.0
    dt: float = 0.03
    integration_steps: int = 1
    seq_length: int = 50
    burn_in: int = 100
    obs_noise_std: float = 1.0
    obs_indices: Optional[Sequence[int]] = None
    initial_mean: float = 0.0
    initial_std: float = 1.0
    seed: int = 0


class Lorenz96Dataset(Dataset):
    """Torch dataset that synthesises Lorenz-96 trajectories on demand."""

    def __init__(self, config: Lorenz96Config, num_samples: int) -> None:
        super().__init__()
        self.config = config
        self.num_samples = num_samples
        self._obs_indices = (
            np.asarray(config.obs_indices, dtype=np.int64)
            if config.obs_indices is not None
            else np.arange(config.state_dim, dtype=np.int64)
        )
        if self._obs_indices.ndim != 1:
            raise ValueError("obs_indices must be a 1-D sequence of integers")
        if np.any((self._obs_indices < 0) | (self._obs_indices >= config.state_dim)):
            raise ValueError("obs_indices contains indices outside of [0, state_dim)")

    def __len__(self) -> int:
        return self.num_samples

    def _integrate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        x = rng.normal(cfg.initial_mean, cfg.initial_std, size=cfg.state_dim)
        dt = cfg.dt / cfg.integration_steps
        for _ in range(cfg.burn_in * cfg.integration_steps):
            x = _rk4_step(x, dt, cfg.forcing)
        states = np.empty((cfg.seq_length, cfg.state_dim), dtype=np.float32)
        for t in range(cfg.seq_length):
            for _ in range(cfg.integration_steps):
                x = _rk4_step(x, dt, cfg.forcing)
            states[t] = x.astype(np.float32, copy=True)
        observations = np.minimum(np.power(states[:, self._obs_indices], 4, dtype=np.float32), 10.0)
        #print(f"{observations.shape=}")
        if cfg.obs_noise_std > 0:
            observations = observations + rng.normal(
                0.0, cfg.obs_noise_std, size=observations.shape
            ).astype(np.float32)
        return observations.astype(np.float32), states

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.config.seed + index)
        obs, states = self._integrate(rng)
        obs_tensor = torch.from_numpy(obs)
        states_tensor = torch.from_numpy(states)
        return obs_tensor, states_tensor
