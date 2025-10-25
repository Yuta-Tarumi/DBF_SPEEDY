"""Run inference on SPEEDY test data using a trained Deep Bayesian Filter.

This script mirrors the evaluation loop that lives in ``train_script.py``
but makes it easy to run the model on the public ``test_data`` bundle.  It
assumes that the archive has already been extracted so that the directory
layout looks like::

    data_mean/
        ps_mean.npy
        t_mean.npy
    test_data/
        ps_data/ps_data_index10000036
        q_data/q_data_index10000036
        ...

The script will normalise every variable in the same way as the training
pipeline, construct the observation tensors for the requested observation
pattern, and finally execute the model to report the test losses.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import blosc2
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
os.environ.setdefault("NVTE_FUSED_ATTN", "0")

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

from DBF.model.dbf import DeepBayesianFilterBlockDiag
from read_config import read_config


LATITUDE = 48
LONGITUDE = 96
LEVELS = 8
STATE_CHANNELS = 33


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def observation_slices(pattern: str) -> Tuple[slice | None, slice | None, slice | None]:
    """Return latitude/longitude slices that define the observation grid."""

    if pattern == "sparsest":
        return slice(7, LATITUDE, 16), slice(7, LATITUDE, 16), slice(0, LONGITUDE, 16)
    if pattern == "sparser":
        return slice(4, LATITUDE, 8), slice(4, LATITUDE, 8), slice(0, LONGITUDE, 8)
    if pattern == "sparse":
        return slice(1, LATITUDE - 2, 4), slice(1, LATITUDE - 2, 4), slice(0, LONGITUDE, 4)
    if pattern == "dense":
        return None, slice(2, LATITUDE - 2, 2), slice(0, LONGITUDE, 2)
    return None, None, None


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def _load_blosc2(path: Path) -> np.ndarray:
    """Load one Blosc2-compressed array from ``path``."""

    with path.open("rb") as fh:
        return blosc2.unpack_array2(fh.read())


@dataclass(frozen=True)
class SampleDescriptor:
    base_dir: Path
    stem: str

    @property
    def index_suffix(self) -> str:
        """Return the numeric suffix (``"10000036"``) for the sample."""

        return self.stem.split("ps_data_index", 1)[1]


class LocalSpeedyDataset(Dataset):
    """Dataset that mirrors :class:`DBF.dataset.speedy.SPEEDY` for local data.

    Parameters
    ----------
    data_root:
        Directory that contains the ``*_data`` folders (``ps_data`` / ``q_data``
        / ``t_data`` / ``u_data`` / ``v_data``).
    mean_root:
        Directory that holds ``ps_mean.npy`` and ``t_mean.npy``.
    n_steps:
        Number of temporal steps contained in each file.
    obs_pattern:
        Observation pattern – must match what the model was trained with.
    index_json:
        Optional JSON file (``index_test*.json``) that lists the filenames to
        load.  When omitted the dataset will read every file present in
        ``ps_data``.
    limit:
        Optional cap on the number of samples.
    """

    def __init__(
        self,
        data_root: Path,
        mean_root: Path,
        n_steps: int,
        obs_pattern: str,
        index_json: Path | None = None,
        limit: int | None = None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.mean_root = mean_root
        self.n_steps = n_steps
        self.obs_pattern = obs_pattern

        ps_dir = self.data_root / "ps_data"
        if ps_dir.exists():
            self.default_base_dir = self.data_root
        else:
            candidates = sorted(
                (path.parent for path in self.data_root.glob("*/ps_data")),
                key=lambda p: str(p),
            )
            if not candidates:
                raise FileNotFoundError(
                    f"Could not find a 'ps_data' directory under {self.data_root}."
                    " Please point --data-root to the extracted test archive."
                )
            self.default_base_dir = candidates[0]

        self.ps_mean = np.load(self.mean_root / "ps_mean.npy").astype(np.float32)
        self.t_mean = np.load(self.mean_root / "t_mean.npy").astype(np.float32)

        if index_json is not None:
            with index_json.open() as fh:
                mapping = json.load(fh)
            samples: List[SampleDescriptor] = []
            for directory, stems in mapping.items():
                base_dir = self.data_root / directory
                if not base_dir.exists():
                    base_dir = self.default_base_dir
                for stem in stems:
                    samples.append(SampleDescriptor(base_dir, stem))
        else:
            ps_dir = self.default_base_dir / "ps_data"
            stems = sorted(p.stem for p in ps_dir.glob("ps_data_index*"))
            samples = [SampleDescriptor(self.default_base_dir, stem) for stem in stems]

        if limit is not None:
            samples = samples[:limit]

        self.samples = samples

    # ------------------------------------------------------------------
    # Helper routines
    # ------------------------------------------------------------------

    def _load_variable(self, base_dir: Path, suffix: str, subdir: str) -> np.ndarray:
        path = base_dir / subdir / f"{subdir}_index{suffix}"
        return _load_blosc2(path)

    def load_raw_state(self, descriptor: SampleDescriptor) -> Dict[str, np.ndarray]:
        """Return the unnormalised SPEEDY state for ``descriptor``."""

        base_dir = descriptor.base_dir
        suffix = descriptor.index_suffix

        ps = self._load_variable(base_dir, suffix, "ps_data").reshape(self.n_steps, LATITUDE, LONGITUDE)
        q = self._load_variable(base_dir, suffix, "q_data").reshape(self.n_steps, LEVELS, LATITUDE, LONGITUDE)
        t = self._load_variable(base_dir, suffix, "t_data").reshape(self.n_steps, LEVELS, LATITUDE, LONGITUDE)
        u = self._load_variable(base_dir, suffix, "u_data").reshape(self.n_steps, LEVELS, LATITUDE, LONGITUDE)
        v = self._load_variable(base_dir, suffix, "v_data").reshape(self.n_steps, LEVELS, LATITUDE, LONGITUDE)

        return {
            "ps": ps.astype(np.float32),
            "q": q.astype(np.float32),
            "t": t.astype(np.float32),
            "u": u.astype(np.float32),
            "v": v.astype(np.float32),
        }

    def _normalise(self, descriptor: SampleDescriptor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.load_raw_state(descriptor)

        ps = ((raw["ps"].reshape(self.n_steps, -1) - self.ps_mean) / 1e4).astype(np.float32)
        q = (raw["q"].reshape(self.n_steps, -1) / 0.003).astype(np.float32)
        t = ((raw["t"].reshape(self.n_steps, -1) - self.t_mean) / 20).astype(np.float32)
        u = (raw["u"].reshape(self.n_steps, -1) / 15).astype(np.float32)
        v = (raw["v"].reshape(self.n_steps, -1) / 15).astype(np.float32)

        ps_out = torch.tensor(rearrange(ps, "t (ll) -> t 1 ll"))
        q_out = torch.tensor(rearrange(q, "t (h ll) -> t h ll", h=LEVELS))
        t_out = torch.tensor(rearrange(t, "t (h ll) -> t h ll", h=LEVELS))
        u_out = torch.tensor(rearrange(u, "t (h ll) -> t h ll", h=LEVELS))
        v_out = torch.tensor(rearrange(v, "t (h ll) -> t h ll", h=LEVELS))

        state = torch.cat([ps_out, u_out, v_out, t_out, q_out], dim=1)

        if self.obs_pattern == "Koopman":
            obs = rearrange(np.array([u, v, q, t]), "a b c -> b (a c)")
            obs = torch.tensor(obs)
        elif self.obs_pattern == "dense":
            obs = self._dense_observation(u, v, t, q)
        elif self.obs_pattern == "sparse":
            obs = self._sparse_observation(
                ps,
                u,
                v,
                t,
                q,
                ps_lat_slice=slice(1, 46, 4),
                uv_lat_slice=slice(1, 46, 4),
                lon_slice=slice(None, None, 4),
            )
        elif self.obs_pattern == "sparser":
            obs = self._sparse_observation(
                ps,
                u,
                v,
                t,
                q,
                ps_lat_slice=slice(4, 48, 8),
                uv_lat_slice=slice(4, 48, 8),
                lon_slice=slice(None, None, 8),
            )
        elif self.obs_pattern == "sparsest":
            obs = self._sparse_observation(
                ps,
                u,
                v,
                t,
                q,
                ps_lat_slice=slice(7, 48, 16),
                uv_lat_slice=slice(7, 48, 16),
                lon_slice=slice(None, None, 16),
            )
        else:
            raise ValueError(f"Unknown observation pattern: {self.obs_pattern}")

        return obs, state

    def _dense_observation(self, u, v, t, q) -> torch.Tensor:
        u_t = torch.tensor(
            rearrange(u * 15, "t (h lat lon) -> t h lat lon", h=LEVELS, lat=LATITUDE, lon=LONGITUDE)[:, :, 2:46:2, ::2]
        )
        v_t = torch.tensor(
            rearrange(v * 15, "t (h lat lon) -> t h lat lon", h=LEVELS, lat=LATITUDE, lon=LONGITUDE)[:, :, 2:46:2, ::2]
        )
        t_t = torch.tensor(
            rearrange(t * 20, "t (h lat lon) -> t h lat lon", h=LEVELS, lat=LATITUDE, lon=LONGITUDE)[:, :, 2:46:2, ::2]
        )
        q_t = torch.tensor(
            rearrange(q * 0.003, "t (h lat lon) -> t h lat lon", h=LEVELS, lat=LATITUDE, lon=LONGITUDE)[:, :, 2:46:2, ::2]
        )
        noise = lambda x, scale: (x + scale * torch.randn_like(x))
        u_obs = rearrange(noise(u_t, 1.0), "t h lat lon -> t (h lat lon)") / 15
        v_obs = rearrange(noise(v_t, 1.0), "t h lat lon -> t (h lat lon)") / 15
        t_obs = rearrange(noise(t_t, 1.0), "t h lat lon -> t (h lat lon)") / 20
        q_obs = rearrange(noise(q_t, 1e-4), "t h lat lon -> t (h lat lon)") / 0.003
        stacked = torch.stack([u_obs, v_obs, t_obs, q_obs], dim=1)
        return rearrange(stacked, "t a c -> t (a c)")

    def _sparse_observation(
        self,
        ps,
        u,
        v,
        t,
        q,
        *,
        ps_lat_slice: slice,
        uv_lat_slice: slice,
        lon_slice: slice,
    ) -> torch.Tensor:
        ps_tensor = torch.tensor(
            rearrange(ps * 1e4, "t (lat lon) -> t lat lon", lat=LATITUDE, lon=LONGITUDE)[:, ps_lat_slice, lon_slice]
        )
        u_tensor = torch.tensor(
            rearrange(u * 15, "t (h lat lon) -> t h lat lon", h=LEVELS, lat=LATITUDE, lon=LONGITUDE)[:, :, uv_lat_slice, lon_slice]
        )
        v_tensor = torch.tensor(
            rearrange(v * 15, "t (h lat lon) -> t h lat lon", h=LEVELS, lat=LATITUDE, lon=LONGITUDE)[:, :, uv_lat_slice, lon_slice]
        )
        t_tensor = torch.tensor(
            rearrange(t * 20, "t (h lat lon) -> t h lat lon", h=LEVELS, lat=LATITUDE, lon=LONGITUDE)[:, :, uv_lat_slice, lon_slice]
        )
        q_tensor = torch.tensor(
            rearrange(q * 0.003, "t (h lat lon) -> t h lat lon", h=LEVELS, lat=LATITUDE, lon=LONGITUDE)[:, :, uv_lat_slice, lon_slice]
        )

        noise = torch.randn_like
        ps_obs = rearrange((ps_tensor + 100 * noise(ps_tensor)) / 1e4, "t lat lon -> t 1 (lat lon)")
        u_obs = rearrange((u_tensor + noise(u_tensor)) / 15, "t h lat lon -> t h (lat lon)")
        v_obs = rearrange((v_tensor + noise(v_tensor)) / 15, "t h lat lon -> t h (lat lon)")
        t_obs = rearrange((t_tensor + noise(t_tensor)) / 20, "t h lat lon -> t h (lat lon)")
        q_obs = rearrange((q_tensor + 1e-4 * noise(q_tensor)) / 0.003, "t h lat lon -> t h (lat lon)")
        obs = torch.cat([ps_obs, u_obs, v_obs, t_obs, q_obs[:, :4, :]], dim=1)
        return obs

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int: 
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        descriptor = self.samples[index]
        obs, state = self._normalise(descriptor)
        return obs, state, index


# ---------------------------------------------------------------------------
# Inference driver
# ---------------------------------------------------------------------------


def build_dataloader(
    data_root: Path,
    mean_root: Path,
    n_steps: int,
    obs_pattern: str,
    index_json: Path | None,
    batch_size: int,
    num_workers: int,
    limit: int | None,
) -> Tuple[LocalSpeedyDataset, DataLoader]:
    dataset = LocalSpeedyDataset(
        data_root=data_root,
        mean_root=mean_root,
        n_steps=n_steps,
        obs_pattern=obs_pattern,
        index_json=index_json,
        limit=limit,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return dataset, dataloader


def load_model(config_path: Path, checkpoint: Path, device: torch.device) -> DeepBayesianFilterBlockDiag:
    config = read_config(str(config_path))
    latent_dim = int(config["model"]["latent_dim"])
    dropout = float(config["model"]["dropout"])
    arch = config["model"]["arch"]
    q_distribution = config["model"]["q_distribution"]
    model_seed = int(config["model"]["model_seed"])

    model = DeepBayesianFilterBlockDiag(
        latent_dim=latent_dim,
        simulation_dimensions=[LEVELS, LATITUDE, LONGITUDE],
        model_seed=model_seed,
        arch=arch,
        dropout=dropout,
        q_distribution=q_distribution,
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    return model


def decode_filtered_latents(
    model: DeepBayesianFilterBlockDiag,
    mu_filtered_list: torch.Tensor,
    *,
    device: torch.device | None = None,
    sample_indices: slice | Sequence[int] | int | None = None,
) -> torch.Tensor:
    """Decode latent posterior means into the normalised state space.

    Parameters
    ----------
    model:
        The trained DBF model whose decoder should be applied.
    mu_filtered_list:
        Tensor of shape ``(batch, steps, blocks, 2)`` containing the posterior
        means produced by the filter.
    device:
        Device that should run the decoder.  Defaults to the decoder module's
        current device.
    """

    if device is None:
        device = next(model.decoder.parameters()).device

    if sample_indices is None:
        latents_source = mu_filtered_list
    elif isinstance(sample_indices, int):
        latents_source = mu_filtered_list[sample_indices : sample_indices + 1]
    else:
        latents_source = mu_filtered_list[sample_indices]

    batch_size, seq_len, num_blocks, two = latents_source.shape
    latents = rearrange(latents_source, "b t blocks two -> (b t) (blocks two)").contiguous()
    if device.type != latents.device.type:
        latents = latents.to(device, non_blocking=True)

    decoded_flat = model.decoder(latents).detach().to("cpu")
    del latents
    decoded = rearrange(
        decoded_flat,
        "(b t) (c h w) -> b t c h w",
        b=batch_size,
        t=seq_len,
        c=STATE_CHANNELS,
        h=LATITUDE,
        w=LONGITUDE,
    )
    del decoded_flat
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return decoded


def denormalise_decoded(
    decoded: torch.Tensor, ps_mean: np.ndarray, t_mean: np.ndarray
) -> Dict[str, np.ndarray]:
    """Convert the decoded normalised state back to physical units."""

    decoded_np = decoded.detach().cpu().numpy()
    ps_mean_grid = ps_mean.reshape(LATITUDE, LONGITUDE)
    t_mean_grid = t_mean.reshape(LEVELS, LATITUDE, LONGITUDE)

    ps = decoded_np[:, :, 0] * 1e4 + ps_mean_grid[None, None, :, :]
    u = decoded_np[:, :, 1 : 1 + LEVELS] * 15
    v = decoded_np[:, :, 1 + LEVELS : 1 + 2 * LEVELS] * 15
    t = decoded_np[:, :, 1 + 2 * LEVELS : 1 + 3 * LEVELS] * 20 + t_mean_grid[None, None, :, :, :]
    q = decoded_np[:, :, 1 + 3 * LEVELS : 1 + 4 * LEVELS] * 0.003

    return {"ps": ps, "u": u, "v": v, "t": t, "q": q}


def prepare_observation_fields(
    obs_sample: torch.Tensor,
    dataset: LocalSpeedyDataset,
    obs_pattern: str,
) -> Dict[str, np.ndarray]:
    """Project sparse observations back onto the physical SPEEDY grid."""

    ps_slice, uv_slice, lon_slice = observation_slices(obs_pattern)
    obs_np = obs_sample.detach().cpu().numpy()
    time_steps, channel_count, _ = obs_np.shape

    lon_indices = np.arange(LONGITUDE)[lon_slice] if lon_slice is not None else np.array([], dtype=int)
    ps_lat_indices = np.arange(LATITUDE)[ps_slice] if ps_slice is not None else np.array([], dtype=int)
    uv_lat_indices = np.arange(LATITUDE)[uv_slice] if uv_slice is not None else np.array([], dtype=int)

    ps_grid = np.full((time_steps, LATITUDE, LONGITUDE), np.nan, dtype=np.float32)
    u_grid = np.full((time_steps, LEVELS, LATITUDE, LONGITUDE), np.nan, dtype=np.float32)
    v_grid = np.full((time_steps, LEVELS, LATITUDE, LONGITUDE), np.nan, dtype=np.float32)
    t_grid = np.full((time_steps, LEVELS, LATITUDE, LONGITUDE), np.nan, dtype=np.float32)
    q_grid = np.full((time_steps, LEVELS, LATITUDE, LONGITUDE), np.nan, dtype=np.float32)

    offset = 0

    if ps_lat_indices.size and lon_indices.size:
        ps_obs = obs_np[:, offset : offset + 1, :]
        offset += 1
        ps_vals = ps_obs[:, 0].reshape(time_steps, ps_lat_indices.size, lon_indices.size)
        ps_mean_grid = dataset.ps_mean.reshape(LATITUDE, LONGITUDE)
        for i, lat in enumerate(ps_lat_indices):
            for j, lon in enumerate(lon_indices):
                ps_grid[:, lat, lon] = ps_vals[:, i, j] * 1e4 + ps_mean_grid[lat, lon]

    if uv_lat_indices.size and lon_indices.size and channel_count > offset:
        u_obs = obs_np[:, offset : offset + LEVELS, :]
        offset += LEVELS
        v_obs = obs_np[:, offset : offset + LEVELS, :]
        offset += LEVELS
        t_obs = obs_np[:, offset : offset + LEVELS, :]
        offset += LEVELS

        u_vals = u_obs.reshape(time_steps, LEVELS, uv_lat_indices.size, lon_indices.size) * 15
        v_vals = v_obs.reshape(time_steps, LEVELS, uv_lat_indices.size, lon_indices.size) * 15
        t_vals = t_obs.reshape(time_steps, LEVELS, uv_lat_indices.size, lon_indices.size) * 20
        t_mean_grid = dataset.t_mean.reshape(LEVELS, LATITUDE, LONGITUDE)

        for level in range(LEVELS):
            for i, lat in enumerate(uv_lat_indices):
                for j, lon in enumerate(lon_indices):
                    u_grid[:, level, lat, lon] = u_vals[:, level, i, j]
                    v_grid[:, level, lat, lon] = v_vals[:, level, i, j]
                    t_grid[:, level, lat, lon] = t_vals[:, level, i, j] + t_mean_grid[level, lat, lon]

    remaining_channels = channel_count - offset
    if remaining_channels > 0 and uv_lat_indices.size and lon_indices.size:
        q_levels_observed = min(remaining_channels, LEVELS)
        q_obs = obs_np[:, offset : offset + q_levels_observed, :]
        q_vals = q_obs.reshape(
            time_steps, q_levels_observed, uv_lat_indices.size, lon_indices.size
        ) * 0.003
        for level in range(q_levels_observed):
            for i, lat in enumerate(uv_lat_indices):
                for j, lon in enumerate(lon_indices):
                    q_grid[:, level, lat, lon] = q_vals[:, level, i, j]

    return {
        "ps": ps_grid[None, ...],
        "u": u_grid[None, ...],
        "v": v_grid[None, ...],
        "t": t_grid[None, ...],
        "q": q_grid[None, ...],
    }


def _expand_time_level(array: np.ndarray) -> np.ndarray:
    """Ensure arrays include a batch dimension for plotting helpers."""

    if array.ndim == 4:  # (T, L, H, W)
        return array[None, ...]
    if array.ndim == 3:  # (T, H, W)
        return array[None, ...]
    raise ValueError(f"Unexpected array shape {array.shape}")


def plot_variable_maps(
    filtered_fields: Dict[str, np.ndarray],
    target_fields: Dict[str, np.ndarray],
    observation_fields: Dict[str, np.ndarray],
    output_dir: Path,
    *,
    batch_idx: int,
    sample_idx: int,
    level_idx: int = 3,
) -> None:
    """Render level-specific comparison maps with observed samples highlighted.

    The routine exports 2x2 panel figures showing the filtered state, target
    state, their difference, and the available observations for the requested
    level (or surface pressure).  Observed panels mask unobserved grid points so
    only measurement locations remain coloured.
    """

    import matplotlib.pyplot as plt
    from matplotlib import cm
    import cartopy.crs as ccrs
    from cartopy.feature import OCEAN

    lon_edges = np.linspace(-180.0, 180.0, LONGITUDE + 1)
    lat_edges = np.linspace(-90.0, 90.0, LATITUDE + 1)

    base_cmap = cm.get_cmap("viridis").copy()
    base_cmap.set_bad("white")
    diff_cmap = cm.get_cmap("coolwarm").copy()
    diff_cmap.set_bad("white")

    def roll_longitude(array: np.ndarray) -> np.ndarray:
        """Rotate longitude so the map spans ``[-180, 180]`` consistently."""

        return np.roll(array, shift=LONGITUDE // 2, axis=-1)

    field_limits = {
        "u": {"field": (-15.0, 15.0), "diff": (-6.0, 6.0)},
        "v": {"field": (-15.0, 15.0), "diff": (-6.0, 6.0)},
        "t": {"field": (220.0, 270.0), "diff": (-5.0, 5.0)},
        "q": {"field": (0.0, 4e-3), "diff": (-8e-4, 8e-4)},
    }

    for var_name in ["ps", "u", "v", "t", "q"]:
        filtered = filtered_fields[var_name][0]
        target = target_fields[var_name][0]
        observed = observation_fields[var_name][0]

        if var_name != "ps":
            filtered = filtered[:, level_idx]
            target = target[:, level_idx]
            observed = observed[:, level_idx]
            suffix = f"level{level_idx}"
        else:
            suffix = "surface"

        var_dir = output_dir / f"{var_name}_{suffix}_maps"
        var_dir.mkdir(parents=True, exist_ok=True)

        time_steps = filtered.shape[0]
        for t_idx in range(time_steps):
            fig, axes = plt.subplots(
                2,
                2,
                figsize=(12, 8),
                subplot_kw={"projection": ccrs.PlateCarree()},
            )
            axes = axes.ravel()

            filtered_level = roll_longitude(filtered[t_idx])
            target_level = roll_longitude(target[t_idx])
            observed_level = np.ma.masked_invalid(roll_longitude(observed[t_idx]))

            diff = filtered_level - target_level

            limits = field_limits.get(var_name)
            if limits is not None:
                vmin, vmax = limits["field"]
                diff_vmin, diff_vmax = limits["diff"]
            else:
                vmin = float(np.nanmin([filtered_level.min(), target_level.min()]))
                vmax = float(np.nanmax([filtered_level.max(), target_level.max()]))
                diff_abs = (
                    float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 0.0
                )
                diff_vmin, diff_vmax = -diff_abs, diff_abs

            ims = []
            ims.append(
                axes[0].pcolormesh(
                    lon_edges,
                    lat_edges,
                    filtered_level,
                    cmap=base_cmap,
                    vmin=vmin,
                    vmax=vmax,
                    shading="auto",
                    transform=ccrs.PlateCarree(),
                )
            )
            axes[0].set_title("Filtered")

            ims.append(
                axes[1].pcolormesh(
                    lon_edges,
                    lat_edges,
                    target_level,
                    cmap=base_cmap,
                    vmin=vmin,
                    vmax=vmax,
                    shading="auto",
                    transform=ccrs.PlateCarree(),
                )
            )
            axes[1].set_title("Target")

            ims.append(
                axes[2].pcolormesh(
                    lon_edges,
                    lat_edges,
                    diff,
                    cmap=diff_cmap,
                    vmin=diff_vmin,
                    vmax=diff_vmax,
                    shading="auto",
                    transform=ccrs.PlateCarree(),
                )
            )
            axes[2].set_title("Filtered - Target")

            ims.append(
                axes[3].pcolormesh(
                    lon_edges,
                    lat_edges,
                    observed_level,
                    cmap=base_cmap,
                    vmin=vmin,
                    vmax=vmax,
                    shading="auto",
                    transform=ccrs.PlateCarree(),
                )
            )
            axes[3].set_title("Observed")

            for ax, im in zip(axes, ims):
                ax.set_global()
                ax.add_feature(OCEAN, facecolor="lightblue", alpha=0.3, zorder=0)
                ax.coastlines(linewidth=0.5)
                fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

            fig.suptitle(
                f"{var_name} {suffix} | batch {batch_idx} sample {sample_idx} | time {t_idx}"
            )
            fig.tight_layout(rect=[0, 0.03, 1, 0.94])
            output_path = (
                var_dir
                / f"{var_name}_{suffix}_batch{batch_idx:04d}_sample{sample_idx:04d}_time{t_idx:02d}.png"
            )
            fig.savefig(output_path, dpi=150)
            plt.close(fig)


def save_filtered_field_sample(
    filtered_fields: Dict[str, np.ndarray],
    output_dir: Path,
    batch_idx: int,
    sample_idx: int,
) -> None:
    """Persist decoded fields for one sample to ``output_dir``."""

    for var_name, array in filtered_fields.items():
        tensor = torch.from_numpy(array[0])
        output_path = (
            output_dir
            / f"filtered_{var_name}_batch{batch_idx:04d}_sample{sample_idx:04d}.pt"
        )
        torch.save(tensor, output_path)


def load_target_fields(
    dataset: LocalSpeedyDataset, sample_idx: int
) -> Dict[str, np.ndarray]:
    """Load and reshape ground-truth SPEEDY fields for comparison plots."""

    descriptor = dataset.samples[sample_idx]
    raw = dataset.load_raw_state(descriptor)

    ps = _expand_time_level(raw["ps"].astype(np.float32))
    u = _expand_time_level(raw["u"].astype(np.float32))
    v = _expand_time_level(raw["v"].astype(np.float32))
    t = _expand_time_level(raw["t"].astype(np.float32))
    q = _expand_time_level(raw["q"].astype(np.float32))

    return {"ps": ps, "u": u, "v": v, "t": t, "q": q}


def run_inference(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    config = read_config(str(args.config))
    n_steps = int(config["model"]["T"])
    batch_size = 1
    obs_pattern = config["data"]["obs_setting"]

    dataset, dataloader = build_dataloader(
        data_root=args.data_root,
        mean_root=args.mean_root,
        n_steps=n_steps,
        obs_pattern=obs_pattern,
        index_json=args.index_json,
        batch_size=batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
    )

    model = load_model(args.config, args.checkpoint, device)

    save_dir = args.save_dir
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    map_dir = args.rmse_dir
    export_maps = map_dir is not None
    if export_maps:
        map_dir.mkdir(parents=True, exist_ok=True)

    try:
        obs_data, target, sample_indices = next(iter(dataloader))
    except StopIteration:
        print("No samples available for inference.")
        return

    if isinstance(sample_indices, torch.Tensor):
        batch_indices = sample_indices.tolist()
    else:
        batch_indices = list(sample_indices)

    batch_idx = 0

    obs_sample_cpu = obs_data[0].detach().clone()

    obs_data = obs_data.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    obs_data = rearrange(obs_data, "bs t h latlon -> bs t (h latlon)")
    target = rearrange(target, "bs t h latlon -> bs t (h latlon)")

    with torch.no_grad():
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast_ctx:
            (
                loss,
                loss_kl,
                loss_integral,
                mu_pred_list,
                mu_filtered_list,
                sigma_pred_list,
                sigma_filtered_list,
                *_,
            ) = model(obs_seq=obs_data, target_seq=target)

    total_loss = float(loss)
    total_loss_kl = float(loss_kl)
    total_loss_integral = float(loss_integral)

    mu_pred_list_cpu = mu_pred_list.detach().cpu()
    mu_filtered_list_cpu = mu_filtered_list.detach().cpu()
    sigma_pred_list_cpu = sigma_pred_list.detach().cpu()
    sigma_filtered_list_cpu = sigma_filtered_list.detach().cpu()

    del mu_pred_list
    del mu_filtered_list
    del sigma_pred_list
    del sigma_filtered_list
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    if save_dir is not None:
        torch.save(mu_pred_list_cpu, save_dir / f"mu_pred_batch{batch_idx:04d}.pt")
        torch.save(mu_filtered_list_cpu, save_dir / f"mu_filtered_batch{batch_idx:04d}.pt")
        torch.save(sigma_pred_list_cpu, save_dir / f"sigma_pred_batch{batch_idx:04d}.pt")
        torch.save(
            sigma_filtered_list_cpu,
            save_dir / f"sigma_filtered_batch{batch_idx:04d}.pt",
        )

    if export_maps and batch_indices:
        observation_fields = prepare_observation_fields(
            obs_sample_cpu,
            dataset,
            obs_pattern,
        )
        decoded_first = decode_filtered_latents(
            model,
            mu_filtered_list_cpu,
            device=device,
            sample_indices=0,
        )
        filtered_first = denormalise_decoded(
            decoded_first,
            dataset.ps_mean,
            dataset.t_mean,
        )
        target_fields = load_target_fields(dataset, batch_indices[0])
        save_filtered_field_sample(
            filtered_first,
            map_dir,
            batch_idx,
            batch_indices[0],
        )
        plot_variable_maps(
            filtered_first,
            target_fields,
            observation_fields,
            map_dir,
            batch_idx=batch_idx,
            sample_idx=batch_indices[0],
            level_idx=3,
        )

        del filtered_first
        del decoded_first

    del mu_pred_list_cpu
    del mu_filtered_list_cpu
    del sigma_pred_list_cpu
    del sigma_filtered_list_cpu
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Inference complete for first batch sample")
    print(f"Total loss: {total_loss:.6f}")
    print(f"Total KL loss: {total_loss_kl:.6f}")
    print(f"Total integral loss: {total_loss_integral:.6f}")

    if export_maps:
        print(f"Saved filtered figures to {map_dir}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/decoder_transformer_dim2048_sparsest_Gaussian.yaml"),
        help="Path to the config file used during training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("trained_weight/model_dim2048.pt"),
        help="Model checkpoint to load.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("test_data"),
        help="Directory that contains the extracted test data folders.",
    )
    parser.add_argument(
        "--mean-root",
        type=Path,
        default=Path("data_mean"),
        help="Directory that stores ps_mean.npy and t_mean.npy.",
    )
    parser.add_argument(
        "--index-json",
        type=Path,
        default=None,
        help="Optional index JSON (e.g. index_test1.json) that lists the sample order.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for inference when the config does not define test_batch_size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for data loading.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (e.g. 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to store per-batch predictions and covariances.",
    )
    parser.add_argument(
        "--rmse-dir",
        type=Path,
        default=None,
        help="Directory to store level-3 comparison maps and filtered tensors for the first sample.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples processed (useful for smoke tests).",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_inference(args)


if __name__ == "__main__": 
    main()
