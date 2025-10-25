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
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import blosc2
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

from DBF.model.dbf import DeepBayesianFilterBlockDiag
from read_config import read_config


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

    def _normalise(self, descriptor: SampleDescriptor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_dir = descriptor.base_dir
        suffix = descriptor.index_suffix

        ps = (
            self._load_variable(base_dir, suffix, "ps_data").reshape(self.n_steps, 4608) - self.ps_mean
        ) / 1e4
        q = self._load_variable(base_dir, suffix, "q_data").reshape(self.n_steps, 36864) / 0.003
        t = (
            self._load_variable(base_dir, suffix, "t_data").reshape(self.n_steps, 36864) - self.t_mean
        ) / 20
        u = self._load_variable(base_dir, suffix, "u_data").reshape(self.n_steps, 36864) / 15
        v = self._load_variable(base_dir, suffix, "v_data").reshape(self.n_steps, 36864) / 15

        ps_out = torch.tensor(rearrange(ps, "t (ll) -> t 1 ll"))
        q_out = torch.tensor(rearrange(q, "t (h ll) -> t h ll", h=8))
        t_out = torch.tensor(rearrange(t, "t (h ll) -> t h ll", h=8))
        u_out = torch.tensor(rearrange(u, "t (h ll) -> t h ll", h=8))
        v_out = torch.tensor(rearrange(v, "t (h ll) -> t h ll", h=8))

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
            rearrange(u * 15, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 2:46:2, ::2]
        )
        v_t = torch.tensor(
            rearrange(v * 15, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 2:46:2, ::2]
        )
        t_t = torch.tensor(
            rearrange(t * 20, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 2:46:2, ::2]
        )
        q_t = torch.tensor(
            rearrange(q * 0.003, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 2:46:2, ::2]
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
            rearrange(ps * 1e4, "t (lat lon) -> t lat lon", lat=48, lon=96)[:, ps_lat_slice, lon_slice]
        )
        u_tensor = torch.tensor(
            rearrange(u * 15, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, uv_lat_slice, lon_slice]
        )
        v_tensor = torch.tensor(
            rearrange(v * 15, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, uv_lat_slice, lon_slice]
        )
        t_tensor = torch.tensor(
            rearrange(t * 20, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, uv_lat_slice, lon_slice]
        )
        q_tensor = torch.tensor(
            rearrange(q * 0.003, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, uv_lat_slice, lon_slice]
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

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        descriptor = self.samples[index]
        obs, state = self._normalise(descriptor)
        return obs, state


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
) -> DataLoader:
    dataset = LocalSpeedyDataset(
        data_root=data_root,
        mean_root=mean_root,
        n_steps=n_steps,
        obs_pattern=obs_pattern,
        index_json=index_json,
        limit=limit,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_model(config_path: Path, checkpoint: Path, device: torch.device) -> DeepBayesianFilterBlockDiag:
    config = read_config(str(config_path))
    latent_dim = int(config["model"]["latent_dim"])
    dropout = float(config["model"]["dropout"])
    arch = config["model"]["arch"]
    q_distribution = config["model"]["q_distribution"]
    model_seed = int(config["model"]["model_seed"])

    model = DeepBayesianFilterBlockDiag(
        latent_dim=latent_dim,
        simulation_dimensions=[8, 48, 96],
        model_seed=model_seed,
        arch=arch,
        dropout=dropout,
        q_distribution=q_distribution,
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_inference(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    config = read_config(str(args.config))
    n_steps = int(config["model"]["T"])
    batch_size = int(config["model"].get("test_batch_size", args.batch_size))
    obs_pattern = config["data"]["obs_setting"]

    dataloader = build_dataloader(
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

    total_loss = 0.0
    total_loss_kl = 0.0
    total_loss_integral = 0.0

    for batch_idx, (obs_data, target) in enumerate(dataloader):
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

        total_loss += float(loss)
        total_loss_kl += float(loss_kl)
        total_loss_integral += float(loss_integral)

        if save_dir is not None:
            torch.save(mu_pred_list.detach().cpu(), save_dir / f"mu_pred_batch{batch_idx:04d}.pt")
            torch.save(mu_filtered_list.detach().cpu(), save_dir / f"mu_filtered_batch{batch_idx:04d}.pt")
            torch.save(sigma_pred_list.detach().cpu(), save_dir / f"sigma_pred_batch{batch_idx:04d}.pt")
            torch.save(sigma_filtered_list.detach().cpu(), save_dir / f"sigma_filtered_batch{batch_idx:04d}.pt")

    num_batches = max(len(dataloader), 1)
    print("Inference complete")
    print(f"Total loss: {total_loss:.6f}")
    print(f"Total KL loss: {total_loss_kl:.6f}")
    print(f"Total integral loss: {total_loss_integral:.6f}")
    print(f"Average loss per batch: {total_loss / num_batches:.6f}")


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
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples processed (useful for smoke tests).",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_inference(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

