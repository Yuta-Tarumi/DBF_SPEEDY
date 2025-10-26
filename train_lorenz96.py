"""Training script for the Lorenz-96 Deep Bayesian Filter example."""
from __future__ import annotations

import argparse
import ast
import importlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from read_config import read_config


def instantiate(config: Any) -> Any:
    """Recursively instantiate objects from a config dictionary."""
    if isinstance(config, dict):
        if "_target_" in config:
            target = config["_target_"]
            module_name, attr = target.rsplit(".", 1)
            module = importlib.import_module(module_name)
            kwargs = {k: instantiate(v) for k, v in config.items() if k != "_target_"}
            return getattr(module, attr)(**kwargs)
        return {k: instantiate(v) for k, v in config.items()}
    if isinstance(config, list):
        return [instantiate(item) for item in config]
    return config


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def instantiate_optimizer(config: Dict[str, Any], params) -> torch.optim.Optimizer:
    target = config["_target_"]
    module_name, attr = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    kwargs = {k: v for k, v in config.items() if k != "_target_"}
    return getattr(module, attr)(params, **kwargs)


def maybe_instantiate_scheduler(config: Dict[str, Any] | None, optimizer: torch.optim.Optimizer):
    if config is None:
        return None
    if isinstance(config, dict):
        enabled = config.get("enabled", True)
        if enabled is False:
            return None
    target = config.get("_target_") if isinstance(config, dict) else None
    if not target:
        return None
    module_name, attr = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    skip_keys = {"_target_", "enabled"}
    kwargs = {k: instantiate(v) for k, v in config.items() if k not in skip_keys}
    return getattr(module, attr)(optimizer, **kwargs)


def _parse_value(value: str) -> Any:
    value = value.strip()
    if not value:
        return value
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _configparser_to_dict(config_parser) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    for section in config_parser.sections():
        section_items = {key: _parse_value(val) for key, val in config_parser.items(section)}
        path = section.split(".")
        target = config
        for key in path[:-1]:
            target = target.setdefault(key, {})
        target[path[-1]] = section_items
    general = config.pop("general", {})
    if isinstance(general, dict):
        config.update(general)
    return config


def train(config: Dict[str, Any]) -> None:
    seed = config.get("seed", 0)
    set_seed(seed)

    device = torch.device(config.get("device", "cpu"))
    train_settings = config.get("train", {})
    batch_size = train_settings.get("batch_size", 32)
    num_workers = train_settings.get("num_workers", 0)
    epochs = train_settings.get("epochs", 10)
    show_progress = train_settings.get("progress_bar", True)
    val_output_dir_cfg = train_settings.get("val_output_dir")
    val_output_dir = Path(val_output_dir_cfg).expanduser() if val_output_dir_cfg else None
    if val_output_dir is not None:
        val_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = instantiate(config["dataset"])
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = None
    if "validation_dataset" in config:
        val_dataset = instantiate(config["validation_dataset"])
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    model_cfg = config["model"]
    encoder = instantiate(model_cfg["encoder"]).to(device)
    decoder = instantiate(model_cfg["decoder"]).to(device)
    filter_kwargs = {k: v for k, v in model_cfg["filter"].items() if k != "_target_"}
    filter_kwargs.update({"encoder": encoder, "decoder": decoder})
    filter_kwargs = instantiate(filter_kwargs)
    filter_cfg = {"_target_": model_cfg["filter"]["_target_"], **filter_kwargs}
    model = instantiate(filter_cfg).to(device)

    optimizer = instantiate_optimizer(config["optimizer"], model.parameters())
    scheduler = maybe_instantiate_scheduler(config.get("scheduler"), optimizer)

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"loss": 0.0, "kl": 0.0, "likelihood": 0.0}
        train_iter = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs} [train]",
            leave=False,
            disable=not show_progress,
        )
        for batch in train_iter:
            obs, target = batch
            obs = obs.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            loss, loss_kl, loss_integral = model(obs, target)
            loss.backward()
            optimizer.step()

            current_metrics = {
                "loss": loss.item(),
                "kl": loss_kl.item(),
                "likelihood": loss_integral.item(),
            }
            running["loss"] += current_metrics["loss"] * obs.size(0)
            running["kl"] += current_metrics["kl"] * obs.size(0)
            running["likelihood"] += current_metrics["likelihood"] * obs.size(0)

            if show_progress:
                train_iter.set_postfix({k: f"{v:.4f}" for k, v in current_metrics.items()})

        if scheduler is not None:
            scheduler.step()

        num_samples = len(train_loader.dataset)
        epoch_metrics = {k: v / num_samples for k, v in running.items()}
        message = {
            "epoch": epoch,
            "train": epoch_metrics,
        }

        if val_loader is not None:
            model.eval()
            val_running = {"loss": 0.0, "kl": 0.0, "likelihood": 0.0}
            epoch_output_dir = None
            if val_output_dir is not None:
                epoch_output_dir = val_output_dir / f"epoch_{epoch:03d}"
                epoch_output_dir.mkdir(parents=True, exist_ok=True)
            val_iter = tqdm(
                val_loader,
                desc=f"Epoch {epoch}/{epochs} [val]",
                leave=False,
                disable=not show_progress,
            )
            with torch.no_grad():
                for batch_idx, (obs, target) in enumerate(val_iter):
                    obs = obs.to(device)
                    target = target.to(device)
                    if epoch_output_dir is not None:
                        loss, loss_kl, loss_integral, recon = model(
                            obs, target, return_reconstruction=True
                        )
                    else:
                        loss, loss_kl, loss_integral = model(obs, target)
                        recon = None

                    current_val_metrics = {
                        "loss": loss.item(),
                        "kl": loss_kl.item(),
                        "likelihood": loss_integral.item(),
                    }
                    val_running["loss"] += current_val_metrics["loss"] * obs.size(0)
                    val_running["kl"] += current_val_metrics["kl"] * obs.size(0)
                    val_running["likelihood"] += current_val_metrics["likelihood"] * obs.size(0)

                    if show_progress:
                        val_iter.set_postfix(
                            {k: f"{v:.4f}" for k, v in current_val_metrics.items()}
                        )

                    if recon is not None:
                        batch_path = epoch_output_dir / f"batch_{batch_idx:04d}.pt"
                        torch.save(
                            {
                                "observations": obs.detach().cpu(),
                                "targets": target.detach().cpu(),
                                "reconstruction": recon.detach().cpu(),
                            },
                            batch_path,
                        )
            val_samples = len(val_loader.dataset)
            message["validation"] = {k: v / val_samples for k, v in val_running.items()}

        print(json.dumps(message))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DBF on the Lorenz-96 system")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/lorenz96_example.yaml"),
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    config_parser = read_config(str(args.config))
    config = _configparser_to_dict(config_parser)

    train(config)


if __name__ == "__main__":
    main()
