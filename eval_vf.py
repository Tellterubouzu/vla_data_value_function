from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.airoa_moma_vf_dataset import AiroaMomaEpisodeStore, AiroaMomaValueFrameDataset
from src.models.vf_model import ValueFunctionModel
from src.utils.metrics import compute_batch_metrics
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AIRoA MoMa value function checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--precision", type=str, choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    return parser.parse_args()


def choose_device_and_amp_dtype(precision: str):
    if torch.cuda.is_available():
        if precision == "fp32":
            return torch.device("cuda"), None, False
        if precision == "bf16":
            return torch.device("cuda"), torch.bfloat16, True
        if precision == "fp16":
            return torch.device("cuda"), torch.float16, True
        if torch.cuda.is_bf16_supported():
            return torch.device("cuda"), torch.bfloat16, True
        return torch.device("cuda"), torch.float16, True
    return torch.device("cpu"), None, False


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {
        "model_inputs": {},
        "labels_bin": batch["labels_bin"].to(device, non_blocking=True),
        "v_norm": batch["v_norm"].to(device, non_blocking=True),
        "task": batch["task"],
        "episode_index": batch["episode_index"],
        "timestep": batch["timestep"],
    }
    for key, value in batch["model_inputs"].items():
        if isinstance(value, torch.Tensor):
            out["model_inputs"][key] = value.to(device, non_blocking=True)
        else:
            out["model_inputs"][key] = value
    return out


def autocast_context(device: torch.device, amp_enabled: bool, amp_dtype: Optional[torch.dtype]):
    if amp_enabled:
        return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True)
    return nullcontext()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    ckpt_dir = Path(args.checkpoint)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_dir}")

    model_config = load_json(ckpt_dir / "model_config.json")
    dataset_config = load_json(ckpt_dir / "dataset_config.json")

    split_seed = int(args.seed if args.seed is not None else dataset_config.get("split_seed", 42))
    frames_per_episode = int(dataset_config.get("frames_per_episode", 8))
    num_bins = int(dataset_config.get("num_bins", model_config.get("num_bins", 201)))

    set_seed(split_seed)

    model = ValueFunctionModel.from_config_dict(model_config)

    full_state_path = ckpt_dir / "model_full_state.pt"
    trainable_state_path = ckpt_dir / "model_trainable_state.pt"

    if full_state_path.exists():
        state = torch.load(full_state_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
    elif trainable_state_path.exists():
        state = torch.load(trainable_state_path, map_location="cpu")
        model.load_trainable_state_dict(state)
    else:
        raise FileNotFoundError(
            f"No model state found. Expected {full_state_path.name} or {trainable_state_path.name}"
        )

    collator = model.build_collator()

    episode_store = AiroaMomaEpisodeStore.load(args.data_root, split_seed=split_seed, val_ratio=0.05)
    val_dataset = AiroaMomaValueFrameDataset(
        episode_store=episode_store,
        split="val",
        frames_per_episode=frames_per_episode,
        sample_seed=split_seed,
        num_bins=num_bins,
    )

    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )

    device, amp_dtype, amp_enabled = choose_device_and_amp_dtype(args.precision)
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            batch = move_batch_to_device(batch, device)
            with autocast_context(device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
                logits = model(batch)
                loss = F.cross_entropy(logits, batch["labels_bin"])

            metrics = compute_batch_metrics(logits=logits, v_norm=batch["v_norm"], num_bins=num_bins)

            batch_size = int(batch["labels_bin"].shape[0])
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_mae += float(metrics["mae"]) * batch_size
            total_count += batch_size

    if total_count == 0:
        raise RuntimeError("No validation samples found")

    loss_avg = total_loss / total_count
    mae_avg = total_mae / total_count

    print(f"val_loss={loss_avg:.6f}")
    print(f"val_mae={mae_avg:.6f}")


if __name__ == "__main__":
    main()
