from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.airoa_moma_vf_dataset import (
    AiroaMomaEpisodeStore,
    AiroaMomaValueFrameDataset,
    DEFAULT_NUM_BINS,
)
from src.models.vf_model import ValueFunctionModel, ValueFunctionModelConfig
from src.utils.metrics import compute_batch_metrics
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AIRoA MoMa distributed value function (201-bin)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=["siglip2_gemma3_270m", "paligemma2_3b_pt224"],
    )
    parser.add_argument("--tune_mode", type=str, choices=["head", "lora", "full"], default=None)

    parser.add_argument("--siglip_model_id", type=str, default="google/siglip2-base-patch16-224")
    parser.add_argument("--gemma_model_id", type=str, default="google/gemma-3-270m")
    parser.add_argument("--paligemma_model_id", type=str, default="google/paligemma2-3b-pt-224")

    parser.add_argument("--frames_per_episode", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_train_steps", type=int, default=None)

    parser.add_argument("--num_bins", type=int, default=DEFAULT_NUM_BINS)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--max_text_length", type=int, default=128)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--precision", type=str, choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--use_tensorboard", action="store_true")

    return parser.parse_args()


def default_tune_mode(backbone: str) -> str:
    if backbone == "siglip2_gemma3_270m":
        return "lora"
    return "head"


def default_lora_rank(backbone: str) -> int:
    if backbone == "siglip2_gemma3_270m":
        return 16
    return 8


def choose_device_and_amp_dtype(precision: str) -> Tuple[torch.device, Optional[torch.dtype], bool]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if precision == "fp32":
            return device, None, False
        if precision == "bf16":
            return device, torch.bfloat16, True
        if precision == "fp16":
            return device, torch.float16, True
        # auto
        if torch.cuda.is_bf16_supported():
            return device, torch.bfloat16, True
        return device, torch.float16, True

    device = torch.device("cpu")
    return device, None, False


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {
        "model_inputs": {},
        "labels_bin": batch["labels_bin"].to(device, non_blocking=True),
        "v_norm": batch["v_norm"].to(device, non_blocking=True),
        "task": batch["task"],
        "episode_index": batch["episode_index"],
        "timestep": batch["timestep"],
    }
    for key, value in batch["model_inputs"].items():
        if isinstance(value, torch.Tensor):
            moved["model_inputs"][key] = value.to(device, non_blocking=True)
        else:
            moved["model_inputs"][key] = value
    return moved


def autocast_context(device: torch.device, amp_enabled: bool, amp_dtype: Optional[torch.dtype]):
    if amp_enabled:
        return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True)
    return nullcontext()


def evaluate(
    model: ValueFunctionModel,
    data_loader: DataLoader,
    device: torch.device,
    num_bins: int,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="eval", leave=False):
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
        return {"loss": float("nan"), "mae": float("nan")}

    return {
        "loss": total_loss / total_count,
        "mae": total_mae / total_count,
    }


def maybe_create_summary_writer(output_dir: Path, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        print("[warn] tensorboard is not available. Proceeding without SummaryWriter.")
        return None
    return SummaryWriter(log_dir=str(output_dir / "tb"))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tune_mode = args.tune_mode or default_tune_mode(args.backbone)
    lora_rank = args.lora_rank
    if tune_mode == "lora" and lora_rank is None:
        lora_rank = default_lora_rank(args.backbone)

    run_config = {
        **vars(args),
        "resolved_tune_mode": tune_mode,
        "resolved_lora_rank": lora_rank,
    }

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    episode_store = AiroaMomaEpisodeStore.load(data_root=args.data_root, split_seed=args.seed, val_ratio=0.05)
    train_dataset = AiroaMomaValueFrameDataset(
        episode_store=episode_store,
        split="train",
        frames_per_episode=args.frames_per_episode,
        sample_seed=args.seed,
        num_bins=args.num_bins,
    )
    val_dataset = AiroaMomaValueFrameDataset(
        episode_store=episode_store,
        split="val",
        frames_per_episode=args.frames_per_episode,
        sample_seed=args.seed,
        num_bins=args.num_bins,
    )

    vf_config = ValueFunctionModelConfig(
        backbone=args.backbone,
        tune_mode=tune_mode,
        num_bins=args.num_bins,
        siglip_model_id=args.siglip_model_id,
        gemma_model_id=args.gemma_model_id,
        paligemma_model_id=args.paligemma_model_id,
        lora_rank=lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    model = ValueFunctionModel(vf_config)
    collator = model.build_collator(max_text_length=args.max_text_length)

    device, amp_dtype, amp_enabled = choose_device_and_amp_dtype(args.precision)
    model = model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
        drop_last=False,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Check tune_mode setup.")

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16 and device.type == "cuda"))

    writer = maybe_create_summary_writer(output_dir, enabled=args.use_tensorboard)

    print(f"device={device}, amp_enabled={amp_enabled}, amp_dtype={amp_dtype}")
    print(f"trainable_params={model.trainable_parameter_count():,} / total={model.total_parameter_count():,}")
    print(f"train_samples={len(train_dataset)}, val_samples={len(val_dataset)}")

    global_step = 0
    stop_training = False

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"train epoch {epoch + 1}/{args.epochs}")

        for batch in progress:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
                logits = model(batch)
                loss = F.cross_entropy(logits, batch["labels_bin"])

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            metrics = compute_batch_metrics(logits=logits.detach(), v_norm=batch["v_norm"], num_bins=args.num_bins)

            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}", mae=f"{metrics['mae']:.4f}")

            if writer is not None:
                writer.add_scalar("train/loss", float(loss.item()), global_step)
                writer.add_scalar("train/mae", float(metrics["mae"]), global_step)

            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                stop_training = True
                break

        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            num_bins=args.num_bins,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        print(
            f"[epoch {epoch + 1}] val_loss={val_metrics['loss']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} step={global_step}"
        )

        if writer is not None:
            writer.add_scalar("val/loss", val_metrics["loss"], global_step)
            writer.add_scalar("val/mae", val_metrics["mae"], global_step)

        if stop_training:
            break

    # Save checkpoint.
    with (output_dir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(model.to_config_dict(), f, indent=2, ensure_ascii=False)

    with (output_dir / "dataset_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "split_seed": args.seed,
                "frames_per_episode": args.frames_per_episode,
                "num_bins": args.num_bins,
                "l_max_by_task": episode_store.l_max_by_task,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    if tune_mode == "full":
        torch.save(model.state_dict(), output_dir / "model_full_state.pt")
    else:
        torch.save(model.get_trainable_state_dict(), output_dir / "model_trainable_state.pt")

    if writer is not None:
        writer.close()

    print(f"checkpoint saved: {output_dir}")


if __name__ == "__main__":
    main()
