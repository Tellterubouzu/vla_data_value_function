from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image

from src.models.vf_backbones import (
    PaliGemma2Backbone,
    Siglip2Gemma3Backbone,
    infer_attention_lora_target_modules,
)


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


def _concat_horizontal(head_img: Image.Image, hand_img: Image.Image) -> Image.Image:
    head_rgb = head_img.convert("RGB")
    hand_rgb = hand_img.convert("RGB")

    if head_rgb.height != hand_rgb.height:
        hand_rgb = hand_rgb.resize((hand_rgb.width, head_rgb.height))

    mosaic = Image.new("RGB", (head_rgb.width + hand_rgb.width, head_rgb.height))
    mosaic.paste(head_rgb, (0, 0))
    mosaic.paste(hand_rgb, (head_rgb.width, 0))
    return mosaic


class Siglip2Gemma3Collator:
    def __init__(self, vision_processor: Any, text_tokenizer: Any, max_text_length: int = 128) -> None:
        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer
        self.max_text_length = int(max_text_length)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        head_images = [x["head_image"] for x in batch]
        hand_images = [x["hand_image"] for x in batch]
        texts = [x["text"] for x in batch]

        head_inputs = self.vision_processor(images=head_images, return_tensors="pt")
        hand_inputs = self.vision_processor(images=hand_images, return_tensors="pt")

        text_inputs = self.text_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        return {
            "model_inputs": {
                "pixel_values_head": head_inputs["pixel_values"],
                "pixel_values_hand": hand_inputs["pixel_values"],
                "input_ids": text_inputs["input_ids"],
                "attention_mask": text_inputs["attention_mask"],
            },
            "labels_bin": torch.tensor([x["label_bin"] for x in batch], dtype=torch.long),
            "v_norm": torch.tensor([x["v_norm"] for x in batch], dtype=torch.float32),
            "task": [x["task"] for x in batch],
            "episode_index": torch.tensor([x["episode_index"] for x in batch], dtype=torch.long),
            "timestep": torch.tensor([x["timestep"] for x in batch], dtype=torch.long),
        }


class PaliGemma2Collator:
    def __init__(self, processor: Any, max_text_length: int = 128) -> None:
        self.processor = processor
        self.max_text_length = int(max_text_length)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        mosaics = [_concat_horizontal(x["head_image"], x["hand_image"]) for x in batch]
        texts = [x["text"] for x in batch]

        model_inputs = self.processor(
            text=texts,
            images=mosaics,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
        )

        return {
            "model_inputs": dict(model_inputs),
            "labels_bin": torch.tensor([x["label_bin"] for x in batch], dtype=torch.long),
            "v_norm": torch.tensor([x["v_norm"] for x in batch], dtype=torch.float32),
            "task": [x["task"] for x in batch],
            "episode_index": torch.tensor([x["episode_index"] for x in batch], dtype=torch.long),
            "timestep": torch.tensor([x["timestep"] for x in batch], dtype=torch.long),
        }


@dataclass
class ValueFunctionModelConfig:
    backbone: str
    tune_mode: str
    num_bins: int
    siglip_model_id: str = "google/siglip2-base-patch16-224"
    gemma_model_id: str = "google/gemma-3-270m"
    paligemma_model_id: str = "google/paligemma2-3b-pt-224"
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.05


class ValueFunctionModel(nn.Module):
    def __init__(self, config: ValueFunctionModelConfig) -> None:
        super().__init__()
        self.config = config

        if config.backbone == "siglip2_gemma3_270m":
            self.backbone = Siglip2Gemma3Backbone(
                num_bins=config.num_bins,
                vision_model_id=config.siglip_model_id,
                text_model_id=config.gemma_model_id,
            )
        elif config.backbone == "paligemma2_3b_pt224":
            self.backbone = PaliGemma2Backbone(
                num_bins=config.num_bins,
                model_id=config.paligemma_model_id,
            )
        else:
            raise ValueError(f"Unsupported backbone: {config.backbone}")

        self._apply_tune_mode()

    def _apply_tune_mode(self) -> None:
        mode = self.config.tune_mode
        if mode == "head":
            self._configure_head_tuning()
            return
        if mode == "full":
            self._configure_full_tuning()
            return
        if mode == "lora":
            self._configure_lora_tuning()
            return
        raise ValueError(f"Unsupported tune_mode: {mode}")

    def _configure_head_tuning(self) -> None:
        _set_requires_grad(self, False)

        if self.config.backbone == "siglip2_gemma3_270m":
            _set_requires_grad(self.backbone.img_proj, True)
            _set_requires_grad(self.backbone.txt_proj, True)
            _set_requires_grad(self.backbone.fuse_mlp, True)
            return

        _set_requires_grad(self.backbone.value_head, True)

    def _configure_full_tuning(self) -> None:
        _set_requires_grad(self, True)

    def _configure_lora_tuning(self) -> None:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as exc:
            raise ModuleNotFoundError(
                "peft is required for tune_mode=lora. Install with: pip install peft"
            ) from exc

        _set_requires_grad(self, False)

        rank = self.config.lora_rank or 16
        alpha = self.config.lora_alpha or (2 * rank)
        dropout = float(self.config.lora_dropout)

        if self.config.backbone == "siglip2_gemma3_270m":
            vision_targets = infer_attention_lora_target_modules(self.backbone.vision_model)
            text_targets = infer_attention_lora_target_modules(self.backbone.text_model)

            vision_lora_cfg = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias="none",
                target_modules=vision_targets,
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            text_lora_cfg = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias="none",
                target_modules=text_targets,
                task_type=TaskType.FEATURE_EXTRACTION,
            )

            self.backbone.vision_model = get_peft_model(self.backbone.vision_model, vision_lora_cfg)
            self.backbone.text_model = get_peft_model(self.backbone.text_model, text_lora_cfg)

            _set_requires_grad(self.backbone.img_proj, True)
            _set_requires_grad(self.backbone.txt_proj, True)
            _set_requires_grad(self.backbone.fuse_mlp, True)
            return

        targets = infer_attention_lora_target_modules(self.backbone.model)
        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=targets,
            task_type=TaskType.CAUSAL_LM,
        )
        self.backbone.model = get_peft_model(self.backbone.model, lora_cfg)
        _set_requires_grad(self.backbone.value_head, True)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.backbone(**batch["model_inputs"])

    def build_collator(self, max_text_length: int = 128):
        bundle = self.backbone.processor_bundle()
        if self.config.backbone == "siglip2_gemma3_270m":
            return Siglip2Gemma3Collator(
                vision_processor=bundle["vision_processor"],
                text_tokenizer=bundle["text_tokenizer"],
                max_text_length=max_text_length,
            )
        return PaliGemma2Collator(processor=bundle["processor"], max_text_length=max_text_length)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                state[name] = param.detach().cpu()
        return state

    def load_trainable_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys while loading trainable state: {unexpected}")

        # Strict=False means frozen backbone keys are expected to be missing.
        for key in missing:
            if key in state_dict:
                raise RuntimeError(f"Missing trainable key after load: {key}")

    def to_config_dict(self) -> Dict[str, Any]:
        return {
            "backbone": self.config.backbone,
            "tune_mode": self.config.tune_mode,
            "num_bins": self.config.num_bins,
            "siglip_model_id": self.config.siglip_model_id,
            "gemma_model_id": self.config.gemma_model_id,
            "paligemma_model_id": self.config.paligemma_model_id,
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
        }

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any]) -> "ValueFunctionModel":
        return cls(ValueFunctionModelConfig(**config_dict))
