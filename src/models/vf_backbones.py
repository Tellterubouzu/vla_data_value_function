from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def get_last_nonpad_hidden(last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return last_hidden_state[:, -1, :]

    lengths = attention_mask.long().sum(dim=1) - 1
    lengths = torch.clamp(lengths, min=0)
    batch_index = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[batch_index, lengths]


def infer_attention_lora_target_modules(model: nn.Module) -> List[str]:
    preferred_suffixes = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "query",
        "key",
        "value",
        "qkv",
        "to_q",
        "to_k",
        "to_v",
    }

    found_suffixes = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        suffix = name.split(".")[-1]
        lname = name.lower()
        if suffix in preferred_suffixes:
            found_suffixes.add(suffix)
            continue

        if any(token in lname for token in ["attn", "attention"]) and any(
            token in lname for token in ["q", "k", "v", "out", "proj", "query", "key", "value"]
        ):
            found_suffixes.add(suffix)

    if not found_suffixes:
        raise RuntimeError(
            "Could not infer LoRA target attention modules. "
            "Inspect model.named_modules() and pass explicit modules if needed."
        )

    return sorted(found_suffixes)


class Siglip2Gemma3Backbone(nn.Module):
    def __init__(
        self,
        num_bins: int,
        vision_model_id: str = "google/siglip2-base-patch16-224",
        text_model_id: str = "google/gemma-3-270m",
        fusion_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.vision_model_id = vision_model_id
        self.text_model_id = text_model_id

        self.vision_processor = AutoProcessor.from_pretrained(vision_model_id)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_id)
        if self.text_tokenizer.pad_token is None:
            if self.text_tokenizer.eos_token is None:
                raise ValueError(
                    f"Tokenizer for {text_model_id} has no pad_token and no eos_token to reuse."
                )
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        self.vision_model = AutoModel.from_pretrained(vision_model_id)
        self.text_model = AutoModel.from_pretrained(text_model_id)

        vision_dim = self._infer_vision_dim()
        text_dim = self._infer_text_dim()

        self.img_proj = nn.Linear(vision_dim * 2, fusion_dim)
        self.txt_proj = nn.Linear(text_dim, fusion_dim)
        self.fuse_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, self.num_bins),
        )

    def _infer_vision_dim(self) -> int:
        candidates = [
            getattr(self.vision_model.config, "projection_dim", None),
            getattr(self.vision_model.config, "hidden_size", None),
            getattr(getattr(self.vision_model.config, "vision_config", None), "hidden_size", None),
        ]
        for dim in candidates:
            if isinstance(dim, int) and dim > 0:
                return dim
        raise ValueError(f"Unable to infer vision feature dim from config for {self.vision_model_id}")

    def _infer_text_dim(self) -> int:
        candidates = [
            getattr(self.text_model.config, "hidden_size", None),
            getattr(getattr(self.text_model.config, "text_config", None), "hidden_size", None),
        ]
        for dim in candidates:
            if isinstance(dim, int) and dim > 0:
                return dim
        raise ValueError(f"Unable to infer text hidden dim from config for {self.text_model_id}")

    def forward(
        self,
        pixel_values_head: torch.Tensor,
        pixel_values_hand: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        head_feat = self.vision_model.get_image_features(pixel_values=pixel_values_head)
        hand_feat = self.vision_model.get_image_features(pixel_values=pixel_values_hand)

        if head_feat.ndim > 2:
            head_feat = head_feat[:, 0, :]
        if hand_feat.ndim > 2:
            hand_feat = hand_feat[:, 0, :]

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_hidden = text_outputs.last_hidden_state
        text_feat = get_last_nonpad_hidden(text_hidden, attention_mask)

        h_img = self.img_proj(torch.cat([head_feat, hand_feat], dim=-1))
        h_txt = self.txt_proj(text_feat)
        logits = self.fuse_mlp(torch.cat([h_img, h_txt], dim=-1))
        return logits

    def processor_bundle(self) -> Dict[str, Any]:
        return {
            "vision_processor": self.vision_processor,
            "text_tokenizer": self.text_tokenizer,
        }


class PaliGemma2Backbone(nn.Module):
    def __init__(
        self,
        num_bins: int,
        model_id: str = "google/paligemma2-3b-pt-224",
    ) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.model_id = model_id

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

        hidden_size = self._infer_hidden_size()
        self.value_head = nn.Linear(hidden_size, self.num_bins)

    def _infer_hidden_size(self) -> int:
        candidates = [
            getattr(self.model.config, "hidden_size", None),
            getattr(getattr(self.model.config, "text_config", None), "hidden_size", None),
            getattr(getattr(self.model.config, "vision_config", None), "hidden_size", None),
        ]
        for dim in candidates:
            if isinstance(dim, int) and dim > 0:
                return dim
        raise ValueError(f"Unable to infer hidden size from config for {self.model_id}")

    def forward(self, **model_inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            hidden_states = getattr(outputs, "hidden_states", None)
            if not hidden_states:
                raise RuntimeError("Model output has neither last_hidden_state nor hidden_states")
            last_hidden_state = hidden_states[-1]

        attention_mask = model_inputs.get("attention_mask")
        feat = get_last_nonpad_hidden(last_hidden_state, attention_mask)
        logits = self.value_head(feat)
        return logits

    def processor_bundle(self) -> Dict[str, Any]:
        return {"processor": self.processor}
