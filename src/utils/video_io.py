from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

try:
    from torch.utils.data import get_worker_info
except Exception:  # pragma: no cover - allows utility tests without torch.
    def get_worker_info() -> Any:  # type: ignore[override]
        return None


class VideoReaderCache:
    """Per-worker LRU cache for video readers/tensors."""

    def __init__(self, max_size: int = 16) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be > 0, got {max_size}")
        self.max_size = int(max_size)
        self.backend = self._choose_backend()
        self._cache: "OrderedDict[str, Any]" = OrderedDict()

    def _choose_backend(self) -> str:
        try:
            import decord  # noqa: F401

            return "decord"
        except Exception:
            pass

        try:
            from torchvision.io import read_video  # noqa: F401

            return "torchvision"
        except Exception:
            pass

        raise ModuleNotFoundError(
            "No video decoder backend available. Install decord (preferred) or torchvision."
        )

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def _open_reader(self, video_path: Path) -> Any:
        if self.backend == "decord":
            import decord

            return decord.VideoReader(str(video_path), ctx=decord.cpu(0))

        # torchvision fallback: cache full decoded tensor for the file.
        from torchvision.io import read_video

        try:
            frames, _, _ = read_video(str(video_path), pts_unit="sec", output_format="THWC")
        except TypeError:
            frames, _, _ = read_video(str(video_path), pts_unit="sec")
        return frames

    def _get_reader(self, video_path: Path) -> Any:
        key = str(video_path)
        reader = self._cache.get(key)
        if reader is not None:
            self._cache.move_to_end(key)
            return reader

        reader = self._open_reader(video_path)
        self._cache[key] = reader
        self._cache.move_to_end(key)
        self._evict_if_needed()
        return reader

    def get_num_frames(self, video_path: Path) -> int:
        reader = self._get_reader(video_path)
        if self.backend == "decord":
            return int(len(reader))

        # torchvision tensor path.
        return int(reader.shape[0])

    def read_frame(self, video_path: Path, frame_index: int) -> Image.Image:
        reader = self._get_reader(video_path)

        if self.backend == "decord":
            n_frames = int(len(reader))
            if n_frames <= 0:
                raise RuntimeError(f"No frame found in video: {video_path}")
            index = int(np.clip(frame_index, 0, n_frames - 1))
            frame = reader[index].asnumpy()
            return _to_pil_rgb(frame)

        n_frames = int(reader.shape[0])
        if n_frames <= 0:
            raise RuntimeError(f"No frame found in video: {video_path}")

        index = int(np.clip(frame_index, 0, n_frames - 1))
        frame = reader[index]
        if hasattr(frame, "cpu"):
            frame = frame.cpu()
        if hasattr(frame, "numpy"):
            frame = frame.numpy()

        return _to_pil_rgb(frame)


def _to_pil_rgb(frame: np.ndarray) -> Image.Image:
    if frame is None:
        raise RuntimeError("Decoded frame is None")

    frame_np = np.asarray(frame)

    if frame_np.ndim != 3:
        raise RuntimeError(f"Expected frame ndim=3, got shape={frame_np.shape}")

    # Convert CHW -> HWC when needed.
    if frame_np.shape[0] in (1, 3) and frame_np.shape[-1] not in (1, 3):
        frame_np = np.transpose(frame_np, (1, 2, 0))

    if frame_np.shape[-1] == 1:
        frame_np = np.repeat(frame_np, 3, axis=-1)

    if frame_np.shape[-1] != 3:
        raise RuntimeError(f"Expected 3 channels, got frame shape={frame_np.shape}")

    if frame_np.dtype != np.uint8:
        frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)

    return Image.fromarray(frame_np, mode="RGB")


_WORKER_VIDEO_CACHES: Dict[Tuple[int, int], VideoReaderCache] = {}


def get_worker_video_cache(max_size: int = 16) -> VideoReaderCache:
    worker_info = get_worker_info()
    worker_id = int(worker_info.id) if worker_info is not None else -1
    key = (worker_id, int(max_size))

    cache = _WORKER_VIDEO_CACHES.get(key)
    if cache is None:
        cache = VideoReaderCache(max_size=max_size)
        _WORKER_VIDEO_CACHES[key] = cache
    return cache
