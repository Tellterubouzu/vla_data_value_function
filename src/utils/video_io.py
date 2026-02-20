from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
        self.backends = self._choose_backends()
        # cache value: (backend, reader_object)
        self._cache: "OrderedDict[str, Tuple[str, Any]]" = OrderedDict()

    def _choose_backends(self) -> List[str]:
        backends: List[str] = []
        try:
            import decord  # noqa: F401

            backends.append("decord")
        except Exception:
            pass

        try:
            from torchvision.io import read_video  # noqa: F401

            backends.append("torchvision")
        except Exception:
            pass

        if not backends:
            raise ModuleNotFoundError(
                "No video decoder backend available. Install decord (preferred) or torchvision."
            )
        return backends

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def _is_lfs_pointer_file(self, video_path: Path) -> bool:
        try:
            with video_path.open("rb") as f:
                head = f.read(256)
        except Exception:
            return False
        return head.startswith(b"version https://git-lfs.github.com/spec/v1")

    def _open_with_backend(self, backend: str, video_path: Path) -> Any:
        if backend == "decord":
            import decord

            return decord.VideoReader(str(video_path), ctx=decord.cpu(0))

        # torchvision fallback: cache full decoded tensor for the file.
        from torchvision.io import read_video

        try:
            frames, _, _ = read_video(str(video_path), pts_unit="sec", output_format="THWC")
        except TypeError:
            frames, _, _ = read_video(str(video_path), pts_unit="sec")
        return frames

    def _open_reader(self, video_path: Path) -> Tuple[str, Any]:
        if self._is_lfs_pointer_file(video_path):
            raise RuntimeError(
                f"Video file is a Git LFS pointer (not actual MP4): {video_path}. "
                "Run 'git lfs pull' in the dataset directory."
            )

        errors: List[str] = []
        for backend in self.backends:
            try:
                return backend, self._open_with_backend(backend=backend, video_path=video_path)
            except Exception as exc:
                errors.append(f"{backend}: {exc}")

        raise RuntimeError(
            f"Failed to open video file with all available backends: {video_path}. "
            f"backend_errors={errors}"
        )

    def _get_reader(self, video_path: Path) -> Tuple[str, Any]:
        key = str(video_path)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        opened = self._open_reader(video_path)
        self._cache[key] = opened
        self._cache.move_to_end(key)
        self._evict_if_needed()
        return opened

    def get_num_frames(self, video_path: Path) -> int:
        backend, reader = self._get_reader(video_path)
        if backend == "decord":
            return int(len(reader))

        # torchvision tensor path.
        return int(reader.shape[0])

    def read_frame(self, video_path: Path, frame_index: int) -> Image.Image:
        backend, reader = self._get_reader(video_path)

        if backend == "decord":
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
