from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - allows importing utility functions without torch.
    class Dataset:  # type: ignore[override]
        pass

from src.utils.video_io import get_worker_video_cache


DEFAULT_NUM_BINS = 201
DEFAULT_VIDEO_FPS = 30.0


@dataclass(frozen=True)
class EpisodeMeta:
    episode_index: int
    length: int
    task_success: bool
    short_horizon_task: str
    head_path: Path
    hand_path: Path
    head_from_ts: Optional[float] = None
    head_to_ts: Optional[float] = None
    hand_from_ts: Optional[float] = None
    hand_to_ts: Optional[float] = None


@dataclass(frozen=True)
class FrameSample:
    episode_index: int
    timestep: int


@dataclass
class AiroaMomaEpisodeStore:
    data_root: Path
    episodes: List[EpisodeMeta]
    episodes_by_index: Dict[int, EpisodeMeta]
    l_max_by_task: Dict[str, int]
    train_episode_indices: List[int]
    val_episode_indices: List[int]

    @classmethod
    def load(
        cls,
        data_root: str | Path,
        split_seed: int,
        val_ratio: float = 0.05,
    ) -> "AiroaMomaEpisodeStore":
        root = Path(data_root)
        if not root.exists():
            raise FileNotFoundError(f"data_root not found: {root}")

        episodes = load_episode_metadata(root)
        if not episodes:
            raise RuntimeError(f"No episodes were found in: {root}")

        episodes_by_index = {ep.episode_index: ep for ep in episodes}
        if len(episodes_by_index) != len(episodes):
            raise RuntimeError("episode_index is not unique in metadata")

        l_max_by_task = compute_l_max_by_task(episodes)
        train_indices, val_indices = split_episode_indices(
            sorted(episodes_by_index.keys()),
            seed=split_seed,
            val_ratio=val_ratio,
        )

        return cls(
            data_root=root,
            episodes=episodes,
            episodes_by_index=episodes_by_index,
            l_max_by_task=l_max_by_task,
            train_episode_indices=train_indices,
            val_episode_indices=val_indices,
        )


class AiroaMomaValueFrameDataset(Dataset):
    def __init__(
        self,
        episode_store: AiroaMomaEpisodeStore,
        split: str,
        frames_per_episode: int = 8,
        sample_seed: int = 0,
        num_bins: int = DEFAULT_NUM_BINS,
        video_fps: float = DEFAULT_VIDEO_FPS,
        video_cache_size: int = 16,
    ) -> None:
        if split not in {"train", "val", "all"}:
            raise ValueError(f"split must be train/val/all, got: {split}")
        if frames_per_episode <= 0:
            raise ValueError("frames_per_episode must be > 0")

        self.episode_store = episode_store
        self.split = split
        self.frames_per_episode = int(frames_per_episode)
        self.sample_seed = int(sample_seed)
        self.num_bins = int(num_bins)
        self.video_fps = float(video_fps)
        self.video_cache_size = int(video_cache_size)

        if split == "train":
            episode_indices = episode_store.train_episode_indices
        elif split == "val":
            episode_indices = episode_store.val_episode_indices
        else:
            episode_indices = sorted(episode_store.episodes_by_index.keys())

        self.episode_indices = episode_indices
        self.samples = build_frame_samples(
            episodes_by_index=episode_store.episodes_by_index,
            episode_indices=episode_indices,
            frames_per_episode=self.frames_per_episode,
            sample_seed=self.sample_seed,
            split=split,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        episode = self.episode_store.episodes_by_index[sample.episode_index]

        l_max = self.episode_store.l_max_by_task.get(episode.short_horizon_task)
        if l_max is None:
            raise KeyError(f"L_max missing for task: {episode.short_horizon_task}")

        v_norm, target_bin = compute_value_and_bin(
            length=episode.length,
            timestep=sample.timestep,
            success=episode.task_success,
            l_max_task=l_max,
            num_bins=self.num_bins,
        )

        cache = get_worker_video_cache(max_size=self.video_cache_size)

        head_image = self._decode_episode_frame(
            cache=cache,
            video_path=episode.head_path,
            from_ts=episode.head_from_ts,
            to_ts=episode.head_to_ts,
            length=episode.length,
            timestep=sample.timestep,
        )
        hand_image = self._decode_episode_frame(
            cache=cache,
            video_path=episode.hand_path,
            from_ts=episode.hand_from_ts,
            to_ts=episode.hand_to_ts,
            length=episode.length,
            timestep=sample.timestep,
        )

        return {
            "head_image": head_image,
            "hand_image": hand_image,
            "text": episode.short_horizon_task,
            "label_bin": target_bin,
            "v_norm": float(v_norm),
            "episode_index": episode.episode_index,
            "timestep": sample.timestep,
            "task": episode.short_horizon_task,
        }

    def _decode_episode_frame(
        self,
        cache: Any,
        video_path: Path,
        from_ts: Optional[float],
        to_ts: Optional[float],
        length: int,
        timestep: int,
    ) -> Image.Image:
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            total_frames = cache.get_num_frames(video_path)
        except Exception as exc:
            raise RuntimeError(f"Failed opening video file: {video_path}. cause={exc}") from exc
        if total_frames <= 0:
            raise RuntimeError(f"Video has no frames: {video_path}")

        seg_start, seg_len = compute_segment_range(
            total_frames=total_frames,
            from_ts=from_ts,
            to_ts=to_ts,
            fps=self.video_fps,
        )
        if seg_len <= 0:
            raise RuntimeError(
                f"Invalid segment length for {video_path}: "
                f"from_ts={from_ts}, to_ts={to_ts}, total_frames={total_frames}"
            )

        seg_index = map_step_to_frame_index(
            timestep=timestep,
            length=length,
            n_video_frames=seg_len,
        )
        abs_index = int(np.clip(seg_start + seg_index, 0, total_frames - 1))

        try:
            return cache.read_frame(video_path, abs_index)
        except Exception as exc:
            raise RuntimeError(
                f"Failed decoding frame. file={video_path}, abs_index={abs_index}, "
                f"episode_length={length}, timestep={timestep}, "
                f"segment_start={seg_start}, segment_len={seg_len}"
            ) from exc


def compute_return(length: int, timestep: int, success: bool, c_fail: int) -> int:
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")
    t_terminal = length - 1
    if timestep < 0 or timestep > t_terminal:
        raise ValueError(
            f"timestep out of range: timestep={timestep}, valid=[0,{t_terminal}]"
        )

    if success:
        return -(t_terminal - timestep)
    return -(t_terminal - timestep) - int(c_fail)


def compute_value_and_bin(
    length: int,
    timestep: int,
    success: bool,
    l_max_task: int,
    num_bins: int = DEFAULT_NUM_BINS,
) -> Tuple[float, int]:
    if l_max_task <= 0:
        raise ValueError(f"l_max_task must be > 0, got {l_max_task}")

    ret = compute_return(length=length, timestep=timestep, success=success, c_fail=l_max_task)
    v_raw = ret / float(l_max_task)
    v_norm = float(np.clip(v_raw, -1.0, 0.0))

    bin_index = int(round((v_norm + 1.0) * (num_bins - 1)))
    bin_index = int(np.clip(bin_index, 0, num_bins - 1))
    return v_norm, bin_index


def map_step_to_frame_index(timestep: int, length: int, n_video_frames: int) -> int:
    if n_video_frames <= 0:
        raise ValueError(f"n_video_frames must be > 0, got {n_video_frames}")
    if length <= 1 or n_video_frames == 1:
        return 0
    idx = round(timestep / float(length - 1) * float(n_video_frames - 1))
    return int(np.clip(idx, 0, n_video_frames - 1))


def compute_segment_range(
    total_frames: int,
    from_ts: Optional[float],
    to_ts: Optional[float],
    fps: float,
) -> Tuple[int, int]:
    if total_frames <= 0:
        raise ValueError(f"total_frames must be > 0, got {total_frames}")

    if from_ts is None or to_ts is None:
        return 0, total_frames

    start = int(round(from_ts * fps))
    end_exclusive = int(round(to_ts * fps))

    start = int(np.clip(start, 0, total_frames - 1))
    end_exclusive = int(np.clip(end_exclusive, start + 1, total_frames))

    return start, end_exclusive - start


def sample_episode_steps(length: int, k: int, rng: random.Random) -> List[int]:
    if length <= 0:
        return []
    if length >= k:
        return list(rng.sample(range(length), k=k))
    return [rng.randrange(length) for _ in range(k)]


def build_frame_samples(
    episodes_by_index: Dict[int, EpisodeMeta],
    episode_indices: Sequence[int],
    frames_per_episode: int,
    sample_seed: int,
    split: str,
) -> List[FrameSample]:
    samples: List[FrameSample] = []
    split_offset = 0 if split == "train" else 10_000_019 if split == "val" else 20_000_033

    for episode_index in episode_indices:
        ep = episodes_by_index[episode_index]
        if ep.length <= 0:
            continue

        seed_i = (sample_seed * 1_000_003) ^ (episode_index * 97_003) ^ split_offset
        rng = random.Random(seed_i)
        timesteps = sample_episode_steps(length=ep.length, k=frames_per_episode, rng=rng)
        for timestep in timesteps:
            samples.append(FrameSample(episode_index=episode_index, timestep=int(timestep)))

    return samples


def compute_l_max_by_task(episodes: Iterable[EpisodeMeta]) -> Dict[str, int]:
    l_max: Dict[str, int] = {}
    for ep in episodes:
        if not ep.short_horizon_task:
            continue
        prev = l_max.get(ep.short_horizon_task, 0)
        if ep.length > prev:
            l_max[ep.short_horizon_task] = int(ep.length)
    return l_max


def split_episode_indices(
    episode_indices: Sequence[int],
    seed: int,
    val_ratio: float = 0.05,
) -> Tuple[List[int], List[int]]:
    if not episode_indices:
        return [], []

    indices = np.array(sorted(episode_indices), dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(indices))

    val_size = max(1, int(len(indices) * val_ratio))
    val_mask = np.zeros(len(indices), dtype=bool)
    val_mask[perm[:val_size]] = True

    val_indices = indices[val_mask].tolist()
    train_indices = indices[~val_mask].tolist()
    return train_indices, val_indices


def load_episode_metadata(data_root: Path) -> List[EpisodeMeta]:
    jsonl_candidates = [data_root / "episodes.jsonl", data_root / "meta" / "episodes.jsonl"]
    for jsonl_path in jsonl_candidates:
        if jsonl_path.exists():
            return _load_episode_metadata_from_jsonl(data_root, jsonl_path)

    parquet_root = data_root / "meta" / "episodes"
    if parquet_root.exists():
        return _load_episode_metadata_from_parquet(data_root, parquet_root)

    candidate_text = " or ".join(str(p) for p in jsonl_candidates)
    raise FileNotFoundError(
        "Episode metadata is missing. Expected one of: "
        f"{candidate_text}, or parquet files under {parquet_root}"
    )


def _load_episode_metadata_from_jsonl(data_root: Path, jsonl_path: Path) -> List[EpisodeMeta]:
    episodes: List[EpisodeMeta] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {jsonl_path}:{line_no}") from exc
            episodes.append(_episode_from_row(data_root, row, source=f"{jsonl_path}:{line_no}"))

    episodes.sort(key=lambda ep: ep.episode_index)
    return episodes


def _load_episode_metadata_from_parquet(data_root: Path, parquet_root: Path) -> List[EpisodeMeta]:
    parquet_files = sorted(parquet_root.glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {parquet_root}")

    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise ModuleNotFoundError(
            "pyarrow is required to load parquet episode metadata. "
            "Install it with: pip install pyarrow"
        ) from exc

    needed_columns = [
        "episode_index",
        "length",
        "task_success",
        "short_horizon_task",
        "tasks",
        "videos/observation.image.head/chunk_index",
        "videos/observation.image.head/file_index",
        "videos/observation.image.head/from_timestamp",
        "videos/observation.image.head/to_timestamp",
        "videos/observation.image.hand/chunk_index",
        "videos/observation.image.hand/file_index",
        "videos/observation.image.hand/from_timestamp",
        "videos/observation.image.hand/to_timestamp",
    ]

    episodes: List[EpisodeMeta] = []
    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        present_columns = [c for c in needed_columns if c in parquet_file.schema.names]
        table = pq.read_table(parquet_path, columns=present_columns)
        rows = table.to_pylist()

        for row_idx, row in enumerate(rows, start=1):
            source = f"{parquet_path}:{row_idx}"
            episodes.append(_episode_from_row(data_root, row, source=source))

    episodes.sort(key=lambda ep: ep.episode_index)
    return episodes


def _episode_from_row(data_root: Path, row: Dict[str, Any], source: str) -> EpisodeMeta:
    try:
        episode_index = int(row["episode_index"])
        length = int(row["length"])
    except Exception as exc:
        raise ValueError(f"Missing episode_index/length in {source}") from exc

    task_success = bool(row.get("task_success", row.get("success_short_horizon_task", False)))

    short_task = row.get("short_horizon_task")
    if not short_task:
        tasks = row.get("tasks")
        if isinstance(tasks, list) and tasks:
            short_task = str(tasks[0])
        elif isinstance(tasks, str) and tasks:
            short_task = tasks
    if not short_task:
        raise ValueError(f"short_horizon_task is missing for episode_index={episode_index} ({source})")

    head_path, head_from_ts, head_to_ts = _resolve_video_view(data_root, row, "head", source)
    hand_path, hand_from_ts, hand_to_ts = _resolve_video_view(data_root, row, "hand", source)

    if not head_path.exists():
        raise FileNotFoundError(
            f"Head video file not found for episode_index={episode_index}: {head_path} (source={source})"
        )
    if not hand_path.exists():
        raise FileNotFoundError(
            f"Hand video file not found for episode_index={episode_index}: {hand_path} (source={source})"
        )

    return EpisodeMeta(
        episode_index=episode_index,
        length=length,
        task_success=task_success,
        short_horizon_task=str(short_task),
        head_path=head_path,
        hand_path=hand_path,
        head_from_ts=head_from_ts,
        head_to_ts=head_to_ts,
        hand_from_ts=hand_from_ts,
        hand_to_ts=hand_to_ts,
    )


def _resolve_video_view(
    data_root: Path,
    row: Dict[str, Any],
    view: str,
    source: str,
) -> Tuple[Path, Optional[float], Optional[float]]:
    view_key = f"observation.image.{view}"

    explicit_path_keys = [
        f"videos/{view_key}/path",
        f"{view_key}_path",
        f"{view_key}",
    ]

    for key in explicit_path_keys:
        value = row.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            if not path.is_absolute():
                path = data_root / path
            from_ts = _optional_float(row.get(f"videos/{view_key}/from_timestamp"))
            to_ts = _optional_float(row.get(f"videos/{view_key}/to_timestamp"))
            return path, from_ts, to_ts

    chunk_key = f"videos/{view_key}/chunk_index"
    file_key = f"videos/{view_key}/file_index"

    if chunk_key in row and file_key in row:
        chunk_index = int(row[chunk_key])
        file_index = int(row[file_key])
        path = (
            data_root
            / "videos"
            / view_key
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:03d}.mp4"
        )
        from_ts = _optional_float(row.get(f"videos/{view_key}/from_timestamp"))
        to_ts = _optional_float(row.get(f"videos/{view_key}/to_timestamp"))
        return path, from_ts, to_ts

    raise ValueError(
        f"Could not resolve video path for {view_key} in {source}. "
        "Expected either explicit path key or chunk/file index keys."
    )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
