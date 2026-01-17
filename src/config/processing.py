"""Configuration dataclasses for video processing."""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple
from src.models.wan_video_dit import AttentionMode, MaskAttentionMode


class OutputMode(Enum):
    """Output mode for processed video."""

    VIDEO = "video"
    FRAMES = "frames"


@dataclass
class ProcessingConfig:
    """Configuration for video processing."""

    scale: int = 4
    color_fix: bool = True
    seed: int = 0
    sparse_ratio: float = 2.0
    kv_ratio: float = 3.0
    local_range: int = 11
    unload_dit: bool = True
    force_offload: bool = True
    attn_mode: AttentionMode = AttentionMode.FLASH
    mask_attn_mode: MaskAttentionMode = None


@dataclass
class SpatialTilingConfig:
    """Configuration for spatial tiling."""

    enabled: bool = True
    tile_size: Tuple[int, int] = (192, 192)
    tile_overlap: int = 24


@dataclass
class TemporalTilingConfig:
    """Configuration for temporal tiling."""

    enabled: bool = True
    tile_size: int = 100
    tile_overlap: int = 4


@dataclass
class IOConfig:
    """Configuration for input/output."""

    input_path: str = "input/video.mp4"
    output_mode: OutputMode = OutputMode.VIDEO
    output_dir: str = "output"
