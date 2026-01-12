import os

import torch

from src.config.processing import (
    IOConfig,
    OutputMode,
    ProcessingConfig,
    SpatialTilingConfig,
    TemporalTilingConfig,
)
from src.models.utils import get_device_list
from src.processing.pipeline import init_pipeline
from src.processing.video import flashvsr


def main():
    model = "FlashVSR-v1.1"

    _device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "auto"
    )
    if _device == "auto" or _device not in get_device_list():
        raise RuntimeError("No devices found to run FlashVSR!")

    input_path = "inputs/example3.mp4"

    video_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = f"{video_name}_output"
    os.makedirs(output_dir, exist_ok=True)

    proc_config = ProcessingConfig(
        scale=4,
        seed=0,
        sparse_ratio=2.0,
        kv_ratio=3.0,
        local_range=11,
        color_fix=True,
        unload_dit=True,
        force_offload=True,
    )
    spatial_tiling_config = SpatialTilingConfig(
        enabled=True,
        tile_size=(192, 192),
        tile_overlap=24,
    )
    temporal_tiling_config = TemporalTilingConfig(
        enabled=True,
        tile_size=100,
        tile_overlap=6,
    )
    io_config = IOConfig(
        input_path=input_path,
        output_mode=OutputMode.VIDEO,
        output_dir=output_dir,
    )

    pipe = init_pipeline(model, _device, torch.float16)

    flashvsr(
        pipe=pipe,
        io_config=io_config,
        proc_config=proc_config,
        spatial_tiling_config=spatial_tiling_config,
        temporal_tiling_config=temporal_tiling_config,
    )


if __name__ == "__main__":
    main()
