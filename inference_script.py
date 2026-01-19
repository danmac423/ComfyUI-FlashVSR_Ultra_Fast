import argparse
import glob
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
from src.models.wan_video_dit import AttentionMode, MaskAttentionMode
from src.processing.pipeline import init_pipeline
from src.processing.video import flashvsr


def parse_args():
    parser = argparse.ArgumentParser(
        description="FlashVSR Ultra Fast - Video Super Resolution"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Ścieżka do katalogu z filmami wejściowymi",
    )
    
    parser.add_argument(
        "--output_mode",
        type=str,
        choices=["video", "frames"],
        default="frames",
        help="Sposób zapisywania wyników: 'video' lub 'frames'",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Ścieżka do katalogu na wyniki",
    )
    
    parser.add_argument(
        "--attn_mode",
        type=str,
        choices=["sage", "flash"],
        default="flash",
        help="Tryb attention: 'sage' lub 'flash'",
    )
    
    parser.add_argument(
        "--mask_attn_mode",
        type=str,
        choices=["sparse_sage", "block_sparse"],
        default="block_sparse",
        help="Tryb mask attention: 'sparse_sage' lub 'block_sparse'",
    )
    
    parser.add_argument(
        "--spatial_tile_size",
        type=int,
        nargs=2,
        default=[192, 192],
        metavar=("HEIGHT", "WIDTH"),
        help="Rozmiar kafelka przestrzennego (wysokość szerokość)",
    )
    
    parser.add_argument(
        "--spatial_tile_overlap",
        type=int,
        default=24,
        help="Nakładanie kafelków przestrzennych",
    )
    
    parser.add_argument(
        "--temporal_tile_size",
        type=int,
        default=100,
        help="Rozmiar kafelka czasowego",
    )
    
    parser.add_argument(
        "--temporal_tile_overlap",
        type=int,
        default=6,
        help="Nakładanie kafelków czasowych",
    )
    
    parser.add_argument(
        "--disable_spatial_tiling",
        action="store_true",
        help="Wyłącz spatial tiling",
    )
    
    parser.add_argument(
        "--disable_temporal_tiling",
        action="store_true",
        help="Wyłącz temporal tiling",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
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

    # Znajdź wszystkie pliki wideo w katalogu wejściowym
    video_extensions = ["*.mp4", "*.mkv", "*.avi", "*.mov", "*.webm", "*.flv"]
    input_videos = []
    for ext in video_extensions:
        input_videos.extend(glob.glob(os.path.join(args.input_dir, ext)))
    input_videos = sorted(input_videos)
    
    if not input_videos:
        raise RuntimeError(f"Nie znaleziono filmów w katalogu: {args.input_dir}")
    
    print(f"Znaleziono {len(input_videos)} filmów do przetworzenia")
    
    # Konwersja argumentów na odpowiednie typy enum
    output_mode = OutputMode.VIDEO if args.output_mode == "video" else OutputMode.FRAMES
    attn_mode = AttentionMode.SAGE if args.attn_mode == "sage" else AttentionMode.FLASH
    mask_attn_mode = (
        MaskAttentionMode.SPARSE_SAGE
        if args.mask_attn_mode == "sparse_sage"
        else MaskAttentionMode.BLOCK_SPARSE
    )
    
    # Konfiguracja przetwarzania
    proc_config = ProcessingConfig(
        scale=4,
        seed=0,
        sparse_ratio=2.0,
        kv_ratio=3.0,
        local_range=11,
        color_fix=True,
        unload_dit=True,
        force_offload=True,
        attn_mode=attn_mode,
        mask_attn_mode=mask_attn_mode,
    )
    
    spatial_tiling_config = SpatialTilingConfig(
        enabled=not args.disable_spatial_tiling,
        tile_size=tuple(args.spatial_tile_size),
        tile_overlap=args.spatial_tile_overlap,
    )
    
    temporal_tiling_config = TemporalTilingConfig(
        enabled=not args.disable_temporal_tiling,
        tile_size=args.temporal_tile_size,
        tile_overlap=args.temporal_tile_overlap,
    )
    
    # Inicjalizacja pipeline
    pipe = init_pipeline(
        model,
        _device,
        torch.float16,
        attn_mode=proc_config.attn_mode,
        mask_attn_mode=proc_config.mask_attn_mode,
    )
    
    # Przetwarzanie każdego filmu
    for i, input_path in enumerate(input_videos):
        print(f"\n[{i}/{len(input_videos)}] Przetwarzanie: {input_path}")
        
        video_name = os.path.splitext(os.path.basename(input_path))[0]
        video_output_dir = os.path.join(args.output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        io_config = IOConfig(
            input_path=input_path,
            output_mode=output_mode,
            output_dir=video_output_dir,
        )
        
        flashvsr(
            pipe=pipe,
            io_config=io_config,
            proc_config=proc_config,
            spatial_tiling_config=spatial_tiling_config,
            temporal_tiling_config=temporal_tiling_config,
        )
        
        print(f"Zakończono przetwarzanie: {video_name}")
    
    print(f"\n✓ Wszystkie filmy przetworzone! Wyniki zapisane w: {args.output_dir}")


if __name__ == "__main__":
    main()