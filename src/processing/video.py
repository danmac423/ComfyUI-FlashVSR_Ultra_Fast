"""Video processing functions for FlashVSR."""

import os

import torch
from einops import rearrange
from torchcodec.decoders import VideoDecoder
from torchcodec.encoders import VideoEncoder
from torchvision.io import write_png
from tqdm import tqdm

from src.config.processing import (
    IOConfig,
    OutputMode,
    ProcessingConfig,
    SpatialTilingConfig,
    TemporalTilingConfig,
)
from src.models.utils import clean_vram
from src.pipelines.base import BasePipeline
from src.utils.dimension import calculate_next_frame_requirement
from src.utils.tensor import convert_tensor_to_video, prepare_input_tensor, remove_padding
from src.utils.tiling import (
    blend_temporal_overlap,
    calculate_spatial_tile_coords,
    calculate_temporal_tile_ranges,
    create_spatial_blend_mask,
)


def process_single_temporal_chunk(
    pipe: BasePipeline,
    frames_chunk: torch.Tensor,
    proc_config: ProcessingConfig,
    spatial_config: SpatialTilingConfig,
) -> torch.Tensor:
    """Process a single temporal chunk of frames.

    Args:
        pipe (BasePipeline): FlashVSR pipeline instance
        frames_chunk (torch.Tensor): Input frames tensor of shape (N, H, W, C)
        proc_config (ProcessingConfig): Processing configuration
        spatial_config (SpatialTilingConfig): Spatial tiling configuration

    Returns:
        torch.Tensor: Processed frames tensor of shape (N, H, W, C)
    """
    _device = pipe.device
    dtype = pipe.torch_dtype

    add = calculate_next_frame_requirement(frames_chunk.shape[0]) - frames_chunk.shape[0]
    if add > 0:
        padding_frames = frames_chunk[-1:, :, :, :].repeat(add, 1, 1, 1)
        _frames = torch.cat([frames_chunk, padding_frames], dim=0)
    else:
        _frames = frames_chunk

    if spatial_config.enabled:
        N, H, W, C = _frames.shape

        final_output_canvas = torch.zeros(
            (N, H * proc_config.scale, W * proc_config.scale, C), dtype=torch.float16, device="cpu"
        )
        weight_sum_canvas = torch.zeros_like(final_output_canvas)
        tile_coords = calculate_spatial_tile_coords(
            H, W, spatial_config.tile_size, spatial_config.tile_overlap
        )

        for i, (x1, y1, x2, y2) in enumerate(tile_coords):
            input_tile = _frames[:, y1:y2, x1:x2, :]

            LQ_tile, sh, sw, th, tw, N, padding = prepare_input_tensor(
                input_tile, _device, scale=proc_config.scale, dtype=dtype
            )
            LQ_tile = LQ_tile.to(_device)

            output_tile_gpu = pipe(
                prompt="",
                negative_prompt="",
                cfg_scale=1.0,
                num_inference_steps=1,
                seed=proc_config.seed,
                LQ_video=LQ_tile,
                num_frames=N,
                height=th,
                width=tw,
                is_full_block=False,
                if_buffer=True,
                topk_ratio=proc_config.sparse_ratio * 768 * 1280 / (th * tw),
                kv_ratio=proc_config.kv_ratio,
                local_range=proc_config.local_range,
                color_fix=proc_config.color_fix,
                unload_dit=proc_config.unload_dit,
                force_offload=proc_config.force_offload,
            )

            processed_tile_cpu = convert_tensor_to_video(output_tile_gpu).to("cpu")

            processed_tile_cpu = remove_padding(processed_tile_cpu, sh, sw, th, tw, padding)

            mask = create_spatial_blend_mask(
                (sh, sw), spatial_config.tile_overlap * proc_config.scale
            ).to("cpu")

            out_x1, out_y1 = x1 * proc_config.scale, y1 * proc_config.scale
            out_x2, out_y2 = out_x1 + sw, out_y1 + sh

            final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask
            weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask

            del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile
            clean_vram()

        weight_sum_canvas[weight_sum_canvas == 0] = 1.0
        final_output = final_output_canvas / weight_sum_canvas
    else:
        LQ, sh, sw, th, tw, N, padding = prepare_input_tensor(
            _frames, _device, scale=proc_config.scale, dtype=dtype
        )
        LQ = LQ.to(_device)

        video = pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=proc_config.seed,
            LQ_video=LQ,
            num_frames=N,
            height=th,
            width=tw,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=proc_config.sparse_ratio * 768 * 1280 / (th * tw),
            kv_ratio=proc_config.kv_ratio,
            local_range=proc_config.local_range,
            color_fix=proc_config.color_fix,
            unload_dit=proc_config.unload_dit,
            force_offload=proc_config.force_offload,
        )

        final_output = convert_tensor_to_video(video).to("cpu")

        final_output = remove_padding(final_output, sh, sw, th, tw, padding)

        del video, LQ
        clean_vram()

    return final_output[: frames_chunk.shape[0], :, :, :]


def flashvsr(
    pipe: BasePipeline,
    io_config: IOConfig,
    proc_config: ProcessingConfig,
    spatial_tiling_config: SpatialTilingConfig,
    temporal_tiling_config: TemporalTilingConfig,
):
    """Process video using FlashVSR with temporal tiling support.
    Writes output incrementally to avoid RAM overflow.

    Args:
        pipe (BasePipeline): FlashVSR pipeline instance
        io_config (IOConfig): Input/output configuration
        proc_config (ProcessingConfig): Processing configuration
        spatial_tiling_config (SpatialTilingConfig): Spatial tiling configuration
        temporal_tiling_config (TemporalTilingConfig): Temporal tiling configuration
    """
    video_decoder = VideoDecoder(io_config.input_path, dimension_order="NHWC")

    if (
        temporal_tiling_config.enabled
        and video_decoder.metadata.num_frames > temporal_tiling_config.tile_size
    ):
        chunks = calculate_temporal_tile_ranges(
            video_decoder.metadata.num_frames,
            temporal_tiling_config.tile_size,
            temporal_tiling_config.tile_overlap,
        )

        prev_chunk_tail = None

        frame_counter = 0

        for chunk_idx, (start, end) in enumerate(tqdm(chunks, desc="Processing Temporal Chunks")):
            frames_chunk = (
                video_decoder.get_frames_in_range(start, end).data.to(torch.float16) / 255.0
            )
            current_chunk_output = process_single_temporal_chunk(
                pipe=pipe,
                frames_chunk=frames_chunk,
                proc_config=proc_config,
                spatial_config=spatial_tiling_config,
            )
            del frames_chunk
            clean_vram()

            if chunk_idx == 0:
                frames_to_write = (
                    current_chunk_output[: -temporal_tiling_config.tile_overlap]
                    if temporal_tiling_config.tile_overlap > 0
                    else current_chunk_output
                )
                prev_chunk_tail = (
                    current_chunk_output[-temporal_tiling_config.tile_overlap :]
                    if temporal_tiling_config.tile_overlap > 0
                    else None
                )
            else:
                overlap_head = current_chunk_output[: temporal_tiling_config.tile_overlap]
                blended_overlap = blend_temporal_overlap(
                    prev_chunk_tail, overlap_head, temporal_tiling_config.tile_overlap
                )

                rest_of_chunk = current_chunk_output[temporal_tiling_config.tile_overlap :]
                if chunk_idx == len(chunks) - 1:
                    frames_to_write = torch.cat([blended_overlap, rest_of_chunk], dim=0)
                    prev_chunk_tail = None
                else:
                    frames_to_write = (
                        torch.cat(
                            [
                                blended_overlap,
                                rest_of_chunk[: -temporal_tiling_config.tile_overlap],
                            ],
                            dim=0,
                        )
                        if temporal_tiling_config.tile_overlap > 0
                        else torch.cat([blended_overlap, rest_of_chunk], dim=0)
                    )
                    prev_chunk_tail = (
                        rest_of_chunk[-temporal_tiling_config.tile_overlap :]
                        if temporal_tiling_config.tile_overlap > 0
                        and rest_of_chunk.shape[0] >= temporal_tiling_config.tile_overlap
                        else rest_of_chunk
                    )

            if io_config.output_mode == OutputMode.VIDEO:
                encoder = VideoEncoder(
                    frames=rearrange(frames_to_write, "N H W C -> N C H W")
                    .clamp(0, 1)
                    .mul_(255)
                    .to(torch.uint8),
                    frame_rate=video_decoder.metadata.average_fps,
                )
                encoder.to_file(
                    os.path.join(io_config.output_dir, f"chunk_{chunk_idx:03d}.mp4"),
                    codec="libx264",
                    crf=0,
                    pixel_format="yuv420p",
                )
            elif io_config.output_mode == OutputMode.FRAMES:
                final_output = (
                    rearrange(frames_to_write, "N H W C -> N C H W")
                    .clamp(0, 1)
                    .mul_(255)
                    .to(torch.uint8)
                )
                N = final_output.shape[0]
                for i in range(N):
                    frame = final_output[i]
                    write_png(
                        frame,
                        os.path.join(io_config.output_dir, f"{frame_counter:08d}.png"),
                        compression_level=0,
                    )
                    frame_counter += 1

            del current_chunk_output, frames_to_write
            clean_vram()

    else:
        frames = (
            video_decoder.get_frames_in_range(0, video_decoder.metadata.num_frames).data.to(
                torch.float16
            )
            / 255.0
        )

        final_output = process_single_temporal_chunk(
            pipe=pipe,
            frames_chunk=frames,
            proc_config=proc_config,
            spatial_config=spatial_tiling_config,
        )

        if io_config.output_mode == OutputMode.VIDEO:
            encoder = VideoEncoder(
                frames=rearrange(final_output, "N H W C -> N C H W")
                .clamp(0, 1)
                .mul_(255)
                .to(torch.uint8),
                frame_rate=video_decoder.metadata.average_fps,
            )
            encoder.to_file(
                os.path.join(io_config.output_dir, "chunk_000.mp4"),
                codec="libx264",
                crf=0,
                pixel_format="yuv420p",
            )
        elif io_config.output_mode == OutputMode.FRAMES:
            final_output = (
                rearrange(final_output, "N H W C -> N C H W").clamp(0, 1).mul_(255).to(torch.uint8)
            )
            N = final_output.shape[0]
            for i in range(N):
                frame = final_output[i]
                write_png(
                    frame,
                    os.path.join(io_config.output_dir, f"{i:08d}.png"),
                    compression_level=0,
                )

        del frames, final_output
        clean_vram()
