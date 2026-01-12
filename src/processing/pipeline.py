"""Pipeline initialization for FlashVSR."""

import os

import torch

from src.models.TCDecoder import build_tcdecoder
from src.models.utils import Causal_LQ4x_Proj
from src.pipelines.flashvsr_tiny import FlashVSRTinyPipeline
from src.models.model_manager import ModelManager


def init_pipeline(model: str, device: torch.device, dtype: torch.dtype) -> FlashVSRTinyPipeline:
    """Initialize FlashVSR pipeline with given model and device.

    Args:
        model (str): Model name.
        device (torch.device): Device to load the model on.
        dtype (torch.dtype): Data type for model weights.

    Returns: FlashVSRTinyPipeline: Initialized pipeline instance.

    Raises:
        RuntimeError: If model directory or required files do not exist.
    """
    model_path = os.path.join("models", model)
    if not os.path.exists(model_path):
        raise RuntimeError(
            f'Model directory does not exist!\nPlease save all weights to "{model_path}"'
        )

    ckpt_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(
            f'"diffusion_pytorch_model_streaming_dmd.safetensors" does not exist!\nPlease save it to "{model_path}"'
        )

    lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
    if not os.path.exists(lq_path):
        raise RuntimeError(f'"LQ_proj_in.ckpt" does not exist!\nPlease save it to "{model_path}"')

    tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
    if not os.path.exists(tcd_path):
        raise RuntimeError(f'"TCDecoder.ckpt" does not exist!\nPlease save it to "{model_path}"')

    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prompt_path = os.path.join(current_dir, "posi_prompt.pth")
    if not os.path.exists(prompt_path):
        raise RuntimeError(f'"posi_prompt.pth" does not exist!\nPlease save it to "{model_path}"')

    mm = ModelManager(torch_dtype=dtype, device="cpu")

    mm.load_models([ckpt_path])

    pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device)

    multi_scale_channels = [512, 256, 128, 128]
    pipe.TCDecoder = build_tcdecoder(
        new_channels=multi_scale_channels,
        device=device,
        dtype=dtype,
        new_latent_channels=16 + 768,
    )
    pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device), strict=False)
    pipe.TCDecoder.clean_mem()

    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(
        device, dtype=dtype
    )
    pipe.denoising_model().LQ_proj_in.load_state_dict(
        torch.load(lq_path, map_location="cpu"), strict=True
    )
    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.to(device, dtype=dtype)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt_path)
    pipe.load_models_to_device(["dit"])
    pipe.offload_model()

    return pipe
