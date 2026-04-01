"""Image conversion utilities for ComfyUI-Sharp."""

import numpy as np
import torch


def comfy_to_numpy_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert ComfyUI image tensor to numpy RGB array.

    Args:
        image_tensor: ComfyUI image tensor with shape [B, H, W, C] in range [0, 1].

    Returns:
        Numpy array with shape [H, W, C] in range [0, 255] as uint8.
    """
    if image_tensor.dim() == 4:
        img = image_tensor[0]  # Take first image from batch
    else:
        img = image_tensor

    img_np = img.cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)

    return img_np


def convert_focallength(width: float, height: float, f_mm: float = 30.0) -> float:
    """Convert focal length from mm (35mm film equivalent) to pixels.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        f_mm: Focal length in mm (35mm film equivalent). Default 30mm.

    Returns:
        Focal length in pixels.
    """
    # 35mm film dimensions: 36mm x 24mm
    return f_mm * np.sqrt(width**2 + height**2) / np.sqrt(36**2 + 24**2)
