"""LoadSharpModel node for ComfyUI-Sharp."""

import os
import logging

import torch
from huggingface_hub import hf_hub_download

log = logging.getLogger("sharp")

# Try to get ComfyUI models directory
try:
    import folder_paths
    MODELS_DIR = os.path.join(folder_paths.models_dir, "sharp")
except ImportError:
    folder_paths = None
    MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "sharp")

SHARP_REPO_ID = "apple/Sharp"
SHARP_FILENAME = "sharp_2572gikvuh.pt"
WARM_READ_CHUNK_SIZE = 32 * 1024 * 1024


def _sharp_model_dirs():
    if folder_paths is not None:
        try:
            return folder_paths.get_folder_paths("sharp")
        except KeyError:
            pass
    return [MODELS_DIR]


def _resolve_sharp_checkpoint():
    for model_dir in _sharp_model_dirs():
        model_path = os.path.join(model_dir, SHARP_FILENAME)
        if os.path.exists(model_path):
            return model_path
    return None


def _is_external_model_path(model_path):
    model_drive = os.path.splitdrive(os.path.abspath(model_path))[0].upper()
    local_drive = os.path.splitdrive(os.path.abspath(MODELS_DIR))[0].upper()
    return bool(model_drive and model_drive != local_drive)


def _warm_file_cache(model_path):
    if not _is_external_model_path(model_path):
        return

    file_size = os.path.getsize(model_path)
    log.info(f"Warming SHARP checkpoint cache ({file_size / (1024 ** 3):.2f} GiB)")
    with open(model_path, "rb") as checkpoint:
        while checkpoint.read(WARM_READ_CHUNK_SIZE):
            pass


class LoadSharpModel:
    """Load SHARP model and wrap with ModelPatcher for ComfyUI-native VRAM management.

    The model is built once, cached by ComfyUI's execution cache, and stays in
    VRAM between runs (no repeated GPU transfers).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "precision": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Model precision. auto: best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."
                }),
                "checkpoint_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to .pt checkpoint. Leave empty to auto-download from Hugging Face."
                }),
            }
        }

    RETURN_TYPES = ("SHARP_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Gaussian Splat Shot"
    DESCRIPTION = "Load the SHARP model for monocular 3D Gaussian Splatting prediction."

    def load_model(self, precision: str = "auto", checkpoint_path: str = ""):
        """Build model, load weights, wrap with ModelPatcher."""
        import comfy.model_management
        import comfy.model_patcher
        import comfy.ops
        import comfy.utils
        from .sharp import PredictorParams, create_predictor

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        # Resolve dtype
        if precision == "auto":
            if comfy.model_management.should_use_bf16(load_device):
                dtype = torch.bfloat16
            elif comfy.model_management.should_use_fp16(load_device):
                dtype = torch.float16
            else:
                dtype = torch.float32
        elif precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Select optimal operations class (enables fp8, nvfp4, CublasOps, etc.)
        manual_cast_dtype = comfy.model_management.unet_manual_cast(dtype, load_device)
        operations = comfy.ops.pick_operations(dtype, manual_cast_dtype)

        # Resolve / download checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            model_path = checkpoint_path
        else:
            model_path = _resolve_sharp_checkpoint()
        if model_path is None:
            download_dir = _sharp_model_dirs()[0]
            os.makedirs(download_dir, exist_ok=True)
            model_path = hf_hub_download(
                repo_id=SHARP_REPO_ID,
                filename=SHARP_FILENAME,
                local_dir=download_dir,
            )
        _warm_file_cache(model_path)

        # Load state dict
        log.info(f"Loading checkpoint from {model_path}")
        state_dict = comfy.utils.load_torch_file(model_path)

        # Build model with native operations (disable_weight_init skips random weight init)
        log.info("Initializing model...")
        predictor = create_predictor(
            PredictorParams(),
            dtype=dtype,
            device=None,
            operations=operations,
        )

        # Load weights (assign=True efficiently replaces nn.Parameter objects)
        # Operations handle dtype casting at forward time — no .to(dtype) needed
        predictor.load_state_dict(state_dict, assign=True)
        predictor.eval()
        if comfy.model_management.force_channels_last():
            predictor.to(memory_format=torch.channels_last)
        comfy.model_management.archive_model_dtypes(predictor)
        log.info(f"Model ready ({dtype})")

        # Wrap with ModelPatcher — ComfyUI manages VRAM from here
        patcher = comfy.model_patcher.ModelPatcher(
            predictor,
            load_device=load_device,
            offload_device=offload_device,
        )

        return (patcher,)


NODE_CLASS_MAPPINGS = {
    "LoadSharpModel": LoadSharpModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSharpModel": "Load SHARP Model",
}
