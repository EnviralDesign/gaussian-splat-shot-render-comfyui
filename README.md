# Gaussian Splat Shot Render (ComfyUI)

Custom node for composing camera shots over 3D Gaussian splatting (PLY), with an interactive WebGL scout viewer and CPU raster output that stay aligned.

## Install

1. Clone (or copy) this repository into your ComfyUI `custom_nodes` folder. The directory name can be anything; ComfyUI uses the folder name for extension URLs.

   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone <your-remote-url> gaussian-splat-shot-render-comfyui
   ```

2. Install Python extras used by the nodes (if your environment does not already have them):

   ```bash
   pip install -r gaussian-splat-shot-render-comfyui/requirements.txt
   ```

3. Restart ComfyUI and refresh the browser.

This pack is **self-contained** for the combined workflow:

- **Load SHARP Model** / **SHARP Predict (Image to PLY)** — monocular Gaussian prediction (checkpoint loads from Hugging Face unless you pass a local path).
- **Gaussian Splat Shot Render** — aligned WebGL scout + CPU raster from the PLY + SHARP camera metadata.

The WebGL viewer uses **`web/js/gsplat-bundle.js`** shipped inside this repository (no ComfyUI-GeometryPack install required). Legacy GeometryPack URLs remain as a fallback in `viewer_gaussian_shot.html` only.

## Developing against a local ComfyUI tree

Keep this repo as the source of truth. Point your Comfy install at it with a junction/symlink or by cloning this repo directly into `custom_nodes`:

```powershell
# Windows example: symlink folder name kept short
cmd /c mklink /J "D:\ComfyUI\custom_nodes\gaussian-splat-shot-render-comfyui" "C:\repos\gaussian-splat-shot-render-comfyui"
```

## Layout

| Path | Role |
|------|------|
| `__init__.py` | ComfyUI node registration and `WEB_DIRECTORY` |
| `gaussian_shot_node.py` | `GaussianShotRender` node, raster, camera / variation math |
| `sharp_nodes/` | `LoadSharpModel`, `SharpPredict`, vendored Apple `sharp/` core |
| `web/gaussian_shot.js` | Frontend: iframe viewer, widget sync, execution hooks |
| `web/viewer_gaussian_shot.html` | WebGL viewer (gsplat) and HUD |
| `web/js/gsplat-bundle.js` | Vendored WebGL splat bundle (from `comfy-3d-viewers`, GPL-3.0-or-later) |

## Performance and memory

- **Why the raster feels slow:** The “capture” path is a **CPU** reference splatter (NumPy, optionally **Numba**-JIT). The inline viewer uses **WebGL** (`gsplat`) on the GPU and only moves splatted fragments—very different work, so realtime preview does not imply a fast final raster.
- **Speed:** Install **`numba`** (`pip install numba` or use `requirements.txt`). The first splat in a Comfy session may pause briefly while LLVM compiles; later runs in the same process stay fast. By default the JIT does **not** use Numba’s on-disk cache, so cloning the same venv or switching between `D:\…` and `C:\…` Comfy trees does not explode with `ModuleNotFoundError` for an old path. To opt into disk cache on a stable machine, set `GAUSSIAN_SHOT_NUMBA_DISK_CACHE=1`.
- **VRAM after a run:** This node does **not** run the splat raster on CUDA. Any **GPU** memory you still see is almost always **SHARP / other models**, **ComfyUI’s graph / preview cache** (it may place `IMAGE` tensors on the GPU), or the **browser** WebGL tab—not the PLY decode itself. **System RAM:** one decoded PLY is cached (path + mtime) so repeat runs skip disk parse; that cache can be large on big clouds.
- **Optional env:** `GAUSSIAN_SHOT_DISABLE_PLY_CACHE=1` — no in-process PLY decode cache (more I/O and CPU on every run, lower steady RAM). `GAUSSIAN_SHOT_DISABLE_NUMBA=1` — force the slow pure-Python splat loop (debug only). `GAUSSIAN_SHOT_NUMBA_DISK_CACHE=1` — enable Numba disk cache (faster first splat after restart if your install path is stable).

## License

See **`LICENSE`** (MIT for original project code) and **`THIRD_PARTY_NOTICES.md`**
(Apple SHARP, GPL-3.0 wrapper nodes, bundled `gsplat-bundle.js`, HF Hub, model weights).
