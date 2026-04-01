# Gaussian Splat Shot Render (ComfyUI)

Custom node for composing camera shots over 3D Gaussian splatting (PLY), with an interactive WebGL scout viewer and CPU raster output that stay aligned.

## Install

1. Clone (or copy) this repository into your ComfyUI `custom_nodes` folder. The directory name can be anything; ComfyUI uses the folder name for extension URLs.

   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone <your-remote-url> gaussian-splat-shot-render-comfyui
   ```

2. Install Python extras used by the node (if your environment does not already have them):

   ```bash
   pip install -r gaussian-splat-shot-render-comfyui/requirements.txt
   ```

3. **Required:** Install **ComfyUI-GeometryPack** (or another extension that serves `gsplat-bundle.js` at `/extensions/ComfyUI-GeometryPack/js/gsplat-bundle.js`). The viewer tries a lowercase path as well for differently cased installs.

4. Restart ComfyUI and refresh the browser.

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
| `web/gaussian_shot.js` | Frontend: iframe viewer, widget sync, execution hooks |
| `web/viewer_gaussian_shot.html` | WebGL viewer (gsplat) and HUD |

## License

Add a `LICENSE` file when you publish this repository.
