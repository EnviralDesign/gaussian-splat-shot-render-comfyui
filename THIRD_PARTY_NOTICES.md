# Third-party notices

This extension bundles or relies on several third-party works. Keep this file
when you redistribute the repository or a derivative.

## Apple ML SHARP (Python, `sharp_nodes/sharp/`)

- **Source:** vendored from the ComfyUI adaptation in
  [comfyui-sharp](https://github.com/PozzettiAndrea/comfyui-sharp), which in
  turn tracks Apple’s [ml-sharp](https://github.com/apple/ml-sharp).
- **License:** **Apple Software License Agreement** (ASL). Full text:
  <https://github.com/apple/ml-sharp/blob/main/LICENSE>
- **Copyright:** Copyright (C) Apple Inc. and contributors.

## SHARP model weights

- **Artifact:** `sharp_2572gikvuh.pt` (downloaded at runtime via Hugging Face Hub
  from `apple/Sharp`, or loaded from a local path you provide).
- **License:** **Apple ML Research Model License**. Full text:
  <https://github.com/apple/ml-sharp/blob/main/LICENSE_MODEL>
- Use of the weights is subject to Apple’s model license in addition to any
  PyTorch / HF Hub terms.

## ComfyUI SHARP wrapper nodes (`sharp_nodes/load_model.py`, `sharp_nodes/predict.py`, `sharp_nodes/utils/image.py`)

- **Source:** adapted from
  [comfyui-sharp](https://github.com/PozzettiAndrea/comfyui-sharp) (same files
  under that project’s `nodes/` tree).
- **License:** **GNU General Public License v3.0** (GPL-3.0). A copy of the
  license is included in this repo as `licenses/GPL-3.0.txt` (or see
  <https://www.gnu.org/licenses/gpl-3.0.html>).
- If you distribute modified versions of these files, GPL-3.0 obligations apply
  to those distributions.

## WebGL Gaussian bundle (`web/js/gsplat-bundle.js`)

- **Origin:** copied from the **`comfy-3d-viewers`** PyPI distribution (viewer
  infrastructure also used by ComfyUI-GeometryPack). Upstream repo:
  <https://github.com/PozzettiAndrea/comfy-3d-viewers>
- **SPDX:** `GPL-3.0-or-later` (see package metadata on PyPI).
- **Full license text:** `licenses/GPL-3.0.txt` in this repository.

The viewer loads this file from the same extension folder as
`web/viewer_gaussian_shot.html` (via `import.meta.url`), so **ComfyUI-GeometryPack
is not required** for the WebGL preview path.

## huggingface-hub (runtime dependency)

- Used by `LoadSharpModel` to download checkpoint files when no local checkpoint
  path is provided.
- **Project:** <https://github.com/huggingface/huggingface_hub> — see that
  project for license terms (Apache-2.0).

## plyfile, numba, numpy (runtime dependencies)

- Standard PyPI packages; see each project for its license (typically BSD/MIT).

---

_This file is provided for attribution and compliance convenience and is not
legal advice._
