import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from plyfile import PlyData

try:
    import folder_paths

    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except Exception:
    COMFYUI_OUTPUT_FOLDER = None

_DISABLE_PLY_CACHE = os.environ.get("GAUSSIAN_SHOT_DISABLE_PLY_CACHE", "").strip().lower() in ("1", "true", "yes")
_DISABLE_NUMBA = os.environ.get("GAUSSIAN_SHOT_DISABLE_NUMBA", "").strip().lower() in ("1", "true", "yes")
# Disk cache binds to how/where Comfy loaded the module; another install path can break unpickle
# (ModuleNotFoundError for the old custom_nodes path). In-process compile is enough per Comfy run.
_NUMBA_DISK_CACHE = os.environ.get("GAUSSIAN_SHOT_NUMBA_DISK_CACHE", "").strip().lower() in ("1", "true", "yes")

_PLY_CACHE: dict[tuple[str, float], dict[str, np.ndarray]] = {}

try:
    if _DISABLE_NUMBA:
        raise ImportError("numba disabled via GAUSSIAN_SHOT_DISABLE_NUMBA")
    from numba import njit

    @njit(cache=_NUMBA_DISK_CACHE, fastmath=True)
    def _accumulate_splats_numba(
        image: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        colors: np.ndarray,
        opacities: np.ndarray,
        sigma2: np.ndarray,
        radius: np.ndarray,
        width: int,
        height: int,
    ) -> None:
        """In-place splat raster (same math as meshgrid path; compiled for speed)."""
        n = u.shape[0]
        for idx in range(n):
            r = int(math.ceil(float(radius[idx])))
            ui = float(u[idx])
            vi = float(v[idx])
            x0 = max(0, int(math.floor(ui - float(r))))
            x1 = min(width, int(math.ceil(ui + float(r) + 1.0)))
            y0 = max(0, int(math.floor(vi - float(r))))
            y1 = min(height, int(math.ceil(vi + float(r) + 1.0)))
            if x0 >= x1 or y0 >= y1:
                continue

            c00 = float(sigma2[idx, 0, 0])
            c01 = float(sigma2[idx, 0, 1])
            c10 = float(sigma2[idx, 1, 0])
            c11 = float(sigma2[idx, 1, 1])
            det2 = c00 * c11 - c01 * c10
            if det2 <= 1e-12:
                continue
            idet = 1.0 / det2
            inv00 = c11 * idet
            inv01 = -c01 * idet
            inv10 = -c10 * idet
            inv11 = c00 * idet
            inv_cross = inv01 + inv10

            opac = float(opacities[idx])
            cr = float(colors[idx, 0])
            cg = float(colors[idx, 1])
            cb = float(colors[idx, 2])

            max_al = 0.0
            for yy in range(y0, y1):
                dys = float(yy) - vi
                for xx in range(x0, x1):
                    dxs = float(xx) - ui
                    q = inv00 * dxs * dxs + inv_cross * dxs * dys + inv11 * dys * dys
                    al = opac * math.exp(-0.5 * q)
                    if al > 1.0:
                        al = 1.0
                    elif al < 0.0:
                        al = 0.0
                    if al > max_al:
                        max_al = al
            if max_al < 1e-4:
                continue

            for yy in range(y0, y1):
                dys = float(yy) - vi
                for xx in range(x0, x1):
                    dxs = float(xx) - ui
                    q = inv00 * dxs * dxs + inv_cross * dxs * dys + inv11 * dys * dys
                    al = opac * math.exp(-0.5 * q)
                    if al > 1.0:
                        al = 1.0
                    elif al < 0.0:
                        al = 0.0
                    om = 1.0 - al
                    image[yy, xx, 0] = image[yy, xx, 0] * om + cr * al
                    image[yy, xx, 1] = image[yy, xx, 1] * om + cg * al
                    image[yy, xx, 2] = image[yy, xx, 2] * om + cb * al

    _HAS_NUMBA_SPLATS = True
except ImportError:
    _accumulate_splats_numba = None  # type: ignore[misc, assignment]
    _HAS_NUMBA_SPLATS = False


def _decode_sh_to_rgb(sh0: np.ndarray) -> np.ndarray:
    coeff_degree0 = math.sqrt(1.0 / (4.0 * math.pi))
    return np.clip(sh0 * coeff_degree0 + 0.5, 0.0, 1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _quat_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Rotation matrix from unit quaternion in (x, y, z, w) / Hamilton convention."""
    norms = np.linalg.norm(quat_xyzw.astype(np.float64), axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    q = (quat_xyzw.astype(np.float64) / norms).astype(np.float32)
    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    mats = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    mats[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    mats[:, 0, 1] = 2.0 * (xy - wz)
    mats[:, 0, 2] = 2.0 * (xz + wy)
    mats[:, 1, 0] = 2.0 * (xy + wz)
    mats[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    mats[:, 1, 2] = 2.0 * (yz - wx)
    mats[:, 2, 0] = 2.0 * (xz - wy)
    mats[:, 2, 1] = 2.0 * (yz + wx)
    mats[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return mats


def _normalize(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        if fallback is None:
            return np.zeros_like(v, dtype=np.float32)
        return fallback.astype(np.float32, copy=True)
    return (v / norm).astype(np.float32)


def _euler_deg_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    ax = math.radians(rx)
    ay = math.radians(ry)
    az = math.radians(rz)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)

    rxm = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    rym = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rzm = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (rzm @ rym @ rxm).astype(np.float32)


def _orbit_basis_from_yp(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    return _euler_deg_to_matrix(pitch_deg, yaw_deg, 0.0)


def _apply_roll_to_basis(basis: np.ndarray, roll_deg: float) -> np.ndarray:
    roll_rad = math.radians(roll_deg)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    ref_right = (basis @ np.array([1.0, 0.0, 0.0], dtype=np.float32)).astype(np.float32)
    ref_down = (basis @ np.array([0.0, 1.0, 0.0], dtype=np.float32)).astype(np.float32)
    forward = (basis @ np.array([0.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)
    right = (ref_right * cr + ref_down * sr).astype(np.float32)
    down = (ref_down * cr - ref_right * sr).astype(np.float32)
    return np.array(
        [
            [right[0], down[0], forward[0]],
            [right[1], down[1], forward[1]],
            [right[2], down[2], forward[2]],
        ],
        dtype=np.float32,
    )


def _hash_unit(seed: int, stream: int) -> float:
    x = (int(seed) & 0xFFFFFFFF) ^ ((stream * 0x9E3779B9) & 0xFFFFFFFF)
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x / 4294967295.0


def _rand_in_range(seed: int, stream: int, vmin: float, vmax: float) -> float:
    """Uniform sample in [vmin, vmax] (inclusive-ish); constant if vmin == vmax."""
    t = _hash_unit(seed, stream)
    return float(vmin + (vmax - vmin) * t)


def _parse_interactive_state(state_text: str | None) -> dict[str, float]:
    default = {
        "pivot_x": 0.0,
        "pivot_y": 0.0,
        "pivot_z": 0.0,
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
        "distance": 0.0,
    }
    if not state_text:
        return default
    try:
        data = json.loads(state_text)
        if not isinstance(data, dict):
            return default
        for key in default:
            if key in data:
                default[key] = float(data[key])
        return default
    except Exception:
        return default


def _camera_position_from_extrinsics(extrinsics: np.ndarray | None) -> np.ndarray:
    if extrinsics is None or extrinsics.shape != (4, 4):
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    r = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    return (-r.T @ t).astype(np.float32)


def _yaw_pitch_from_position(position: np.ndarray, pivot: np.ndarray) -> tuple[float, float, float]:
    forward = _normalize(pivot - position, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    yaw = math.degrees(math.atan2(float(forward[0]), float(forward[2])))
    pitch = math.degrees(math.asin(float(np.clip(forward[1], -1.0, 1.0))))
    distance = float(np.linalg.norm(pivot - position))
    return yaw, pitch, distance


def _mat_mul_3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)


def _rotation_matrix_to_ypr_deg(rot: np.ndarray) -> tuple[float, float, float]:
    """Recover yaw/pitch/roll (degrees) consistent with _camera_axes / forward = rot @ e_z."""
    r = rot.astype(np.float64)
    f = r @ np.array([0.0, 0.0, 1.0])
    fn = float(np.linalg.norm(f))
    if fn < 1e-10:
        return 0.0, 0.0, 0.0
    f = f / fn
    yaw = math.degrees(math.atan2(float(f[0]), float(f[2])))
    pitch = math.degrees(math.asin(float(np.clip(f[1], -1.0, 1.0))))
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(f, world_up))) > 0.99:
        ref_right = np.array([1.0, 0.0, 0.0])
    else:
        ref_right = np.cross(world_up, f)
        ref_right = ref_right / max(float(np.linalg.norm(ref_right)), 1e-8)
    cam_r = r @ np.array([1.0, 0.0, 0.0])
    r_proj = cam_r - f * float(np.dot(cam_r, f))
    rn = float(np.linalg.norm(r_proj))
    if rn < 1e-8:
        roll = 0.0
    else:
        r_proj = r_proj / rn
        ref_proj = ref_right - f * float(np.dot(ref_right, f))
        refn = float(np.linalg.norm(ref_proj))
        if refn < 1e-8:
            roll = 0.0
        else:
            ref_proj = ref_proj / refn
            roll = math.degrees(
                math.atan2(float(np.dot(np.cross(ref_proj, r_proj), f)), float(np.dot(ref_proj, r_proj)))
            )
    return yaw, pitch, roll


def _camera_axes(yaw_deg: float, pitch_deg: float, roll_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = _apply_roll_to_basis(_orbit_basis_from_yp(yaw_deg, pitch_deg), roll_deg)
    right = rot @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    down = rot @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
    forward = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return right.astype(np.float32), down.astype(np.float32), forward.astype(np.float32)


def _view_rotation_from_ypr(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    right, down, forward = _camera_axes(yaw_deg, pitch_deg, roll_deg)
    return np.stack([right, down, forward], axis=0).astype(np.float32)


def _view_rotation_from_camera_state(state: dict[str, float]) -> np.ndarray:
    """3×3 world-from-view rows [right, down, forward] — same layout as _view_rotation_from_ypr."""
    if state.get("cam_r00") is None:
        return _view_rotation_from_ypr(
            float(state["yaw_deg"]),
            float(state["pitch_deg"]),
            float(state["roll_deg"]),
        )
    return np.array(
        [
            [float(state["cam_r00"]), float(state["cam_r01"]), float(state["cam_r02"])],
            [float(state["cam_r10"]), float(state["cam_r11"]), float(state["cam_r12"])],
            [float(state["cam_r20"]), float(state["cam_r21"]), float(state["cam_r22"])],
        ],
        dtype=np.float32,
    )


def _state_to_camera(state: dict[str, float]) -> dict[str, Any]:
    pivot = np.array([state["pivot_x"], state["pivot_y"], state["pivot_z"]], dtype=np.float32)
    cx = state.get("cam_pos_x")
    cy = state.get("cam_pos_y")
    cz = state.get("cam_pos_z")
    if cx is not None and cy is not None and cz is not None:
        position = np.array([float(cx), float(cy), float(cz)], dtype=np.float32)
    else:
        orbit_basis = _orbit_basis_from_yp(state["yaw_deg"], state["pitch_deg"])
        forward = (orbit_basis @ np.array([0.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)
        position = (pivot - forward * float(state["distance"])).astype(np.float32)
    return {
        "position": position,
        "pivot": pivot,
        "yaw_deg": float(state["yaw_deg"]),
        "pitch_deg": float(state["pitch_deg"]),
        "roll_deg": float(state["roll_deg"]),
        "distance": float(state["distance"]),
    }


def _camera_to_state(position: np.ndarray, pivot: np.ndarray, roll_deg: float = 0.0) -> dict[str, float]:
    yaw, pitch, distance = _yaw_pitch_from_position(position, pivot)
    return {
        "pivot_x": float(pivot[0]),
        "pivot_y": float(pivot[1]),
        "pivot_z": float(pivot[2]),
        "yaw_deg": float(yaw),
        "pitch_deg": float(pitch),
        "roll_deg": float(roll_deg),
        "distance": float(distance),
    }


def _scene_radius_from_xyz(xyz: np.ndarray) -> float:
    if xyz.size == 0:
        return 1.0
    bbox_size = xyz.max(axis=0) - xyz.min(axis=0)
    radius = max(float(np.linalg.norm(bbox_size) * 0.5), float(np.max(bbox_size) * 0.5), 1e-3)
    return max(radius, 0.5)


def _fit_distance_for_radius(radius: float, intrinsics: np.ndarray | None) -> float:
    if intrinsics is not None and intrinsics.shape == (3, 3):
        fx = max(float(intrinsics[0, 0]), 1e-3)
        fy = max(float(intrinsics[1, 1]), 1e-3)
        width = max(float(intrinsics[0, 2]) * 2.0, 1.0)
        height = max(float(intrinsics[1, 2]) * 2.0, 1.0)
        half_fov_x = math.atan(width / (2.0 * fx))
        half_fov_y = math.atan(height / (2.0 * fy))
        half_fov = max(min(half_fov_x, half_fov_y), math.radians(10.0))
        return max(radius / math.tan(half_fov) + radius * 1.25, radius * 2.5)
    return radius * 3.5


def _build_framed_state(auto_pivot: np.ndarray, xyz: np.ndarray, intrinsics: np.ndarray | None) -> dict[str, float]:
    radius = _scene_radius_from_xyz(xyz)
    distance = _fit_distance_for_radius(radius, intrinsics)
    position = auto_pivot + np.array([0.0, 0.0, -distance], dtype=np.float32)
    return _camera_to_state(position, auto_pivot, 0.0)


def _camera_front_fraction(camera_state: dict[str, float], xyz: np.ndarray) -> float:
    if xyz.size == 0:
        return 0.0
    cam = _state_to_camera(camera_state)
    cam_pos = cam["position"]
    r_wc = _view_rotation_from_camera_state(camera_state)
    cam_xyz = (xyz - cam_pos[None, :]) @ r_wc.T
    return float(np.mean(cam_xyz[:, 2] > 1e-3))


def _build_source_state(extrinsics: np.ndarray | None, auto_pivot: np.ndarray, xyz: np.ndarray, intrinsics: np.ndarray | None) -> dict[str, float]:
    framed_state = _build_framed_state(auto_pivot, xyz, intrinsics)
    position = _camera_position_from_extrinsics(extrinsics)
    if not np.all(np.isfinite(position)):
        return framed_state

    extrinsic_state = _camera_to_state(position, auto_pivot, 0.0)
    radius = _scene_radius_from_xyz(xyz)
    distance = float(extrinsic_state["distance"])
    front_fraction = _camera_front_fraction(extrinsic_state, xyz)

    if (
        distance < radius * 0.35
        or distance > radius * 200.0
        or front_fraction < 0.15
        or np.linalg.norm(position - auto_pivot) < 1e-5
    ):
        return framed_state
    return extrinsic_state


def _build_parameter_state(
    pivot: np.ndarray,
    cam_yaw_deg: float,
    cam_pitch_deg: float,
    cam_roll_deg: float,
    cam_distance: float,
    source_state: dict[str, float],
) -> dict[str, float]:
    if abs(cam_distance) > 1e-5:
        return {
            "pivot_x": float(pivot[0]),
            "pivot_y": float(pivot[1]),
            "pivot_z": float(pivot[2]),
            "yaw_deg": float(cam_yaw_deg),
            "pitch_deg": float(cam_pitch_deg),
            "roll_deg": float(cam_roll_deg),
            "distance": float(cam_distance),
        }

    state = dict(source_state)
    state["pivot_x"] = float(pivot[0])
    state["pivot_y"] = float(pivot[1])
    state["pivot_z"] = float(pivot[2])
    return state


def _apply_shot_variation(
    base_state: dict[str, float],
    pivot: np.ndarray,
    seed: int,
    rand_tx_min: float,
    rand_tx_max: float,
    rand_ty_min: float,
    rand_ty_max: float,
    rand_tz_min: float,
    rand_tz_max: float,
    rand_yaw_min: float,
    rand_yaw_max: float,
    rand_pitch_min: float,
    rand_pitch_max: float,
    rand_roll_min: float,
    rand_roll_max: float,
    rand_loc_pitch_min: float,
    rand_loc_pitch_max: float,
    rand_loc_yaw_min: float,
    rand_loc_yaw_max: float,
) -> dict[str, float]:
    """
    Locked-shot variation order (matches viewer):
    1) R_w = R_delta_world @ R_base with the same Rz·Ry·Rx(pitch,yaw,rolldeg) convention as the panel.
    2) R = R_w @ R_loc using only camera-local yaw/pitch jitter.
    3) Apply the single downstream roll jitter from rand_roll.
    4) Place on pivot orbit: position = pivot - forward(R_orbit)·distance, then add pure world XYZ translation.
    Stored yaw/pitch/roll are recovered from R for HUD; explicit cam_pos_* always set when non-idle.
    """
    dx = _rand_in_range(seed, 1, rand_tx_min, rand_tx_max)
    dy = _rand_in_range(seed, 2, rand_ty_min, rand_ty_max)
    dz = _rand_in_range(seed, 3, rand_tz_min, rand_tz_max)
    dyaw = _rand_in_range(seed, 4, rand_yaw_min, rand_yaw_max)
    dpitch = _rand_in_range(seed, 5, rand_pitch_min, rand_pitch_max)
    droll = _rand_in_range(seed, 6, rand_roll_min, rand_roll_max)
    lpitch = _rand_in_range(seed, 7, rand_loc_pitch_min, rand_loc_pitch_max)
    lyaw = _rand_in_range(seed, 8, rand_loc_yaw_min, rand_loc_yaw_max)

    base_yaw = float(base_state["yaw_deg"])
    base_pitch = float(base_state["pitch_deg"])
    base_roll = float(base_state.get("roll_deg", 0.0))
    base_dist = float(base_state["distance"])

    idle = (
        abs(dx) < 1e-8
        and abs(dy) < 1e-8
        and abs(dz) < 1e-8
        and abs(dyaw) < 1e-8
        and abs(dpitch) < 1e-8
        and abs(droll) < 1e-8
        and abs(lpitch) < 1e-8
        and abs(lyaw) < 1e-8
    )
    if idle:
        out = dict(base_state)
        for k in ("cam_pos_x", "cam_pos_y", "cam_pos_z"):
            out.pop(k, None)
        for i in range(3):
            for j in range(3):
                out.pop(f"cam_r{i}{j}", None)
        return out

    r_base = _orbit_basis_from_yp(base_yaw, base_pitch)
    r_world_delta = _orbit_basis_from_yp(dyaw, dpitch)
    r_mid = _mat_mul_3(r_world_delta, r_base)
    r_loc = _orbit_basis_from_yp(lyaw, lpitch)
    r_orbit = _mat_mul_3(r_mid, r_loc)
    r_final = _apply_roll_to_basis(r_orbit, base_roll + droll)

    forward = (r_orbit @ np.array([0.0, 0.0, 1.0], dtype=np.float64)).astype(np.float32)
    fn = float(np.linalg.norm(forward))
    if fn < 1e-10:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        forward = (forward / fn).astype(np.float32)

    pivot_f = pivot.astype(np.float32)
    orbit_pos = (pivot_f - forward * np.float32(base_dist)).astype(np.float32)
    t_world = np.array([dx, dy, dz], dtype=np.float32)
    pos1 = (orbit_pos + t_world).astype(np.float32)

    yaw1, pitch1, roll1 = _rotation_matrix_to_ypr_deg(r_final)
    chord = float(np.linalg.norm((pivot_f - pos1).astype(np.float64)))

    r_wc = r_final.T.astype(np.float32)

    out = dict(base_state)
    out["yaw_deg"] = yaw1
    out["pitch_deg"] = pitch1
    out["roll_deg"] = roll1
    out["distance"] = chord
    out["cam_pos_x"] = float(pos1[0])
    out["cam_pos_y"] = float(pos1[1])
    out["cam_pos_z"] = float(pos1[2])
    for i in range(3):
        for j in range(3):
            out[f"cam_r{i}{j}"] = float(r_wc[i, j])
    return out


def _resolve_intrinsics(
    intrinsics: np.ndarray | None,
    output_width: int,
    output_height: int,
    use_source_resolution: bool,
) -> tuple[int, int, float, float, float, float]:
    if intrinsics is not None and intrinsics.shape == (3, 3):
        source_width = max(int(round(float(intrinsics[0, 2] * 2.0))), 1)
        source_height = max(int(round(float(intrinsics[1, 2] * 2.0))), 1)
        if use_source_resolution:
            width = source_width
            height = source_height
        else:
            width = max(int(output_width), 1)
            height = max(int(output_height), 1)
        # Square-pixel pinhole: single focal length. Independent fx=u-scale and fy=v-scale with
        # fx/fy = width/height locked the horizontal and vertical *angular* FOV to be equal on every
        # aspect, which squashed the scene for non-square outputs and mismatched typical gsplat fx≈fy.
        f_ref = (float(intrinsics[0, 0]) + float(intrinsics[1, 1])) * 0.5
        min_src = min(source_width, source_height)
        min_out = min(width, height)
        f = f_ref * (min_out / max(min_src, 1))
        fx = f
        fy = f
        cx = width * 0.5
        cy = height * 0.5
        return width, height, fx, fy, cx, cy

    width = max(int(output_width), 1)
    height = max(int(output_height), 1)
    f = min(width, height) * 0.9
    fx = f
    fy = f
    cx = width * 0.5
    cy = height * 0.5
    return width, height, fx, fy, cx, cy


def _load_gaussian_ply(ply_path: str) -> dict[str, np.ndarray]:
    path = Path(ply_path)
    cache_key = (str(path), path.stat().st_mtime)
    if not _DISABLE_PLY_CACHE:
        cached = _PLY_CACHE.get(cache_key)
        if cached is not None:
            return cached

    plydata = PlyData.read(path)
    vertices = next(element for element in plydata.elements if element.name == "vertex")

    xyz = np.stack(
        [
            np.asarray(vertices["x"], dtype=np.float32),
            np.asarray(vertices["y"], dtype=np.float32),
            np.asarray(vertices["z"], dtype=np.float32),
        ],
        axis=1,
    )
    sh0 = np.stack(
        [
            np.asarray(vertices["f_dc_0"], dtype=np.float32),
            np.asarray(vertices["f_dc_1"], dtype=np.float32),
            np.asarray(vertices["f_dc_2"], dtype=np.float32),
        ],
        axis=1,
    )
    colors = _decode_sh_to_rgb(sh0).astype(np.float32)
    opacities = _sigmoid(np.asarray(vertices["opacity"], dtype=np.float32)).astype(np.float32)
    scales = np.exp(
        np.stack(
            [
                np.asarray(vertices["scale_0"], dtype=np.float32),
                np.asarray(vertices["scale_1"], dtype=np.float32),
                np.asarray(vertices["scale_2"], dtype=np.float32),
            ],
            axis=1,
        )
    ).astype(np.float32)
    # PLY / Graphdeco / gsplat: rot_0=w, rot_1=x, rot_2=y, rot_3=z — _quat_to_matrix expects (x,y,z,w).
    quats = np.stack(
        [
            np.asarray(vertices["rot_1"], dtype=np.float32),
            np.asarray(vertices["rot_2"], dtype=np.float32),
            np.asarray(vertices["rot_3"], dtype=np.float32),
            np.asarray(vertices["rot_0"], dtype=np.float32),
        ],
        axis=1,
    )

    rot_mats = _quat_to_matrix(quats)
    diag = np.zeros((xyz.shape[0], 3, 3), dtype=np.float32)
    diag[:, 0, 0] = scales[:, 0] ** 2
    diag[:, 1, 1] = scales[:, 1] ** 2
    diag[:, 2, 2] = scales[:, 2] ** 2
    covariances = np.einsum("nij,njk,nlk->nil", rot_mats, diag, rot_mats).astype(np.float32)

    result = {
        "xyz": xyz,
        "colors": colors,
        "opacities": opacities,
        "covariances": covariances,
    }
    if not _DISABLE_PLY_CACHE:
        _PLY_CACHE.clear()
        _PLY_CACHE[cache_key] = result
    return result


def _render_gaussians(
    ply_path: str,
    camera_state: dict[str, float],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    gaussian_scale: float,
    max_gaussians: int,
    background: str,
    ply_data: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    data = ply_data if ply_data is not None else _load_gaussian_ply(ply_path)
    xyz = data["xyz"]
    colors = data["colors"]
    opacities = data["opacities"]
    covariances = data["covariances"]

    cam = _state_to_camera(camera_state)
    cam_pos = cam["position"]
    r_wc = _view_rotation_from_camera_state(camera_state)

    rel = xyz - cam_pos[None, :]
    cam_xyz = rel @ r_wc.T
    z = cam_xyz[:, 2]
    visible = z > 1e-3
    if not np.any(visible):
        bg = np.zeros((height, width, 3), dtype=np.float32)
        if background == "white":
            bg.fill(1.0)
        return bg

    cam_xyz = cam_xyz[visible]
    colors = colors[visible]
    opacities = opacities[visible]
    covariances = covariances[visible]
    r_cov = np.einsum("ij,njk,lk->nil", r_wc, covariances, r_wc).astype(np.float32)

    x = cam_xyz[:, 0]
    y = cam_xyz[:, 1]
    z = cam_xyz[:, 2]
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy

    j = np.zeros((cam_xyz.shape[0], 2, 3), dtype=np.float32)
    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z
    j[:, 0, 0] = fx * inv_z
    j[:, 0, 2] = -fx * x * inv_z2
    j[:, 1, 1] = fy * inv_z
    j[:, 1, 2] = -fy * y * inv_z2

    sigma2 = np.einsum("nij,njk,nlk->nil", j, r_cov, j).astype(np.float32)
    sigma2 *= max(float(gaussian_scale), 1e-4) ** 2

    trace = sigma2[:, 0, 0] + sigma2[:, 1, 1]
    det = sigma2[:, 0, 0] * sigma2[:, 1, 1] - sigma2[:, 0, 1] * sigma2[:, 1, 0]
    det = np.clip(det, 1e-10, None)
    disc = np.sqrt(np.clip(trace * trace - 4.0 * det, 0.0, None))
    sigma_major = np.sqrt(np.clip((trace + disc) * 0.5, 1e-10, None))
    radius = np.clip(3.0 * sigma_major, 1.0, 96.0)

    in_frame = (
        (u + radius >= 0)
        & (u - radius < width)
        & (v + radius >= 0)
        & (v - radius < height)
        & (opacities > 1e-4)
    )
    if not np.any(in_frame):
        bg = np.zeros((height, width, 3), dtype=np.float32)
        if background == "white":
            bg.fill(1.0)
        return bg

    u = u[in_frame]
    v = v[in_frame]
    z = z[in_frame]
    colors = colors[in_frame]
    opacities = opacities[in_frame]
    sigma2 = sigma2[in_frame]
    radius = radius[in_frame]

    importance = opacities * radius * radius / np.maximum(z, 1e-4)
    if max_gaussians > 0 and importance.shape[0] > max_gaussians:
        keep = np.argpartition(importance, -max_gaussians)[-max_gaussians:]
        u = u[keep]
        v = v[keep]
        z = z[keep]
        colors = colors[keep]
        opacities = opacities[keep]
        sigma2 = sigma2[keep]
        radius = radius[keep]

    order = np.argsort(z)[::-1]
    u = u[order]
    v = v[order]
    colors = colors[order]
    opacities = opacities[order]
    sigma2 = sigma2[order]
    radius = radius[order]

    image = np.zeros((height, width, 3), dtype=np.float32)
    if background == "white":
        image.fill(1.0)

    if _HAS_NUMBA_SPLATS and _accumulate_splats_numba is not None:
        _accumulate_splats_numba(
            image,
            np.ascontiguousarray(u),
            np.ascontiguousarray(v),
            np.ascontiguousarray(colors),
            np.ascontiguousarray(opacities),
            np.ascontiguousarray(sigma2),
            np.ascontiguousarray(radius),
            width,
            height,
        )
    else:
        for idx in range(u.shape[0]):
            r = int(math.ceil(float(radius[idx])))
            x0 = max(0, int(math.floor(float(u[idx]) - r)))
            x1 = min(width, int(math.ceil(float(u[idx]) + r + 1)))
            y0 = max(0, int(math.floor(float(v[idx]) - r)))
            y1 = min(height, int(math.ceil(float(v[idx]) + r + 1)))
            if x0 >= x1 or y0 >= y1:
                continue

            cov = sigma2[idx]
            det2 = float(cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0])
            if det2 <= 1e-12:
                continue
            inv = np.array([[cov[1, 1], -cov[0, 1]], [-cov[1, 0], cov[0, 0]]], dtype=np.float32) / det2

            xs = np.arange(x0, x1, dtype=np.float32) - float(u[idx])
            ys = np.arange(y0, y1, dtype=np.float32) - float(v[idx])
            dx, dy = np.meshgrid(xs, ys)
            quad = inv[0, 0] * dx * dx + (inv[0, 1] + inv[1, 0]) * dx * dy + inv[1, 1] * dy * dy
            alpha = float(opacities[idx]) * np.exp(-0.5 * quad)
            alpha = np.clip(alpha, 0.0, 1.0)
            if float(alpha.max()) < 1e-4:
                continue

            patch = image[y0:y1, x0:x1]
            patch *= 1.0 - alpha[..., None]
            patch += colors[idx][None, None, :] * alpha[..., None]

    return np.clip(image, 0.0, 1.0)


def _make_ui_path(ply_path: str) -> str:
    filename = os.path.basename(ply_path)
    if COMFYUI_OUTPUT_FOLDER and ply_path.startswith(COMFYUI_OUTPUT_FOLDER):
        return os.path.relpath(ply_path, COMFYUI_OUTPUT_FOLDER)
    return filename


class GaussianShotRenderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_locked": ("BOOLEAN", {"default": False}),
                "pivot_x": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0e6,
                        "max": 1.0e6,
                        "step": 0.0001,
                        "tooltip": "Orbit / look-at pivot in scene units (PLY space).",
                    },
                ),
                "pivot_y": ("FLOAT", {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.0001}),
                "pivot_z": ("FLOAT", {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.0001}),
                "cam_yaw_deg": ("FLOAT", {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.01}),
                "cam_pitch_deg": ("FLOAT", {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.01}),
                "cam_roll_deg": ("FLOAT", {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.01}),
                "cam_distance": ("FLOAT", {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.0001}),
                "interactive_state": ("STRING", {"default": "{\"pivot_x\":0,\"pivot_y\":0,\"pivot_z\":0,\"yaw_deg\":0,\"pitch_deg\":0,\"roll_deg\":0,\"distance\":0}", "multiline": False}),
                "output_width": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 4096, "step": 64, "tooltip": "Raster width; viewer uses same aspect."},
                ),
                "output_height": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 4096, "step": 64, "tooltip": "Raster height; viewer uses same aspect."},
                ),
                "use_source_resolution": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If True and SHARP intrinsics are connected, render at the source image size (ignores width/height).",
                    },
                ),
                "show_viewer_hud": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Show the interactive viewer info overlay (camera, pivot, resolution).",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Locked-shot variation seed.",
                    },
                ),
                "rand_tx_min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0e6,
                        "max": 1.0e6,
                        "step": 0.0001,
                        "tooltip": "Locked shot: world-space X translation (scene units) added last, after orbit placement along composed view axis.",
                    },
                ),
                "rand_tx_max": ("FLOAT", {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.0001}),
                "rand_ty_min": ("FLOAT", {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.0001}),
                "rand_ty_max": ("FLOAT", {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.0001}),
                "rand_tz_min": ("FLOAT", {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.0001}),
                "rand_tz_max": ("FLOAT", {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.0001}),
                "rand_yaw_min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0e6,
                        "max": 1.0e6,
                        "step": 0.01,
                        "tooltip": "Locked shot: world-stage yaw (deg) as part of Rδ (Rz·Ry·Rx) left-applied after panel rotation.",
                    },
                ),
                "rand_yaw_max": ("FLOAT", {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.01}),
                "rand_pitch_min": ("FLOAT", {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.01}),
                "rand_pitch_max": ("FLOAT", {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.01}),
                "rand_roll_min": ("FLOAT", {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.01, "tooltip": "Final downstream roll jitter (deg)."}),
                "rand_roll_max": ("FLOAT", {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.01}),
                "rand_loc_pitch_min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0e6,
                        "max": 1.0e6,
                        "step": 0.01,
                        "tooltip": "Locked shot: camera-local pitch jitter (deg), right-applied after world Rδ (same Rz·Ry·Rx order in camera frame).",
                    },
                ),
                "rand_loc_pitch_max": ("FLOAT", {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.01}),
                "rand_loc_yaw_min": ("FLOAT", {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.01}),
                "rand_loc_yaw_max": ("FLOAT", {"default": 0.0, "min": -1.0e6, "max": 1.0e6, "step": 0.01}),
                "gaussian_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 64.0, "step": 0.1}),
                "max_gaussians": (
                    "INT",
                    {"default": 0, "min": 0, "max": 200000, "step": 1000, "tooltip": "0 = no cap (all in-frustum splats). >0 keeps top-N by importance."},
                ),
                "background": (["black", "white"], {"default": "black"}),
            },
            "optional": {
                "ply_path": ("STRING", {"forceInput": True, "tooltip": "Path to a Gaussian Splatting PLY file"}),
                "extrinsics": ("EXTRINSICS", {"tooltip": "Camera extrinsics from SHARP"}),
                "intrinsics": ("INTRINSICS", {"tooltip": "Camera intrinsics from SHARP"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render"
    CATEGORY = "Gaussian Splat Shot"
    OUTPUT_NODE = True

    def render(
        self,
        camera_locked: bool,
        pivot_x: float,
        pivot_y: float,
        pivot_z: float,
        cam_yaw_deg: float,
        cam_pitch_deg: float,
        cam_roll_deg: float,
        cam_distance: float,
        interactive_state: str,
        output_width: int,
        output_height: int,
        use_source_resolution: bool,
        show_viewer_hud: bool,
        seed: int,
        rand_tx_min: float,
        rand_tx_max: float,
        rand_ty_min: float,
        rand_ty_max: float,
        rand_tz_min: float,
        rand_tz_max: float,
        rand_yaw_min: float,
        rand_yaw_max: float,
        rand_pitch_min: float,
        rand_pitch_max: float,
        rand_roll_min: float,
        rand_roll_max: float,
        rand_loc_pitch_min: float,
        rand_loc_pitch_max: float,
        rand_loc_yaw_min: float,
        rand_loc_yaw_max: float,
        gaussian_scale: float,
        max_gaussians: int,
        background: str,
        ply_path=None,
        extrinsics=None,
        intrinsics=None,
        unique_id=None,
    ):
        if not ply_path:
            raise ValueError("No PLY path provided")
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        ply_data = _load_gaussian_ply(ply_path)
        xyz = ply_data["xyz"]
        auto_pivot = ((xyz.min(axis=0) + xyz.max(axis=0)) * 0.5).astype(np.float32)

        pivot = np.array([pivot_x, pivot_y, pivot_z], dtype=np.float32)

        extrinsics_np = None
        if extrinsics is not None:
            extrinsics_np = np.asarray(extrinsics, dtype=np.float32)
            if extrinsics_np.ndim == 3:
                extrinsics_np = extrinsics_np[0]
            if extrinsics_np.shape != (4, 4):
                extrinsics_np = None

        intrinsics_np = None
        if intrinsics is not None:
            intrinsics_np = np.asarray(intrinsics, dtype=np.float32)
            if intrinsics_np.ndim == 3:
                intrinsics_np = intrinsics_np[0]
            if intrinsics_np.shape != (3, 3):
                intrinsics_np = None

        source_state = _build_source_state(extrinsics_np, auto_pivot, xyz, intrinsics_np)
        parameter_state = _build_parameter_state(
            pivot=pivot,
            cam_yaw_deg=cam_yaw_deg,
            cam_pitch_deg=cam_pitch_deg,
            cam_roll_deg=cam_roll_deg,
            cam_distance=cam_distance,
            source_state=source_state,
        )
        interactive = _parse_interactive_state(interactive_state)

        if abs(interactive["distance"]) <= 1e-5:
            preview_state = dict(parameter_state)
        else:
            preview_state = dict(interactive)

        output_state = _apply_shot_variation(
            base_state=parameter_state,
            pivot=pivot,
            seed=seed,
            rand_tx_min=rand_tx_min,
            rand_tx_max=rand_tx_max,
            rand_ty_min=rand_ty_min,
            rand_ty_max=rand_ty_max,
            rand_tz_min=rand_tz_min,
            rand_tz_max=rand_tz_max,
            rand_yaw_min=rand_yaw_min,
            rand_yaw_max=rand_yaw_max,
            rand_pitch_min=rand_pitch_min,
            rand_pitch_max=rand_pitch_max,
            rand_roll_min=rand_roll_min,
            rand_roll_max=rand_roll_max,
            rand_loc_pitch_min=rand_loc_pitch_min,
            rand_loc_pitch_max=rand_loc_pitch_max,
            rand_loc_yaw_min=rand_loc_yaw_min,
            rand_loc_yaw_max=rand_loc_yaw_max,
        )

        active_state = dict(output_state if camera_locked else preview_state)

        output_width = max(int(output_width), 1)
        output_height = max(int(output_height), 1)
        width, height, fx, fy, cx, cy = _resolve_intrinsics(intrinsics_np, output_width, output_height, use_source_resolution)

        # Match the interactive viewer: scout mode uses preview/interactive camera; locked shot uses output (procedural).
        raster_camera_state = output_state if camera_locked else preview_state

        image = _render_gaussians(
            ply_path=ply_path,
            camera_state=raster_camera_state,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            gaussian_scale=gaussian_scale,
            max_gaussians=max_gaussians,
            background=background,
            ply_data=ply_data,
        )
        image = torch.from_numpy(image.astype(np.float32)[None, ...])

        ui_data = {
            "ply_file": [_make_ui_path(ply_path)],
            "filename": [os.path.basename(ply_path)],
            "camera_locked": [camera_locked],
            "active_camera_state": [json.dumps(active_state)],
            "preview_camera_state": [json.dumps(preview_state)],
            "raster_camera_state": [json.dumps(raster_camera_state)],
            "parameter_camera_state": [json.dumps(parameter_state)],
            "output_camera_state": [json.dumps(output_state)],
            "source_camera_state": [json.dumps(source_state)],
            "render_size": [json.dumps({"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy})],
            "gaussian_scale": [gaussian_scale],
            "max_gaussians": [max_gaussians],
            "unique_id": [str(unique_id) if unique_id is not None else ""],
        }
        return {"ui": ui_data, "result": (image,)}


NODE_CLASS_MAPPINGS = {"GaussianShotRender": GaussianShotRenderNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GaussianShotRender": "Gaussian Shot Render"}
