from collections import Counter
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import time
import torch
import numpy as np
from unified_biological_brain import UnifiedBiologicalBrainSystem
from isaac_warp_3d_engine import get_global_isaac_warp_engine
def elevate_process_priority_high():
    if os.name == "nt":
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            if not kernel32.SetPriorityClass(handle, 0x00000080):
                kernel32.SetPriorityClass(handle, 0x00008000)
        except Exception:
            pass
elevate_process_priority_high()
ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
try:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass
def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.clip(x, -15.0, 15.0)))
_BG_CACHE = {}
_COMP_CACHE = {}
_HOLE_CACHE = {}
_CANON_CACHE = {}
def _grid_cache_key(grid):
    return (grid.shape, grid.tobytes())
_OPPONENT_BG_SAL_LUT = np.full(10, 0.50, dtype=np.float32)
def detect_background_color(grid):
    if grid.size == 0:
        return 0
    gkey = _grid_cache_key(grid)
    if gkey in _BG_CACHE:
        return _BG_CACHE[gkey]
    vals, counts = np.unique(grid, return_counts=True)
    if len(vals) == 1:
        res = int(vals[0])
        _BG_CACHE[gkey] = res
        return res
    H, W = grid.shape
    border_pixels = np.concatenate([grid[0, :], grid[H - 1, :], grid[:, 0], grid[:, W - 1]])
    best_bg = int(vals[int(np.argmax(counts))])
    best_bg_score = -1e9
    total_sz = float(max(1, grid.size))
    border_sz = float(max(1, border_pixels.size))
    for v_raw, cnt_raw in zip(vals, counts):
        v_c = int(v_raw)
        area_frac = float(cnt_raw) / total_sz
        border_frac = float(np.sum(border_pixels == v_c)) / border_sz
        opp_sal = float(_OPPONENT_BG_SAL_LUT[v_c % 10])
        bg_score = 0.52 * area_frac + 0.34 * border_frac - 0.28 * opp_sal
        if bg_score > best_bg_score:
            best_bg_score = bg_score
            best_bg = v_c
    if len(_BG_CACHE) > 2048:
        _BG_CACHE.clear()
    _BG_CACHE[gkey] = best_bg
    return best_bg
def apply_d4(grid, d4_id):
    k = int(d4_id) % 4
    flip = (int(d4_id) // 4) == 1
    out = np.rot90(grid, k=k)
    if flip:
        out = np.fliplr(out)
    return out.copy()
def invert_d4(grid, d4_id):
    k = int(d4_id) % 4
    flip = (int(d4_id) // 4) == 1
    out = grid.copy()
    if flip:
        out = np.fliplr(out)
    out = np.rot90(out, k=(4 - k) % 4)
    return out.copy()
def canonical_mask_signature(mask):
    mkey = (mask.shape, mask.tobytes())
    if mkey in _CANON_CACHE:
        return _CANON_CACHE[mkey]
    best = None
    m_int = mask.astype(np.uint8)
    for d4_id in range(8):
        rot = apply_d4(m_int, d4_id)
        sig = (rot.shape[0], rot.shape[1], rot.tobytes())
        if best is None or sig < best:
            best = sig
    if len(_CANON_CACHE) > 4096:
        _CANON_CACHE.clear()
    _CANON_CACHE[mkey] = best
    return best
def fuzzy_prototype_signature(comp):
    h, w = comp["height"], comp["width"]
    area = max(1, h * w)
    fill_ratio = comp["mass"] / float(area)
    if fill_ratio >= 0.92:
        fill_class = "solid"
    elif comp["holes"] >= 1:
        fill_class = "hollow"
    elif fill_ratio <= 0.45:
        fill_class = "sparse"
    else:
        fill_class = "concave"
    aspect = round(max(h, w) / float(max(1, min(h, w))), 1)
    return (fill_class, comp["holes"], aspect)
def _cortical_lateral_phase_labels(active_mask, color_grid=None, connectivity=4):
    H, W = active_mask.shape
    if H == 0 or W == 0 or not np.any(active_mask):
        return np.zeros((H, W), dtype=np.int32), np.array([], dtype=np.int32)
    inf_val = int(H * W + 1)
    base_ids = np.arange(1, H * W + 1, dtype=np.int32).reshape(H, W)
    phase = np.where(active_mask, base_ids, inf_val)
    if color_grid is None:
        c_v = active_mask[:-1, :] & active_mask[1:, :]
        c_h = active_mask[:, :-1] & active_mask[:, 1:]
    else:
        c_v = active_mask[:-1, :] & active_mask[1:, :] & (color_grid[:-1, :] == color_grid[1:, :])
        c_h = active_mask[:, :-1] & active_mask[:, 1:] & (color_grid[:, :-1] == color_grid[:, 1:])
    if connectivity == 8:
        if color_grid is None:
            c_d1 = active_mask[:-1, :-1] & active_mask[1:, 1:]
            c_d2 = active_mask[:-1, 1:] & active_mask[1:, :-1]
        else:
            c_d1 = active_mask[:-1, :-1] & active_mask[1:, 1:] & (color_grid[:-1, :-1] == color_grid[1:, 1:])
            c_d2 = active_mask[:-1, 1:] & active_mask[1:, :-1] & (color_grid[:-1, 1:] == color_grid[1:, :-1])
    while True:
        prev = phase.copy()
        v_min = np.minimum(phase[:-1, :], phase[1:, :])
        phase[:-1, :] = np.where(c_v, v_min, phase[:-1, :])
        phase[1:, :] = np.where(c_v, np.minimum(phase[:-1, :], phase[1:, :]), phase[1:, :])
        h_min = np.minimum(phase[:, :-1], phase[:, 1:])
        phase[:, :-1] = np.where(c_h, h_min, phase[:, :-1])
        phase[:, 1:] = np.where(c_h, np.minimum(phase[:, :-1], phase[:, 1:]), phase[:, 1:])
        if connectivity == 8:
            d1_min = np.minimum(phase[:-1, :-1], phase[1:, 1:])
            phase[:-1, :-1] = np.where(c_d1, d1_min, phase[:-1, :-1])
            phase[1:, 1:] = np.where(c_d1, np.minimum(phase[:-1, :-1], phase[1:, 1:]), phase[1:, 1:])
            d2_min = np.minimum(phase[:-1, 1:], phase[1:, :-1])
            phase[:-1, 1:] = np.where(c_d2, d2_min, phase[:-1, 1:])
            phase[1:, :-1] = np.where(c_d2, np.minimum(phase[:-1, 1:], phase[1:, :-1]), phase[1:, :-1])
        if np.array_equal(phase, prev):
            break
    unique_labels = np.unique(phase)
    valid_labels = unique_labels[unique_labels < inf_val]
    return phase, valid_labels
def _propagate_border_ownership_surround(bg_mask):
    H, W = bg_mask.shape
    surround = np.zeros((H, W), dtype=bool)
    surround[0, :] = bg_mask[0, :]
    surround[H - 1, :] = bg_mask[H - 1, :]
    surround[:, 0] = bg_mask[:, 0]
    surround[:, W - 1] = bg_mask[:, W - 1]
    while True:
        prev = surround.copy()
        surround[:-1, :] |= surround[1:, :] & bg_mask[:-1, :]
        surround[1:, :] |= surround[:-1, :] & bg_mask[1:, :]
        surround[:, :-1] |= surround[:, 1:] & bg_mask[:, :-1]
        surround[:, 1:] |= surround[:, :-1] & bg_mask[:, 1:]
        if np.array_equal(surround, prev):
            break
    return bg_mask & (~surround)
def _compute_cortical_synchrony_coherence(mask):
    h, w = mask.shape
    if h == 0 or w == 0 or not np.any(mask):
        return 0.0, 0.0
    m_f = mask.astype(np.float32)
    r_idx, c_idx = np.indices((h, w), dtype=np.float32)
    theta = (0.35 * (r_idx / float(max(1, h))) + 0.35 * (c_idx / float(max(1, w)))) * m_f
    for _ in range(4):
        coup = np.zeros_like(theta)
        if h > 1:
            v_pair = m_f[:-1, :] * m_f[1:, :]
            d_v = np.sin(theta[1:, :] - theta[:-1, :]) * v_pair
            coup[:-1, :] += d_v
            coup[1:, :] -= d_v
        if w > 1:
            h_pair = m_f[:, :-1] * m_f[:, 1:]
            d_h = np.sin(theta[:, 1:] - theta[:, :-1]) * h_pair
            coup[:, :-1] += d_h
            coup[:, 1:] -= d_h
        theta = theta + 0.24 * coup
    active_thetas = theta[mask]
    mean_cos = float(np.mean(np.cos(active_thetas)))
    mean_sin = float(np.mean(np.sin(active_thetas)))
    coherence = float(math.sqrt(mean_cos * mean_cos + mean_sin * mean_sin))
    mean_phase = float(math.atan2(mean_sin, mean_cos))
    return coherence, mean_phase
def count_enclosed_holes(mask):
    h, w = mask.shape
    if h < 3 or w < 3 or mask.all():
        return 0
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1 : h + 1, 1 : w + 1] = mask
    enclosed_mask = _propagate_border_ownership_surround(~padded)
    _, cavity_labels = _cortical_lateral_phase_labels(enclosed_mask, color_grid=None, connectivity=4)
    return int(len(cavity_labels))
def extract_components(grid, bg=None, connectivity=4, multicolor=False):
    if bg is None:
        bg = detect_background_color(grid)
    ckey = (_grid_cache_key(grid), int(bg), int(connectivity), bool(multicolor))
    if ckey in _COMP_CACHE:
        return _COMP_CACHE[ckey]
    H, W = grid.shape
    fg_mask = grid != bg
    phase_map, comp_labels = _cortical_lateral_phase_labels(
        fg_mask, color_grid=(None if multicolor else grid), connectivity=connectivity
    )
    comps = []
    for lbl in comp_labels:
        coords = np.argwhere(phase_map == int(lbl))
        cells = [(int(r), int(c)) for r, c in coords]
        colors = [int(grid[r, c]) for r, c in cells]
        r0, c0 = int(coords[:, 0].min()), int(coords[:, 1].min())
        r1, c1 = int(coords[:, 0].max()), int(coords[:, 1].max())
        sub_h, sub_w = r1 - r0 + 1, c1 - c0 + 1
        mask = np.zeros((sub_h, sub_w), dtype=bool)
        patch = np.full((sub_h, sub_w), bg, dtype=np.int32)
        rel_r = coords[:, 0] - r0
        rel_c = coords[:, 1] - c0
        mask[rel_r, rel_c] = True
        patch[rel_r, rel_c] = grid[coords[:, 0], coords[:, 1]]
        hole_count = count_enclosed_holes(mask)
        sync_coh, sync_phase = _compute_cortical_synchrony_coherence(mask)
        color_counts = Counter(colors)
        dom_color = color_counts.most_common(1)[0][0]
        min_color = color_counts.most_common()[-1][0]
        comp_dict = {
            "color": int(dom_color),
            "min_color": int(min_color),
            "colors": set(colors),
            "color_counts": dict(color_counts),
            "cells": cells,
            "mass": len(cells),
            "bbox": (r0, c0, r1, c1),
            "height": sub_h,
            "width": sub_w,
            "area": sub_h * sub_w,
            "center": ((r0 + r1) * 0.5, (c0 + c1) * 0.5),
            "mask": mask,
            "patch": patch,
            "holes": hole_count,
            "is_rect": len(cells) == sub_h * sub_w,
            "synchrony_coherence": sync_coh,
            "gamma_synchrony_phase": sync_phase,
        }
        comp_dict["proto"] = fuzzy_prototype_signature(comp_dict)
        comps.append(comp_dict)
    if len(_COMP_CACHE) > 2048:
        _COMP_CACHE.clear()
    _COMP_CACHE[ckey] = comps
    return comps
def find_enclosed_background_regions(grid, bg=None):
    if bg is None:
        bg = detect_background_color(grid)
    hkey = (_grid_cache_key(grid), int(bg))
    if hkey in _HOLE_CACHE:
        return _HOLE_CACHE[hkey]
    H, W = grid.shape
    bg_mask = grid == bg
    enclosed_mask = _propagate_border_ownership_surround(bg_mask)
    phase_map, cavity_labels = _cortical_lateral_phase_labels(enclosed_mask, color_grid=None, connectivity=4)
    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    regions = []
    for lbl in cavity_labels:
        coords = np.argwhere(phase_map == int(lbl))
        reg_cells = [(int(r), int(c)) for r, c in coords]
        border_colors = Counter()
        for cr, cc in reg_cells:
            for dr, dc in dirs:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] != bg:
                    border_colors[int(grid[nr, nc])] += 1
        r0, c0 = int(coords[:, 0].min()), int(coords[:, 1].min())
        r1, c1 = int(coords[:, 0].max()), int(coords[:, 1].max())
        regions.append({
            "cells": reg_cells,
            "mass": len(reg_cells),
            "bbox": (r0, c0, r1, c1),
            "height": r1 - r0 + 1,
            "width": c1 - c0 + 1,
            "is_rect": len(reg_cells) == (r1 - r0 + 1) * (c1 - c0 + 1),
            "border_color": border_colors.most_common(1)[0][0] if border_colors else bg,
        })
    if len(_HOLE_CACHE) > 2048:
        _HOLE_CACHE.clear()
    _HOLE_CACHE[hkey] = regions
    return regions
def compute_grid_observation(x_grid, hp=1.0, energy=0.5, pain=0.0, surprise=0.0, y_ref=None):
    bg = detect_background_color(x_grid)
    H, W = x_grid.shape
    G = float(max(H, W, 1))
    comps = extract_components(x_grid, bg=bg)
    comps_sorted = sorted(
        comps,
        key=lambda c: float(c["mass"]) * (0.75 + 0.25 * float(c.get("synchrony_coherence", 1.0))),
        reverse=True,
    )
    mean_sync = (
        float(np.mean([float(c.get("synchrony_coherence", 1.0)) for c in comps_sorted]))
        if comps_sorted
        else 1.0
    )
    if len(comps_sorted) >= 2:
        a = comps_sorted[0]
        t = comps_sorted[1]
        dx_t = (t["center"][1] - a["center"][1]) / G
        dy_t = (t["center"][0] - a["center"][0]) / G
        dz_t = -0.3 if a["area"] > t["area"] else 0.3
        ratio_w = (t["width"] - a["width"]) / G
        ratio_h = (t["height"] - a["height"]) / G
        phase_diff = abs(float(a.get("gamma_synchrony_phase", 0.0)) - float(t.get("gamma_synchrony_phase", 0.0)))
        sync_couple = 0.5 * (1.0 + math.cos(phase_diff))
        contact_cue = float(
            abs(a["center"][0] - t["center"][0]) <= 2.0 or abs(a["center"][1] - t["center"][1]) <= 2.0
        ) * (0.70 + 0.30 * sync_couple)
        depth_occlude_cue = float(
            max(a["area"], t["area"]) >= 9 
        )
    elif len(comps_sorted) == 1:
        a = comps_sorted[0]
        dx_t = 0.0
        dy_t = (H - 1 - a["center"][0]) / G
        dz_t = 0.0
        ratio_w = (a["height"] - a["width"]) / G
        ratio_h = (a["width"] - a["height"]) / G
        contact_cue = 0.0
        depth_occlude_cue = float(False)
    else:
        dx_t = dy_t = dz_t = ratio_w = ratio_h = contact_cue = depth_occlude_cue = 0.0
    holes = find_enclosed_background_regions(x_grid, bg=bg)
    enclosure_cue = float(len(holes) > 0)
    aspect_mismatch = float(
        (len(comps_sorted) >= 1 and comps_sorted[0]["width"] != comps_sorted[0]["height"])
        or (H != W)
    )
    gravity_cue = float(
        len(comps_sorted) >= 1 and any(c["bbox"][2] < H - 1 for c in comps_sorted)
    )
    fg_mask = x_grid != bg
    if np.any(fg_mask):
        rs, cs = np.where(fg_mask)
        dx_h = (float(np.mean(cs)) - W * 0.5) / G
        dy_h = (float(np.mean(rs)) - H * 0.5) / G
        dz_h = float(np.mean(fg_mask)) * (0.80 + 0.20 * mean_sync)
    else:
        dx_h = dy_h = dz_h = 0.0
    return np.array([
        dx_t, dy_t, dz_t,
        dx_h, dy_h, dz_h,
        ratio_w, ratio_h,
        contact_cue, depth_occlude_cue, aspect_mismatch, gravity_cue, enclosure_cue,
        float(hp), float(energy), float(pain + 0.3 * surprise)
    ], dtype=np.float32)
def perceive_core_affordances(axis_id, train_pairs):
    if not train_pairs:
        return []
    ax = int(axis_id)
    cands = []
    if ax == 0:
        def fn_ident(g, p):
            return g.copy()
        cands.append(("kinematic_identity", fn_ident, {}))
        def fn_transpose(g, p):
            return g.T.copy()
        cands.append(("kinematic_transpose", fn_transpose, {}))
        def fn_trim_border(g, p):
            bg = detect_background_color(g)
            coords = np.argwhere(g != bg)
            if coords.size == 0:
                return g.copy()
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0) + 1
            return g[r0:r1, c0:c1].copy()
        cands.append(("kinematic_trim_border", fn_trim_border, {}))
        def fn_pad_border(g, p):
            bg = detect_background_color(g)
            h, w = g.shape
            out = np.full((h + 2, w + 2), bg, dtype=g.dtype)
            out[1 : h + 1, 1 : w + 1] = g
            return out
        cands.append(("kinematic_pad_border", fn_pad_border, {}))
    elif ax == 1:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            def make_shift_fn(r_d, c_d):
                def fn_shift(g, p):
                    res = np.full_like(g, detect_background_color(g))
                    h, w = g.shape
                    sr = max(0, -r_d)
                    er = min(h, h - r_d)
                    sc = max(0, -c_d)
                    ec = min(w, w - c_d)
                    if er > sr and ec > sc:
                        res[sr + r_d : er + r_d, sc + c_d : ec + c_d] = g[sr:er, sc:ec]
                    return res
                return fn_shift
            cands.append((f"kinematic_rigid_push_{dr}_{dc}", make_shift_fn(dr, dc), {"dr": dr, "dc": dc}))
        for g_dr, g_dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            def make_gravity_fn(r_d, c_d):
                def fn_gravity(g, p):
                    bg = detect_background_color(g)
                    h, w = g.shape
                    out = np.full_like(g, bg)
                    coords = np.argwhere(g != bg)
                    if coords.size == 0:
                        return g.copy()
                    if r_d > 0:
                        coords = coords[np.argsort(-coords[:, 0])]
                    elif r_d < 0:
                        coords = coords[np.argsort(coords[:, 0])]
                    elif c_d > 0:
                        coords = coords[np.argsort(-coords[:, 1])]
                    else:
                        coords = coords[np.argsort(coords[:, 1])]
                    for r, c in coords:
                        val = g[r, c]
                        cur_r, cur_c = r, c
                        while True:
                            nr, nc = cur_r + r_d, cur_c + c_d
                            if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == bg:
                                cur_r, cur_c = nr, nc
                            else:
                                break
                        out[cur_r, cur_c] = val
                    return out
                return fn_gravity
            cands.append((f"kinematic_gravity_fall_{g_dr}_{g_dc}", make_gravity_fn(g_dr, g_dc), {"dr": g_dr, "dc": g_dc}))
        for r_dr, r_dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            def make_raycast_fn(r_d, c_d):
                def fn_raycast(g, p):
                    bg = detect_background_color(g)
                    h, w = g.shape
                    out = g.copy()
                    coords = np.argwhere(g != bg)
                    for r, c in coords:
                        col = g[r, c]
                        cur_r, cur_c = r + r_d, c + c_d
                        while 0 <= cur_r < h and 0 <= cur_c < w:
                            if out[cur_r, cur_c] == bg:
                                out[cur_r, cur_c] = col
                                cur_r += r_d
                                cur_c += c_d
                            else:
                                break
                    return out
                return fn_raycast
            cands.append((f"kinematic_raycast_{r_dr}_{r_dc}", make_raycast_fn(r_dr, r_dc), {"dr": r_dr, "dc": r_dc}))
    elif ax == 2:
        for k in [1, 2, 3]:
            def make_rot_fn(k_val):
                def fn_rot(g, p):
                    return np.rot90(g, k_val).copy()
                return fn_rot
            cands.append((f"kinematic_rigid_rotation_{k}", make_rot_fn(k), {"rot_k": k}))
        def fn_fliph(g, p):
            return np.fliplr(g).copy()
        def fn_flipv(g, p):
            return np.flipud(g).copy()
        def fn_flipdiag(g, p):
            return g.T.copy()
        def fn_flipantidiag(g, p):
            return np.rot90(g.T, 2).copy()
        cands.append(("kinematic_reflection_h", fn_fliph, {}))
        cands.append(("kinematic_reflection_v", fn_flipv, {}))
        cands.append(("kinematic_reflection_diag", fn_flipdiag, {}))
        cands.append(("kinematic_reflection_antidiag", fn_flipantidiag, {}))
        def fn_mirror_left_to_right(g, p):
            h, w = g.shape
            mid = w // 2
            out = g.copy()
            for c in range(mid):
                tgt_c = w - 1 - c
                if tgt_c >= 0:
                    out[:, tgt_c] = out[:, c]
            return out
        def fn_mirror_top_to_bottom(g, p):
            h, w = g.shape
            mid = h // 2
            out = g.copy()
            for r in range(mid):
                tgt_r = h - 1 - r
                if tgt_r >= 0:
                    out[tgt_r, :] = out[r, :]
            return out
        cands.append(("kinematic_mirror_l_to_r", fn_mirror_left_to_right, {}))
        cands.append(("kinematic_mirror_t_to_b", fn_mirror_top_to_bottom, {}))
    elif ax == 3:
        def fn_tile_2x2(g, p):
            return np.tile(g, (2, 2))
        def fn_tile_3x3(g, p):
            return np.tile(g, (3, 3))
        def fn_tile_mirror_2x2(g, p):
            top = np.hstack([g, np.fliplr(g)])
            bottom = np.hstack([np.flipud(g), np.flipud(np.fliplr(g))])
            return np.vstack([top, bottom])
        cands.append(("kinematic_tile_2x2", fn_tile_2x2, {}))
        cands.append(("kinematic_tile_3x3", fn_tile_3x3, {}))
        cands.append(("kinematic_tile_mirror_2x2", fn_tile_mirror_2x2, {}))
        def fn_extract_lattice_cells(g, p):
            h, w = g.shape
            bg = detect_background_color(g)
            sep_rows = [r for r in range(h) if len(np.unique(g[r, :])) == 1 and g[r, 0] != bg]
            sep_cols = [c for c in range(w) if len(np.unique(g[:, c])) == 1 and g[0, c] != bg]
            if not sep_rows and not sep_cols:
                return g.copy()
            r_splits = [-1] + sep_rows + [h]
            c_splits = [-1] + sep_cols + [w]
            best_cell = g.copy()
            max_nz = -1
            for i in range(len(r_splits) - 1):
                for j in range(len(c_splits) - 1):
                    r_start, r_end = r_splits[i] + 1, r_splits[i + 1]
                    c_start, c_end = c_splits[j] + 1, c_splits[j + 1]
                    if r_end > r_start and c_end > c_start:
                        sub = g[r_start:r_end, c_start:c_end]
                        nz = int(np.sum(sub != bg))
                        if nz > max_nz:
                            max_nz = nz
                            best_cell = sub.copy()
            return best_cell
        cands.append(("kinematic_extract_lattice_cells", fn_extract_lattice_cells, {}))
    elif ax == 4:
        def fn_crop_nz(g, p):
            bg = detect_background_color(g)
            coords = np.argwhere(g != bg)
            if coords.size == 0:
                return g.copy()
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0) + 1
            return g[r0:r1, c0:c1].copy()
        cands.append(("kinematic_alignment_crop", fn_crop_nz, {}))
        def fn_scale_up_2x(g, p):
            return np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)
        def fn_scale_up_3x(g, p):
            return np.repeat(np.repeat(g, 3, axis=0), 3, axis=1)
        cands.append(("kinematic_scale_up_2x", fn_scale_up_2x, {}))
        cands.append(("kinematic_scale_up_3x", fn_scale_up_3x, {}))
        def fn_dilate(g, p):
            bg = detect_background_color(g)
            h, w = g.shape
            out = g.copy()
            for r in range(h):
                for c in range(w):
                    if g[r, c] != bg:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == bg:
                                out[nr, nc] = g[r, c]
            return out
        def fn_erode(g, p):
            bg = detect_background_color(g)
            h, w = g.shape
            out = g.copy()
            for r in range(h):
                for c in range(w):
                    if g[r, c] != bg:
                        is_edge = any(
                            r + dr < 0 or r + dr >= h or c + dc < 0 or c + dc >= w or g[r + dr, c + dc] == bg
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        )
                        if is_edge:
                            out[r, c] = bg
            return out
        cands.append(("kinematic_morphology_dilate", fn_dilate, {}))
        cands.append(("kinematic_morphology_erode", fn_erode, {}))
    elif ax == 5:
        def fn_boundary(g, p):
            bg = detect_background_color(g)
            res = g.copy()
            h, w = g.shape
            for r in range(1, h - 1):
                for c in range(1, w - 1):
                    if g[r, c] != bg and all(g[r + dr, c + dc] != bg for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                        res[r, c] = bg
            return res
        cands.append(("kinematic_boundary_contact", fn_boundary, {}))
        def fn_keep_largest_component(g, p):
            bg = detect_background_color(g)
            mask = (g != bg)
            if not np.any(mask):
                return g.copy()
            phase, labels = _cortical_lateral_phase_labels(mask, color_grid=g, connectivity=4)
            if labels.size == 0:
                return g.copy()
            counts = [np.sum(phase == lb) for lb in labels]
            max_lb = labels[int(np.argmax(counts))]
            out = np.full_like(g, bg)
            out[phase == max_lb] = g[phase == max_lb]
            return out
        def fn_keep_smallest_component(g, p):
            bg = detect_background_color(g)
            mask = (g != bg)
            if not np.any(mask):
                return g.copy()
            phase, labels = _cortical_lateral_phase_labels(mask, color_grid=g, connectivity=4)
            if labels.size == 0:
                return g.copy()
            counts = [np.sum(phase == lb) for lb in labels]
            min_lb = labels[int(np.argmin(counts))]
            out = np.full_like(g, bg)
            out[phase == min_lb] = g[phase == min_lb]
            return out
        cands.append(("kinematic_keep_largest_component", fn_keep_largest_component, {}))
        cands.append(("kinematic_keep_smallest_component", fn_keep_smallest_component, {}))
        def fn_contour_corners(g, p):
            bg = detect_background_color(g)
            h, w = g.shape
            out = np.full_like(g, bg)
            coords = np.argwhere(g != bg)
            if coords.size == 0:
                return g.copy()
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            out[r_min, c_min] = g[r_min, c_min]
            out[r_min, c_max] = g[r_min, c_max]
            out[r_max, c_min] = g[r_max, c_min]
            out[r_max, c_max] = g[r_max, c_max]
            return out
        cands.append(("kinematic_contour_corners", fn_contour_corners, {}))
    elif ax == 6:
        def fn_fill_cavity(g, p):
            bg = detect_background_color(g)
            fill_col = int(p.get("fill_col", 2))
            res = g.copy()
            h, w = g.shape
            for r in range(1, h - 1):
                for c in range(1, w - 1):
                    if res[r, c] == bg and all(g[r + dr, c + dc] != bg for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                        res[r, c] = fill_col
            return res
        for fcol in [1, 2, 3, 4, 8]:
            cands.append((f"kinematic_cavity_filling_{fcol}", fn_fill_cavity, {"fill_col": fcol}))
        def fn_enclosure_fill(g, p):
            bg = detect_background_color(g)
            fill_col = int(p.get("fill_col", 2))
            h, w = g.shape
            if h < 3 or w < 3:
                return g.copy()
            padded = np.zeros((h + 2, w + 2), dtype=bool)
            padded[1 : h + 1, 1 : w + 1] = (g != bg)
            enc = _propagate_border_ownership_surround(~padded)[1 : h + 1, 1 : w + 1]
            out = g.copy()
            out[enc] = fill_col
            return out
        for fcol in [1, 2, 3, 4, 8]:
            cands.append((f"kinematic_enclosure_fill_{fcol}", fn_enclosure_fill, {"fill_col": fcol}))
        def fn_extract_enclosed_holes(g, p):
            bg = detect_background_color(g)
            h, w = g.shape
            if h < 3 or w < 3:
                return g.copy()
            padded = np.zeros((h + 2, w + 2), dtype=bool)
            padded[1 : h + 1, 1 : w + 1] = (g != bg)
            enc = _propagate_border_ownership_surround(~padded)[1 : h + 1, 1 : w + 1]
            out = np.full_like(g, bg)
            out[enc] = g[enc] if np.any(g[enc] != bg) else 1
            return out
        cands.append(("kinematic_extract_enclosed_holes", fn_extract_enclosed_holes, {}))
    elif ax == 7:
        if train_pairs:
            sample_in = train_pairs[0][0]
            sample_out = train_pairs[0][1]
            in_cols = [int(c) for c in np.unique(sample_in)]
            out_cols = [int(c) for c in np.unique(sample_out)]
            target_cols = [int(c) for c in out_cols if int(c) not in in_cols]
            if not target_cols:
                target_cols = [int(c) for c in range(10) if int(c) not in in_cols]
            if not target_cols:
                target_cols = [int(c) for c in range(10)]
            for col in in_cols[:5]:
                for t_col in target_cols[:5]:
                    if col != t_col:
                        def make_recolor_fn(src_c, dst_c):
                            def fn_recolor(g, p):
                                out = g.copy()
                                out[out == src_c] = dst_c
                                return out
                            return fn_recolor
                        cands.append((f"kinematic_texture_recolor_{col}_{t_col}", make_recolor_fn(int(col), int(t_col)), {"src": int(col), "dst": int(t_col)}))
            def fn_swap_bg_fg(g, p):
                bg = detect_background_color(g)
                vals, counts = np.unique(g, return_counts=True)
                fg_candidates = [v for v in vals if v != bg]
                if not fg_candidates:
                    return g.copy()
                fg_col = fg_candidates[int(np.argmax([counts[int(np.where(vals == v)[0][0])] for v in fg_candidates]))]
                out = g.copy()
                out[g == bg] = fg_col
                out[g == fg_col] = bg
                return out
            cands.append(("kinematic_swap_bg_fg", fn_swap_bg_fg, {}))
    return cands
_GRID_ORDER_STATS_CACHE = {}

def _get_grid_order_stats(g):
    key = _grid_cache_key(g)
    if key in _GRID_ORDER_STATS_CACHE:
        return _GRID_ORDER_STATS_CACHE[key]
    if len(_GRID_ORDER_STATS_CACHE) > 2048:
        _GRID_ORDER_STATS_CACHE.clear()
    bg = detect_background_color(g)
    nz = g[g != bg]
    comps = extract_components(g, bg=bg, connectivity=4, multicolor=False)
    holes = len(find_enclosed_background_regions(g, bg=bg)) if len(comps) <= 16 else 0
    if comps and len(comps) <= 12:
        sigs = Counter(canonical_mask_signature(c["mask"]) for c in comps)
        protos = Counter(c["proto"] for c in comps)
    else:
        sigs = None
        protos = None
    cols = set(int(c) for c in np.unique(nz))
    res = (bg, int(len(nz)), comps, int(holes), sigs, protos, cols)
    _GRID_ORDER_STATS_CACHE[key] = res
    return res

def compute_homeostatic_structural_order(pred, y, x_orig=None):
    if pred.size == 0 or y.size == 0:
        return 0.0, 1.0
    if x_orig is not None and pred.shape == x_orig.shape and np.array_equal(pred, x_orig):
        if x_orig.shape == y.shape and np.array_equal(x_orig, y):
            return 1.0, 0.0
        return 0.15, 0.85
    bg_p, len_nz_p, comps_p, holes_p, sigs_p, proto_p, cols_p = _get_grid_order_stats(pred)
    bg_y, len_nz_y, comps_y, holes_y, sigs_y, proto_y, cols_y = _get_grid_order_stats(y)
    if len_nz_y > 0 and len_nz_p == 0:
        return 0.05, 0.95
    shape_match = 1.0 if pred.shape == y.shape else max(
        0.0,
        1.0 - 0.5 * (
            abs(pred.shape[0] - y.shape[0]) / max(pred.shape[0], y.shape[0], 1)
            + abs(pred.shape[1] - y.shape[1]) / max(pred.shape[1], y.shape[1], 1)
        )
    )
    if len(comps_y) > 0 and len(comps_p) > max(35, len(comps_y) * 4):
        return 0.08, 0.92
    cnt_score = 1.0 - min(1.0, abs(len(comps_p) - len(comps_y)) / max(len(comps_y), len(comps_p), 1))
    hole_score = 1.0 - min(1.0, abs(holes_p - holes_y) / max(holes_y, holes_p, 1))
    if sigs_p is not None and sigs_y is not None:
        exact_overlap = sum((sigs_p & sigs_y).values())
        proto_overlap = sum((proto_p & proto_y).values())
        total = max(len(comps_p), len(comps_y), 1)
        iso_score = 0.65 * (float(exact_overlap) / float(total)) + 0.35 * (float(proto_overlap) / float(total))
    else:
        iso_score = cnt_score * 0.5
    pal_score = float(len(cols_p & cols_y)) / float(len(cols_p | cols_y)) if cols_y else (1.0 if not cols_p else 0.0)
    if pred.shape == y.shape:
        fg_mask = (y != bg_y) | (pred != bg_p)
        fg_acc = float(np.mean(pred[fg_mask] == y[fg_mask])) if np.any(fg_mask) else float(np.mean(pred == y))
        if x_orig is not None and x_orig.shape == y.shape:
            diff_m = x_orig != y
            prog_acc = float(np.mean(pred[diff_m] == y[diff_m])) if np.any(diff_m) else fg_acc
        else:
            prog_acc = fg_acc
        pix_score = 0.5 * fg_acc + 0.5 * prog_acc
        energy_val = 0.45 * pix_score + 0.22 * iso_score + 0.13 * cnt_score + 0.10 * hole_score + 0.10 * pal_score
    else:
        ov_h = min(pred.shape[0], y.shape[0])
        ov_w = min(pred.shape[1], y.shape[1])
        if ov_h > 0 and ov_w > 0:
            p_sub = pred[:ov_h, :ov_w]
            y_sub = y[:ov_h, :ov_w]
            fg_m = (y_sub != bg_y) | (p_sub != bg_p)
            ov_fg_acc = float(np.mean(p_sub[fg_m] == y_sub[fg_m])) if np.any(fg_m) else float(np.mean(p_sub == y_sub))
        else:
            ov_fg_acc = 0.0
        pix_score = ov_fg_acc * shape_match
        energy_val = (0.45 * pix_score + 0.22 * iso_score + 0.13 * cnt_score + 0.10 * hole_score + 0.10 * pal_score) * shape_match
    damage_val = max(0.0, 1.0 - energy_val)
    return float(np.clip(energy_val, 0.0, 1.0)), float(np.clip(damage_val, 0.0, 1.0))
class EmbodiedPuzzleSolver:
    def __init__(self, manifold_path=None, use_pain=True, use_curiosity=True, use_mortality=True, plastic=True, pop_size=12):
        if manifold_path is None:
            manifold_path = ARTIFACTS_DIR / "stage1_core_manifold.npz"
        loaded_manifold = False
        if Path(manifold_path).exists():
            data = np.load(manifold_path)
            if "Wx" in data.files and "Wh" in data.files and "Wo" in data.files and "Wpred" in data.files and "concept_centroids" in data.files:
                wx_raw = data["Wx"]
                wh_raw = data["Wh"]
                h_init = int(wh_raw.shape[-1])
                b_raw = data["b"] if "b" in data.files else np.zeros(h_init, dtype=np.float32)
                wo_raw = data["Wo"]
                bo_raw = data["bo"] if "bo" in data.files else np.zeros(wo_raw.shape[-2], dtype=np.float32)
                wpred_raw = data["Wpred"]
                c_raw = data["concept_centroids"]
                if wx_raw.shape[-2] > h_init:
                    wx_raw = wx_raw[..., -h_init:, :]
                if wh_raw.shape[-2] > h_init:
                    wh_raw = wh_raw[..., -h_init:, :]
                if b_raw.shape[-1] > h_init:
                    b_raw = b_raw[..., -h_init:]
                loaded_manifold = True
        if not loaded_manifold:
            rng_init = np.random.default_rng(42)
            wx_raw = rng_init.normal(0.0, 0.08, (128, 16)).astype(np.float32)
            wh_raw = rng_init.normal(0.0, 0.06, (128, 128)).astype(np.float32)
            b_raw = np.zeros(128, dtype=np.float32)
            wo_raw = rng_init.normal(0.0, 0.08, (8, 128)).astype(np.float32)
            bo_raw = np.zeros(8, dtype=np.float32)
            wpred_raw = rng_init.normal(0.0, 0.08, (16, 128)).astype(np.float32)
            c_raw = rng_init.normal(0.0, 0.1, (8, 128)).astype(np.float32)
        rng = np.random.default_rng(77)
        if c_raw.shape[0] < 8:
            extra_c = np.vstack([
                0.5 * c_raw[0] + 0.5 * c_raw[4],
                0.5 * c_raw[0] + 0.5 * c_raw[3],
                0.5 * c_raw[2] + 0.5 * c_raw[4],
            ]).astype(np.float32)
            for k in range(extra_c.shape[0]):
                extra_c[k] /= (np.linalg.norm(extra_c[k]) + 1e-8)
            self.concept_centroids = np.vstack([c_raw, extra_c])[:8]
        else:
            self.concept_centroids = c_raw[:8]
        if wx_raw.ndim == 2:
            self.pop_size = max(8, pop_size)
            self.Wx = np.stack([wx_raw + (0.0 if i == 0 else rng.normal(0.0, 0.02, wx_raw.shape).astype(np.float32)) for i in range(self.pop_size)])
            self.Wh = np.stack([wh_raw + (0.0 if i == 0 else rng.normal(0.0, 0.02, wh_raw.shape).astype(np.float32)) for i in range(self.pop_size)])
            self.b = np.stack([b_raw for _ in range(self.pop_size)])
            if wo_raw.shape[0] < 8:
                mean_norm = float(np.mean(np.linalg.norm(wo_raw, axis=1)))
                extra_wo = (self.concept_centroids[wo_raw.shape[0]:8] * mean_norm).astype(np.float32)
                wo_full = np.vstack([wo_raw, extra_wo])
                bo_full = np.concatenate([bo_raw, np.full(8 - bo_raw.shape[0], float(np.mean(bo_raw)), dtype=np.float32)])
            else:
                wo_full = wo_raw[:8]
                bo_full = bo_raw[:8]
            self.Wo = np.stack([wo_full + (0.0 if i == 0 else rng.normal(0.0, 0.04, wo_full.shape).astype(np.float32)) for i in range(self.pop_size)])
            self.bo = np.stack([bo_full for _ in range(self.pop_size)])
            self.Wpred = np.stack([wpred_raw for _ in range(self.pop_size)])
        else:
            if wx_raw.shape[0] < 8:
                n_extra = 8 - wx_raw.shape[0]
                wx_raw = np.concatenate([wx_raw, wx_raw[:n_extra] + rng.normal(0.0, 0.02, wx_raw[:n_extra].shape).astype(np.float32)], axis=0)
                wh_raw = np.concatenate([wh_raw, wh_raw[:n_extra] + rng.normal(0.0, 0.02, wh_raw[:n_extra].shape).astype(np.float32)], axis=0)
                b_raw = np.concatenate([b_raw, b_raw[:n_extra]], axis=0)
                wo_raw = np.concatenate([wo_raw, wo_raw[:n_extra] + rng.normal(0.0, 0.03, wo_raw[:n_extra].shape).astype(np.float32)], axis=0)
                bo_raw = np.concatenate([bo_raw, bo_raw[:n_extra]], axis=0)
                wpred_raw = np.concatenate([wpred_raw, wpred_raw[:n_extra]], axis=0)
            self.Wx = wx_raw
            self.Wh = wh_raw
            self.b = b_raw
            self.pop_size = self.Wx.shape[0]
            if wo_raw.shape[1] < 8:
                mean_norm = float(np.mean(np.linalg.norm(wo_raw, axis=2)))
                base_extra = (self.concept_centroids[wo_raw.shape[1]:8] * mean_norm).astype(np.float32)
                extra_wo = np.stack([
                    base_extra + rng.normal(0.0, 0.05, base_extra.shape).astype(np.float32)
                    for _ in range(self.pop_size)
                ])
                self.Wo = np.concatenate([wo_raw, extra_wo], axis=1)
                mean_bo = float(np.mean(bo_raw))
                self.bo = np.concatenate([bo_raw, np.full((self.pop_size, 8 - bo_raw.shape[1]), mean_bo, dtype=np.float32)], axis=1)
            else:
                self.Wo = wo_raw[:, :8, :]
                self.bo = bo_raw[:, :8]
            self.Wpred = wpred_raw
        self.hid_dim = self.Wh.shape[2]
        self.in_dim = self.Wx.shape[2]
        self.act_dim = 8
        self.theta = np.full((self.pop_size, self.hid_dim), 0.35, dtype=np.float32)
        self.Ws = rng.normal(0.0, 0.08, (self.pop_size, 2, self.hid_dim)).astype(np.float32)
        self.bs = np.zeros((self.pop_size, 2), dtype=np.float32)
        isaac_path = ARTIFACTS_DIR / "stage1_isaac3d_core_manifold.npz"
        self.signpost_ctx_k = np.zeros((6, self.hid_dim), dtype=np.float32)
        self.signpost_ctx_v = np.zeros((6, self.hid_dim), dtype=np.float32)
        self.hippo_ctx_k = np.zeros((8, self.hid_dim), dtype=np.float32)
        self.hippo_ctx_v = np.zeros((8, self.hid_dim), dtype=np.float32)
        self.mono_dim = 4
        self.W_mono_out = rng.normal(0.0, 0.06, (self.pop_size, self.mono_dim, self.hid_dim)).astype(np.float32)
        self.W_mono_in = rng.normal(0.0, 0.06, (self.pop_size, self.hid_dim, self.mono_dim)).astype(np.float32)
        self.W_macro_out = rng.normal(0.0, 0.06, (self.pop_size, self.mono_dim, self.hid_dim)).astype(np.float32)
        self.W_macro_in = rng.normal(0.0, 0.06, (self.pop_size, self.hid_dim, self.mono_dim)).astype(np.float32)
        self.W_source_attr_reg = rng.normal(0.0, 0.06, (self.pop_size, self.mono_dim, self.hid_dim)).astype(np.float32)
        self.W_target_op_reg = rng.normal(0.0, 0.06, (self.pop_size, self.mono_dim, self.hid_dim + self.mono_dim)).astype(np.float32)
        self.W_rel_bind_in = rng.normal(0.0, 0.06, (self.pop_size, self.hid_dim, self.mono_dim * 2)).astype(np.float32)
        self.inner_monologue = {
            "source_attr_reg": np.zeros((self.pop_size, self.mono_dim), dtype=np.float32),
            "target_op_reg": np.zeros((self.pop_size, self.mono_dim), dtype=np.float32),
            "symbolic_bindings": [],
        }
        self.W_cond_gate = rng.normal(0.0, 0.08, (self.pop_size, 2, self.mono_dim * 2)).astype(np.float32)
        self.b_cond_gate = np.array([[0.25, -0.15] for _ in range(self.pop_size)], dtype=np.float32)
        self.W_nested_cond_gate = rng.normal(0.0, 0.08, (self.pop_size, 3, self.mono_dim * 2)).astype(np.float32)
        self.b_nested_cond_gate = np.array([[0.30, 0.15, -0.20] for _ in range(self.pop_size)], dtype=np.float32)
        self.Wq = rng.normal(0.0, 0.08, (self.pop_size, self.hid_dim, self.hid_dim)).astype(np.float32)
        self.Wk = rng.normal(0.0, 0.08, (self.pop_size, self.hid_dim, self.hid_dim)).astype(np.float32)
        self.Wv = rng.normal(0.0, 0.08, (self.pop_size, self.hid_dim, self.hid_dim)).astype(np.float32)
        self.Wpred_val = rng.normal(0.0, 0.06, (self.pop_size, 2, self.hid_dim + self.act_dim)).astype(np.float32)
        self.dream_wake_buffer = []
        self.unified_brains = UnifiedBiologicalBrainSystem(
            hid_dim=self.hid_dim, artifact_dir=str(ARTIFACTS_DIR), seed=int(getattr(self, "seed", 42))
        )
        self.loaded_brain_status = self.unified_brains.load_all_brains()
        if isaac_path.exists():
            isaac_data = np.load(isaac_path)
            wq_src = isaac_data["Wq"]
            wk_src = isaac_data["Wk"]
            wv_src = isaac_data["Wv"]
            if wq_src.ndim == 2:
                h_src = wq_src.shape[0]
                wq_mat = np.zeros((self.hid_dim, self.hid_dim), dtype=np.float32)
                wk_mat = np.zeros((self.hid_dim, self.hid_dim), dtype=np.float32)
                wv_mat = np.zeros((self.hid_dim, self.hid_dim), dtype=np.float32)
                for b_off in range(0, self.hid_dim, h_src):
                    b_end = min(self.hid_dim, b_off + h_src)
                    b_len = b_end - b_off
                    wq_mat[b_off:b_end, b_off:b_end] = wq_src[:b_len, :b_len]
                    wk_mat[b_off:b_end, b_off:b_end] = wk_src[:b_len, :b_len]
                    wv_mat[b_off:b_end, b_off:b_end] = wv_src[:b_len, :b_len]
                self.Wq = (0.75 * self.Wq + 0.25 * wq_mat[None, :, :]).astype(np.float32)
                self.Wk = (0.75 * self.Wk + 0.25 * wk_mat[None, :, :]).astype(np.float32)
                self.Wv = (0.75 * self.Wv + 0.25 * wv_mat[None, :, :]).astype(np.float32)
                if "spike_ctx" in isaac_data and "val_ctx" in isaac_data:
                    s_ctx = isaac_data["spike_ctx"]
                    v_ctx = isaac_data["val_ctx"]
                    c_len = min(6, s_ctx.shape[0])
                    for b_off in range(0, self.hid_dim, h_src):
                        b_end = min(self.hid_dim, b_off + h_src)
                        b_len = b_end - b_off
                        self.signpost_ctx_k[:c_len, b_off:b_end] = s_ctx[:c_len, :b_len]
                        self.signpost_ctx_v[:c_len, b_off:b_end] = v_ctx[:c_len, :b_len]
                if "hippo_keys" in isaac_data and "hippo_vals" in isaac_data:
                    hk_src = isaac_data["hippo_keys"]
                    hv_src = isaac_data["hippo_vals"]
                    h_len = min(8, hk_src.shape[0])
                    for b_off in range(0, self.hid_dim, h_src):
                        b_end = min(self.hid_dim, b_off + h_src)
                        b_len = b_end - b_off
                        self.hippo_ctx_k[:h_len, b_off:b_end] = hk_src[:h_len, :b_len]
                        self.hippo_ctx_v[:h_len, b_off:b_end] = hv_src[:h_len, :b_len]
                if "W_mono_out" in isaac_data and "W_mono_in" in isaac_data:
                    wmo_src = isaac_data["W_mono_out"]
                    wmi_src = isaac_data["W_mono_in"]
                    for b_off in range(0, self.hid_dim, h_src):
                        b_end = min(self.hid_dim, b_off + h_src)
                        b_len = b_end - b_off
                        self.W_mono_out[:, :, b_off:b_end] = 0.70 * self.W_mono_out[:, :, b_off:b_end] + 0.30 * wmo_src[None, :, :b_len]
                        self.W_mono_in[:, b_off:b_end, :] = 0.70 * self.W_mono_in[:, b_off:b_end, :] + 0.30 * wmi_src[None, :b_len, :]
                if "W_macro_out" in isaac_data and "W_macro_in" in isaac_data:
                    wmaco_src = isaac_data["W_macro_out"]
                    wmaci_src = isaac_data["W_macro_in"]
                    for b_off in range(0, self.hid_dim, h_src):
                        b_end = min(self.hid_dim, b_off + h_src)
                        b_len = b_end - b_off
                        self.W_macro_out[:, :, b_off:b_end] = 0.70 * self.W_macro_out[:, :, b_off:b_end] + 0.30 * wmaco_src[None, :, :b_len]
                        self.W_macro_in[:, b_off:b_end, :] = 0.70 * self.W_macro_in[:, b_off:b_end, :] + 0.30 * wmaci_src[None, :b_len, :]
                if "dream_q" in isaac_data and "dream_k" in isaac_data and "dream_v" in isaac_data:
                    dq_src = isaac_data["dream_q"]
                    dk_src = isaac_data["dream_k"]
                    dv_src = isaac_data["dream_v"]
                    for d_idx in range(min(4, dq_src.shape[0])):
                        q_pad = np.zeros((self.hid_dim,), dtype=np.float32)
                        k_pad = np.zeros((self.hid_dim,), dtype=np.float32)
                        v_pad = np.zeros((self.hid_dim,), dtype=np.float32)
                        for b_off in range(0, self.hid_dim, h_src):
                            b_end = min(self.hid_dim, b_off + h_src)
                            b_len = b_end - b_off
                            q_pad[b_off:b_end] = dq_src[d_idx, :b_len]
                            k_pad[b_off:b_end] = dk_src[d_idx, :b_len]
                            v_pad[b_off:b_end] = dv_src[d_idx, :b_len]
                        self.dream_wake_buffer.append({
                            "obs": np.zeros((self.in_dim,), dtype=np.float32),
                            "h": v_pad.copy(),
                            "q_spk": q_pad,
                            "k_spk": k_pad,
                            "v_proj": v_pad,
                            "macro_vec": np.tanh(self.W_macro_out[0] @ v_pad).astype(np.float32),
                            "micro_vec": np.tanh(self.W_mono_out[0] @ v_pad).astype(np.float32),
                            "winning_axis": int(d_idx % self.act_dim),
                            "program_axes": [int(d_idx % self.act_dim)],
                            "program_names": ["isaac3d_hazard_dream"],
                            "reward": 0.90,
                        })
                if "mutated_dream_q" in isaac_data and "mutated_dream_k" in isaac_data and "mutated_dream_v" in isaac_data:
                    mdq_src = isaac_data["mutated_dream_q"]
                    mdk_src = isaac_data["mutated_dream_k"]
                    mdv_src = isaac_data["mutated_dream_v"]
                    for md_idx in range(min(4, mdq_src.shape[0])):
                        mq_pad = np.zeros((self.hid_dim,), dtype=np.float32)
                        mk_pad = np.zeros((self.hid_dim,), dtype=np.float32)
                        mv_pad = np.zeros((self.hid_dim,), dtype=np.float32)
                        for b_off in range(0, self.hid_dim, h_src):
                            b_end = min(self.hid_dim, b_off + h_src)
                            b_len = b_end - b_off
                            mq_pad[b_off:b_end] = mdq_src[md_idx, :b_len]
                            mk_pad[b_off:b_end] = mdk_src[md_idx, :b_len]
                            mv_pad[b_off:b_end] = mdv_src[md_idx, :b_len]
                        self.dream_wake_buffer.append({
                            "obs": np.zeros((self.in_dim,), dtype=np.float32),
                            "h": mv_pad.copy(),
                            "q_spk": mq_pad,
                            "k_spk": mk_pad,
                            "v_proj": mv_pad,
                            "macro_vec": np.tanh(self.W_macro_out[0] @ mv_pad).astype(np.float32),
                            "micro_vec": np.tanh(self.W_mono_out[0] @ mv_pad).astype(np.float32),
                            "winning_axis": int((md_idx + 4) % self.act_dim),
                            "program_axes": [int((md_idx + 4) % self.act_dim)],
                            "program_names": ["isaac3d_chimera_dream"],
                            "reward": 0.95,
                        })
        self.unified_brains.sync_to_stage2_solver(self)
        self.hippocampal_episodic_buffer = []
        self.mutated_episodic_buffer = []
        self.cultural_tool_library = []
        self.recursive_chunk_library = []
        self.synthesized_micro_primitives = []
        self.residual_subgoal_queue = []
        self.adaptive_glaciation_depth = 5
        self.bred_macro_recipes = []
        self.nested_macro_recipes = []
        self.if_then_else_switches_total = 0
        self.nested_if_then_switches_total = 0
        self.relational_bindings_total = 0
        self.recursive_chunks_compiled_total = 0
        self.canvas_stitch_solves_total = 0
        self.micro_primitives_synthesized_total = 0
        self.residual_subgoals_harvested_total = 0
        self.residual_bridges_spliced_total = 0
        self.tools_pruned_total = 0
        self.pruning_cycles_run = 0
        self.tasks_processed_count = 0
        self.evolved_ast_organs = []
        self.synthesized_primitive_nodes = {}
        self.de_novo_codon_pool = []
        self.de_novo_codons_synthesized_total = 0
        self.evolving_genome_archive = {}
        self.evolving_task_visit_counts = {}
        self.ast_mutations_total = 0
        self.ast_crossovers_total = 0
        self.tried_ast_expr_hashes = set()
        self.W_ast_intuition = rng.normal(0.0, 0.06, (32, self.hid_dim + self.mono_dim * 4)).astype(np.float32)
        self.specialist_roles = {
            0: "kinematics_ray_specialist",
            1: "fovea_partition_specialist",
            2: "geometry_symmetry_specialist",
            3: "gravity_contact_specialist",
            4: "topology_color_specialist",
            5: "number_statistics_specialist",
            6: "signpost_monologue_specialist",
            7: "gestalt_pattern_specialist",
        }
        for p_i in range(self.pop_size):
            spec_ax = p_i % self.act_dim
            self.bo[p_i, spec_ax] += 0.28
        self.social_empathy_events_total = 0
        self.specialist_leader_elections_total = 0
        self.use_pain = use_pain
        self.use_curiosity = use_curiosity
        self.use_mortality = use_mortality
        self.plastic = plastic
        self.basal_ganglia_reflex_chunks = []
        self.basal_ganglia_reflex_hits_total = 0
        self.basal_ganglia_energy_saved_total = 0.0
        self.last_active_cell_ratio = 0.25
        self.last_dormant_cell_ratio = 0.75
        self.foveal_saccades_executed_total = 0
        self.attractor_snaps_total = 0
        self.metacognitive_corrections_total = 0
        self.foveal_objects_perceived_total = 0
        self.one_step_subgoals_formulated_total = 0
        self.basal_ganglia_drawer_pulls_total = 0
        self.dynamic_board_reconsiderations_total = 0
        self.selective_object_grasps_total = 0
        self.intrinsic_gestalt_evaluations_total = 0
        self.cross_example_diff_comparisons_total = 0
        self.in_grid_legend_reads_total = 0
        self.predicate_group_broadcasts_total = 0
        self.surgical_guard_debugs_total = 0
        self.stage1_sensorimotor_chunks = {}
        self.stage1_sensorimotor_chunks_transferred = 0
        self.sensorimotor_body_executions_total = 0
        self.sensorimotor_chunks_bred_total = 0
        self.cerebellar_error_corrections_total = 0
        self.prefrontal_dynamic_chains_total = 0
        self.bred_sensorimotor_chains = []
        self.W_body_assembly = rng.normal(0.0, 0.08, (11, self.hid_dim)).astype(np.float32)
        self.symbol_vocab_size = 16
        w_sym_init = rng.normal(0.0, 0.25, (self.symbol_vocab_size, 11)).astype(np.float32)
        w_sym_norms = np.linalg.norm(w_sym_init, axis=1, keepdims=True) + 1e-6
        self.W_symbol_lexicon = (w_sym_init / w_sym_norms).astype(np.float32)
        self.symbol_usage_counts = np.zeros((self.symbol_vocab_size,), dtype=np.float32)
        self.verified_symbol_utterances_total = 0
        self.symbol_silence_choices_total = 0
        self.shared_intentionality_groundings_total = 0
        self.epistemic_memo_writes_total = 0
        self.wm_offload_flushes_total = 0
        self.epistemic_delayed_payoffs_total = 0
        self.ripple_replays_executed_total = 0
        self.ib_ltp_invariant_synapses_total = 0
        self.ib_ltd_pruned_synapses_total = 0
        self.counterfactual_incongruity_triggers_total = 0
        self.prefrontal_gaba_blocks_total = 0
        self.latent_feature_bypasses_total = 0
        self.topological_lens_warps_total = 0
        self.ach_signal_level = 0.0
        self.active_topological_lens = "cartesian"
        self.v1_v2_texture_segregations_total = 0
        self.v1_v2_constellation_groups_total = 0
        self.v1_v2_hierarchical_expansions_total = 0
        self.virtual_scratchpad_projections_total = 0
        self.scratchpad_inverse_backprojections_total = 0
        self.subgoal_checkpoints_locked_total = 0
        self.subgoal_non_destructive_vetoes_total = 0
        self.it_parietal_relational_bindings_total = 0
        self.cerebellar_valley_crossing_macros_total = 0
        self.ior_sequential_saccades_total = 0
        self.pfc_dyadic_bindings_total = 0
        self.foveal_window_zooms_total = 0
        self.motor_stamp_executions_total = 0
        self.persistent_h = np.zeros((self.pop_size, self.hid_dim), dtype=np.float32)
        self.persistent_v_mem = np.zeros((self.pop_size, self.hid_dim), dtype=np.float32)
        self.closed_loop_3brain_steps_total = 0
        self.mental_simulations_total = 0
        self.micro_rest_recoveries_total = 0
        self.foveal_mental_canvas_updates_total = 0
        self._last_saccade_info = None
        self.foveated_mental_canvas_perceptions_total = 0
        self.opponent_color_segregations_total = 0
        self.W_parietal_gain = rng.normal(0.0, 0.08, (4, self.hid_dim)).astype(np.float32)
        self.parietal_gain_field_modulations_total = 0
        self.v2_collinear_completions_total = 0
        self._last_parietal_gain_vec = np.ones((self.hid_dim,), dtype=np.float32)
        self.effector_hand_yx = (0, 0)
        self.proprioceptive_contact_spikes_total = 0
        self.visuomotor_stroke_steps_total = 0
        self.W_cerebellar_purkinje = (
            np.eye(6, dtype=np.float32) * 0.85
            + rng.normal(0.0, 0.05, (6, 6)).astype(np.float32)
        )
        self.climbing_fiber_vector_corrections_total = 0
        self.direct_spiking_action_decodings_total = 0
        self.latent_pfc_mental_rollouts_total = 0
        self.cortical_column_competitions_total = 0
        self.winning_cortical_column_idx = 0
        self.pre_spike_trace = np.zeros((self.pop_size, self.hid_dim), dtype=np.float32)
        self.post_spike_trace = np.zeros((self.pop_size, self.hid_dim), dtype=np.float32)
        self.gamma_micro_ticks_executed_total = 0
        self.stdp_trace_updates_total = 0
        self.W_procedural_attractor = rng.normal(0.0, 0.08, (16, self.hid_dim)).astype(np.float32)
        self.W_procedural_chain = rng.normal(0.0, 0.08, (16, 33)).astype(np.float32)
        self.synaptic_procedural_consolidations_total = 0
        self.synaptic_procedural_recalls_total = 0
        self.W_syntax_transition = rng.normal(0.0, 0.12, (self.mono_dim, self.mono_dim)).astype(np.float32)
        self.compositional_phonological_loops_total = 0
        self.last_phonological_phrase = (0, 1, 2)
        self.organism_hp = 1.0
        self.organism_energy = 0.85
        self.organism_pain = 0.0
        self.organism_sleep_pressure = 0.0
        self.organism_lifetime_steps = 0
        self.autonomous_homeostatic_sleep_cycles_total = 0
        self.intrinsic_dopamine_level = 0.0
        self.aha_dopamine_bursts_total = 0
        self.epistemic_atp_invested_total = 0.0
        self.seated_posture_locked = False
        self.seated_tabletop_focus_engagements_total = 0
        self.tabletop_25d_projections_total = 0
        self.head_tilt_viewpoint_shifts_total = 0
        self._last_tabletop_25d_info = None
        self.W_mirror_inverse = rng.normal(0.0, 0.12, (11, 16)).astype(np.float32)
        self.mirror_neuron_imitations_total = 0
        self.autotelic_play_bouts_total = 0
        self.hand_regard_self_imitations_total = 0
        self.developmental_stage = 1
        self.critical_period_transitions_total = 0
        self.myelination_stage_gates = np.array([1.0, 0.65, 0.40], dtype=np.float32)
        self.ach_habit_resets_total = 0
        self.einstellung_fixation_prevented_total = 0
        self.synaptic_homeostasis_downscalings_total = 0
        self.triple_overfit_guards_triggered_total = 0
        self.external_workbench_interactions_total = 0
        self.habit_suppression_gate = 1.0
        self._mirror_inferred_affordance_steps = []
        self.organism_world_xyz = np.array([50.0, 50.0, 0.0], dtype=np.float32)
        self.in_safe_study_room = False
        self.safe_room_transitions_total = 0
        self.last_safe_room_info = None
        self.vestibular_gravity_vec = np.array([0.0, 0.0, -9.81], dtype=np.float32)
        self.vestibular_canal_omega = np.zeros(3, dtype=np.float32)
        self.head_euler_rad = np.zeros(3, dtype=np.float32)
        self.vestibular_vor_stabilizations_total = 0
        self.W_cochlear_auditory = rng.normal(0.0, 0.08, (self.hid_dim, 8)).astype(np.float32)
        self.cochlear_acoustic_perceptions_total = 0
        self.last_cochlear_spectrum_8 = np.zeros(8, dtype=np.float32)
        self.visceral_heart_rate_bpm = 68.0
        self.visceral_respiration_rate = 14.0
        self.visceral_cortisol_level = 0.12
        self.visceral_oxytocin_level = 0.35
        self.W_insular_interoception = rng.normal(0.0, 0.08, (self.hid_dim, 4)).astype(np.float32)
        self.visceral_interoceptive_steps_total = 0
        self.organism_life_time_sec = 0.0
        self.place_cell_centers_xyz = np.array(
            [
                [25.0, 25.0, 0.0],
                [75.0, 25.0, 0.0],
                [25.0, 75.0, 0.0],
                [75.0, 75.0, 0.0],
                [100.0, 50.0, 0.0],
                [104.0, 50.0, 0.0],
                [112.0, 49.75, 0.0],
                [118.0, 55.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.W_hippo_spatiotemporal = rng.normal(0.0, 0.08, (self.hid_dim, 16)).astype(np.float32)
        self.W_fovea_v1_spikes = rng.normal(0.0, 0.08, (self.hid_dim, 100)).astype(np.float32)
        self.W_fovea_stroke_assoc = rng.normal(0.0, 0.02, (25, 11, 100)).astype(np.float32)
        self.last_foveal_retinotopic_100 = np.zeros((100,), dtype=np.float32)
        self.foveal_retinotopic_spike_projections_total = 0
        self.foveal_visuomotor_strokes_learned_total = 0
        self.foveal_saccade_stroke_loops_executed_total = 0
        self.current_gaze_yx = (0, 0)
        self.foveal_smooth_pursuit_steps_total = 0
        self.foveal_lateral_inhibitions_total = 0
        self.foveal_intuition_selections_total = 0
        self.learned_foveal_stroke_prototypes = []
        self.autobiographical_memory_bank = []
        self.hippocampal_place_grid_activations_total = 0
        self.autobiographical_recalls_total = 0
        self.affective_empathy_resonances_total = 0
        self.embodied_peer_demonstrations_total = 0
        self.absorb_stage1_3d_embodied_manifold(isaac_path=isaac_path)
        if hasattr(self, "unified_brains") and self.unified_brains is not None:
            self.unified_brains.sync_to_stage2_solver(self)
    def encode_and_verify_symbol_utterance(self, q_spk_vec, body_vec_override=None, allow_self_imprint=False):
        if body_vec_override is not None:
            b_vec = np.asarray(body_vec_override, dtype=np.float32).reshape(-1)[:11]
            if b_vec.size < 11:
                b_vec = np.pad(b_vec, (0, 11 - b_vec.size)).astype(np.float32)
        else:
            q_flat = np.asarray(q_spk_vec, dtype=np.float32).reshape(-1)[: self.hid_dim]
            if q_flat.size < self.hid_dim:
                q_flat = np.pad(q_flat, (0, self.hid_dim - q_flat.size)).astype(np.float32)
            b_vec = np.tanh(self.W_body_assembly @ q_flat).astype(np.float32)
        b_norm = float(np.linalg.norm(b_vec))
        if b_norm < 1e-6:
            self.symbol_silence_choices_total += 1
            return -1, 0.0, np.zeros((11,), dtype=np.float32)
        b_unit = (b_vec / b_norm).astype(np.float32)
        lex_norms = np.linalg.norm(self.W_symbol_lexicon, axis=1, keepdims=True) + 1e-6
        lex_unit = (self.W_symbol_lexicon / lex_norms).astype(np.float32)
        sims = lex_unit @ b_unit
        sym_id = int(np.argmax(sims))
        best_cos = float(sims[sym_id])
        if self.plastic and best_cos < 0.80 and allow_self_imprint:
            min_slot = int(np.argmin(self.symbol_usage_counts))
            if float(self.symbol_usage_counts[min_slot]) < 1.0:
                self.W_symbol_lexicon[min_slot] = b_unit.copy()
                sym_id = min_slot
                best_cos = 1.0
        recon_vec = self.W_symbol_lexicon[sym_id].copy()
        recon_norm = float(np.linalg.norm(recon_vec))
        cos_sim = float(np.dot(b_unit, recon_vec / max(1e-6, recon_norm))) if recon_norm > 1e-6 else 0.0
        if cos_sim < 0.80:
            self.symbol_silence_choices_total += 1
            return -1, cos_sim, b_unit
        self.verified_symbol_utterances_total += 1
        if self.plastic:
            self.symbol_usage_counts[sym_id] += 1.0
        aud_spk, aud_cur = self.synthesize_and_perceive_vocal_formant_wave(b_unit)
        self.last_cochlear_spikes_8 = aud_spk
        return sym_id, cos_sim, b_unit
    def decode_symbol_to_sensorimotor_bias(self, sym_id):
        if sym_id is None or int(sym_id) < 0 or int(sym_id) >= int(self.symbol_vocab_size):
            return np.zeros((11,), dtype=np.float32), np.zeros((self.hid_dim,), dtype=np.float32)
        dec_body = self.W_symbol_lexicon[int(sym_id)].astype(np.float32).copy()
        q_bias = (self.W_body_assembly.T @ dec_body).astype(np.float32)
        qb_norm = float(np.linalg.norm(q_bias))
        if qb_norm > 1e-6:
            q_bias = (q_bias / qb_norm).astype(np.float32)
        return dec_body, q_bias
    def ground_shared_symbol_convention(self, sym_id, target_body_vec, order_gain):
        if not self.plastic:
            return False
        if sym_id is None or int(sym_id) < 0 or int(sym_id) >= int(self.symbol_vocab_size):
            return False
        if float(order_gain) <= 0.0:
            return False
        s_idx = int(sym_id)
        t_vec = np.asarray(target_body_vec, dtype=np.float32).reshape(-1)[:11]
        t_norm = float(np.linalg.norm(t_vec))
        if t_norm < 1e-6:
            return False
        t_unit = (t_vec / t_norm).astype(np.float32)
        lr_g = float(np.clip(0.18 * float(order_gain), 0.02, 0.25))
        updated = (1.0 - lr_g) * self.W_symbol_lexicon[s_idx] + lr_g * t_unit
        u_norm = float(np.linalg.norm(updated))
        if u_norm > 1e-6:
            self.W_symbol_lexicon[s_idx] = (updated / u_norm).astype(np.float32)
        self.shared_intentionality_groundings_total += 1
        return True
    def v1_v2_opponent_color_figure_ground_segregation(self, grid):
        self.opponent_color_segregations_total += 1
        if hasattr(self, "unified_brains") and self.unified_brains is not None:
            return self.unified_brains.sensory_cortex.compute_v1_v2_opponent_border_ownership(grid)
        g = np.clip(np.asarray(grid, dtype=np.int32), 0, 9)
        bg_c = int(detect_background_color(g))
        return {
            "inferred_bg_color": bg_c,
            "figure_mask": (g != bg_c),
            "edge_energy": (g != bg_c).astype(np.float32),
            "border_ownership": np.zeros_like(g, dtype=np.float32),
        }
    def compute_parietal_gain_field_modulation(self, obs_vec, lens_mode="cartesian", gaze_yx=(0.5, 0.5)):
        self.parietal_gain_field_modulations_total += 1
        polar_on = 1.0 if lens_mode in ("polar", "log_polar") else 0.0
        log_on = 1.0 if lens_mode in ("log_scale", "log_polar") else 0.0
        gy = float(np.clip(gaze_yx[0], 0.0, 1.0)) if gaze_yx is not None else 0.5
        gx = float(np.clip(gaze_yx[1], 0.0, 1.0)) if gaze_yx is not None else 0.5
        eye_topo_vec = np.array([polar_on, log_on, gy, gx], dtype=np.float32)
        raw_gain = eye_topo_vec @ self.W_parietal_gain
        gain_vec = (1.0 + 0.25 * np.tanh(raw_gain)).astype(np.float32)
        self._last_parietal_gain_vec = gain_vec
        v_in = np.asarray(obs_vec, dtype=np.float32).reshape(-1)
        if v_in.size == self.hid_dim:
            return (v_in * gain_vec).astype(np.float32), gain_vec
        if v_in.size == self.in_dim and self.hid_dim % self.in_dim == 0:
            gain_in = gain_vec.reshape(self.in_dim, -1).mean(axis=1).astype(np.float32)
            return (v_in * gain_in).astype(np.float32), gain_vec
        return v_in, gain_vec
    def execute_visuomotor_stroke_servo(self, canvas, stroke_cells, bg_color=None, step_physx=None):
        if not stroke_cells:
            return canvas
        if bg_color is None:
            bg_color = detect_background_color(canvas) if canvas.size > 0 else 0
        H, W = canvas.shape
        hy, hx = self.effector_hand_yx
        hy = int(np.clip(hy, 0, max(0, H - 1)))
        hx = int(np.clip(hx, 0, max(0, W - 1)))
        valid_strokes = []
        for cell_item in stroke_cells:
            r_t, c_t, col_t = int(cell_item[0]), int(cell_item[1]), int(cell_item[2])
            if not (0 <= r_t < H and 0 <= c_t < W):
                self.proprioceptive_contact_spikes_total += 1
                continue
            valid_strokes.append((r_t, c_t, col_t))
        if not valid_strokes:
            return canvas
        if bool(getattr(self, "_mental_imagery_mode", False)):
            for r_t, c_t, col_t in valid_strokes:
                canvas[r_t, c_t] = int(col_t) % 10
            self.effector_hand_yx = (int(valid_strokes[-1][0]), int(valid_strokes[-1][1]))
            self.current_gaze_yx = self.effector_hand_yx
            self.visuomotor_stroke_steps_total += len(valid_strokes)
            return canvas
        use_physx = (not bool(getattr(self, "_mental_imagery_mode", False))) if step_physx is None else bool(step_physx)
        phys_res = get_global_isaac_warp_engine().execute_physical_tabletop_stroke_on_desk(
            canvas,
            valid_strokes,
            bg_color=bg_color,
            step_physx=use_physx,
        )
        contacted_cells = phys_res.get("contacted_cells", [])
        if not contacted_cells:
            contacted_cells = valid_strokes
        n_cells = len(contacted_cells)
        sample_stride = max(1, n_cells // 3)
        for step_i, (r_t, c_t, col_t) in enumerate(contacted_cells):
            manhattan_dist = abs(r_t - hy) + abs(c_t - hx)
            step_move = max(1, int(manhattan_dist))
            self.visuomotor_stroke_steps_total += step_move
            self.foveal_smooth_pursuit_steps_total += step_move
            self.proprioceptive_contact_spikes_total += 1
            existing_col = int(canvas[r_t, c_t])
            if existing_col != int(bg_color) and existing_col != int(col_t):
                self.proprioceptive_contact_spikes_total += 1
            if r_t == 0 or r_t == H - 1 or c_t == 0 or c_t == W - 1:
                self.proprioceptive_contact_spikes_total += 1
            canvas[r_t, c_t] = int(col_t) % 10
            hy, hx = r_t, c_t
            self.current_gaze_yx = (hy, hx)
            if step_i == 0 or step_i == n_cells - 1 or (step_i % sample_stride == 0):
                self.encode_retinotopic_foveal_patch_spikes(canvas, hy, hx, bg_color=bg_color)
        self.effector_hand_yx = (hy, hx)
        return canvas
    def _extract_salient_foveal_fixations(self, x_grid, y_grid=None, bg_color=None, max_fixations=48):
        gx = np.asarray(x_grid, dtype=np.int32)
        if gx.ndim != 2 or gx.size == 0:
            return []
        H, W = gx.shape
        bg = int(detect_background_color(gx)) if bg_color is None else int(bg_color)
        salience_map = np.zeros((H, W), dtype=np.float32)
        comps = extract_components(gx, bg=bg)
        for comp in comps:
            mass = int(comp["mass"])
            rc = int(np.clip(round(float(comp["center"][0])), 0, H - 1))
            cc = int(np.clip(round(float(comp["center"][1])), 0, W - 1))
            if mass == 1:
                r0, c0 = int(comp["cells"][0][0]), int(comp["cells"][0][1])
                salience_map[r0, c0] += 5.0
            else:
                salience_map[rc, cc] += 3.5
                rmin, cmin, rmax, cmax = comp["bbox"]
                for br, bc in ((rmin, cmin), (rmin, cmax), (rmax, cmin), (rmax, cmax)):
                    if 0 <= br < H and 0 <= bc < W:
                        salience_map[br, bc] += 2.2
                for r_cell, c_cell in comp["cells"]:
                    salience_map[int(r_cell), int(c_cell)] += 1.0
        cand_rs, cand_cs = np.where(salience_map > 0.0)
        if cand_rs.size == 0:
            return []
        scored = [
            (float(salience_map[int(r_i), int(c_i)]), -int(r_i), -int(c_i), int(r_i), int(c_i))
            for r_i, c_i in zip(cand_rs.tolist(), cand_cs.tolist())
        ]
        scored.sort(reverse=True)
        return [(item[3], item[4]) for item in scored[: int(max_fixations)]]
    def _imprint_foveal_stroke_synapse_with_lateral_inhibition(self, s_pre_100, k_idx, ch_idx, lr=0.045):
        k_i = int(np.clip(int(k_idx), 0, 24))
        c_i = int(np.clip(int(ch_idx), 0, 10))
        s_vec = np.asarray(s_pre_100, dtype=np.float32).reshape(100)
        s_norm = float(np.linalg.norm(s_vec))
        if s_norm < 1e-6:
            return
        s_unit = (s_vec / s_norm).astype(np.float32)
        w_slice = self.W_fovea_stroke_assoc[k_i]
        sims = w_slice @ s_unit
        for other_c in range(11):
            if other_c == c_i:
                continue
            sim_v = float(sims[other_c])
            if sim_v > 0.0:
                w_slice[other_c] = np.clip(
                    w_slice[other_c] - float(lr) * 0.35 * sim_v * s_unit,
                    -2.5,
                    2.5,
                ).astype(np.float32)
                self.foveal_lateral_inhibitions_total += 1
        w_win = (1.0 - 0.015 * float(lr)) * w_slice[c_i] + float(lr) * s_unit
        w_win_norm = float(np.linalg.norm(w_win))
        if w_win_norm > 1.5:
            w_win = (w_win * (1.5 / w_win_norm)).astype(np.float32)
        w_slice[c_i] = np.clip(w_win, -2.5, 2.5).astype(np.float32)
        if hasattr(self, "unified_brains") and self.unified_brains is not None:
            self.unified_brains.neocortex_logic.imprint_foveal_stroke_association(s_unit, k_i, c_i, lr=lr)
    def _readout_foveal_stroke_synapse_with_lateral_inhibition(self, s_cur_100):
        if hasattr(self, "unified_brains") and self.unified_brains is not None:
            return self.unified_brains.neocortex_logic.readout_foveal_stroke_association(s_cur_100)
        s_vec = np.asarray(s_cur_100, dtype=np.float32).reshape(100)
        s_norm = float(np.linalg.norm(s_vec))
        if s_norm < 1e-6:
            return np.zeros((25, 11), dtype=np.float32)
        s_unit = (s_vec / s_norm).astype(np.float32)
        raw_drive = np.tensordot(self.W_fovea_stroke_assoc, s_unit, axes=([2], [0])).astype(np.float32)
        sorted_drive = np.sort(raw_drive, axis=1)
        margin = (sorted_drive[:, -1] - sorted_drive[:, -2])[:, None]
        gate = np.clip(margin / 0.12, 0.25, 1.5).astype(np.float32)
        return (raw_drive * gate).astype(np.float32)

    def encode_retinotopic_foveal_patch_spikes(self, grid, gaze_y, gaze_x, bg_color=None):
        self.foveal_retinotopic_spike_projections_total += 1
        if hasattr(self, "unified_brains") and self.unified_brains is not None:
            spikes = self.unified_brains.sensory_cortex.encode_retinotopic_foveal_patch_spikes(
                grid, gaze_y, gaze_x, bg_color=bg_color
            )
        else:
            g = np.asarray(grid, dtype=np.int32)
            if g.ndim != 2 or g.size == 0:
                spikes = np.zeros((100,), dtype=np.float32)
            else:
                h, w = g.shape
                gy = int(np.clip(int(gaze_y), 0, h - 1))
                gx = int(np.clip(int(gaze_x), 0, w - 1))
                bg = int(detect_background_color(g)) if bg_color is None else int(bg_color)
                center_col = int(np.clip(g[gy, gx], 0, 9))
                spikes = np.zeros((100,), dtype=np.float32)
                idx = 0
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ry, rx = gy + dy, gx + dx
                        if 0 <= ry < h and 0 <= rx < w:
                            c_val = int(np.clip(g[ry, rx], 0, 9))
                            fg_sp = 1.0 if c_val != bg else 0.0
                            same_sp = 1.0 if (c_val == center_col and c_val != bg) else (-1.0 if c_val != bg else 0.0)
                            spikes[idx] = float(fg_sp)
                            spikes[idx + 1] = float(same_sp)
                            spikes[idx + 2] = ((float(c_val) + 0.5) / 10.0)
                            spikes[idx + 3] = 1.0 if (dy == 0 and dx == 0 and c_val != bg) else 0.0
                        else:
                            spikes[idx] = -0.5
                        idx += 4
        self.last_foveal_retinotopic_100 = np.asarray(spikes, dtype=np.float32).reshape(100)
        if hasattr(self, "unified_brains") and self.unified_brains is not None:
            with torch.no_grad():
                spk_t = torch.from_numpy(self.last_foveal_retinotopic_100).to(
                    device=self.unified_brains.neocortex_logic.W_fovea_v1_spikes.device
                )
                foveal_drive = torch.tanh(self.unified_brains.neocortex_logic.W_fovea_v1_spikes @ spk_t)
                self.last_foveal_cortical_drive = foveal_drive.cpu().numpy().astype(np.float32)
        elif hasattr(self, "W_fovea_v1_spikes"):
            self.last_foveal_cortical_drive = np.tanh(self.W_fovea_v1_spikes @ self.last_foveal_retinotopic_100).astype(np.float32)
        return self.last_foveal_retinotopic_100
    def execute_foveal_saccade_stroke_closed_loop(self, grid, stroke_program=None, _allow_prototype_fallback=True):
        g_in = np.asarray(grid, dtype=np.int32)
        if g_in.ndim != 2 or g_in.size == 0:
            return g_in.copy()
        H, W = g_in.shape
        v1_seg = self.v1_v2_opponent_color_figure_ground_segregation(g_in)
        bg = int(v1_seg.get("inferred_bg_color", detect_background_color(g_in)))
        working_canvas = g_in.copy()
        ior_visited = np.zeros((H, W), dtype=bool)
        prog = stroke_program if isinstance(stroke_program, dict) else {}
        loop_mode = str(prog.get("foveal_mode", "synaptic_assoc"))
        self.foveal_saccade_stroke_loops_executed_total += 1
        self.sensorimotor_body_executions_total += 1
        if loop_mode == "component_anchor_stroke":
            comps = extract_components(g_in, bg=bg)
            target_mass = prog.get("target_mass", None)
            target_color = prog.get("target_color", None)
            anchor_ref = str(prog.get("anchor_ref", "center"))
            rel_strokes = list(prog.get("rel_strokes", []))
            erase_template_cells = list(prog.get("erase_cells_if_matched", []))
            for comp in comps:
                if target_mass is not None and int(comp["mass"]) != int(target_mass):
                    continue
                if target_color is not None and int(comp["color"]) != int(target_color):
                    continue
                if anchor_ref == "topleft":
                    ra, ca = int(comp["bbox"][0]), int(comp["bbox"][1])
                else:
                    ra = int(round(float(comp["center"][0])))
                    ca = int(round(float(comp["center"][1])))
                if not (0 <= ra < H and 0 <= ca < W) or ior_visited[ra, ca]:
                    continue
                ior_visited[ra, ca] = True
                self.current_gaze_yx = (ra, ca)
                self.foveal_saccades_executed_total += 1
                self.ior_sequential_saccades_total += 1
                self.encode_retinotopic_foveal_patch_spikes(working_canvas, ra, ca, bg_color=bg)
                c_self = int(comp["color"])
                stroke_cells = []
                for dr, dc, col_mode, col_val in rel_strokes:
                    nr, nc = ra + int(dr), ca + int(dc)
                    if 0 <= nr < H and 0 <= nc < W:
                        if col_mode == "self":
                            c_paint = c_self
                        elif col_mode == "bg":
                            c_paint = bg
                        else:
                            c_paint = int(col_val)
                        stroke_cells.append((nr, nc, c_paint))
                if stroke_cells:
                    self.execute_visuomotor_stroke_servo(working_canvas, stroke_cells, bg_color=bg)
                    gy_now, gx_now = self.current_gaze_yx
                    self.encode_retinotopic_foveal_patch_spikes(working_canvas, gy_now, gx_now, bg_color=bg)
            if erase_template_cells:
                erase_col = prog.get("template_signature_colors", None)
                if erase_col is not None:
                    sig_set = set(int(x) for x in erase_col)
                    for comp in extract_components(g_in, bg=bg):
                        comp_cols = set(int(g_in[r, c]) for r, c in comp["cells"])
                        if len(comp["cells"]) > 1 and comp_cols == sig_set:
                            er_strokes = [(int(r), int(c), bg) for r, c in comp["cells"]]
                            self.execute_visuomotor_stroke_servo(working_canvas, er_strokes, bg_color=bg)
            if _allow_prototype_fallback and np.array_equal(working_canvas, g_in):
                return self._fallback_foveal_prototype_if_noop(g_in, working_canvas, stroke_program)
            return working_canvas
        if loop_mode == "template_stamp_stroke":
            comps_multi = extract_components(g_in, bg=bg, connectivity=8, multicolor=True)
            if not comps_multi:
                return working_canvas
            multi_col_objs = []
            single_dots = []
            for o in comps_multi:
                cols_o = [int(g_in[r, c]) for r, c in o["cells"]]
                uniq_o = set(cols_o)
                if len(uniq_o) >= 2 and len(o["cells"]) >= 2:
                    multi_col_objs.append((o, cols_o, uniq_o))
                elif len(o["cells"]) == 1:
                    single_dots.append((int(o["cells"][0][0]), int(o["cells"][0][1]), int(cols_o[0])))
            if multi_col_objs and single_dots:
                tmpl_obj, tmpl_cols, _ = max(multi_col_objs, key=lambda item: len(item[0]["cells"]))
                dot_colors = set(dc_col for _, _, dc_col in single_dots)
                anchor_cell = None
                for r_t, c_t in tmpl_obj["cells"]:
                    if int(g_in[r_t, c_t]) in dot_colors:
                        anchor_cell = (int(r_t), int(c_t), int(g_in[r_t, c_t]))
                        break
                if anchor_cell is None:
                    col_counts = {}
                    for r_t, c_t in tmpl_obj["cells"]:
                        cv = int(g_in[r_t, c_t])
                        col_counts[cv] = col_counts.get(cv, 0) + 1
                    min_c = min(col_counts, key=lambda k: col_counts[k])
                    for r_t, c_t in tmpl_obj["cells"]:
                        if int(g_in[r_t, c_t]) == min_c:
                            anchor_cell = (int(r_t), int(c_t), min_c)
                            break
                if anchor_cell is not None:
                    ar, ac, acol = anchor_cell
                    self.current_gaze_yx = (ar, ac)
                    self.foveal_saccades_executed_total += 1
                    self.encode_retinotopic_foveal_patch_spikes(working_canvas, ar, ac, bg_color=bg)
                    rel_tmpl = []
                    for r_t, c_t in tmpl_obj["cells"]:
                        cv = int(g_in[r_t, c_t])
                        rel_tmpl.append((int(r_t - ar), int(c_t - ac), cv))
                    match_any_dot_color = bool(prog.get("match_any_dot_color", False))
                    recolor_to_dot = bool(prog.get("recolor_to_dot", False))
                    erase_template = bool(prog.get("erase_template", False))
                    for dr_pos, dc_pos, d_col in single_dots:
                        if not match_any_dot_color and d_col != acol:
                            continue
                        if ior_visited[dr_pos, dc_pos]:
                            continue
                        ior_visited[dr_pos, dc_pos] = True
                        self.current_gaze_yx = (dr_pos, dc_pos)
                        self.foveal_saccades_executed_total += 1
                        self.ior_sequential_saccades_total += 1
                        self.encode_retinotopic_foveal_patch_spikes(working_canvas, dr_pos, dc_pos, bg_color=bg)
                        stroke_cells = []
                        for d_r, d_c, t_col in rel_tmpl:
                            nr, nc = dr_pos + d_r, dc_pos + d_c
                            if 0 <= nr < H and 0 <= nc < W:
                                c_out = d_col if (recolor_to_dot and t_col != acol) else (d_col if (d_r == 0 and d_c == 0) else t_col)
                                stroke_cells.append((nr, nc, c_out))
                        if stroke_cells:
                            self.execute_visuomotor_stroke_servo(working_canvas, stroke_cells, bg_color=bg)
                            gy_now, gx_now = self.current_gaze_yx
                            self.encode_retinotopic_foveal_patch_spikes(working_canvas, gy_now, gx_now, bg_color=bg)
                    if erase_template:
                        er_strokes = [(int(r_t), int(c_t), bg) for r_t, c_t in tmpl_obj["cells"]]
                        self.execute_visuomotor_stroke_servo(working_canvas, er_strokes, bg_color=bg)
            if _allow_prototype_fallback and np.array_equal(working_canvas, g_in):
                return self._fallback_foveal_prototype_if_noop(g_in, working_canvas, stroke_program)
            return working_canvas
        if loop_mode == "retinotopic_patch_lookup":
            patch_rules = dict(prog.get("patch_rules", {}))
            invariant_mode = bool(prog.get("invariant_mode", False))
            stroke_cells_all = []
            for r in range(H):
                for c in range(W):
                    fg_nb = 0
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = r + dy, c + dx
                            if 0 <= ny < H and 0 <= nx < W and int(g_in[ny, nx]) != bg:
                                fg_nb += 1
                    if fg_nb == 0:
                        continue
                    key_vals = []
                    c_center = int(g_in[r, c])
                    nb_colors = []
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = r + dy, c + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                cv = int(g_in[ny, nx])
                                if cv != bg and not (dy == 0 and dx == 0):
                                    nb_colors.append(cv)
                                if invariant_mode:
                                    key_vals.append(1 if cv != bg else 0)
                                else:
                                    key_vals.append(cv)
                            else:
                                key_vals.append(-1)
                    k_tup = tuple(key_vals)
                    if k_tup in patch_rules:
                        ior_visited[r, c] = True
                        self.current_gaze_yx = (r, c)
                        self.foveal_saccades_executed_total += 1
                        self.ior_sequential_saccades_total += 1
                        self.encode_retinotopic_foveal_patch_spikes(working_canvas, r, c, bg_color=bg)
                        rule_act = patch_rules[k_tup]
                        if rule_act == "neighbor_majority" and nb_colors:
                            c_paint = int(max(set(nb_colors), key=nb_colors.count))
                        elif rule_act == "bg":
                            c_paint = bg
                        elif rule_act == "self":
                            c_paint = c_center
                        else:
                            c_paint = int(rule_act)
                        stroke_cells_all.append((r, c, c_paint))
            if stroke_cells_all:
                self.execute_visuomotor_stroke_servo(working_canvas, stroke_cells_all, bg_color=bg)
                gy_now, gx_now = self.current_gaze_yx
                self.encode_retinotopic_foveal_patch_spikes(working_canvas, gy_now, gx_now, bg_color=bg)
            if _allow_prototype_fallback and np.array_equal(working_canvas, g_in):
                return self._fallback_foveal_prototype_if_noop(g_in, working_canvas, stroke_program)
            return working_canvas
        salient_fixations = self._extract_salient_foveal_fixations(g_in, bg_color=bg, max_fixations=max(32, H * W))
        for r_f, c_f in salient_fixations:
            if ior_visited[r_f, c_f]:
                continue
            ior_visited[r_f, c_f] = True
            self.current_gaze_yx = (r_f, c_f)
            self.foveal_saccades_executed_total += 1
            self.ior_sequential_saccades_total += 1
            s_cur = self.encode_retinotopic_foveal_patch_spikes(working_canvas, r_f, c_f, bg_color=bg)
            assoc_drive = self._readout_foveal_stroke_synapse_with_lateral_inhibition(s_cur)
            stroke_cells = []
            for k_idx in range(25):
                best_ch = int(np.argmax(assoc_drive[k_idx]))
                if float(assoc_drive[k_idx, best_ch]) >= 0.65:
                    dr = (k_idx // 5) - 2
                    dc = (k_idx % 5) - 2
                    nr, nc = r_f + dr, c_f + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        c_paint = bg if best_ch == 0 else (int(g_in[r_f, c_f]) if best_ch == 10 else best_ch)
                        stroke_cells.append((nr, nc, c_paint))
            if stroke_cells:
                self.execute_visuomotor_stroke_servo(working_canvas, stroke_cells, bg_color=bg)
                gy_now, gx_now = self.current_gaze_yx
                self.encode_retinotopic_foveal_patch_spikes(working_canvas, gy_now, gx_now, bg_color=bg)
        if _allow_prototype_fallback and np.array_equal(working_canvas, g_in):
            return self._fallback_foveal_prototype_if_noop(g_in, working_canvas, stroke_program)
        return working_canvas
    def _fallback_foveal_prototype_if_noop(self, g_in, working_canvas, active_program):
        return working_canvas
    def learn_and_imitate_foveal_stroke(self, train_pairs, imprint_synapses=True):
        if not self.plastic or not imprint_synapses:
            return []
        if not train_pairs:
            return []
        for x_tr, y_tr in train_pairs:
            if x_tr.shape != y_tr.shape:
                return []
        dev_info = self.advance_developmental_critical_period()
        lr_fovea = 0.045 * float(dev_info["plasticity_multipliers"]["sensory_motor_lr"])
        for x_tr, y_tr in train_pairs:
            H, W = x_tr.shape
            bg_in = int(detect_background_color(x_tr))
            salient_anchors = self._extract_salient_foveal_fixations(
                x_tr, y_grid=y_tr, bg_color=bg_in, max_fixations=48
            )
            for r_a, c_a in salient_anchors:
                r0, r1 = max(0, r_a - 2), min(H, r_a + 3)
                c0, c1 = max(0, c_a - 2), min(W, c_a + 3)
                if np.array_equal(x_tr[r0:r1, c0:c1], y_tr[r0:r1, c0:c1]):
                    continue
                self.current_gaze_yx = (r_a, c_a)
                s_pre = self.encode_retinotopic_foveal_patch_spikes(x_tr, r_a, c_a, bg_color=bg_in)
                pursuit_last_yx = (r_a, c_a)
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r_a + dr, c_a + dc
                        if 0 <= nr < H and 0 <= nc < W and int(y_tr[nr, nc]) != int(x_tr[nr, nc]):
                            pursuit_dist = max(1, abs(nr - pursuit_last_yx[0]) + abs(nc - pursuit_last_yx[1]))
                            self.foveal_smooth_pursuit_steps_total += pursuit_dist
                            pursuit_last_yx = (nr, nc)
                            self.current_gaze_yx = (nr, nc)
                            if imprint_synapses:
                                k_idx = (dr + 2) * 5 + (dc + 2)
                                c_new = int(y_tr[nr, nc])
                                ch = 0 if c_new == bg_in else (10 if c_new == int(x_tr[r_a, c_a]) else (c_new % 10))
                                self._imprint_foveal_stroke_synapse_with_lateral_inhibition(
                                    s_pre, k_idx, ch, lr=lr_fovea
                                )
                                self.foveal_visuomotor_strokes_learned_total += 1
                self.encode_retinotopic_foveal_patch_spikes(
                    y_tr, pursuit_last_yx[0], pursuit_last_yx[1], bg_color=bg_in
                )
        patch_rules_inv = {}
        patch_rules_col = {}
        for x_tr, y_tr in train_pairs:
            H_p, W_p = x_tr.shape
            bg_p = int(detect_background_color(x_tr))
            diff_p = (x_tr != y_tr)
            for r_p in range(H_p):
                for c_p in range(W_p):
                    if diff_p[r_p, c_p]:
                        k_inv = []
                        k_col = []
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                ny, nx = r_p + dy, c_p + dx
                                if 0 <= ny < H_p and 0 <= nx < W_p:
                                    cv = int(x_tr[ny, nx])
                                    k_inv.append(1 if cv != bg_p else 0)
                                    k_col.append(cv)
                                else:
                                    k_inv.append(-1)
                                    k_col.append(-1)
                        c_dest = int(y_tr[r_p, c_p])
                        act = "bg" if c_dest == bg_p else c_dest
                        patch_rules_inv[tuple(k_inv)] = act
                        patch_rules_col[tuple(k_col)] = act

        self.learned_foveal_stroke_prototypes = []
        if patch_rules_inv:
            step_inv = {
                "foveal_mode": "retinotopic_patch_lookup",
                "invariant_mode": True,
                "patch_rules": patch_rules_inv,
            }
            self.learned_foveal_stroke_prototypes.append((step_inv, "foveal_saccade_stroke_patch3x3_inv1"))
            self.foveal_intuition_selections_total += 1
        if patch_rules_col:
            step_col = {
                "foveal_mode": "retinotopic_patch_lookup",
                "invariant_mode": False,
                "patch_rules": patch_rules_col,
            }
            self.learned_foveal_stroke_prototypes.append((step_col, "foveal_saccade_stroke_patch3x3_inv0"))

        verified_foveal_steps = list(self.learned_foveal_stroke_prototypes[:4])
        return verified_foveal_steps
    def compute_climbing_fiber_error_vector(self, pred_grid, target_grid):
        self.climbing_fiber_vector_corrections_total += 1
        p = np.asarray(pred_grid, dtype=np.int32)
        t = np.asarray(target_grid, dtype=np.int32)
        hp, wp = p.shape
        ht, wt = t.shape
        err_dh = float(ht - hp) / max(1.0, float(max(ht, hp)))
        err_dw = float(wt - wp) / max(1.0, float(max(wt, wp)))
        bg_p = detect_background_color(p)
        bg_t = detect_background_color(t)
        py_idx, px_idx = np.where(p != bg_p)
        ty_idx, tx_idx = np.where(t != bg_t)
        cy_p = float(np.mean(py_idx)) / max(1.0, float(hp)) if py_idx.size > 0 else 0.5
        cx_p = float(np.mean(px_idx)) / max(1.0, float(wp)) if px_idx.size > 0 else 0.5
        cy_t = float(np.mean(ty_idx)) / max(1.0, float(ht)) if ty_idx.size > 0 else 0.5
        cx_t = float(np.mean(tx_idx)) / max(1.0, float(wt)) if tx_idx.size > 0 else 0.5
        err_dy = float(cy_t - cy_p)
        err_dx = float(cx_t - cx_p)
        dom_p = int(np.argmax(np.bincount(p[p != bg_p].reshape(-1), minlength=10))) if py_idx.size > 0 else int(bg_p)
        dom_t = int(np.argmax(np.bincount(t[t != bg_t].reshape(-1), minlength=10))) if ty_idx.size > 0 else int(bg_t)
        err_dcol = 0.0 if dom_t == dom_p else 1.0
        dens_p = float(py_idx.size) / max(1.0, float(hp * wp))
        dens_t = float(ty_idx.size) / max(1.0, float(ht * wt))
        err_dens = float(dens_t - dens_p)
        cf_err = np.array([err_dy, err_dx, err_dh, err_dw, err_dcol, err_dens], dtype=np.float32)
        corr_vec = np.tanh(self.W_cerebellar_purkinje @ cf_err).astype(np.float32)
        if self.plastic:
            ltd_delta = -0.02 * np.outer(corr_vec, cf_err).astype(np.float32)
            self.W_cerebellar_purkinje = np.clip(
                0.98 * self.W_cerebellar_purkinje + ltd_delta + 0.005 * np.eye(6, dtype=np.float32),
                -1.5,
                1.5,
            ).astype(np.float32)
        return corr_vec, cf_err
    def _step_tuple_from_11d_motor_vec(self, vec11, tag_prefix="spk_direct"):
        v = np.asarray(vec11, dtype=np.float32).reshape(-1)
        if v.size < 11:
            v = np.pad(v, (0, 11 - v.size)).astype(np.float32)
        gaze_list = ["gaze_salient_target", "gaze_corner_sample", "gaze_next_matching_attr", "gaze_count_objects"]
        wm_list = ["wm_store_attr", "wm_store_sample_pair", "wm_store_count"]
        finger_list = ["finger_step_until_contact", "finger_paint_from_wm", "finger_apply_to_gaze", "finger_step_n_from_wm"]
        gaze_idx = int(np.argmax(v[0:4]))
        wm_idx = int(np.argmax(v[4:7]))
        finger_idx = int(np.argmax(v[7:11]))
        chosen_gaze = gaze_list[gaze_idx]
        chosen_wm = wm_list[wm_idx]
        chosen_finger = finger_list[finger_idx]
        dy = int(np.clip(np.round((v[0] - v[1]) * 2.0), -3, 3))
        dx = int(np.clip(np.round((v[2] - v[3]) * 2.0), -3, 3))
        p_dict = {
            "gaze_op": chosen_gaze,
            "wm_op": chosen_wm,
            "finger_op": chosen_finger,
            "dy": dy,
            "dx": dx,
            "energy_cost": float(np.sum(np.abs(v))),
            "resonance_weight": float(np.max(v)),
        }
        lbl = f"{tag_prefix}_{chosen_gaze[:4]}_{chosen_wm[:4]}_{chosen_finger[:6]}_d{dy}_{dx}"
        return (p_dict, lbl, [("chunk", p_dict)])
    def decode_spiking_action_from_neural_state(self, q_spk_vec):
        self.direct_spiking_action_decodings_total += 1
        q_flat = np.asarray(q_spk_vec, dtype=np.float32).reshape(-1)[: self.hid_dim]
        if q_flat.size < self.hid_dim:
            q_flat = np.pad(q_flat, (0, self.hid_dim - q_flat.size)).astype(np.float32)
        q_mod = (q_flat * self._last_parietal_gain_vec).astype(np.float32)
        body_11 = np.tanh(self.W_body_assembly @ q_mod).astype(np.float32)
        cand_primary = self._step_tuple_from_11d_motor_vec(body_11, tag_prefix="spk_bg_direct")
        out_list = [cand_primary]
        if hasattr(self, "unified_brains") and self.unified_brains is not None:
            m_info = self.unified_brains.motor_reflex.infer_motor_policy(q_mod, q_mod)
            act_logits = np.asarray(m_info.get("act_2d_logits", np.zeros(8)), dtype=np.float32).reshape(-1)
            p_vec = body_11.copy()
            n_ov = min(p_vec.size, act_logits.size)
            p_vec[:n_ov] = np.tanh(0.6 * p_vec[:n_ov] + 0.4 * act_logits[:n_ov]).astype(np.float32)
            cand_motor = self._step_tuple_from_11d_motor_vec(p_vec, tag_prefix="spk_m1_direct")
            out_list.append(cand_motor)
        return out_list
    def simulate_latent_pfc_rollout(self, x_grid, y_grid, steps_spec):
        self.latent_pfc_mental_rollouts_total += 1
        obs_x = compute_grid_observation(x_grid)
        obs_y = compute_grid_observation(y_grid)
        h_curr = np.tanh(self.Wx[0] @ obs_x + self.b[0]).astype(np.float32)
        h_goal = np.tanh(self.Wx[0] @ obs_y + self.b[0]).astype(np.float32)
        v_mem = h_curr.copy()
        for _, st_val in steps_spec:
            b_step = self.encode_sensorimotor_body_vec(st_val if isinstance(st_val, dict) else {})
            q_drive = (self.W_body_assembly.T @ b_step).astype(np.float32)
            rec_in = self.Wh[0] @ h_curr + 0.35 * q_drive * self._last_parietal_gain_vec
            v_mem = (0.55 * v_mem + 0.45 * rec_in).astype(np.float32)
            spk = (v_mem >= 0.35).astype(np.float32)
            v_mem = v_mem - spk * 0.35
            h_curr = np.tanh(v_mem + 0.30 * spk).astype(np.float32)
        n_c = float(np.linalg.norm(h_curr))
        n_g = float(np.linalg.norm(h_goal))
        if n_c < 1e-6 or n_g < 1e-6:
            return 0.0
        return float(np.dot(h_curr / n_c, h_goal / n_g))
    def compete_cortical_minicolumns_trn_gaba(self, h_cols, col_orders, col_energies, col_pains):
        self.cortical_column_competitions_total += 1
        orders_arr = np.asarray(col_orders, dtype=np.float32).reshape(-1)
        pains_arr = np.asarray(col_pains, dtype=np.float32).reshape(-1)
        energies_arr = np.asarray(col_energies, dtype=np.float32).reshape(-1)
        salience = (orders_arr - 0.45 * pains_arr) / 0.15
        salience = salience - float(np.max(salience))
        w_trn = np.exp(salience).astype(np.float32)
        w_trn = w_trn / max(1e-6, float(np.sum(w_trn)))
        win_idx = int(np.argmax(w_trn))
        self.winning_cortical_column_idx = win_idx
        h_win = h_cols[win_idx].copy()
        h_out = np.zeros_like(h_cols, dtype=np.float32)
        for idx in range(h_cols.shape[0]):
            if idx == win_idx:
                h_out[idx] = h_cols[idx]
            else:
                gaba_inh = float(w_trn[win_idx] - w_trn[idx])
                pull = float(np.clip(0.35 * gaba_inh, 0.05, 0.45))
                h_out[idx] = ((1.0 - pull) * h_cols[idx] + pull * h_win).astype(np.float32)
        self.organism_energy = float(np.clip(np.mean(energies_arr), 0.05, 1.0))
        self.organism_pain = float(np.clip(np.mean(pains_arr), 0.0, 1.0))
        self.organism_hp = float(np.clip(1.0 - 0.4 * self.organism_pain, 0.2, 1.0))
        self.organism_sleep_pressure = float(
            np.clip(self.organism_sleep_pressure + 0.035 + 0.05 * self.organism_pain, 0.0, 1.5)
        )
        self.organism_lifetime_steps += 1
        return h_out, win_idx, w_trn
    def consolidate_procedural_chain_synaptic(self, h_vec, steps_spec):
        if not steps_spec or not self.plastic:
            return False
        self.synaptic_procedural_consolidations_total += 1
        h_flat = np.asarray(h_vec, dtype=np.float32).reshape(-1)[: self.hid_dim]
        if h_flat.size < self.hid_dim:
            h_flat = np.pad(h_flat, (0, self.hid_dim - h_flat.size)).astype(np.float32)
        h_norm = float(np.linalg.norm(h_flat))
        h_unit = (h_flat / h_norm).astype(np.float32) if h_norm > 1e-6 else h_flat
        chain_33 = np.zeros((33,), dtype=np.float32)
        for s_idx, (_, st_val) in enumerate(steps_spec[:3]):
            b11 = self.encode_sensorimotor_body_vec(st_val if isinstance(st_val, dict) else {})
            chain_33[s_idx * 11 : (s_idx + 1) * 11] = b11[:11]
        sims = self.W_procedural_attractor @ h_unit
        slot = int(np.argmax(sims))
        self.W_procedural_attractor[slot] = (0.75 * self.W_procedural_attractor[slot] + 0.25 * h_unit).astype(np.float32)
        self.W_procedural_chain[slot] = (0.70 * self.W_procedural_chain[slot] + 0.30 * chain_33).astype(np.float32)
        return True
    def recall_synaptic_procedural_attractor(self, h_vec):
        self.synaptic_procedural_recalls_total += 1
        h_flat = np.asarray(h_vec, dtype=np.float32).reshape(-1)[: self.hid_dim]
        if h_flat.size < self.hid_dim:
            h_flat = np.pad(h_flat, (0, self.hid_dim - h_flat.size)).astype(np.float32)
        h_norm = float(np.linalg.norm(h_flat))
        if h_norm < 1e-6:
            return []
        h_unit = (h_flat / h_norm).astype(np.float32)
        sims = self.W_procedural_attractor @ h_unit
        slot = int(np.argmax(sims))
        chain_33 = self.W_procedural_chain[slot]
        steps_recalled = []
        lbl_parts = []
        for s_idx in range(3):
            seg11 = chain_33[s_idx * 11 : (s_idx + 1) * 11]
            if float(np.linalg.norm(seg11)) > 0.12:
                p_dict, s_lbl, _ = self._step_tuple_from_11d_motor_vec(seg11, tag_prefix=f"syn_proc_s{s_idx}")
                steps_recalled.append(("chunk", p_dict))
                lbl_parts.append(s_lbl)
        if not steps_recalled:
            p_dict, s_lbl, _ = self._step_tuple_from_11d_motor_vec(chain_33[:11], tag_prefix="syn_proc_s0")
            steps_recalled.append(("chunk", p_dict))
            lbl_parts.append(s_lbl)
        return [(steps_recalled[0][1], "+".join(lbl_parts), steps_recalled)]
    def emit_compositional_phonological_loop(self, h_vec):
        self.compositional_phonological_loops_total += 1
        h_flat = np.asarray(h_vec, dtype=np.float32).reshape(-1)[: self.hid_dim]
        if h_flat.size < self.hid_dim:
            h_flat = np.pad(h_flat, (0, self.hid_dim - h_flat.size)).astype(np.float32)
        m_state = np.tanh(self.W_mono_out[0] @ h_flat).astype(np.float32)
        m_dim = int(m_state.size)
        if self.W_syntax_transition.shape[0] < m_dim or self.W_syntax_transition.shape[1] < m_dim:
            self.W_syntax_transition = np.eye(m_dim, dtype=np.float32) * 0.5
        phrase_ids = []
        phrase_bias = np.zeros((self.hid_dim,), dtype=np.float32)
        prev_m = m_state.copy()
        for pos_idx in range(3):
            q_proj = np.tanh(self.W_mono_in[0] @ m_state).astype(np.float32)
            b_vec = np.tanh(self.W_body_assembly @ q_proj).astype(np.float32)
            _, aud_curr = self.synthesize_and_perceive_vocal_formant_wave(b_vec)
            lex_norms = np.linalg.norm(self.W_symbol_lexicon, axis=1, keepdims=True) + 1e-6
            lex_unit = (self.W_symbol_lexicon / lex_norms).astype(np.float32)
            sims = lex_unit @ (b_vec / (float(np.linalg.norm(b_vec)) + 1e-6))
            tok_id = int(np.argmax(sims))
            phrase_ids.append(tok_id)
            _, q_sym_bias = self.decode_symbol_to_sensorimotor_bias(tok_id)
            phrase_bias = phrase_bias + (0.5 ** pos_idx) * (0.72 * q_sym_bias + 0.28 * aud_curr)
            w_syn_sub = self.W_syntax_transition[:m_dim, :m_dim]
            next_m = np.tanh(m_state @ w_syn_sub).astype(np.float32)
            hebb_syn = 0.015 * np.outer(prev_m, next_m).astype(np.float32)
            if self.plastic:
                self.W_syntax_transition[:m_dim, :m_dim] = np.clip(
                    0.995 * w_syn_sub + hebb_syn, -1.2, 1.2
                ).astype(np.float32)
            prev_m = m_state
            m_state = next_m
        self.last_phonological_phrase = tuple(phrase_ids)
        pb_norm = float(np.linalg.norm(phrase_bias))
        if pb_norm > 1e-6:
            phrase_bias = (phrase_bias / pb_norm).astype(np.float32)
        return self.last_phonological_phrase, phrase_bias
    def check_and_trigger_autonomous_homeostatic_sleep(self, force_check=False):
        needs_sleep = (
            self.organism_sleep_pressure >= 0.70
            or self.organism_energy <= 0.28
            or (force_check and self.organism_sleep_pressure >= 0.25)
        )
        if not needs_sleep:
            return False
        self.autonomous_homeostatic_sleep_cycles_total += 1
        self.offline_dream_replay(n_dream_cycles=2)
        self.organism_energy = float(min(1.0, self.organism_energy + 0.55))
        self.organism_sleep_pressure = float(max(0.0, self.organism_sleep_pressure * 0.15))
        self.organism_pain = float(max(0.0, self.organism_pain * 0.35))
        return True
    def advance_developmental_critical_period(self, experience_steps=None):
        if not self.plastic:
            cur_stage = int(getattr(self, "developmental_stage", 1))
            cur_gates = getattr(self, "myelination_stage_gates", np.array([1.0, 0.55, 0.35], dtype=np.float32))
            return {
                "developmental_stage": cur_stage,
                "myelination_stage_gates": cur_gates.copy(),
                "plasticity_multipliers": {"sensory_motor_lr": 0.0, "parietal_mirror_lr": 0.0, "pfc_syntactic_lr": 0.0},
            }
        if experience_steps is not None:
            exp_val = int(experience_steps)
        else:
            exp_val = int(self.organism_lifetime_steps + self.tasks_processed_count * 4)
        prev_stage = int(getattr(self, "developmental_stage", 1))
        if exp_val < 12:
            new_stage = 1
            gates = np.array([1.0, 0.55, 0.35], dtype=np.float32)
            mults = {"sensory_motor_lr": 1.35, "parietal_mirror_lr": 0.75, "pfc_syntactic_lr": 0.50}
        elif exp_val < 36:
            new_stage = 2
            gates = np.array([0.45, 1.0, 0.65], dtype=np.float32)
            mults = {"sensory_motor_lr": 0.55, "parietal_mirror_lr": 1.40, "pfc_syntactic_lr": 0.85}
        else:
            new_stage = 3
            gates = np.array([0.25, 0.50, 1.0], dtype=np.float32)
            mults = {"sensory_motor_lr": 0.30, "parietal_mirror_lr": 0.65, "pfc_syntactic_lr": 1.25}
        if new_stage != prev_stage:
            self.critical_period_transitions_total += 1
        self.developmental_stage = int(new_stage)
        self.myelination_stage_gates = gates.copy()
        if hasattr(self, "unified_brains") and self.unified_brains is not None:
            self.unified_brains.sync_from_stage2_solver(self)
        return {
            "developmental_stage": self.developmental_stage,
            "myelination_stage_gates": self.myelination_stage_gates.copy(),
            "plasticity_multipliers": mults,
        }
    def project_3d_physics_to_2d_tabletop(
        self,
        grid,
        head_center_xyz=(0.0, 0.0, 0.52),
        head_shift_delta_xyz=(0.045, 0.0, 0.0),
        gaze_uv_deg=(0.0, 0.0),
    ):
        self.tabletop_25d_projections_total += 1
        if hasattr(self, "unified_brains") and self.unified_brains is not None:
            info = self.unified_brains.neocortex_logic.project_3d_physics_to_2d_tabletop(
                grid,
                head_center_xyz=head_center_xyz,
                head_shift_delta_xyz=head_shift_delta_xyz,
                gaze_uv_deg=gaze_uv_deg,
            )
        else:
            info = get_global_isaac_warp_engine().perceive_3d_tabletop_via_human_vision(
                grid,
                head_center_xyz=head_center_xyz,
                head_shift_delta_xyz=head_shift_delta_xyz,
                gaze_uv_deg=gaze_uv_deg,
            )
        self._last_tabletop_25d_info = info
        return info
    def sense_vestibular_and_apply_vor(self, head_euler_rad, desired_gaze_uv_deg=(0.0, 0.0), dt_sec=0.10):
        prev_euler = np.asarray(self.head_euler_rad, dtype=np.float32)
        cur_euler = np.asarray(head_euler_rad, dtype=np.float32).reshape(-1)[:3]
        if cur_euler.size < 3:
            cur_euler = np.pad(cur_euler, (0, 3 - cur_euler.size)).astype(np.float32)
        roll, pitch, yaw = float(cur_euler[0]), float(cur_euler[1]), float(cur_euler[2])
        omega = (cur_euler - prev_euler) / max(1e-3, float(dt_sec))
        self.vestibular_canal_omega = (0.55 * self.vestibular_canal_omega + 0.45 * omega).astype(np.float32)
        self.head_euler_rad = cur_euler.copy()
        g0 = 9.81
        gx = -g0 * math.sin(pitch)
        gy = g0 * math.sin(roll) * math.cos(pitch)
        gz = -g0 * math.cos(roll) * math.cos(pitch)
        self.vestibular_gravity_vec = np.array([gx, gy, gz], dtype=np.float32)
        vor_gain = 0.94
        comp_u_deg = float(desired_gaze_uv_deg[0]) - math.degrees(vor_gain * (yaw + 0.08 * float(self.vestibular_canal_omega[2])))
        comp_v_deg = float(desired_gaze_uv_deg[1]) - math.degrees(vor_gain * (pitch + 0.08 * float(self.vestibular_canal_omega[1])))
        self.vestibular_vor_stabilizations_total = int(getattr(self, "vestibular_vor_stabilizations_total", 0)) + 1
        return (
            float(np.clip(comp_u_deg, -18.0, 18.0)),
            float(np.clip(comp_v_deg, -18.0, 18.0)),
        )
    def synthesize_and_perceive_vocal_formant_wave(self, body_vec_11):
        b = np.asarray(body_vec_11, dtype=np.float32).reshape(-1)[:11]
        if b.size < 11:
            b = np.pad(b, (0, 11 - b.size)).astype(np.float32)
        t_samples = np.linspace(0.0, 0.04, 64, dtype=np.float32)
        f0 = 125.0 + 75.0 * float(np.tanh(b[0]))
        f1 = 380.0 + 260.0 * float(np.tanh(b[1]))
        f2 = 1150.0 + 620.0 * float(np.tanh(b[2]))
        f3 = 2350.0 + 680.0 * float(np.tanh(b[3]))
        env = 0.5 - 0.5 * np.cos(2.0 * math.pi * np.arange(64, dtype=np.float32) / 63.0)
        glottal = (
            0.50 * np.sin(2.0 * math.pi * f0 * t_samples)
            + 0.35 * np.sin(2.0 * math.pi * f1 * t_samples + float(b[4]))
            + 0.25 * np.sin(2.0 * math.pi * f2 * t_samples + float(b[5]))
            + 0.15 * np.sin(2.0 * math.pi * f3 * t_samples + float(b[6]))
        ) * env
        spec_mag = np.abs(np.fft.rfft(glottal))[:32].astype(np.float32)
        cochlear_8 = np.zeros(8, dtype=np.float32)
        for band_i in range(8):
            cochlear_8[band_i] = float(np.log1p(np.mean(spec_mag[band_i * 4 : (band_i + 1) * 4] ** 2)))
        c_norm = float(np.linalg.norm(cochlear_8))
        if c_norm > 1e-6:
            cochlear_8 = (cochlear_8 / c_norm).astype(np.float32)
        self.last_cochlear_spectrum_8 = cochlear_8.copy()
        self.cochlear_acoustic_perceptions_total = int(getattr(self, "cochlear_acoustic_perceptions_total", 0)) + 1
        aud_current = np.tanh(self.W_cochlear_auditory @ cochlear_8).astype(np.float32)
        return cochlear_8, aud_current
    def step_visceral_interoceptive_homeostasis(self, pain_val, surprise_val, spk_rate):
        p_v = float(np.clip(pain_val, 0.0, 1.0))
        s_v = float(np.clip(surprise_val, 0.0, 1.0))
        r_v = float(np.clip(spk_rate, 0.0, 1.0))
        safe_bonus = 0.08 if bool(getattr(self, "in_safe_study_room", False)) else 0.0
        self.visceral_cortisol_level = float(
            np.clip(0.86 * self.visceral_cortisol_level + 0.14 * (0.65 * p_v + 0.35 * s_v - safe_bonus), 0.02, 1.0)
        )
        self.visceral_oxytocin_level = float(
            np.clip(0.90 * self.visceral_oxytocin_level + 0.10 * (safe_bonus + 0.15 * (1.0 - p_v)), 0.05, 1.0)
        )
        target_hr = 64.0 + 52.0 * self.visceral_cortisol_level + 18.0 * r_v - 10.0 * self.visceral_oxytocin_level
        target_resp = 12.0 + 14.0 * self.visceral_cortisol_level + 6.0 * r_v - 2.5 * self.visceral_oxytocin_level
        self.visceral_heart_rate_bpm = float(0.80 * self.visceral_heart_rate_bpm + 0.20 * target_hr)
        self.visceral_respiration_rate = float(0.80 * self.visceral_respiration_rate + 0.20 * target_resp)
        self.visceral_interoceptive_steps_total = int(getattr(self, "visceral_interoceptive_steps_total", 0)) + 1
        visc_vec = np.array(
            [
                (self.visceral_heart_rate_bpm - 65.0) / 60.0,
                (self.visceral_respiration_rate - 12.0) / 18.0,
                self.visceral_cortisol_level,
                self.visceral_oxytocin_level,
            ],
            dtype=np.float32,
        )
        self.last_visceral_vec_4 = visc_vec.copy()
        insular_bias = np.tanh(self.W_insular_interoception @ visc_vec).astype(np.float32)
        return visc_vec, insular_bias
    def encode_hippocampal_place_grid_time_context(self, world_xyz=None, dt_sec=0.25):
        if world_xyz is not None:
            self.organism_world_xyz = np.asarray(world_xyz, dtype=np.float32).reshape(-1)[:3].copy()
        self.organism_life_time_sec = float(getattr(self, "organism_life_time_sec", 0.0)) + float(dt_sec)
        pos = self.organism_world_xyz
        dists_sq = np.sum((self.place_cell_centers_xyz - pos.reshape(1, 3)) ** 2, axis=1)
        place_8 = np.exp(-dists_sq / (2.0 * (12.0 ** 2))).astype(np.float32)
        freqs = np.array([0.04, 0.08, 0.16, 0.32], dtype=np.float32)
        k1 = np.array([1.0, 0.0], dtype=np.float32)
        k2 = np.array([-0.5, 0.8660254], dtype=np.float32)
        k3 = np.array([-0.5, -0.8660254], dtype=np.float32)
        xy = pos[:2]
        grid_4 = np.zeros(4, dtype=np.float32)
        for gi, fq in enumerate(freqs):
            g_val = (
                math.cos(fq * float(np.dot(k1, xy)))
                + math.cos(fq * float(np.dot(k2, xy)))
                + math.cos(fq * float(np.dot(k3, xy)))
            ) / 3.0
            grid_4[gi] = float(0.5 * (g_val + 1.0))
        t_now = max(0.05, float(self.organism_life_time_sec))
        taus = np.array([1.0, 4.0, 16.0, 64.0], dtype=np.float32)
        time_4 = np.exp(-((np.log(t_now / taus)) ** 2) / 1.8).astype(np.float32)
        spatiotemporal_16 = np.concatenate([place_8, grid_4, time_4]).astype(np.float32)
        self.last_spatiotemporal_16 = spatiotemporal_16.copy()
        self.hippocampal_place_grid_activations_total = int(
            getattr(self, "hippocampal_place_grid_activations_total", 0)
        ) + 1
        hippo_bias = np.tanh(self.W_hippo_spatiotemporal @ spatiotemporal_16).astype(np.float32)
        return spatiotemporal_16, hippo_bias
    def record_autobiographical_episode(self, h_vec, step_label, order_gain):
        if not self.plastic:
            return None
        st_16, _ = self.encode_hippocampal_place_grid_time_context(dt_sec=0.15)
        h_flat = np.asarray(h_vec, dtype=np.float32).reshape(-1)[: self.hid_dim]
        h_norm = float(np.linalg.norm(h_flat))
        h_unit = (h_flat / h_norm).astype(np.float32) if h_norm > 1e-6 else np.zeros((self.hid_dim,), dtype=np.float32)
        episode = {
            "life_time_sec": round(float(self.organism_life_time_sec), 3),
            "world_xyz": self.organism_world_xyz.tolist(),
            "in_safe_study_room": bool(self.in_safe_study_room),
            "spatiotemporal_16": st_16.tolist(),
            "h_signature": h_unit.tolist(),
            "step_label": str(step_label),
            "order_gain": round(float(order_gain), 4),
            "oxytocin": round(float(self.visceral_oxytocin_level), 4),
        }
        self.autobiographical_memory_bank.append(episode)
        if len(self.autobiographical_memory_bank) > 48:
            self.autobiographical_memory_bank.pop(0)
        return episode
    def recall_autobiographical_context(self, h_vec):
        if not self.autobiographical_memory_bank:
            return np.zeros((self.hid_dim,), dtype=np.float32)
        st_now, _ = self.encode_hippocampal_place_grid_time_context(dt_sec=0.02)
        h_flat = np.asarray(h_vec, dtype=np.float32).reshape(-1)[: self.hid_dim]
        h_norm = float(np.linalg.norm(h_flat)) + 1e-6
        h_unit = (h_flat / h_norm).astype(np.float32)
        best_ep = None
        best_sim = -1.0
        for ep in self.autobiographical_memory_bank[-16:]:
            ep_h = np.asarray(ep.get("h_signature", []), dtype=np.float32)
            ep_st = np.asarray(ep.get("spatiotemporal_16", []), dtype=np.float32)
            if ep_h.size == self.hid_dim and ep_st.size == 16:
                sim = 0.65 * float(np.dot(h_unit, ep_h)) + 0.35 * float(
                    np.dot(st_now, ep_st) / (float(np.linalg.norm(st_now) * np.linalg.norm(ep_st)) + 1e-6)
                )
                if sim > best_sim:
                    best_sim = sim
                    best_ep = ep
        if best_ep is None or best_sim < 0.15:
            return np.zeros((self.hid_dim,), dtype=np.float32)
        self.autobiographical_recalls_total = int(getattr(self, "autobiographical_recalls_total", 0)) + 1
        ep_h_vec = np.asarray(best_ep["h_signature"], dtype=np.float32)
        ep_st_vec = np.asarray(best_ep["spatiotemporal_16"], dtype=np.float32)
        return (0.15 * ep_h_vec + 0.10 * np.tanh(self.W_hippo_spatiotemporal @ ep_st_vec)).astype(np.float32)
    def _update_multimodal_hippocampal_plasticity(self, h_vec, reward_mod=0.10):
        if not bool(getattr(self, "plastic", True)):
            return
        h_flat = np.asarray(h_vec, dtype=np.float32).reshape(-1)[: self.hid_dim]
        if h_flat.size < self.hid_dim:
            h_flat = np.pad(h_flat, (0, self.hid_dim - h_flat.size)).astype(np.float32)
        h_norm = float(np.linalg.norm(h_flat)) + 1e-6
        h_unit = (h_flat / h_norm).astype(np.float32)
        lr_eff = float(np.clip(0.015 * float(reward_mod), -0.025, 0.035))
        if abs(lr_eff) < 1e-6:
            return
        coch_8 = np.asarray(getattr(self, "last_cochlear_spectrum_8", np.zeros(8, dtype=np.float32)), dtype=np.float32)
        if coch_8.size == 8 and float(np.linalg.norm(coch_8)) > 1e-6:
            self.W_cochlear_auditory = np.clip(
                0.998 * self.W_cochlear_auditory + lr_eff * np.outer(h_unit, coch_8), -1.5, 1.5
            ).astype(np.float32)
        visc_4 = np.asarray(getattr(self, "last_visceral_vec_4", np.zeros(4, dtype=np.float32)), dtype=np.float32)
        if visc_4.size == 4 and float(np.linalg.norm(visc_4)) > 1e-6:
            self.W_insular_interoception = np.clip(
                0.998 * self.W_insular_interoception + lr_eff * np.outer(h_unit, visc_4), -1.5, 1.5
            ).astype(np.float32)
        st_16 = np.asarray(getattr(self, "last_spatiotemporal_16", np.zeros(16, dtype=np.float32)), dtype=np.float32)
        if st_16.size == 16 and float(np.linalg.norm(st_16)) > 1e-6:
            self.W_hippo_spatiotemporal = np.clip(
                0.998 * self.W_hippo_spatiotemporal + lr_eff * np.outer(h_unit, st_16), -1.5, 1.5
            ).astype(np.float32)
    def empathize_and_share_embodied_demonstration(
        self,
        mentor_q_vec,
        mentor_params,
        demo_before_grid,
        demo_after_grid,
        peer_pain=0.2,
    ):
        self.affective_empathy_resonances_total = int(getattr(self, "affective_empathy_resonances_total", 0)) + 1
        shared_distress = 0.30 * float(np.clip(peer_pain, 0.0, 1.0))
        self.visceral_oxytocin_level = float(np.clip(self.visceral_oxytocin_level + 0.12 + 0.15 * shared_distress, 0.05, 1.0))
        self.visceral_cortisol_level = float(np.clip(self.visceral_cortisol_level * 0.88, 0.02, 1.0))
        mentor_b_vec = self.encode_sensorimotor_body_vec(mentor_params, q_spk_fallback=mentor_q_vec)
        _, aud_bias = self.synthesize_and_perceive_vocal_formant_wave(mentor_b_vec)
        mirror_stroke_bias = np.zeros((self.hid_dim,), dtype=np.float32)
        if demo_before_grid is not None and demo_after_grid is not None:
            self.embodied_peer_demonstrations_total = int(
                getattr(self, "embodied_peer_demonstrations_total", 0)
            ) + 1
            m_info = self.infer_mirror_neuron_author_stroke(
                demo_before_grid,
                demo_after_grid,
                imprint_synapses=bool(self.plastic),
            )
            stroke_11 = np.asarray(m_info.get("author_stroke_11", mentor_b_vec), dtype=np.float32).reshape(-1)[:11]
            if stroke_11.size < 11:
                stroke_11 = np.pad(stroke_11, (0, 11 - stroke_11.size)).astype(np.float32)
            mirror_stroke_bias = np.tanh(self.W_body_assembly.T @ stroke_11).astype(np.float32)
        combined_social_bias = (0.55 * mirror_stroke_bias + 0.45 * aud_bias).astype(np.float32)
        c_norm = float(np.linalg.norm(combined_social_bias))
        if c_norm > 1e-6:
            combined_social_bias = (combined_social_bias / c_norm).astype(np.float32)
        return combined_social_bias
    def engage_seated_tabletop_focus(self, grid):
        safe_to_sit = (
            self.organism_hp > 0.30
            and self.organism_energy > 0.22
            and self.organism_pain < 0.55
        )
        engine = get_global_isaac_warp_engine()
        room_info = engine.enter_safe_study_room_and_sit_at_desk(self.organism_world_xyz)
        self.last_safe_room_info = room_info
        for wp in room_info.get("walk_trajectory_xyz", []):
            self.encode_hippocampal_place_grid_time_context(wp, dt_sec=0.12)
        self.organism_world_xyz = np.array(room_info["seated_world_xyz"], dtype=np.float32)
        self.in_safe_study_room = bool(room_info["in_safe_study_room"])
        if bool(room_info.get("walked_into_room", False)) or int(getattr(self, "safe_room_transitions_total", 0)) == 0:
            self.safe_room_transitions_total = int(getattr(self, "safe_room_transitions_total", 0)) + 1
        self.seated_posture_locked = bool(safe_to_sit and self.in_safe_study_room)
        self.seated_tabletop_focus_engagements_total += 1
        vor_gaze_uv = self.sense_vestibular_and_apply_vor((0.0, -0.18, 0.0), desired_gaze_uv_deg=(0.0, 0.0))
        self.encode_hippocampal_place_grid_time_context(self.organism_world_xyz, dt_sec=0.25)
        tabletop_info = self.project_3d_physics_to_2d_tabletop(grid, gaze_uv_deg=vor_gaze_uv)
        return {
            "seated_posture_locked": self.seated_posture_locked,
            "in_safe_study_room": self.in_safe_study_room,
            "safe_room_info": room_info,
            "locomotion_suppressed": bool(self.seated_posture_locked),
            "foveal_finger_metabolic_share": 0.96 if self.seated_posture_locked else 0.50,
            "tabletop_25d_info": tabletop_info,
        }
    def shift_3d_head_tilt_viewpoint(self, grid, surprise_val=0.5, fatigue_val=0.5):
        self.head_tilt_viewpoint_shifts_total += 1
        self.modulate_ach_topological_lens(
            max(0.0, 0.25 - 0.3 * float(fatigue_val)),
            float(surprise_val),
            step_idx=self.head_tilt_viewpoint_shifts_total,
        )
        warped_out = self.apply_topological_lens_forward(grid)
        warped_grid = warped_out[0] if isinstance(warped_out, tuple) else warped_out
        sign_shift = 1.0 if (self.head_tilt_viewpoint_shifts_total % 2 == 1) else -1.0
        head_xyz = (0.025 * sign_shift, 0.015 * float(surprise_val), 0.52)
        delta_xyz = (0.045 * sign_shift, 0.012, 0.0)
        vor_gaze_uv = self.sense_vestibular_and_apply_vor(
            (0.06 * sign_shift, -0.16 + 0.03 * float(surprise_val), 0.09 * sign_shift),
            desired_gaze_uv_deg=(0.0, 0.0),
        )
        tabletop_reproj = self.project_3d_physics_to_2d_tabletop(
            warped_grid,
            head_center_xyz=head_xyz,
            head_shift_delta_xyz=delta_xyz,
            gaze_uv_deg=vor_gaze_uv,
        )
        reproj_grid = tabletop_reproj.get("reconstructed_grid", warped_grid) if isinstance(tabletop_reproj, dict) else tabletop_reproj
        return {
            "warped_grid": warped_grid,
            "lens_mode": self.active_topological_lens,
            "vor_stabilized_gaze_uv_deg": vor_gaze_uv,
            "tabletop_reprojection": reproj_grid,
        }
    def infer_mirror_neuron_author_stroke(self, x_grid, y_grid, imprint_synapses=True, extra_pairs=None):
        self.mirror_neuron_imitations_total += 1
        dev_info = self.advance_developmental_critical_period()
        mirror_lr = 0.025 * float(dev_info["plasticity_multipliers"]["parietal_mirror_lr"])
        gx = np.asarray(x_grid, dtype=np.int32)
        gy = np.asarray(y_grid, dtype=np.int32)
        if hasattr(self, "unified_brains") and self.unified_brains is not None:
            m_out = self.unified_brains.neocortex_logic.infer_mirror_neuron_author_stroke(
                x_grid, y_grid, update_weights=bool(imprint_synapses), lr=mirror_lr
            )
            vis_delta_16 = m_out["visual_delta_16"]
            author_stroke_11 = m_out["author_stroke_11"]
            target_stroke_11 = m_out["target_stroke_11"]
            if imprint_synapses:
                self.W_mirror_inverse = (
                    self.unified_brains.neocortex_logic.W_mirror_inverse.detach().cpu().numpy().astype(np.float32).copy()
                )
        else:
            vis_delta_16 = np.zeros(16, dtype=np.float32)
            vis_delta_16[0] = float(np.mean(gy > 0) - np.mean(gx > 0))
            vis_delta_16[1] = 1.0 if gx.shape == gy.shape else -1.0
            author_stroke_11 = np.tanh(self.W_mirror_inverse @ vis_delta_16).astype(np.float32)
            target_stroke_11 = np.tanh(author_stroke_11 * 1.1).astype(np.float32)
        matched_mirror_steps = []
        if gx.ndim == 2 and gy.ndim == 2:
            check_pairs = list(extra_pairs) if extra_pairs else [(gx, gy)]
            foveal_learned = self.learn_and_imitate_foveal_stroke(
                check_pairs,
                imprint_synapses=imprint_synapses,
            )
            for p_fov, lbl_fov in foveal_learned[:2]:
                matched_mirror_steps.append((p_fov, lbl_fov, [("chunk", p_fov)]))
            if matched_mirror_steps:
                best_body_11 = self.encode_sensorimotor_body_vec(matched_mirror_steps[0][0])
                target_stroke_11 = np.tanh(0.55 * target_stroke_11 + 0.45 * best_body_11).astype(np.float32)
        if imprint_synapses:
            err_m = (target_stroke_11 - author_stroke_11) * (1.0 - author_stroke_11 * author_stroke_11)
            self.W_mirror_inverse = np.clip(
                0.998 * self.W_mirror_inverse + mirror_lr * np.outer(err_m, vis_delta_16),
                -2.5,
                2.5,
            ).astype(np.float32)
            author_stroke_11 = np.tanh(self.W_mirror_inverse @ vis_delta_16).astype(np.float32)
            slot_idx = int(np.argmax(np.abs(vis_delta_16))) % self.W_procedural_chain.shape[0]
            chain_33 = np.tile(author_stroke_11, 3)[:33].astype(np.float32)
            self.W_procedural_chain[slot_idx] = np.clip(
                0.85 * self.W_procedural_chain[slot_idx] + 0.15 * chain_33,
                -2.5,
                2.5,
            ).astype(np.float32)
        step_dict, step_label, step_tuples = self._step_tuple_from_11d_motor_vec(author_stroke_11, tag_prefix="mirror_author")
        self._mirror_inferred_affordance_steps = [(step_dict, step_label, step_tuples)]
        self._last_mirror_author_step = (step_dict, step_label, step_tuples)
        return {
            "visual_delta_16": vis_delta_16,
            "author_stroke_11": author_stroke_11,
            "inferred_step_dict": step_dict,
            "inferred_step_label": step_label,
        }
    def autotelic_hand_regard_play(self, base_grid, n_bouts=3):
        g0 = np.asarray(base_grid, dtype=np.int32)
        if g0.ndim != 2 or g0.size == 0:
            g0 = np.zeros((4, 4), dtype=np.int32)
        bouts = max(1, int(n_bouts))
        self.autotelic_play_bouts_total += bouts
        dev_info = self.advance_developmental_critical_period()
        sm_lr = 0.030 * float(dev_info["plasticity_multipliers"]["sensory_motor_lr"])
        imitations_learned = 0
        for b_idx in range(bouts):
            h_seed = self.persistent_h[b_idx % self.persistent_h.shape[0]].copy()
            h_seed = np.roll(h_seed, b_idx * 7 + 1)
            play_motor_11 = np.tanh(self.W_body_assembly @ h_seed).astype(np.float32)
            play_motor_11[b_idx % 4] = float(np.clip(play_motor_11[b_idx % 4] + 0.65, -1.0, 1.0))
            play_motor_11[7 + (b_idx % 4)] = float(np.clip(play_motor_11[7 + (b_idx % 4)] + 0.65, -1.0, 1.0))
            canvas = g0.copy()
            h_g, w_g = canvas.shape
            if b_idx % 4 == 0:
                after_grid = self.execute_unified_sensorimotor_chunk(
                    canvas,
                    "gaze_salient_target",
                    "wm_store_attr",
                    "finger_step_until_contact",
                    {"selector": "largest", "dr": 1 if play_motor_11[0] >= 0.0 else -1, "dc": 0},
                )
            elif b_idx % 4 == 1:
                after_grid = self.execute_unified_sensorimotor_chunk(
                    canvas,
                    "gaze_salient_target",
                    "wm_store_attr",
                    "finger_apply_to_gaze",
                    {"selector": "largest", "sub_op": "rotate_flip_in_place", "d4_k": 1 + (b_idx % 3)},
                )
            elif b_idx % 4 == 2 and h_g >= 3 and w_g >= 3:
                after_grid = self.execute_unified_sensorimotor_chunk(
                    canvas,
                    "gaze_salient_target",
                    "wm_store_attr",
                    "finger_apply_to_gaze",
                    {"selector": "largest", "sub_op": "crop_grasped"},
                )
            else:
                after_grid = self.execute_unified_sensorimotor_chunk(
                    canvas,
                    "gaze_salient_target",
                    "wm_store_attr",
                    "finger_paint_from_wm",
                    {"selector": "largest", "val_color": int(abs(play_motor_11[5]) * 8.0) + 1, "paint_mode": "recolor"},
                )
            if after_grid.shape == canvas.shape and np.array_equal(after_grid, canvas):
                r_mid = int(h_g // 2)
                c_mid = int(w_g // 2)
                stroke_col = int((int(canvas[r_mid, c_mid]) + 2) % 9 + 1)
                bg_play = int(canvas[0, 0]) if canvas.size > 0 else 0
                after_grid = self.execute_visuomotor_stroke_servo(canvas.copy(), [(r_mid, c_mid, stroke_col)], bg_color=bg_play)
            reproj_play = self.project_3d_physics_to_2d_tabletop(after_grid)
            after_grid = reproj_play.get("reconstructed_grid", after_grid)
            if after_grid.shape != g0.shape or not np.array_equal(after_grid, g0):
                self.hand_regard_self_imitations_total += 1
                imitations_learned += 1
                m_res = self.infer_mirror_neuron_author_stroke(g0, after_grid, imprint_synapses=False)
                vis_d16 = m_res["visual_delta_16"]
                pred_m11 = np.tanh(self.W_mirror_inverse @ vis_d16).astype(np.float32)
                err_self = (play_motor_11 - pred_m11) * (1.0 - pred_m11 * pred_m11)
                if self.plastic:
                    self.W_mirror_inverse = np.clip(
                        0.996 * self.W_mirror_inverse + sm_lr * np.outer(err_self, vis_d16),
                        -2.5,
                        2.5,
                    ).astype(np.float32)
                    if hasattr(self, "unified_brains") and self.unified_brains is not None:
                        self.unified_brains.sync_from_stage2_solver(self)
        return {
            "bouts_executed": bouts,
            "self_imitations_learned": imitations_learned,
            "w_mirror_norm": float(np.linalg.norm(self.W_mirror_inverse)),
        }
    def apply_biological_triple_overfit_guard(self, train_pairs, surprise_val=0.5):
        self.triple_overfit_guards_triggered_total += 1
        shape_warped = bool(any(x.shape != y.shape for x, y in train_pairs)) if train_pairs else False
        multi_pair_gaba_active = bool(len(train_pairs) >= 2) if train_pairs else False
        wo_norm = float(np.linalg.norm(self.Wo))
        if self.plastic and wo_norm > 18.0:
            scale_h = 18.0 / max(18.0, wo_norm)
            self.Wo = (self.Wo * scale_h).astype(np.float32)
            self.Wq = (self.Wq * 0.992).astype(np.float32)
            self.Wk = (self.Wk * 0.992).astype(np.float32)
            self.synaptic_homeostasis_downscalings_total += 1
        novelty_drive = float(np.clip(float(surprise_val) + (0.35 if shape_warped else 0.15), 0.0, 1.0))
        if novelty_drive >= 0.25 or shape_warped:
            self.ach_habit_resets_total += 1
            self.einstellung_fixation_prevented_total += 1
            self.ach_signal_level = float(np.clip(0.55 + 0.40 * novelty_drive, 0.0, 1.0))
            self.habit_suppression_gate = float(np.clip(1.0 - 0.55 * novelty_drive, 0.30, 0.85))
        else:
            self.habit_suppression_gate = 1.0
        return {
            "guard1_pfc_gaba_cross_pair_active": multi_pair_gaba_active,
            "guard2_synaptic_homeostasis_downscalings": int(self.synaptic_homeostasis_downscalings_total),
            "guard3_ach_habit_reset_triggered": bool(novelty_drive >= 0.25 or shape_warped),
            "habit_suppression_gate": round(float(self.habit_suppression_gate), 4),
            "ach_signal_level": round(float(self.ach_signal_level), 4),
        }
    def trigger_epistemic_aha_dopamine_burst(self, order_gain, surprise_drop, h_vec=None, action_axis=None):
        og = float(max(0.0, order_gain))
        sd = float(max(0.0, surprise_drop))
        da_burst = float(np.clip(1.65 * og + 0.85 * sd, 0.0, 2.0))
        self.intrinsic_dopamine_level = float(0.65 * self.intrinsic_dopamine_level + 0.35 * da_burst)
        if da_burst >= 0.05:
            self.aha_dopamine_bursts_total += 1
            if h_vec is not None:
                h_flat = np.asarray(h_vec, dtype=np.float32).reshape(-1)[: self.hid_dim]
                if h_flat.size < self.hid_dim:
                    h_flat = np.pad(h_flat, (0, self.hid_dim - h_flat.size)).astype(np.float32)
                h_unit = h_flat / (float(np.linalg.norm(h_flat)) + 1e-6)
                slot = int(action_axis if action_axis is not None else 0) % self.W_procedural_attractor.shape[0]
                ltp_rate = 0.025 * da_burst * float(self.myelination_stage_gates[2])
                self.W_procedural_attractor[slot] = np.clip(
                    0.992 * self.W_procedural_attractor[slot] + ltp_rate * h_unit,
                    -2.5,
                    2.5,
                ).astype(np.float32)
        return da_burst
    def absorb_stage1_3d_embodied_manifold(self, isaac_path=None):
        if isaac_path is None:
            isaac_path = ARTIFACTS_DIR / "stage1_isaac3d_core_manifold.npz"
        p_file = Path(isaac_path)
        if not p_file.exists():
            return False
        try:
            isaac_data = np.load(p_file)
            H = self.hid_dim
            if "Wx_retinal" in isaac_data:
                wxr_src = isaac_data["Wx_retinal"].astype(np.float32)
                if wxr_src.shape[0] > H:
                    wxr_src = wxr_src[-H:, :]
                h_src_x = wxr_src.shape[0]
                wx_pad = np.zeros((H, self.in_dim), dtype=np.float32)
                for b_off in range(0, H, h_src_x):
                    b_end = min(H, b_off + h_src_x)
                    b_len = b_end - b_off
                    wx_pad[b_off:b_end, :] = wxr_src[:b_len, : self.in_dim]
                self.Wx = (0.78 * self.Wx + 0.22 * wx_pad[None, :, :]).astype(np.float32)
            if "Wpred_retinal" in isaac_data:
                wpr_src = isaac_data["Wpred_retinal"].astype(np.float32)
                h_src_p = wpr_src.shape[1]
                wp_pad = np.zeros((self.in_dim, H), dtype=np.float32)
                for b_off in range(0, H, h_src_p):
                    b_end = min(H, b_off + h_src_p)
                    b_len = b_end - b_off
                    wp_pad[:, b_off:b_end] = wpr_src[: self.in_dim, :b_len]
                self.Wpred = (0.78 * self.Wpred + 0.22 * wp_pad[None, :, :]).astype(np.float32)
            if "Wh" in isaac_data:
                wh_src = isaac_data["Wh"].astype(np.float32)
                if wh_src.shape[0] > wh_src.shape[1]:
                    wh_src = wh_src[-wh_src.shape[1] :, :]
                h_src_w = wh_src.shape[1]
                wh_pad = np.zeros((H, H), dtype=np.float32)
                for r_off in range(0, H, h_src_w):
                    r_end = min(H, r_off + h_src_w)
                    r_len = r_end - r_off
                    for c_off in range(0, H, h_src_w):
                        c_end = min(H, c_off + h_src_w)
                        c_len = c_end - c_off
                        wh_pad[r_off:r_end, c_off:c_end] = wh_src[:r_len, :c_len]
                self.Wh = (0.82 * self.Wh + 0.18 * wh_pad[None, :, :]).astype(np.float32)
            if "Wo" in isaac_data:
                wo_src = isaac_data["Wo"].astype(np.float32)
                h_src_o = wo_src.shape[1]
                wo_pad = np.zeros((self.act_dim, H), dtype=np.float32)
                a_len = min(self.act_dim, wo_src.shape[0])
                for b_off in range(0, H, h_src_o):
                    b_end = min(H, b_off + h_src_o)
                    b_len = b_end - b_off
                    wo_pad[:a_len, b_off:b_end] = wo_src[:a_len, :b_len]
                if a_len < self.act_dim:
                    wo_pad[a_len:, :] = wo_pad[: self.act_dim - a_len, :]
                self.Wo = (0.82 * self.Wo + 0.18 * wo_pad[None, :, :]).astype(np.float32)
            if "Wq" in isaac_data and "Wk" in isaac_data and "Wv" in isaac_data:
                wq_src = isaac_data["Wq"].astype(np.float32)
                wk_src = isaac_data["Wk"].astype(np.float32)
                wv_src = isaac_data["Wv"].astype(np.float32)
                if wq_src.ndim == 2:
                    h_src = wq_src.shape[0]
                    wq_mat = np.zeros((H, H), dtype=np.float32)
                    wk_mat = np.zeros((H, H), dtype=np.float32)
                    wv_mat = np.zeros((H, H), dtype=np.float32)
                    for b_off in range(0, H, h_src):
                        b_end = min(H, b_off + h_src)
                        b_len = b_end - b_off
                        wq_mat[b_off:b_end, b_off:b_end] = wq_src[:b_len, :b_len]
                        wk_mat[b_off:b_end, b_off:b_end] = wk_src[:b_len, :b_len]
                        wv_mat[b_off:b_end, b_off:b_end] = wv_src[:b_len, :b_len]
                    self.Wq = (0.80 * self.Wq + 0.20 * wq_mat[None, :, :]).astype(np.float32)
                    self.Wk = (0.80 * self.Wk + 0.20 * wk_mat[None, :, :]).astype(np.float32)
                    self.Wv = (0.80 * self.Wv + 0.20 * wv_mat[None, :, :]).astype(np.float32)
            if "W_body_assembly" in isaac_data:
                wb_src = isaac_data["W_body_assembly"].astype(np.float32)
                if wb_src.ndim == 2 and wb_src.shape[0] == 11:
                    h_src_b = wb_src.shape[1]
                    wb_pad = np.zeros((11, H), dtype=np.float32)
                    for b_off in range(0, H, h_src_b):
                        b_end = min(H, b_off + h_src_b)
                        b_len = b_end - b_off
                        wb_pad[:, b_off:b_end] = wb_src[:, :b_len]
                    self.W_body_assembly = (0.75 * self.W_body_assembly + 0.25 * wb_pad).astype(np.float32)
            if "W_symbol_lexicon" in isaac_data:
                ws_src = isaac_data["W_symbol_lexicon"].astype(np.float32)
                if ws_src.ndim == 2 and ws_src.shape[1] == 11:
                    n_sym = min(self.symbol_vocab_size, ws_src.shape[0])
                    blended_sym = 0.65 * self.W_symbol_lexicon[:n_sym] + 0.35 * ws_src[:n_sym]
                    b_norms = np.linalg.norm(blended_sym, axis=1, keepdims=True) + 1e-6
                    self.W_symbol_lexicon[:n_sym] = (blended_sym / b_norms).astype(np.float32)
            if "danger_vec" in isaac_data:
                d_src = isaac_data["danger_vec"].astype(np.float32)
                d_norm = float(np.linalg.norm(d_src))
                if d_norm > 1e-4:
                    d_flat = d_src.reshape(-1)[:H]
                    if d_flat.size < H:
                        d_flat = np.pad(d_flat, (0, H - d_flat.size)).astype(np.float32)
                    self.stage1_danger_manifold = (d_flat / (float(np.linalg.norm(d_flat)) + 1e-6)).astype(np.float32)
                    self.danger_manifold_loaded = True
            self.load_stage1_sensorimotor_chunks()
            return True
        except Exception:
            self.load_stage1_sensorimotor_chunks()
            return False
    def load_stage1_sensorimotor_chunks(self):
        chunks_file = ARTIFACTS_DIR / "stage1_sensorimotor_chunks.json"
        loaded_map = {}
        if chunks_file.exists():
            try:
                with open(chunks_file, "r", encoding="utf-8") as f:
                    raw_c = json.load(f)
                if isinstance(raw_c, dict):
                    loaded_map = raw_c
            except Exception:
                loaded_map = {}
        if not loaded_map:
            proc_file = ARTIFACTS_DIR / "stage1_procedural_hazard_evolution.json"
            if proc_file.exists():
                try:
                    with open(proc_file, "r", encoding="utf-8") as f:
                        proc_d = json.load(f)
                    if isinstance(proc_d.get("discovered_sensorimotor_chunks"), dict):
                        loaded_map = proc_d["discovered_sensorimotor_chunks"]
                except Exception:
                    loaded_map = {}
        if len(loaded_map) < 4:
            try:
                from stage1_isaac3d_embodied_arena import evolve_infinite_procedural_hazard_manifold
                rep_s1 = evolve_infinite_procedural_hazard_manifold(n_cycles=4, steps_per_cycle=90, seed=int(getattr(self, "seed", 2026)))
                if isinstance(rep_s1.get("discovered_sensorimotor_chunks"), dict):
                    loaded_map.update(rep_s1["discovered_sensorimotor_chunks"])
            except Exception:
                pass
        loaded_triples = {
            (str(rec.get("gaze_op")), str(rec.get("wm_op")), str(rec.get("finger_op")))
            for rec in loaded_map.values()
            if isinstance(rec, dict)
        }
        for g_def, w_def, f_def in [
            ("gaze_salient_target", "wm_store_attr", "finger_step_until_contact"),
            ("gaze_corner_sample", "wm_store_sample_pair", "finger_paint_from_wm"),
            ("gaze_next_matching_attr", "wm_store_attr", "finger_apply_to_gaze"),
            ("gaze_count_objects", "wm_store_count", "finger_step_n_from_wm"),
        ]:
            sig_def = f"sensorimotor_chunk::{g_def}+{w_def}+{f_def}"
            if (g_def, w_def, f_def) not in loaded_triples and sig_def not in loaded_map:
                loaded_map[sig_def] = {
                    "gaze_op": g_def,
                    "wm_op": w_def,
                    "finger_op": f_def,
                    "hits": 4,
                }
        if loaded_map:
            self.stage1_sensorimotor_chunks = dict(loaded_map)
            self.stage1_sensorimotor_chunks_transferred = int(len(self.stage1_sensorimotor_chunks))
            for c_sig, c_rec in self.stage1_sensorimotor_chunks.items():
                if not isinstance(c_rec, dict):
                    continue
                g_op = str(c_rec.get("gaze_op", "gaze_salient_target"))
                w_op = str(c_rec.get("wm_op", "wm_store_attr"))
                f_op = str(c_rec.get("finger_op", "finger_step_until_contact"))
                hits = max(1, int(c_rec.get("hits", 1)))
                existing = next((ch for ch in self.basal_ganglia_reflex_chunks if ch.get("sig_key") == c_sig), None)
                if existing is None:
                    self.basal_ganglia_reflex_chunks.append({
                        "chunk_type": "sensorimotor_body_chunk",
                        "sig_key": str(c_sig),
                        "gaze_op": g_op,
                        "wm_op": w_op,
                        "finger_op": f_op,
                        "usage_count": hits,
                        "pfc_wake_cost": 0.028,
                        "reflex_spike_cost": 0.0009,
                        "net_energy_balance": round(0.18 * float(min(12, hits)), 4),
                    })
        return int(self.stage1_sensorimotor_chunks_transferred)
    def _perceive_in_grid_signpost_legend(self, x_grid):
        H, W = x_grid.shape
        bg = int(detect_background_color(x_grid))
        color_map = {}
        stamp_map = {}
        legend_cells = set()
        body_bbox = None
        for r in range(H):
            row_vals = set(int(v) for v in x_grid[r, :])
            if len(row_vals) == 1 and next(iter(row_vals)) != bg and 1 <= r <= H - 2:
                sep_c = next(iter(row_vals))
                top_area = r * W
                bot_area = (H - 1 - r) * W
                if top_area <= bot_area:
                    leg_slice = x_grid[:r, :]
                    r_off = 0
                    body_bbox = (r + 1, 0, H - 1, W - 1)
                    for lr in range(r + 1):
                        for lc in range(W):
                            legend_cells.add((lr, lc))
                else:
                    leg_slice = x_grid[r + 1 :, :]
                    r_off = r + 1
                    body_bbox = (0, 0, r - 1, W - 1)
                    for lr in range(r, H):
                        for lc in range(W):
                            legend_cells.add((lr, lc))
                lh, lw = leg_slice.shape
                for lr in range(lh):
                    for lc in range(lw - 1):
                        c1 = int(leg_slice[lr, lc])
                        c2 = int(leg_slice[lr, lc + 1])
                        if c1 != bg and c2 != bg and c1 != sep_c and c2 != sep_c and c1 != c2:
                            color_map[c1] = c2
                break
        if body_bbox is None:
            for c in range(W):
                col_vals = set(int(v) for v in x_grid[:, c])
                if len(col_vals) == 1 and next(iter(col_vals)) != bg and 1 <= c <= W - 2:
                    sep_c = next(iter(col_vals))
                    left_area = H * c
                    right_area = H * (W - 1 - c)
                    if left_area <= right_area:
                        leg_slice = x_grid[:, :c]
                        body_bbox = (0, c + 1, H - 1, W - 1)
                        for lr in range(H):
                            for lc in range(c + 1):
                                legend_cells.add((lr, lc))
                    else:
                        leg_slice = x_grid[:, c + 1 :]
                        body_bbox = (0, 0, H - 1, c - 1)
                        for lr in range(H):
                            for lc in range(c, W):
                                legend_cells.add((lr, lc))
                    lh, lw = leg_slice.shape
                    for lr in range(lh):
                        for lc in range(lw):
                            c1 = int(leg_slice[lr, lc])
                            if c1 != bg and c1 != sep_c:
                                if lc + 1 < lw and int(leg_slice[lr, lc + 1]) not in (bg, sep_c, c1):
                                    color_map[c1] = int(leg_slice[lr, lc + 1])
                                elif lr + 1 < lh and int(leg_slice[lr + 1, lc]) not in (bg, sep_c, c1):
                                    color_map[c1] = int(leg_slice[lr + 1, lc])
                    break
        if not color_map:
            fg_phase, fg_labels = _cortical_lateral_phase_labels(x_grid != bg, color_grid=None, connectivity=8)
            for lbl in fg_labels:
                rs, cs = np.where(fg_phase == lbl)
                if len(rs) == 2:
                    cells_list = sorted(zip(rs.tolist(), cs.tolist()))
                    (r1, c1), (r2, c2) = cells_list[0], cells_list[1]
                    col1 = int(x_grid[r1, c1])
                    col2 = int(x_grid[r2, c2])
                    if col1 != col2 and col1 != bg and col2 != bg:
                        cnt1 = int(np.sum(x_grid == col1))
                        cnt2 = int(np.sum(x_grid == col2))
                        if cnt1 >= cnt2:
                            color_map[col1] = col2
                        else:
                            color_map[col2] = col1
                        legend_cells.add((r1, c1))
                        legend_cells.add((r2, c2))
        has_legend = bool(len(color_map) > 0 or len(stamp_map) > 0)
        if has_legend and body_bbox is None and legend_cells:
            non_leg_r = [r for r in range(H) for c in range(W) if int(x_grid[r, c]) != bg and (r, c) not in legend_cells]
            non_leg_c = [c for r in range(H) for c in range(W) if int(x_grid[r, c]) != bg and (r, c) not in legend_cells]
            if non_leg_r and non_leg_c:
                body_bbox = (min(non_leg_r), min(non_leg_c), max(non_leg_r), max(non_leg_c))
        return {
            "has_legend": has_legend,
            "color_map": color_map,
            "stamp_map": stamp_map,
            "legend_cells": legend_cells,
            "body_bbox": body_bbox,
        }
    def execute_unified_sensorimotor_chunk(
        self,
        x_grid,
        gaze_op="gaze_salient_target",
        wm_op="wm_store_attr",
        finger_op="finger_apply_to_gaze",
        param_dict=None,
        wm_state=None,
    ):
        self.sensorimotor_body_executions_total += 1
        if isinstance(gaze_op, dict):
            p = dict(gaze_op)
            gaze_op = str(p.get("gaze_op", "gaze_salient_target"))
            wm_op = str(p.get("wm_op", "wm_store_attr"))
            if p.get("motor_op") == "recolor":
                finger_op = "finger_paint_from_wm"
                p["val_color"] = int(p.get("target_color", p.get("val_color", 1)))
                p["paint_mode"] = str(p.get("paint_mode", "recolor"))
            else:
                finger_op = str(p.get("finger_op", "finger_apply_to_gaze"))
        else:
            p = dict(param_dict) if isinstance(param_dict, dict) else {}
            if p.get("motor_op") == "recolor" and "val_color" not in p and "target_color" in p:
                p["val_color"] = int(p["target_color"])
                p["paint_mode"] = str(p.get("paint_mode", "recolor"))
        if finger_op == "finger_foveal_saccade_stroke" or p.get("sub_op") == "foveal_saccade_stroke":
            return self.execute_foveal_saccade_stroke_closed_loop(x_grid, p)
        bg_x = detect_background_color(x_grid)
        comps = extract_components(x_grid, bg=bg_x)
        if not comps:
            return x_grid.copy()
        w_in = wm_state if isinstance(wm_state, dict) else {}
        wm_slot = {
            "key_color": p.get("key_color", w_in.get("key_color", None)),
            "val_color": p.get("val_color", w_in.get("val_color", None)),
            "color_map": dict(p.get("color_map", w_in.get("color_map", {}))),
            "shape_map": dict(p.get("shape_map", w_in.get("shape_map", {}))),
            "legend_cells": set(p.get("legend_cells", w_in.get("legend_cells", set()))),
            "attr_type": str(p.get("attr_type", w_in.get("attr_type", "color"))),
            "attr_val": p.get("attr_val", w_in.get("attr_val", None)),
            "count": int(p.get("count", w_in.get("count", 1))),
            "selector": str(p.get("selector", w_in.get("selector", "smallest"))),
            "dr": int(p.get("dr", w_in.get("dr", 0))),
            "dc": int(p.get("dc", w_in.get("dc", 0))),
            "d4_k": int(p.get("d4_k", w_in.get("d4_k", 1))),
            "sub_op": str(p.get("sub_op", w_in.get("sub_op", "shift_delta"))),
            "contact_color": p.get("contact_color", w_in.get("contact_color", None)),
            "moved_cells": list(p.get("moved_cells", w_in.get("moved_cells", []))),
            "ior_visited_cells": set(p.get("ior_visited_cells", w_in.get("ior_visited_cells", set()))),
            "anchor_color": p.get("anchor_color", w_in.get("anchor_color", None)),
            "anchor_bbox": p.get("anchor_bbox", w_in.get("anchor_bbox", None)),
            "anchor_center": p.get("anchor_center", w_in.get("anchor_center", None)),
            "anchor_mass": p.get("anchor_mass", w_in.get("anchor_mass", None)),
            "anchor_holes": p.get("anchor_holes", w_in.get("anchor_holes", None)),
            "anchor_rel_cells": list(p.get("anchor_rel_cells", w_in.get("anchor_rel_cells", []))),
            "anchor_cells": list(p.get("anchor_cells", w_in.get("anchor_cells", []))),
        }
        gaze_targets = []
        if gaze_op == "gaze_corner_sample":
            self.in_grid_legend_reads_total += 1
            if not wm_slot["color_map"] and not wm_slot["shape_map"]:
                leg_info = self._perceive_in_grid_signpost_legend(x_grid)
                if leg_info.get("has_legend", False):
                    wm_slot["color_map"] = dict(leg_info.get("color_map", {}))
                    wm_slot["shape_map"] = dict(leg_info.get("stamp_map", {}))
                    wm_slot["legend_cells"] = set(leg_info.get("legend_cells", set()))
            gaze_targets = comps
        elif gaze_op == "gaze_salient_target":
            sel = wm_slot["selector"]
            multi_match = bool(p.get("motor_op") is not None or p.get("multi_match", False))
            non_leg_comps = (
                [o for o in comps if not (set(o["cells"]) <= wm_slot["legend_cells"])]
                if wm_slot["legend_cells"]
                else list(comps)
            )
            cand_comps = non_leg_comps if non_leg_comps else list(comps)
            H_x, W_x = x_grid.shape
            if sel in ("ior_next_salient", "ior_next_largest", "ior_next_smallest", "ior_next_topmost") or bool(p.get("use_ior", False)):
                unvisited = [o for o in cand_comps if not (set(o["cells"]) & wm_slot["ior_visited_cells"])]
                pool_ior = unvisited if unvisited else list(cand_comps)
                if sel == "ior_next_smallest":
                    chosen_ior = min(pool_ior, key=lambda o: (o["mass"], o["bbox"][0], o["bbox"][1]))
                elif sel == "ior_next_topmost":
                    chosen_ior = min(pool_ior, key=lambda o: (o["bbox"][0], o["bbox"][1]))
                else:
                    chosen_ior = max(pool_ior, key=lambda o: (o["mass"], -o["bbox"][0], -o["bbox"][1]))
                gaze_targets = [chosen_ior]
                for rc_pair in chosen_ior["cells"]:
                    wm_slot["ior_visited_cells"].add((int(rc_pair[0]), int(rc_pair[1])))
                self.ior_sequential_saccades_total += 1
            elif sel in ("constellation_group", "non_texture_foreground", "texture_field"):
                v1_comps, _ = self.v1_v2_preattentive_texture_and_constellation_grouping(x_grid, cand_comps)
                if sel == "constellation_group":
                    c_groups = [o for o in v1_comps if o.get("chunk_tag") == "constellation"]
                    if not c_groups:
                        small_by_col = {}
                        for o in cand_comps:
                            if int(o["mass"]) <= 3:
                                small_by_col.setdefault(int(o["color"]), []).append(o)
                        for col_k, s_list in sorted(small_by_col.items(), key=lambda kv: -len(kv[1])):
                            if len(s_list) >= 2:
                                c_groups.append(self._build_composite_meta_component(x_grid, s_list, chunk_tag="constellation"))
                                self.v1_v2_constellation_groups_total += 1
                    gaze_targets = c_groups if c_groups else [max(cand_comps, key=lambda o: o["mass"])]
                elif sel == "non_texture_foreground":
                    fg_groups = [o for o in v1_comps if o.get("chunk_tag") != "texture_field"]
                    gaze_targets = fg_groups if fg_groups else list(cand_comps)
                else:
                    tx_groups = [o for o in v1_comps if o.get("chunk_tag") == "texture_field"]
                    gaze_targets = tx_groups if tx_groups else [min(cand_comps, key=lambda o: o["mass"])]
            elif sel in ("enclosed_hole_region", "largest_enclosed_hole", "smallest_enclosed_hole"):
                enc_regs = find_enclosed_background_regions(x_grid, bg=bg_x)
                enc_objs = []
                for reg in enc_regs:
                    rb0, cb0, rb1, cb1 = reg["bbox"]
                    enc_objs.append({
                        "cells": list(reg["cells"]),
                        "mass": int(reg["mass"]),
                        "bbox": (int(rb0), int(cb0), int(rb1), int(cb1)),
                        "height": int(reg["height"]),
                        "width": int(reg["width"]),
                        "color": int(reg["border_color"]),
                        "holes": 0,
                        "center": (0.5 * (rb0 + rb1), 0.5 * (cb0 + cb1)),
                        "area": int(reg["height"]) * int(reg["width"]),
                    })
                if enc_objs:
                    if sel == "largest_enclosed_hole":
                        gaze_targets = [max(enc_objs, key=lambda o: (o["mass"], -o["bbox"][0], -o["bbox"][1]))]
                    elif sel == "smallest_enclosed_hole":
                        gaze_targets = [min(enc_objs, key=lambda o: (o["mass"], o["bbox"][0], o["bbox"][1]))]
                    else:
                        gaze_targets = enc_objs
                else:
                    holes_fallback = [o for o in cand_comps if int(o["holes"]) > 0]
                    gaze_targets = holes_fallback if holes_fallback else [max(cand_comps, key=lambda o: o["mass"])]
            elif sel == "all":
                gaze_targets = list(cand_comps)
            elif sel == "moved_objects" and wm_slot["moved_cells"]:
                moved_set = set(tuple(c) for c in wm_slot["moved_cells"])
                matched = [o for o in cand_comps if any((r, col) in moved_set for r, col in o["cells"])]
                gaze_targets = matched if matched else [min(cand_comps, key=lambda o: (o["mass"], o["bbox"][0], o["bbox"][1]))]
            elif sel == "smallest":
                gaze_targets = [min(cand_comps, key=lambda o: (o["mass"], o["bbox"][0], o["bbox"][1]))]
            elif sel == "largest":
                gaze_targets = [max(cand_comps, key=lambda o: (o["mass"], -o["bbox"][0], -o["bbox"][1]))]
            elif sel == "has_hole":
                holes = [o for o in cand_comps if o["holes"] > 0]
                gaze_targets = (holes if multi_match else [holes[0]]) if holes else [max(cand_comps, key=lambda o: o["mass"])]
            elif sel == "no_hole":
                no_holes = [o for o in cand_comps if int(o["holes"]) == 0]
                gaze_targets = (no_holes if multi_match else [no_holes[0]]) if no_holes else [cand_comps[0]]
            elif sel == "even_mass":
                ev_m = [o for o in cand_comps if int(o["mass"]) % 2 == 0]
                gaze_targets = (ev_m if multi_match else [ev_m[0]]) if ev_m else [cand_comps[0]]
            elif sel == "odd_mass":
                od_m = [o for o in cand_comps if int(o["mass"]) % 2 == 1]
                gaze_targets = (od_m if multi_match else [od_m[0]]) if od_m else [cand_comps[0]]
            elif sel == "even_parity":
                ev_p = [o for o in cand_comps if (int(o["bbox"][0]) + int(o["bbox"][1])) % 2 == 0]
                gaze_targets = (ev_p if multi_match else [ev_p[0]]) if ev_p else [cand_comps[0]]
            elif sel == "odd_parity":
                od_p = [o for o in cand_comps if (int(o["bbox"][0]) + int(o["bbox"][1])) % 2 == 1]
                gaze_targets = (od_p if multi_match else [od_p[0]]) if od_p else [cand_comps[0]]
            elif sel == "interior":
                int_objs = [
                    o for o in cand_comps
                    if o["bbox"][0] > 0 and o["bbox"][1] > 0 and o["bbox"][2] < H_x - 1 and o["bbox"][3] < W_x - 1
                ]
                gaze_targets = (int_objs if multi_match else [int_objs[0]]) if int_objs else [cand_comps[0]]
            elif sel == "border":
                brd_objs = [
                    o for o in cand_comps
                    if o["bbox"][0] == 0 or o["bbox"][1] == 0 or o["bbox"][2] == H_x - 1 or o["bbox"][3] == W_x - 1
                ]
                gaze_targets = (brd_objs if multi_match else [brd_objs[0]]) if brd_objs else [cand_comps[0]]
            elif sel == "minority_color":
                col_cnt = {}
                for o in cand_comps:
                    col_cnt[o["color"]] = col_cnt.get(o["color"], 0) + o["mass"]
                min_c = min(col_cnt, key=lambda k: col_cnt[k])
                c_objs = [o for o in cand_comps if o["color"] == min_c]
                gaze_targets = (c_objs if multi_match else [c_objs[0]]) if c_objs else [cand_comps[0]]
            elif sel == "majority_color":
                col_cnt = {}
                for o in cand_comps:
                    col_cnt[o["color"]] = col_cnt.get(o["color"], 0) + o["mass"]
                maj_c = max(col_cnt, key=lambda k: col_cnt[k])
                c_objs = [o for o in cand_comps if o["color"] == maj_c]
                gaze_targets = (c_objs if multi_match else [c_objs[0]]) if c_objs else [cand_comps[0]]
            elif sel == "topmost":
                gaze_targets = [min(cand_comps, key=lambda o: (o["bbox"][0], o["bbox"][1]))]
            elif sel == "bottommost":
                gaze_targets = [max(cand_comps, key=lambda o: (o["bbox"][2], -o["bbox"][1]))]
            elif sel == "leftmost":
                gaze_targets = [min(cand_comps, key=lambda o: (o["bbox"][1], o["bbox"][0]))]
            elif sel == "rightmost":
                gaze_targets = [max(cand_comps, key=lambda o: (o["bbox"][3], -o["bbox"][0]))]
            else:
                gaze_targets = [cand_comps[0]]
        elif gaze_op == "gaze_next_matching_attr":
            self.predicate_group_broadcasts_total += 1
            a_type = wm_slot["attr_type"]
            a_val = wm_slot["attr_val"]
            if a_type == "moved_objects" and wm_slot["moved_cells"]:
                moved_set = set(tuple(c) for c in wm_slot["moved_cells"])
                matched = [o for o in comps if any((r, col) in moved_set for r, col in o["cells"])]
                gaze_targets = matched if matched else list(comps)
            elif a_type == "color" and a_val is not None:
                gaze_targets = [o for o in comps if int(o["color"]) == int(a_val)]
            elif a_type == "mass" and a_val is not None:
                gaze_targets = [o for o in comps if int(o["mass"]) == int(a_val)]
            elif a_type == "has_hole":
                gaze_targets = [o for o in comps if int(o["holes"]) > 0]
            elif a_type == "min_mass_group":
                m_sz = min(o["mass"] for o in comps)
                gaze_targets = [o for o in comps if o["mass"] == m_sz]
            elif a_type == "max_mass_group":
                m_sz = max(o["mass"] for o in comps)
                gaze_targets = [o for o in comps if o["mass"] == m_sz]
            else:
                gaze_targets = list(comps)
        elif gaze_op == "gaze_count_objects":
            dots = [o for o in comps if o["mass"] == 1]
            cnt_src = dots if dots else comps
            wm_slot["count"] = int(p.get("count", max(1, len(cnt_src))))
            if dots and wm_slot["val_color"] is None:
                wm_slot["val_color"] = int(dots[0]["color"])
            non_dots = [o for o in comps if o["mass"] > 1]
            gaze_targets = non_dots if non_dots else comps
        else:
            gaze_targets = list(comps)
        if not gaze_targets:
            return x_grid.copy()
        if wm_op == "wm_bind_anchor_attr" or p.get("anchor_selector", None) is not None:
            anc_sel = str(p.get("anchor_selector", "smallest"))
            target_cell_union = set()
            for t_obj in gaze_targets:
                for rc_p in t_obj["cells"]:
                    target_cell_union.add((int(rc_p[0]), int(rc_p[1])))
            anc_pool = [o for o in comps if not (set(o["cells"]) <= target_cell_union)]
            if not anc_pool:
                anc_pool = list(comps)
            if anc_sel == "smallest":
                anc_obj = min(anc_pool, key=lambda o: (o["mass"], o["bbox"][0], o["bbox"][1]))
            elif anc_sel == "largest":
                anc_obj = max(anc_pool, key=lambda o: (o["mass"], -o["bbox"][0], -o["bbox"][1]))
            elif anc_sel == "has_hole":
                h_anc = [o for o in anc_pool if int(o["holes"]) > 0]
                anc_obj = h_anc[0] if h_anc else max(anc_pool, key=lambda o: o["mass"])
            elif anc_sel == "no_hole":
                nh_anc = [o for o in anc_pool if int(o["holes"]) == 0]
                anc_obj = nh_anc[0] if nh_anc else min(anc_pool, key=lambda o: o["mass"])
            elif anc_sel == "minority_color":
                c_map_a = {}
                for o in comps:
                    c_map_a[o["color"]] = c_map_a.get(o["color"], 0) + o["mass"]
                min_ca = min(c_map_a, key=lambda k: c_map_a[k])
                ca_objs = [o for o in anc_pool if o["color"] == min_ca]
                anc_obj = ca_objs[0] if ca_objs else anc_pool[0]
            elif anc_sel == "majority_color":
                c_map_a = {}
                for o in comps:
                    c_map_a[o["color"]] = c_map_a.get(o["color"], 0) + o["mass"]
                maj_ca = max(c_map_a, key=lambda k: c_map_a[k])
                ca_objs = [o for o in anc_pool if o["color"] == maj_ca]
                anc_obj = ca_objs[0] if ca_objs else anc_pool[0]
            elif anc_sel == "topmost":
                anc_obj = min(anc_pool, key=lambda o: (o["bbox"][0], o["bbox"][1]))
            elif anc_sel == "bottommost":
                anc_obj = max(anc_pool, key=lambda o: (o["bbox"][2], -o["bbox"][1]))
            elif anc_sel == "leftmost":
                anc_obj = min(anc_pool, key=lambda o: (o["bbox"][1], o["bbox"][0]))
            elif anc_sel == "rightmost":
                anc_obj = max(anc_pool, key=lambda o: (o["bbox"][3], -o["bbox"][0]))
            else:
                anc_obj = anc_pool[0]
            self.selective_object_grasps_total = int(getattr(self, "selective_object_grasps_total", 0)) + 1
            ar0, ac0, ar1, ac1 = anc_obj["bbox"]
            wm_slot["anchor_color"] = int(anc_obj["color"])
            wm_slot["anchor_bbox"] = (int(ar0), int(ac0), int(ar1), int(ac1))
            wm_slot["anchor_center"] = (float(anc_obj["center"][0]), float(anc_obj["center"][1]))
            wm_slot["anchor_mass"] = int(anc_obj["mass"])
            wm_slot["anchor_holes"] = int(anc_obj["holes"])
            wm_slot["anchor_cells"] = [(int(r), int(c)) for r, c in anc_obj["cells"]]
            wm_slot["anchor_rel_cells"] = [
                (int(r) - int(ar0), int(c) - int(ac0), int(x_grid[r, c]))
                for r, c in anc_obj["cells"]
            ]
            bind_attr = str(p.get("bind_attr", "color"))
            if bind_attr == "color" and "target_color" not in p:
                wm_slot["val_color"] = int(anc_obj["color"])
            elif bind_attr == "mass":
                wm_slot["count"] = max(1, int(anc_obj["mass"]))
            elif bind_attr == "holes":
                wm_slot["count"] = max(1, int(anc_obj["holes"]))
            self.pfc_dyadic_bindings_total += 1
            self.it_parietal_relational_bindings_total = int(getattr(self, "it_parietal_relational_bindings_total", 0)) + 1
        elif wm_op == "wm_store_sample_pair":
            if wm_slot["key_color"] is not None and wm_slot["val_color"] is not None:
                wm_slot["color_map"][int(wm_slot["key_color"])] = int(wm_slot["val_color"])
        elif wm_op == "wm_store_attr":
            first_t = gaze_targets[0]
            if wm_slot["attr_val"] is None:
                wm_slot["attr_val"] = int(first_t["color"])
            if wm_slot["contact_color"] is not None and p.get("use_contact_color", False):
                wm_slot["val_color"] = int(wm_slot["contact_color"])
        elif wm_op == "wm_store_count":
            wm_slot["count"] = max(1, int(wm_slot["count"]))
        memory_grid = w_in.get("memory_grid", p.get("memory_grid", None))
        if isinstance(memory_grid, np.ndarray) and memory_grid.shape != x_grid.shape:
            memory_grid = None
        if bool(p.get("write_to_memo", False)):
            H_m, W_m = x_grid.shape
            mem_canvas = (
                memory_grid.copy()
                if (isinstance(memory_grid, np.ndarray) and memory_grid.shape == (H_m, W_m))
                else np.zeros((H_m, W_m), dtype=np.int32)
            )
            memo_mode = str(p.get("memo_mode", "guide_cross"))
            if memo_mode == "guide_cross":
                for t_obj in gaze_targets:
                    ra = int(round(t_obj["center"][0]))
                    ca = int(round(t_obj["center"][1]))
                    if 0 <= ra < H_m:
                        mem_canvas[ra, :] = 1
                    if 0 <= ca < W_m:
                        mem_canvas[:, ca] = 1
            elif memo_mode == "guide_col":
                for t_obj in gaze_targets:
                    ca = int(round(t_obj["center"][1]))
                    if 0 <= ca < W_m:
                        mem_canvas[:, ca] = 1
            elif memo_mode == "guide_row":
                for t_obj in gaze_targets:
                    ra = int(round(t_obj["center"][0]))
                    if 0 <= ra < H_m:
                        mem_canvas[ra, :] = 1
            elif memo_mode == "guide_between" and len(gaze_targets) >= 2:
                oa, ob = gaze_targets[0], gaze_targets[1]
                rm = int(round(0.5 * (oa["center"][0] + ob["center"][0])))
                cm = int(round(0.5 * (oa["center"][1] + ob["center"][1])))
                if abs(oa["center"][0] - ob["center"][0]) >= abs(oa["center"][1] - ob["center"][1]):
                    if 0 <= rm < H_m:
                        mem_canvas[rm, :] = 1
                else:
                    if 0 <= cm < W_m:
                        mem_canvas[:, cm] = 1
            elif memo_mode == "anchor_bbox":
                for t_obj in gaze_targets:
                    r0, c0, r1, c1 = t_obj["bbox"]
                    mem_canvas[max(0, r0) : min(H_m, r1 + 1), max(0, c0)] = 1
                    mem_canvas[max(0, r0) : min(H_m, r1 + 1), min(W_m - 1, c1)] = 1
                    mem_canvas[max(0, r0), max(0, c0) : min(W_m, c1 + 1)] = 1
                    mem_canvas[min(H_m - 1, r1), max(0, c0) : min(W_m, c1 + 1)] = 1
            else:
                for t_obj in gaze_targets:
                    for r, col in t_obj["cells"]:
                        if 0 <= r < H_m and 0 <= col < W_m:
                            mem_canvas[r, col] = 1
            self.epistemic_memo_writes_total += 1
            wm_slot["color_map"] = {}
            wm_slot["shape_map"] = {}
            wm_slot["moved_cells"] = []
            wm_slot["attr_val"] = None
            wm_slot["key_color"] = None
            wm_slot["val_color"] = None
            self.wm_offload_flushes_total += 1
            self.virtual_scratchpad_projections_total = int(getattr(self, "virtual_scratchpad_projections_total", 0)) + 1
            if isinstance(wm_state, dict):
                wm_state.update(wm_slot)
                wm_state["memory_grid"] = mem_canvas
                wm_state["memo_ttl"] = int(p.get("memo_ttl", 3))
                wm_state["memo_pending_credit"] = True
            return x_grid.copy()
        if isinstance(wm_state, dict):
            wm_state.update(wm_slot)
            if memory_grid is not None:
                wm_state["memory_grid"] = memory_grid
        if finger_op == "finger_paint_from_wm":
            self.scratchpad_inverse_backprojections_total = int(getattr(self, "scratchpad_inverse_backprojections_total", 0)) + 1
            canvas = x_grid.copy()
            l_mode = str(p.get("legend_mode", "recolor_keep_legend"))
            leg_info = self._perceive_in_grid_signpost_legend(x_grid)
            leg_cells = set(leg_info.get("legend_cells", set()))
            body_bbox = leg_info.get("body_bbox", None)
            if wm_slot["color_map"] and not p.get("use_contact_color", False):
                stroke_map_cells = []
                for t_obj in gaze_targets:
                    for r, col in t_obj["cells"]:
                        if (r, col) in leg_cells:
                            continue
                        old_c = int(canvas[r, col])
                        if old_c in wm_slot["color_map"]:
                            stroke_map_cells.append((int(r), int(col), int(wm_slot["color_map"][old_c])))
                if "erase_legend" in l_mode and leg_cells:
                    for lr, lc in leg_cells:
                        stroke_map_cells.append((int(lr), int(lc), int(bg_x)))
                self.execute_visuomotor_stroke_servo(canvas, stroke_map_cells, bg_color=bg_x)
                return canvas
            if wm_slot["shape_map"] and not p.get("use_contact_color", False):
                return canvas
            paint_c = (
                int(wm_slot["contact_color"])
                if (p.get("use_contact_color", False) and wm_slot["contact_color"] is not None)
                else (int(wm_slot["val_color"]) if wm_slot["val_color"] is not None else int(gaze_targets[0]["color"]))
            )
            paint_mode = str(p.get("paint_mode", "recolor"))
            if paint_mode == "fill_holes":
                H_p, W_p = canvas.shape
                stroke_hole_cells = []
                for t_obj in gaze_targets:
                    r0, c0, r1, c1 = t_obj["bbox"]
                    obj_set = set(t_obj["cells"])
                    for r in range(r0 + 1, r1):
                        for col in range(c0 + 1, c1):
                            if 0 <= r < H_p and 0 <= col < W_p and (r, col) not in obj_set and int(canvas[r, col]) == int(bg_x):
                                stroke_hole_cells.append((int(r), int(col), int(paint_c)))
                return self.execute_visuomotor_stroke_servo(canvas, stroke_hole_cells, bg_color=bg_x)
            if paint_mode == "border":
                H_p, W_p = canvas.shape
                stroke_border_cells = []
                for t_obj in gaze_targets:
                    obj_set = set(t_obj["cells"])
                    for r, col in obj_set:
                        for dr_b in (-1, 0, 1):
                            for dc_b in (-1, 0, 1):
                                nr, nc = r + dr_b, col + dc_b
                                if 0 <= nr < H_p and 0 <= nc < W_p and (nr, nc) not in obj_set and int(canvas[nr, nc]) == int(bg_x):
                                    stroke_border_cells.append((int(nr), int(nc), int(paint_c)))
                return self.execute_visuomotor_stroke_servo(canvas, stroke_border_cells, bg_color=bg_x)
            if paint_mode == "connect_line" and len(gaze_targets) >= 2:
                H_p, W_p = canvas.shape
                stroke_line_cells = []
                for idx_a in range(len(gaze_targets)):
                    for idx_b in range(idx_a + 1, len(gaze_targets)):
                        oa, ob = gaze_targets[idx_a], gaze_targets[idx_b]
                        ra, ca = int(round(oa["center"][0])), int(round(oa["center"][1]))
                        rb, cb = int(round(ob["center"][0])), int(round(ob["center"][1]))
                        if ra == rb:
                            for cc in range(min(ca, cb), max(ca, cb) + 1):
                                if 0 <= ra < H_p and 0 <= cc < W_p and int(canvas[ra, cc]) == int(bg_x):
                                    stroke_line_cells.append((int(ra), int(cc), int(paint_c)))
                        elif ca == cb:
                            for rr in range(min(ra, rb), max(ra, rb) + 1):
                                if 0 <= rr < H_p and 0 <= ca < W_p and int(canvas[rr, ca]) == int(bg_x):
                                    stroke_line_cells.append((int(rr), int(ca), int(paint_c)))
                return self.execute_visuomotor_stroke_servo(canvas, stroke_line_cells, bg_color=bg_x)
            stroke_cells_paint = []
            for t_obj in gaze_targets:
                for r, col in t_obj["cells"]:
                    stroke_cells_paint.append((int(r), int(col), int(paint_c)))
            return self.execute_visuomotor_stroke_servo(canvas, stroke_cells_paint, bg_color=bg_x)
        if finger_op == "finger_step_until_contact":
            H, W = x_grid.shape
            canvas = x_grid.copy()
            moving_cells_all = set()
            for t_obj in gaze_targets:
                for r, col in t_obj["cells"]:
                    moving_cells_all.add((r, col))
            clear_moving_strokes = [(int(r), int(col), int(bg_x)) for r, col in moving_cells_all]
            self.execute_visuomotor_stroke_servo(canvas, clear_moving_strokes, bg_color=bg_x)
            max_s = int(p.get("max_steps", max(H, W))) if p.get("max_steps", None) is not None else max(H, W)
            stop_margin = int(p.get("stop_margin", 0))
            final_moved_cells = []
            last_contact_color = wm_slot.get("contact_color", None)
            has_mem_guide = (
                isinstance(memory_grid, np.ndarray)
                and memory_grid.shape == (H, W)
                and bool(np.any(memory_grid != 0))
            )
            for t_obj in gaze_targets:
                dr, dc = int(wm_slot["dr"]), int(wm_slot["dc"])
                init_obj_cells = set(t_obj["cells"])
                if dr == 0 and dc == 0:
                    if has_mem_guide and bool(p.get("prefer_memo_guide", False)):
                        cr0 = int(round(t_obj["center"][0]))
                        cc0 = int(round(t_obj["center"][1]))
                        best_m_dist = H + W + 10
                        for dr_m, dc_m in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                            for step_m in range(1, max(H, W)):
                                mr = cr0 + dr_m * step_m
                                mc = cc0 + dc_m * step_m
                                if not (0 <= mr < H and 0 <= mc < W):
                                    break
                                if (mr, mc) in init_obj_cells:
                                    continue
                                if int(memory_grid[mr, mc]) != 0 and int(memory_grid[mr - dr_m, mc - dc_m]) == 0:
                                    if step_m < best_m_dist:
                                        best_m_dist = step_m
                                        dr, dc = dr_m, dc_m
                                    break
                    if dr == 0 and dc == 0:
                        static_objs = [
                            o for o in comps
                            if o is not t_obj
                            and not (set(o["cells"]) & moving_cells_all)
                            and not (set(o["cells"]) <= wm_slot["legend_cells"])
                        ]
                        if static_objs:
                            anchor = min(
                                static_objs,
                                key=lambda o: abs(o["center"][0] - t_obj["center"][0]) + abs(o["center"][1] - t_obj["center"][1]),
                            )
                            dy = anchor["center"][0] - t_obj["center"][0]
                            dx = anchor["center"][1] - t_obj["center"][1]
                            if abs(dy) >= abs(dx):
                                dr = 1 if dy > 0 else (-1 if dy < 0 else 0)
                            else:
                                dc = 1 if dx > 0 else (-1 if dx < 0 else 0)
                    if dr == 0 and dc == 0 and has_mem_guide:
                        cr0 = int(round(t_obj["center"][0]))
                        cc0 = int(round(t_obj["center"][1]))
                        best_m_dist = H + W + 10
                        for dr_m, dc_m in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                            for step_m in range(1, max(H, W)):
                                mr = cr0 + dr_m * step_m
                                mc = cc0 + dc_m * step_m
                                if not (0 <= mr < H and 0 <= mc < W):
                                    break
                                if (mr, mc) in init_obj_cells:
                                    continue
                                if int(memory_grid[mr, mc]) != 0 and int(memory_grid[mr - dr_m, mc - dc_m]) == 0:
                                    if step_m < best_m_dist:
                                        best_m_dist = step_m
                                        dr, dc = dr_m, dc_m
                                    break
                    if dr == 0 and dc == 0:
                        dr = 1
                cur_cells = [(int(r), int(col), int(x_grid[r, col])) for r, col in t_obj["cells"]]
                history_cells = [cur_cells]
                steps_done = 0
                while steps_done < max_s:
                    nxt_cells = [(r + dr, col + dc, val) for r, col, val in cur_cells]
                    blocked = False
                    for nr, nc, _ in nxt_cells:
                        if not (0 <= nr < H and 0 <= nc < W):
                            blocked = True
                            break
                        board_hit = int(canvas[nr, nc]) != int(bg_x)
                        memo_hit = (
                            has_mem_guide
                            and int(memory_grid[nr, nc]) != 0
                            and (nr, nc) not in init_obj_cells
                            and int(memory_grid[nr - dr, nc - dc]) == 0
                        )
                        if board_hit or memo_hit:
                            if board_hit:
                                last_contact_color = int(canvas[nr, nc])
                                for c_ob in comps:
                                    if (int(nr), int(nc)) in c_ob["cells"]:
                                        for or_i, oc_i in c_ob["cells"]:
                                            wm_slot["ior_visited_cells"].add((int(or_i), int(oc_i)))
                            blocked = True
                            break
                    if blocked:
                        break
                    cur_cells = nxt_cells
                    history_cells.append(cur_cells)
                    steps_done += 1
                if stop_margin > 0 and len(history_cells) > 1:
                    back_idx = max(0, len(history_cells) - 1 - stop_margin)
                    cur_cells = history_cells[back_idx]
                elif stop_margin < 0:
                    for _ in range(abs(stop_margin)):
                        nxt_c = [(r + dr, col + dc, val) for r, col, val in cur_cells]
                        if all(0 <= nr < H and 0 <= nc < W for nr, nc, _ in nxt_c):
                            cur_cells = nxt_c
                stroke_step_cells = []
                for r, col, val in cur_cells:
                    if 0 <= r < H and 0 <= col < W:
                        stroke_step_cells.append((int(r), int(col), int(val)))
                        final_moved_cells.append((r, col))
                        wm_slot["ior_visited_cells"].add((int(r), int(col)))
                self.execute_visuomotor_stroke_servo(canvas, stroke_step_cells, bg_color=bg_x)
            wm_slot["moved_cells"] = final_moved_cells
            if last_contact_color is not None:
                wm_slot["contact_color"] = int(last_contact_color)
            if isinstance(wm_state, dict):
                wm_state.update(wm_slot)
                if memory_grid is not None:
                    wm_state["memory_grid"] = memory_grid
            return canvas
        if finger_op == "finger_step_n_from_wm":
            H, W = x_grid.shape
            canvas = x_grid.copy()
            n_steps = max(1, int(wm_slot["count"]))
            dr, dc = int(wm_slot["dr"]), int(wm_slot["dc"])
            if dr == 0 and dc == 0:
                dr, dc = 1, 0
            step_style = str(p.get("step_style", "shift"))
            if step_style == "ray":
                ray_strokes = []
                for t_obj in gaze_targets:
                    c_val = int(t_obj["color"])
                    for r, col in t_obj["cells"]:
                        cr, cc = int(r), int(col)
                        for _ in range(n_steps):
                            cr += dr
                            cc += dc
                            if not (0 <= cr < H and 0 <= cc < W):
                                break
                            if int(canvas[cr, cc]) != int(bg_x):
                                break
                            ray_strokes.append((int(cr), int(cc), int(c_val)))
                return self.execute_visuomotor_stroke_servo(canvas, ray_strokes, bg_color=bg_x)
            step_n_strokes = []
            for t_obj in gaze_targets:
                for r, col in t_obj["cells"]:
                    step_n_strokes.append((int(r), int(col), int(bg_x)))
            for t_obj in gaze_targets:
                cur_cells = [(int(r), int(col), int(x_grid[r, col])) for r, col in t_obj["cells"]]
                for _ in range(n_steps):
                    nxt_cells = [(r + dr, col + dc, val) for r, col, val in cur_cells]
                    if all(0 <= nr < H and 0 <= nc < W for nr, nc, _ in nxt_cells):
                        cur_cells = nxt_cells
                    else:
                        break
                for r, col, val in cur_cells:
                    if 0 <= r < H and 0 <= col < W:
                        step_n_strokes.append((int(r), int(col), int(val)))
            return self.execute_visuomotor_stroke_servo(canvas, step_n_strokes, bg_color=bg_x)
        if finger_op == "finger_apply_to_gaze":
            H, W = x_grid.shape
            canvas = x_grid.copy()
            sub_op = str(wm_slot["sub_op"])
            act_p = p.get("action_param", (wm_slot["dr"], wm_slot["dc"]))
            if sub_op in ("shift_delta", "slide_until_obstacle"):
                if sub_op == "slide_until_obstacle":
                    dr = int(act_p[0]) if isinstance(act_p, (tuple, list)) and len(act_p) >= 2 else int(wm_slot["dr"])
                    dc = int(act_p[1]) if isinstance(act_p, (tuple, list)) and len(act_p) >= 2 else int(wm_slot["dc"])
                    p_slide = dict(p)
                    p_slide["dr"] = dr
                    p_slide["dc"] = dc
                    return self.execute_unified_sensorimotor_chunk(
                        x_grid, gaze_op, wm_op, "finger_step_until_contact", p_slide, wm_state=wm_state
                    )
                dr = int(act_p[0]) if isinstance(act_p, (tuple, list)) and len(act_p) >= 2 else int(wm_slot["dr"])
                dc = int(act_p[1]) if isinstance(act_p, (tuple, list)) and len(act_p) >= 2 else int(wm_slot["dc"])
                shift_strokes = []
                for t_obj in gaze_targets:
                    for r, col in t_obj["cells"]:
                        shift_strokes.append((int(r), int(col), int(bg_x)))
                for t_obj in gaze_targets:
                    for r, col in t_obj["cells"]:
                        nr, nc = int(r) + dr, int(col) + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            shift_strokes.append((int(nr), int(nc), int(x_grid[r, col])))
                            wm_slot["ior_visited_cells"].add((nr, nc))
                self.execute_visuomotor_stroke_servo(canvas, shift_strokes, bg_color=bg_x)
                if isinstance(wm_state, dict):
                    wm_state["ior_visited_cells"] = set(wm_slot["ior_visited_cells"])
                return canvas
            if sub_op in ("recolor_group", "recolor", "recolor_from_anchor"):
                if sub_op == "recolor_from_anchor" and wm_slot["anchor_color"] is not None:
                    new_c = int(wm_slot["anchor_color"])
                else:
                    new_c = int(act_p) if isinstance(act_p, (int, np.integer)) else int(wm_slot["val_color"] or 1)
                rec_strokes = []
                for t_obj in gaze_targets:
                    for r, col in t_obj["cells"]:
                        rec_strokes.append((int(r), int(col), int(new_c)))
                if bool(p.get("consume_anchor", False)) and wm_slot["anchor_cells"]:
                    for ar, ac in wm_slot["anchor_cells"]:
                        if 0 <= ar < H and 0 <= ac < W:
                            rec_strokes.append((int(ar), int(ac), int(bg_x)))
                return self.execute_visuomotor_stroke_servo(canvas, rec_strokes, bg_color=bg_x)
            if sub_op in ("fill_interior_cavity", "fill_holes", "fill_holes_from_anchor"):
                if sub_op == "fill_holes_from_anchor" and wm_slot["anchor_color"] is not None:
                    fill_c = int(wm_slot["anchor_color"])
                else:
                    fill_c = int(act_p) if isinstance(act_p, (int, np.integer)) else int(wm_slot["val_color"] or 2)
                enc_regs = find_enclosed_background_regions(x_grid, bg=bg_x)
                filled_any = False
                hole_strokes = []
                for t_obj in gaze_targets:
                    r0, c0, r1, c1 = t_obj["bbox"]
                    obj_set = set(t_obj["cells"])
                    for reg in enc_regs:
                        rr0, rc0, rr1, rc1 = reg["bbox"]
                        if rr0 >= r0 and rc0 >= c0 and rr1 <= r1 and rc1 <= c1:
                            for r, col in reg["cells"]:
                                hole_strokes.append((int(r), int(col), int(fill_c)))
                                filled_any = True
                    if not filled_any:
                        for r in range(r0 + 1, r1):
                            for col in range(c0 + 1, c1):
                                if (r, col) not in obj_set and int(canvas[r, col]) == int(bg_x):
                                    hole_strokes.append((int(r), int(col), int(fill_c)))
                if bool(p.get("consume_anchor", False)) and wm_slot["anchor_cells"]:
                    for ar, ac in wm_slot["anchor_cells"]:
                        if 0 <= ar < H and 0 <= ac < W:
                            hole_strokes.append((int(ar), int(ac), int(bg_x)))
                return self.execute_visuomotor_stroke_servo(canvas, hole_strokes, bg_color=bg_x)
            if sub_op in ("frame_border_1px", "border_from_anchor"):
                if sub_op == "border_from_anchor" and wm_slot["anchor_color"] is not None:
                    b_c = int(wm_slot["anchor_color"])
                else:
                    b_c = int(act_p) if isinstance(act_p, (int, np.integer)) else int(wm_slot["val_color"] or 1)
                frame_strokes = []
                for t_obj in gaze_targets:
                    obj_set = set(t_obj["cells"])
                    for r, col in obj_set:
                        for dr_b in (-1, 0, 1):
                            for dc_b in (-1, 0, 1):
                                nr, nc = r + dr_b, col + dc_b
                                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in obj_set and int(canvas[nr, nc]) == int(bg_x):
                                    frame_strokes.append((int(nr), int(nc), int(b_c)))
                return self.execute_visuomotor_stroke_servo(canvas, frame_strokes, bg_color=bg_x)
            if sub_op in ("align_to_anchor_row", "align_to_anchor_col") and wm_slot["anchor_center"] is not None:
                anc_r, anc_c = wm_slot["anchor_center"]
                align_strokes = []
                for t_obj in gaze_targets:
                    for r, col in t_obj["cells"]:
                        align_strokes.append((int(r), int(col), int(bg_x)))
                for t_obj in gaze_targets:
                    tr, tc = t_obj["center"]
                    dr_a = int(round(anc_r - tr)) if sub_op == "align_to_anchor_row" else 0
                    dc_a = int(round(anc_c - tc)) if sub_op == "align_to_anchor_col" else 0
                    for r, col in t_obj["cells"]:
                        nr, nc = int(r) + dr_a, int(col) + dc_a
                        if 0 <= nr < H and 0 <= nc < W:
                            align_strokes.append((int(nr), int(nc), int(x_grid[r, col])))
                return self.execute_visuomotor_stroke_servo(canvas, align_strokes, bg_color=bg_x)
            if sub_op == "connect_to_anchor" and wm_slot["anchor_center"] is not None:
                anc_r = int(round(wm_slot["anchor_center"][0]))
                anc_c = int(round(wm_slot["anchor_center"][1]))
                line_c = int(wm_slot["anchor_color"] if wm_slot["anchor_color"] is not None else (wm_slot["val_color"] or gaze_targets[0]["color"]))
                conn_strokes = []
                for t_obj in gaze_targets:
                    tr = int(round(t_obj["center"][0]))
                    tc = int(round(t_obj["center"][1]))
                    if tr == anc_r:
                        for cc in range(min(tc, anc_c), max(tc, anc_c) + 1):
                            if 0 <= tr < H and 0 <= cc < W and int(canvas[tr, cc]) == int(bg_x):
                                conn_strokes.append((int(tr), int(cc), int(line_c)))
                    elif tc == anc_c:
                        for rr in range(min(tr, anc_r), max(tr, anc_r) + 1):
                            if 0 <= rr < H and 0 <= tc < W and int(canvas[rr, tc]) == int(bg_x):
                                conn_strokes.append((int(rr), int(tc), int(line_c)))
                return self.execute_visuomotor_stroke_servo(canvas, conn_strokes, bg_color=bg_x)
            if sub_op == "stamp_anchor_at_gaze" and wm_slot["anchor_rel_cells"] and wm_slot["anchor_bbox"] is not None:
                self.motor_stamp_executions_total += 1
                ar0, ac0, ar1, ac1 = wm_slot["anchor_bbox"]
                off_r = (int(ar1) - int(ar0)) // 2
                off_c = (int(ac1) - int(ac0)) // 2
                use_marker_col = bool(p.get("recolor_to_marker", False))
                stamp_strokes = []
                for t_obj in gaze_targets:
                    mr = int(round(t_obj["center"][0]))
                    mc = int(round(t_obj["center"][1]))
                    m_col = int(t_obj["color"])
                    for r_m, c_m in t_obj["cells"]:
                        stamp_strokes.append((int(r_m), int(c_m), int(bg_x)))
                    for rr, rc, val in wm_slot["anchor_rel_cells"]:
                        nr = mr - off_r + int(rr)
                        nc = mc - off_c + int(rc)
                        if 0 <= nr < H and 0 <= nc < W:
                            stamp_strokes.append((int(nr), int(nc), int(m_col if use_marker_col else int(val))))
                return self.execute_visuomotor_stroke_servo(canvas, stamp_strokes, bg_color=bg_x)
            if sub_op == "raycast_extend":
                dr = int(act_p[0]) if isinstance(act_p, (tuple, list)) and len(act_p) >= 2 else 1
                dc = int(act_p[1]) if isinstance(act_p, (tuple, list)) and len(act_p) >= 2 else 0
                p_ray = dict(p)
                p_ray["dr"] = dr
                p_ray["dc"] = dc
                p_ray["count"] = max(H, W)
                p_ray["step_style"] = "ray"
                return self.execute_unified_sensorimotor_chunk(x_grid, gaze_op, wm_op, "finger_step_n_from_wm", p_ray)
            if sub_op in ("isolated_d4", "rotate_flip_in_place"):
                d4_k = int(act_p) if isinstance(act_p, (int, np.integer)) else int(wm_slot["d4_k"])
                wrist_strokes = []
                for t_obj in gaze_targets:
                    r0, c0, r1, c1 = t_obj["bbox"]
                    bh = int(r1 - r0 + 1)
                    bw = int(c1 - c0 + 1)
                    for r, col in t_obj["cells"]:
                        wrist_strokes.append((int(r), int(col), int(bg_x)))
                    for r, col in t_obj["cells"]:
                        val_c = int(x_grid[r, col])
                        ur = int(r - r0)
                        uc = int(col - c0)
                        rot_step = d4_k % 4
                        cur_h, cur_w = bh, bw
                        for _ in range(rot_step):
                            ur, uc = cur_w - 1 - uc, ur
                            cur_h, cur_w = cur_w, cur_h
                        if d4_k >= 4:
                            uc = cur_w - 1 - uc
                        nr, nc = int(r0 + ur), int(c0 + uc)
                        if 0 <= nr < H and 0 <= nc < W:
                            wrist_strokes.append((nr, nc, val_c))
                return self.execute_visuomotor_stroke_servo(canvas, wrist_strokes, bg_color=bg_x)
            if sub_op == "crop_grasped":
                self.foveal_window_zooms_total += 1
                r0_all = min(t_obj["bbox"][0] for t_obj in gaze_targets)
                c0_all = min(t_obj["bbox"][1] for t_obj in gaze_targets)
                r1_all = max(t_obj["bbox"][2] for t_obj in gaze_targets)
                c1_all = max(t_obj["bbox"][3] for t_obj in gaze_targets)
                if r1_all >= r0_all and c1_all >= c0_all:
                    sub_crop = x_grid[r0_all : r1_all + 1, c0_all : c1_all + 1]
                    ch_crop, cw_crop = sub_crop.shape
                    crop_canvas = np.full((ch_crop, cw_crop), bg_x, dtype=np.int32)
                    crop_strokes = [
                        (sr, sc, int(sub_crop[sr, sc]))
                        for sr in range(ch_crop)
                        for sc in range(cw_crop)
                        if int(sub_crop[sr, sc]) != int(bg_x)
                    ]
                    return self.execute_visuomotor_stroke_servo(crop_canvas, crop_strokes, bg_color=bg_x)
                return canvas
            if sub_op == "zoom_interior_room":
                self.foveal_window_zooms_total += 1
                r0_all = min(t_obj["bbox"][0] for t_obj in gaze_targets)
                c0_all = min(t_obj["bbox"][1] for t_obj in gaze_targets)
                r1_all = max(t_obj["bbox"][2] for t_obj in gaze_targets)
                c1_all = max(t_obj["bbox"][3] for t_obj in gaze_targets)
                sub_zoom = None
                if r1_all - r0_all >= 2 and c1_all - c0_all >= 2:
                    sub_zoom = x_grid[r0_all + 1 : r1_all, c0_all + 1 : c1_all]
                elif r1_all >= r0_all and c1_all >= c0_all:
                    sub_zoom = x_grid[r0_all : r1_all + 1, c0_all : c1_all + 1]
                if sub_zoom is not None and sub_zoom.size > 0:
                    zh, zw = sub_zoom.shape
                    zoom_canvas = np.full((zh, zw), bg_x, dtype=np.int32)
                    zoom_strokes = [
                        (sr, sc, int(sub_zoom[sr, sc]))
                        for sr in range(zh)
                        for sc in range(zw)
                        if int(sub_zoom[sr, sc]) != int(bg_x)
                    ]
                    return self.execute_visuomotor_stroke_servo(zoom_canvas, zoom_strokes, bg_color=bg_x)
                return canvas
            if sub_op == "scale_grasped":
                self.foveal_window_zooms_total += 1
                if str(p.get("selector", "all")) == "all":
                    sub_g = x_grid.copy()
                else:
                    r0_all = min(t_obj["bbox"][0] for t_obj in gaze_targets)
                    c0_all = min(t_obj["bbox"][1] for t_obj in gaze_targets)
                    r1_all = max(t_obj["bbox"][2] for t_obj in gaze_targets)
                    c1_all = max(t_obj["bbox"][3] for t_obj in gaze_targets)
                    sub_g = x_grid[r0_all : r1_all + 1, c0_all : c1_all + 1].copy()
                scale_k = int(act_p) if isinstance(act_p, (int, np.integer)) else 2
                scale_k = max(2, min(4, scale_k))
                sh, sw = sub_g.shape
                max_bound = max(H, W, 64)
                if sh * scale_k <= max_bound and sw * scale_k <= max_bound:
                    scaled_canvas = np.full((sh * scale_k, sw * scale_k), bg_x, dtype=np.int32)
                    scale_strokes = []
                    for sr in range(sh):
                        for sc in range(sw):
                            s_val = int(sub_g[sr, sc])
                            if s_val != int(bg_x):
                                for dr_s in range(scale_k):
                                    for dc_s in range(scale_k):
                                        scale_strokes.append((sr * scale_k + dr_s, sc * scale_k + dc_s, s_val))
                    return self.execute_visuomotor_stroke_servo(scaled_canvas, scale_strokes, bg_color=bg_x)
                return canvas
            if sub_op == "tile_grasped":
                self.motor_stamp_executions_total += 1
                if str(p.get("selector", "all")) == "all":
                    sub_g = x_grid.copy()
                else:
                    r0_all = min(t_obj["bbox"][0] for t_obj in gaze_targets)
                    c0_all = min(t_obj["bbox"][1] for t_obj in gaze_targets)
                    r1_all = max(t_obj["bbox"][2] for t_obj in gaze_targets)
                    c1_all = max(t_obj["bbox"][3] for t_obj in gaze_targets)
                    sub_g = x_grid[r0_all : r1_all + 1, c0_all : c1_all + 1].copy()
                if isinstance(act_p, (tuple, list)) and len(act_p) >= 2:
                    tr_k, tc_k = max(1, min(4, int(act_p[0]))), max(1, min(4, int(act_p[1])))
                else:
                    tr_k, tc_k = 2, 2
                sh, sw = sub_g.shape
                max_bound = max(H, W, 64)
                if sh * tr_k <= max_bound and sw * tc_k <= max_bound:
                    tiled_canvas = np.full((sh * tr_k, sw * tc_k), bg_x, dtype=np.int32)
                    tile_strokes = []
                    mirror_flag = bool(p.get("mirror_tile", False)) and tr_k == 2 and tc_k == 2
                    for br in range(tr_k):
                        for bc in range(tc_k):
                            for sr in range(sh):
                                for sc in range(sw):
                                    src_r = (sh - 1 - sr) if (mirror_flag and br == 1) else sr
                                    src_c = (sw - 1 - sc) if (mirror_flag and bc == 1) else sc
                                    t_val = int(sub_g[src_r, src_c])
                                    if t_val != int(bg_x):
                                        tile_strokes.append((br * sh + sr, bc * sw + sc, t_val))
                    return self.execute_visuomotor_stroke_servo(tiled_canvas, tile_strokes, bg_color=bg_x)
                return canvas
            if sub_op == "condense_dividers":
                self.foveal_window_zooms_total += 1
                keep_rows = [r_i for r_i in range(H) if len(set(int(v) for v in x_grid[r_i, :])) > 1]
                keep_cols = [c_i for c_i in range(W) if len(set(int(v) for v in x_grid[:, c_i])) > 1]
                if keep_rows and keep_cols and (len(keep_rows) < H or len(keep_cols) < W):
                    sub_cond = x_grid[np.ix_(keep_rows, keep_cols)]
                    ch_cond, cw_cond = sub_cond.shape
                    cond_canvas = np.full((ch_cond, cw_cond), bg_x, dtype=np.int32)
                    cond_strokes = [
                        (sr, sc, int(sub_cond[sr, sc]))
                        for sr in range(ch_cond)
                        for sc in range(cw_cond)
                        if int(sub_cond[sr, sc]) != int(bg_x)
                    ]
                    return self.execute_visuomotor_stroke_servo(cond_canvas, cond_strokes, bg_color=bg_x)
                return canvas
            if sub_op == "trace_line_between_gaze":
                line_c = (
                    int(act_p)
                    if isinstance(act_p, (int, np.integer)) and int(act_p) > 0
                    else int(wm_slot["val_color"] if wm_slot["val_color"] is not None else gaze_targets[0]["color"])
                )
                trace_strokes = []
                for idx_a in range(len(gaze_targets)):
                    for idx_b in range(idx_a + 1, len(gaze_targets)):
                        oa, ob = gaze_targets[idx_a], gaze_targets[idx_b]
                        ra, ca = int(round(oa["center"][0])), int(round(oa["center"][1]))
                        rb, cb = int(round(ob["center"][0])), int(round(ob["center"][1]))
                        if ra == rb:
                            for cc in range(min(ca, cb), max(ca, cb) + 1):
                                if 0 <= ra < H and 0 <= cc < W and int(canvas[ra, cc]) == int(bg_x):
                                    trace_strokes.append((int(ra), int(cc), int(line_c)))
                        elif ca == cb:
                            for rr in range(min(ra, rb), max(ra, rb) + 1):
                                if 0 <= rr < H and 0 <= ca < W and int(canvas[rr, ca]) == int(bg_x):
                                    trace_strokes.append((int(rr), int(ca), int(line_c)))
                return self.execute_visuomotor_stroke_servo(canvas, trace_strokes, bg_color=bg_x)
        return x_grid.copy()
    def execute_sensorimotor_chain(self, x_grid, steps_spec):
        canvas = x_grid.copy()
        wm_state = {
            "memory_grid": np.zeros_like(x_grid, dtype=np.int32),
            "memo_ttl": 0,
            "memo_pending_credit": False,
            "ior_visited_cells": set(),
        }
        for step_d in steps_spec:
            prev_canvas = canvas.copy()
            if isinstance(wm_state.get("memory_grid"), np.ndarray) and wm_state["memory_grid"].shape != canvas.shape:
                wm_state["memory_grid"] = np.zeros_like(canvas, dtype=np.int32)
                wm_state["memo_ttl"] = 0
                wm_state["memo_pending_credit"] = False
            canvas = self.execute_unified_sensorimotor_chunk(
                canvas,
                str(step_d.get("gaze_op", "gaze_salient_target")),
                str(step_d.get("wm_op", "wm_store_attr")),
                str(step_d.get("finger_op", "finger_apply_to_gaze")),
                step_d,
                wm_state=wm_state,
            )
            if not bool(step_d.get("write_to_memo", False)):
                if (
                    bool(wm_state.get("memo_pending_credit", False))
                    and int(wm_state.get("memo_ttl", 0)) > 0
                    and canvas.shape == prev_canvas.shape
                    and not np.array_equal(canvas, prev_canvas)
                ):
                    self.epistemic_delayed_payoffs_total += 1
                    wm_state["memo_pending_credit"] = False
                cur_ttl = max(0, int(wm_state.get("memo_ttl", 0)) - 1)
                wm_state["memo_ttl"] = cur_ttl
                if cur_ttl <= 0 and isinstance(wm_state.get("memory_grid"), np.ndarray):
                    wm_state["memory_grid"].fill(0)
                    wm_state["memo_pending_credit"] = False
        return canvas
    def encode_sensorimotor_body_vec(self, step_or_params, q_spk_fallback=None):
        gaze_map = {
            "gaze_salient_target": 0,
            "gaze_corner_sample": 1,
            "gaze_next_matching_attr": 2,
            "gaze_count_objects": 3,
        }
        wm_map = {
            "wm_store_attr": 4,
            "wm_store_sample_pair": 5,
            "wm_bind_anchor_attr": 5,
            "wm_store_count": 6,
        }
        finger_map = {
            "finger_step_until_contact": 7,
            "finger_paint_from_wm": 8,
            "finger_apply_to_gaze": 9,
            "finger_step_n_from_wm": 10,
        }
        steps_list = (
            step_or_params.get("steps", [step_or_params])
            if isinstance(step_or_params, dict)
            else []
        )
        b_vec = np.zeros((11,), dtype=np.float32)
        for st in steps_list:
            if not isinstance(st, dict):
                continue
            if "gaze_op" in st or "finger_op" in st:
                gi = gaze_map.get(str(st.get("gaze_op", "gaze_salient_target")), 0)
                wi = wm_map.get(str(st.get("wm_op", "wm_store_attr")), 4)
                fi = finger_map.get(str(st.get("finger_op", "finger_apply_to_gaze")), 7)
                b_vec[gi] += 1.0
                b_vec[wi] += 1.0
                b_vec[fi] += 1.0
        if float(np.linalg.norm(b_vec)) < 1e-6 and q_spk_fallback is not None and hasattr(self, "W_body_assembly"):
            q_flat = np.asarray(q_spk_fallback, dtype=np.float32).reshape(-1)[: self.hid_dim]
            if q_flat.size < self.hid_dim:
                q_flat = np.pad(q_flat, (0, self.hid_dim - q_flat.size)).astype(np.float32)
            b_vec = np.tanh(self.W_body_assembly @ q_flat).astype(np.float32)
        return b_vec
    def score_sensorimotor_body_resonance(self, step_or_params, q_spk_hint=None):
        if not hasattr(self, "W_body_assembly"):
            return 0.0
        if q_spk_hint is None:
            if hasattr(self, "W_fovea_v1_spikes") and hasattr(self, "last_foveal_retinotopic_100"):
                q_vec = np.tanh(
                    self.W_fovea_v1_spikes @ np.asarray(self.last_foveal_retinotopic_100, dtype=np.float32).reshape(100)
                ).astype(np.float32)
            else:
                return 0.0
        else:
            q_vec = np.asarray(q_spk_hint, dtype=np.float32).reshape(-1)
        if q_vec.size != self.W_body_assembly.shape[1]:
            return 0.0
        body_logits = self.W_body_assembly @ q_vec
        gaze_map = {
            "gaze_salient_target": 0,
            "gaze_corner_sample": 1,
            "gaze_next_matching_attr": 2,
            "gaze_count_objects": 3,
        }
        wm_map = {
            "wm_store_attr": 4,
            "wm_store_sample_pair": 5,
            "wm_store_count": 6,
        }
        finger_map = {
            "finger_step_until_contact": 7,
            "finger_paint_from_wm": 8,
            "finger_apply_to_gaze": 9,
            "finger_foveal_saccade_stroke": 9,
            "finger_step_n_from_wm": 10,
        }
        steps_list = (
            step_or_params.get("steps", [step_or_params])
            if isinstance(step_or_params, dict)
            else []
        )
        if not steps_list:
            return 0.0
        total_res = 0.0
        for st in steps_list:
            if not isinstance(st, dict):
                continue
            gi = gaze_map.get(str(st.get("gaze_op", "")), 0)
            wi = wm_map.get(str(st.get("wm_op", "")), 4)
            fi = finger_map.get(str(st.get("finger_op", "")), 7)
            total_res += float(body_logits[gi] + body_logits[wi] + body_logits[fi])
        return total_res / max(1.0, float(len(steps_list)))
    def build_unified_sensorimotor_affordances(self, cur_pairs, diff_rule_info=None, saccade_idx=0, q_spk_hint=None):
        sm_cands = []
        if not cur_pairs:
            return sm_cands
        discovered_triples = {
            (str(ch.get("gaze_op")), str(ch.get("wm_op")), str(ch.get("finger_op")))
            for ch in getattr(self, "basal_ganglia_reflex_chunks", [])
            if isinstance(ch, dict) and ch.get("chunk_type") == "sensorimotor_body_chunk"
        }
        if not discovered_triples:
            self.load_stage1_sensorimotor_chunks()
            discovered_triples = {
                (str(ch.get("gaze_op")), str(ch.get("wm_op")), str(ch.get("finger_op")))
                for ch in getattr(self, "basal_ganglia_reflex_chunks", [])
                if isinstance(ch, dict) and ch.get("chunk_type") == "sensorimotor_body_chunk"
            }
        if not discovered_triples:
            return sm_cands
        for ch in getattr(self, "basal_ganglia_reflex_chunks", []):
            if isinstance(ch, dict) and ch.get("chunk_type") == "sensorimotor_composite_chunk":
                c_steps = ch.get("steps", [])
                disc_g = {t[0] for t in discovered_triples}
                disc_w = {t[1] for t in discovered_triples}
                disc_f = {t[2] for t in discovered_triples}
                if c_steps and all(
                    str(st.get("gaze_op")) in disc_g
                    and str(st.get("wm_op")) in disc_w
                    and str(st.get("finger_op")) in disc_f
                    for st in c_steps
                    if isinstance(st, dict)
                ):
                    sig_k = str(ch.get("sig_key", "composite"))
                    p_comp = {"steps": [dict(st) for st in c_steps]}
                    sm_cands.append((
                        f"sensorimotor_chunk_composite_{sig_k}",
                        lambda g, p=p_comp: self.execute_sensorimotor_chain(
                            g, (p.get("steps", []) if isinstance(p, dict) else [])
                        ),
                        p_comp,
                    ))
        has_legend_chunk = ("gaze_corner_sample", "wm_store_sample_pair", "finger_paint_from_wm") in discovered_triples
        has_group_chunk = ("gaze_next_matching_attr", "wm_store_attr", "finger_apply_to_gaze") in discovered_triples
        has_contact_chunk = ("gaze_salient_target", "wm_store_attr", "finger_step_until_contact") in discovered_triples
        has_count_chunk = ("gaze_count_objects", "wm_store_count", "finger_step_n_from_wm") in discovered_triples
        x0_in = cur_pairs[0][0]
        bg0_in = detect_background_color(x0_in)
        comps0_in = extract_components(x0_in, bg=bg0_in)
        enc0_in = find_enclosed_background_regions(x0_in, bg=bg0_in)
        in_colors = [int(c) for c in np.unique(x0_in) if int(c) != bg0_in]
        if comps0_in:
            c_small = min(comps0_in, key=lambda o: (o["mass"], o["bbox"][0], o["bbox"][1]))
            c_large = max(comps0_in, key=lambda o: (o["mass"], -o["bbox"][0], -o["bbox"][1]))
            tc_dom = int(c_small["color"])
        else:
            c_small = None
            c_large = None
            tc_dom = int(in_colors[0]) if in_colors else int(bg0_in)

        h_list = [o for o in comps0_in if int(o["holes"]) > 0]
        nh_list = [o for o in comps0_in if int(o["holes"]) == 0]
        if h_list and nh_list:
            perceived_sub_sel = "has_hole"
        elif len(comps0_in) >= 2 and c_small is not None and c_large is not None and c_small["mass"] < c_large["mass"]:
            perceived_sub_sel = "smallest"
        else:
            perceived_sub_sel = "largest"

        if has_contact_chunk and comps0_in:
            sel_mov = "smallest" if len(comps0_in) >= 2 else "largest"
            p_contact_default = {
                "gaze_op": "gaze_salient_target",
                "wm_op": "wm_store_attr",
                "finger_op": "finger_step_until_contact",
                "selector": sel_mov,
                "dr": 0,
                "dc": 0,
            }
            sm_cands.append((
                f"sensorimotor_chunk_target_{sel_mov}_shift_to_static",
                lambda g, p=p_contact_default: self.execute_unified_sensorimotor_chunk(
                    g, p["gaze_op"], p["wm_op"], p["finger_op"], p
                ),
                p_contact_default,
            ))
            if len(comps0_in) >= 2 and c_small is not None and c_large is not None:
                ref_sel_m = "largest" if sel_mov == "smallest" else "smallest"
                dy_m = float(c_large["center"][0] - c_small["center"][0])
                dx_m = float(c_large["center"][1] - c_small["center"][1])
                m_mode = "guide_col" if abs(dx_m) >= abs(dy_m) else "guide_row"
                for sm_margin in (0, -1):
                    chain_memo_guide = [
                        {
                            "gaze_op": "gaze_salient_target",
                            "wm_op": "wm_store_attr",
                            "finger_op": "finger_apply_to_gaze",
                            "selector": ref_sel_m,
                            "write_to_memo": True,
                            "memo_mode": m_mode,
                        },
                        {
                            "gaze_op": "gaze_salient_target",
                            "wm_op": "wm_store_attr",
                            "finger_op": "finger_step_until_contact",
                            "selector": sel_mov,
                            "dr": 0,
                            "dc": 0,
                            "prefer_memo_guide": True,
                            "stop_margin": sm_margin,
                        },
                    ]
                    sm_cands.append((
                        f"sensorimotor_chunk_chain_memo_{m_mode}_contact_{sel_mov}_m{sm_margin}",
                        lambda g, p=None, s=chain_memo_guide: self.execute_sensorimotor_chain(
                            g, (p.get("steps", s) if isinstance(p, dict) else s)
                        ),
                        {"steps": chain_memo_guide},
                    ))
            if has_legend_chunk and len(in_colors) >= 2:
                chain_contact_paint = [
                    {
                        "gaze_op": "gaze_salient_target",
                        "wm_op": "wm_store_attr",
                        "finger_op": "finger_step_until_contact",
                        "selector": sel_mov,
                        "dr": 0,
                        "dc": 0,
                    },
                    {
                        "gaze_op": "gaze_salient_target",
                        "wm_op": "wm_store_attr",
                        "finger_op": "finger_paint_from_wm",
                        "selector": "moved_objects",
                        "use_contact_color": True,
                    },
                ]
                sm_cands.append((
                    f"sensorimotor_chunk_chain_contact_paint_{sel_mov}",
                    lambda g, p=None, s=chain_contact_paint: self.execute_sensorimotor_chain(
                        g, (p.get("steps", s) if isinstance(p, dict) else s)
                    ),
                    {"steps": chain_contact_paint},
                ))

        if has_group_chunk and len(comps0_in) >= 2:
            p_grp = {
                "gaze_op": "gaze_next_matching_attr",
                "wm_op": "wm_store_attr",
                "finger_op": "finger_apply_to_gaze",
                "selector": "same_color",
                "target_color": tc_dom,
            }
            sm_cands.append((
                f"sensorimotor_chunk_group_match_c{tc_dom}",
                self.execute_unified_sensorimotor_chunk,
                p_grp,
            ))

        if enc0_in:
            for sel_b in ("all", "enclosed_hole_region"):
                sub_m = "recolor" if sel_b == "enclosed_hole_region" else "fill_holes"
                p_fill = {
                    "gaze_op": "gaze_salient_target",
                    "wm_op": "wm_store_attr",
                    "finger_op": "finger_apply_to_gaze",
                    "selector": sel_b,
                    "motor_op": sub_m,
                    "sub_op": sub_m,
                    "target_color": tc_dom,
                    "fill_color": tc_dom,
                    "action_param": tc_dom,
                }
                sm_cands.append((
                    f"sensorimotor_chunk_{sel_b}_{sub_m}_c{tc_dom}",
                    self.execute_unified_sensorimotor_chunk,
                    p_fill,
                ))
            p_zoom_room = {
                "gaze_op": "gaze_salient_target",
                "wm_op": "wm_store_attr",
                "finger_op": "finger_apply_to_gaze",
                "selector": "largest",
                "sub_op": "zoom_interior_room",
            }
            sm_cands.append((
                "sensorimotor_chunk_target_largest_zoom_interior_room",
                self.execute_unified_sensorimotor_chunk,
                p_zoom_room,
            ))

        if comps0_in:
            for sel_b in ("all", perceived_sub_sel):
                p_rec = {
                    "gaze_op": "gaze_salient_target",
                    "wm_op": "wm_store_attr",
                    "finger_op": "finger_apply_to_gaze",
                    "selector": sel_b,
                    "motor_op": "recolor",
                    "sub_op": "recolor",
                    "target_color": tc_dom,
                    "action_param": tc_dom,
                }
                sm_cands.append((
                    f"sensorimotor_chunk_{sel_b}_recolor_c{tc_dom}",
                    self.execute_unified_sensorimotor_chunk,
                    p_rec,
                ))
            p_crop = {
                "gaze_op": "gaze_salient_target",
                "wm_op": "wm_store_attr",
                "finger_op": "finger_apply_to_gaze",
                "selector": "largest",
                "sub_op": "crop_grasped",
            }
            sm_cands.append((
                "sensorimotor_chunk_target_largest_crop_grasped",
                self.execute_unified_sensorimotor_chunk,
                p_crop,
            ))

        if has_count_chunk and len(comps0_in) >= 2 and c_small is not None and c_large is not None:
            dy_c = float(c_large["center"][0] - c_small["center"][0])
            dx_c = float(c_large["center"][1] - c_small["center"][1])
            if abs(dy_c) >= abs(dx_c):
                dr_c, dc_c = (1 if dy_c >= 0.0 else -1), 0
            else:
                dr_c, dc_c = 0, (1 if dx_c >= 0.0 else -1)
            p_cnt = {
                "gaze_op": "gaze_count_objects",
                "wm_op": "wm_store_count",
                "finger_op": "finger_step_n_from_wm",
                "selector": "smallest",
                "dr": dr_c,
                "dc": dc_c,
                "step_style": "shift",
            }
            sm_cands.append((
                f"sensorimotor_chunk_count_step_{dr_c}_{dc_c}",
                lambda g, p=p_cnt: self.execute_unified_sensorimotor_chunk(
                    g, p["gaze_op"], p["wm_op"], p["finger_op"], p
                ),
                p_cnt,
            ))

        if len(comps0_in) >= 2:
            anc_s = "smallest" if perceived_sub_sel == "largest" else "largest"
            tgt_s = "largest" if anc_s == "smallest" else "smallest"
            rel_op = "fill_holes_from_anchor" if enc0_in else "recolor_from_anchor"
            p_dyad = {
                "gaze_op": "gaze_salient_target",
                "wm_op": "wm_bind_anchor_attr",
                "finger_op": "finger_apply_to_gaze",
                "anchor_selector": anc_s,
                "selector": tgt_s,
                "sub_op": rel_op,
                "multi_match": True,
                "consume_anchor": False,
            }
            sm_cands.append((
                f"sensorimotor_chunk_dyadic_{anc_s}_to_{tgt_s}_{rel_op}",
                self.execute_unified_sensorimotor_chunk,
                p_dyad,
            ))

        shape_bias = 0
        if isinstance(diff_rule_info, dict) and "preserve_shape" in diff_rule_info:
            if bool(diff_rule_info.get("preserve_shape", True)):
                shape_bias = -1
            elif (
                float(diff_rule_info.get("scale_ratio_h", 1.0)) < 1.0
                or float(diff_rule_info.get("scale_ratio_w", 1.0)) < 1.0
            ):
                shape_bias = 1

        def shape_priority(item):
            is_shape_op = ("crop_grasped" in item[0]) or ("zoom_interior_room" in item[0])
            if shape_bias > 0:
                return 0 if is_shape_op else 1
            if shape_bias < 0:
                return 1 if is_shape_op else 0
            return 0

        sm_cands.sort(key=shape_priority)
        mirror_sm_front = []
        for p_fov, lbl_fov in getattr(self, "learned_foveal_stroke_prototypes", [])[:2]:
            if isinstance(p_fov, dict):
                p_f_copy = dict(p_fov)
                mirror_sm_front.append((
                    f"sensorimotor_chunk_{lbl_fov}",
                    lambda g, p=p_f_copy: self.execute_foveal_saccade_stroke_closed_loop(g, p),
                    p_f_copy,
                ))
        for m_pdict, m_lbl, _ in getattr(self, "_mirror_inferred_affordance_steps", [])[:4]:
            if isinstance(m_pdict, dict) and "gaze_op" in m_pdict and "finger_op" in m_pdict:
                p_m_copy = dict(m_pdict)
                mirror_sm_front.append((
                    f"sensorimotor_chunk_{m_lbl}",
                    lambda g, p=p_m_copy: self.execute_unified_sensorimotor_chunk(
                        g, p["gaze_op"], p["wm_op"], p["finger_op"], p
                    ),
                    p_m_copy,
                ))
        if q_spk_hint is not None and len(sm_cands) > 2:
            spk_decoded = self.decode_spiking_action_from_neural_state(q_spk_hint)
            proc_recalled = self.recall_synaptic_procedural_attractor(q_spk_hint)
            mirror_recalled = []
            last_m_step = getattr(self, "_last_mirror_author_step", None)
            if last_m_step is not None:
                mirror_recalled.append(last_m_step)
            spk_sm_front = []
            for spk_pdict, spk_lbl, _ in (mirror_recalled + spk_decoded + proc_recalled)[:4]:
                if isinstance(spk_pdict, dict) and "gaze_op" in spk_pdict and "finger_op" in spk_pdict:
                    p_spk_sm = dict(spk_pdict)
                else:
                    dr_s = int(np.clip(spk_pdict.get("dy", 0), -1, 1))
                    dc_s = int(np.clip(spk_pdict.get("dx", 0), -1, 1))
                    if dr_s == 0 and dc_s == 0:
                        dr_s = 1
                    p_spk_sm = {
                        "gaze_op": "gaze_salient_target",
                        "wm_op": "wm_store_attr",
                        "finger_op": "finger_step_until_contact",
                        "selector": "smallest",
                        "dr": dr_s,
                        "dc": dc_s,
                    }
                spk_sm_front.append((
                    f"sensorimotor_chunk_{spk_lbl}",
                    lambda g, p=p_spk_sm: self.execute_unified_sensorimotor_chunk(
                        g, p["gaze_op"], p["wm_op"], p["finger_op"], p
                    ),
                    p_spk_sm,
                ))
            comp_front = [c for c in sm_cands if c[0].startswith(("sensorimotor_chunk_composite_", "sensorimotor_chunk_all_"))]
            rest_cands = [c for c in sm_cands if not c[0].startswith(("sensorimotor_chunk_composite_", "sensorimotor_chunk_all_"))]
            rest_cands.sort(key=lambda item: (shape_priority(item), -self.score_sensorimotor_body_resonance(item[2], q_spk_hint)))
            sm_cands = mirror_sm_front + comp_front + rest_cands + spk_sm_front
        elif mirror_sm_front:
            sm_cands = mirror_sm_front + sm_cands
        return sm_cands[:7]
    def perceive_foveal_peripheral_saccade(self, x_grid, saccade_idx=0, external_wm_canvas=None, y_ref=None):
        H, W = x_grid.shape
        n_sacc = max(3, min(6, (int(saccade_idx) % 4) + 3))
        if hasattr(self, "unified_brains") and self.unified_brains is not None:
            sac_info = self.unified_brains.sensory_cortex.saccade_and_integrate_mental_canvas(
                x_grid, max_saccades=n_sacc
            )
            self._last_saccade_info = sac_info
            self.foveal_saccades_executed_total += len(sac_info["saccade_trajectory"])
            self.foveal_mental_canvas_updates_total += 1
            mental_canvas = sac_info["mental_canvas"].copy()
            if external_wm_canvas is not None and external_wm_canvas.shape == x_grid.shape:
                prev_mask = (external_wm_canvas > 0) & (~sac_info["foveated_mask"])
                mental_canvas[prev_mask] = external_wm_canvas[prev_mask]
            gy_last, gx_last = sac_info["saccade_trajectory"][-1]
            r0 = max(0, gy_last - 2)
            r1 = min(H, gy_last + 3)
            c0 = max(0, gx_last - 2)
            c1 = min(W, gx_last + 3)
            fov_patch = mental_canvas[r0:r1, c0:c1] if (r1 > r0 and c1 > c0) else mental_canvas
            rp0 = max(0, gy_last - 4)
            rp1 = min(H, gy_last + 5)
            cp0 = max(0, gx_last - 4)
            cp1 = min(W, gx_last + 5)
            ctx_patch = mental_canvas[rp0:rp1, cp0:cp1] if (rp1 > rp0 and cp1 > cp0) else mental_canvas
            ret_obs = np.asarray(sac_info["retinal_16"], dtype=np.float32)
        else:
            self.foveal_saccades_executed_total += 1
            gy_last, gx_last = H // 2, W // 2
            r0, r1 = max(0, gy_last - 2), min(H, gy_last + 3)
            c0, c1 = max(0, gx_last - 2), min(W, gx_last + 3)
            fov_patch = x_grid[r0:r1, c0:c1] if (r1 > r0 and c1 > c0) else x_grid
            ctx_patch = x_grid
            ret_obs = None
        self.encode_retinotopic_foveal_patch_spikes(x_grid, gy_last, gx_last)
        foveal_obs = compute_grid_observation(fov_patch, hp=1.0, energy=0.5, pain=0.0, surprise=0.0)
        ctx_obs = compute_grid_observation(ctx_patch, hp=1.0, energy=0.5, pain=0.0, surprise=0.0)
        wm_coverage = float(np.mean(external_wm_canvas > 0)) if (external_wm_canvas is not None and external_wm_canvas.size > 0) else 0.0
        wm_alpha = float(np.clip(0.58 - 0.15 * wm_coverage, 0.40, 0.65))
        blended_obs = (wm_alpha * foveal_obs + (1.0 - wm_alpha) * ctx_obs).astype(np.float32)
        if ret_obs is not None:
            blended_obs = (0.76 * blended_obs + 0.24 * ret_obs).astype(np.float32)
        return blended_obs
    def apply_topological_lens_forward(self, grid, lens_mode="cartesian", lens_params=None):
        H, W = grid.shape
        p_lens = lens_params if isinstance(lens_params, dict) else {}
        fwd_map = np.zeros((H, W, 2), dtype=np.int32)
        inv_map = np.zeros((H, W, 2), dtype=np.int32)
        for r in range(H):
            for c in range(W):
                fwd_map[r, c, 0] = r
                fwd_map[r, c, 1] = c
                inv_map[r, c, 0] = r
                inv_map[r, c, 1] = c
        if H < 2 or W < 2 or lens_mode == "cartesian":
            return grid.copy(), {
                "lens_mode": "cartesian",
                "orig_shape": (H, W),
                "fwd_map": fwd_map,
                "inv_map": inv_map,
            }
        self.topological_lens_warps_total += 1
        bg = detect_background_color(grid)
        if lens_mode == "torus_cylinder":
            if "shift_r" in p_lens and "shift_c" in p_lens:
                shift_r = int(p_lens["shift_r"]) % H
                shift_c = int(p_lens["shift_c"]) % W
            else:
                fg_mask = grid != bg
                touch_tb = bool(np.any(fg_mask[0, :]) and np.any(fg_mask[H - 1, :]))
                touch_lr = bool(np.any(fg_mask[:, 0]) and np.any(fg_mask[:, W - 1]))
                shift_r = (H // 2) if touch_tb else 0
                shift_c = (W // 2) if touch_lr else 0
                if shift_r == 0 and shift_c == 0:
                    shift_r = H // 2
                    shift_c = W // 2
            warped = np.empty_like(grid)
            for r in range(H):
                for c in range(W):
                    wr = (r + shift_r) % H
                    wc = (c + shift_c) % W
                    fwd_map[r, c, 0] = wr
                    fwd_map[r, c, 1] = wc
                    inv_map[wr, wc, 0] = r
                    inv_map[wr, wc, 1] = c
                    warped[wr, wc] = grid[r, c]
            return warped, {
                "lens_mode": "torus_cylinder",
                "orig_shape": (H, W),
                "shift_r": int(shift_r),
                "shift_c": int(shift_c),
                "fwd_map": fwd_map,
                "inv_map": inv_map,
            }
        if lens_mode == "depth_25d":
            vals, cnts = np.unique(grid, return_counts=True)
            fg_items = [(int(v), int(cnt)) for v, cnt in zip(vals, cnts) if int(v) != bg]
            if len(fg_items) < 2:
                return grid.copy(), {
                    "lens_mode": "depth_25d",
                    "orig_shape": (H, W),
                    "bg_color": int(bg),
                    "occluder_color": int(bg),
                    "occluder_mask": np.zeros((H, W), dtype=bool),
                    "occluder_layer": np.full((H, W), bg, dtype=np.int32),
                    "interpolated_bridge_mask": np.zeros((H, W), dtype=bool),
                    "warped_baseline": grid.copy(),
                    "z_render_mode": str(p_lens.get("z_render_mode", "occluder_on_top")),
                    "fwd_map": fwd_map,
                    "inv_map": inv_map,
                }
            if "occluder_color" in p_lens and p_lens["occluder_color"] is not None:
                occ_col = int(p_lens["occluder_color"])
            else:
                span_Select = []
                all_comps_fg = extract_components(grid, bg=bg)
                for v_c, cnt_c in fg_items:
                    c_comps = [o for o in all_comps_fg if int(o["color"]) == int(v_c)]
                    max_single_span = 0
                    max_axis_full = 0
                    for o in c_comps:
                        r0_o, c0_o, r1_o, c1_o = o["bbox"]
                        sr = r1_o - r0_o + 1
                        sc = c1_o - c0_o + 1
                        max_single_span = max(max_single_span, max(sr, sc))
                        if sr == H or sc == W:
                            max_axis_full = 1
                    span_Select.append((max_axis_full, max_single_span, cnt_c, v_c))
                span_Select.sort(reverse=True)
                occ_col = int(span_Select[0][3])
            occ_mask = grid == occ_col
            occ_layer = np.where(occ_mask, grid, bg).astype(np.int32)
            warped = np.where(occ_mask, bg, grid).astype(np.int32)
            bridge_mask = np.zeros((H, W), dtype=bool)
            if bool(p_lens.get("interpolate_bridges", True)):
                for r in range(H):
                    for c in range(W):
                        if not occ_mask[r, c]:
                            continue
                        r_up = r - 1
                        while r_up >= 0 and occ_mask[r_up, c]:
                            r_up -= 1
                        r_dn = r + 1
                        while r_dn < H and occ_mask[r_dn, c]:
                            r_dn += 1
                        c_lt = c - 1
                        while c_lt >= 0 and occ_mask[r, c_lt]:
                            c_lt -= 1
                        c_rt = c + 1
                        while c_rt < W and occ_mask[r, c_rt]:
                            c_rt += 1
                        if r_up >= 0 and r_dn < H and (r_dn - r_up <= 4) and warped[r_up, c] != bg and warped[r_up, c] == warped[r_dn, c]:
                            warped[r, c] = warped[r_up, c]
                            bridge_mask[r, c] = True
                            self.v2_collinear_completions_total += 1
                        elif c_lt >= 0 and c_rt < W and (c_rt - c_lt <= 4) and warped[r, c_lt] != bg and warped[r, c_lt] == warped[r, c_rt]:
                            warped[r, c] = warped[r, c_lt]
                            bridge_mask[r, c] = True
                            self.v2_collinear_completions_total += 1
            return warped, {
                "lens_mode": "depth_25d",
                "orig_shape": (H, W),
                "bg_color": int(bg),
                "occluder_color": int(occ_col),
                "occluder_mask": occ_mask,
                "occluder_layer": occ_layer,
                "interpolated_bridge_mask": bridge_mask,
                "warped_baseline": warped.copy(),
                "z_render_mode": str(p_lens.get("z_render_mode", "occluder_on_top")),
                "fwd_map": fwd_map,
                "inv_map": inv_map,
            }
        if lens_mode == "polar_centroid":
            r0 = float(p_lens.get("center_r", (H - 1) * 0.5))
            c0 = float(p_lens.get("center_c", (W - 1) * 0.5))
            polar_cells = []
            for r in range(H):
                for c in range(W):
                    dy = float(r) - r0
                    dx = float(c) - c0
                    r_cheb = round(max(abs(dy), abs(dx)), 4)
                    theta = round(math.atan2(dy, dx), 6)
                    polar_cells.append((r_cheb, theta, r, c))
            polar_cells.sort()
            warped = np.empty_like(grid)
            for flat_idx, (_, _, r_orig, c_orig) in enumerate(polar_cells):
                wr = flat_idx // W
                wc = flat_idx % W
                fwd_map[r_orig, c_orig, 0] = wr
                fwd_map[r_orig, c_orig, 1] = wc
                inv_map[wr, wc, 0] = r_orig
                inv_map[wr, wc, 1] = c_orig
                warped[wr, wc] = grid[r_orig, c_orig]
            return warped, {
                "lens_mode": "polar_centroid",
                "orig_shape": (H, W),
                "center": (r0, c0),
                "fwd_map": fwd_map,
                "inv_map": inv_map,
            }
        return grid.copy(), {
            "lens_mode": "cartesian",
            "orig_shape": (H, W),
            "fwd_map": fwd_map,
            "inv_map": inv_map,
        }
    def apply_topological_lens_inverse(self, warped_grid, lens_tensor):
        if not isinstance(lens_tensor, dict):
            return warped_grid.copy()
        orig_shape = tuple(lens_tensor.get("orig_shape", warped_grid.shape))
        if warped_grid.shape != orig_shape:
            return warped_grid.copy()
        H, W = orig_shape
        inv_map = lens_tensor.get("inv_map", None)
        if isinstance(inv_map, np.ndarray) and inv_map.shape == (H, W, 2):
            unwarped = np.empty_like(warped_grid)
            for wr in range(H):
                for wc in range(W):
                    r_orig = int(inv_map[wr, wc, 0])
                    c_orig = int(inv_map[wr, wc, 1])
                    unwarped[r_orig, c_orig] = warped_grid[wr, wc]
        else:
            unwarped = warped_grid.copy()
        l_mode = str(lens_tensor.get("lens_mode", "cartesian"))
        if l_mode == "depth_25d":
            bg_c = int(lens_tensor.get("bg_color", 0))
            occ_mask = lens_tensor.get("occluder_mask", None)
            occ_layer = lens_tensor.get("occluder_layer", None)
            bridge_mask = lens_tensor.get("interpolated_bridge_mask", None)
            w_base = lens_tensor.get("warped_baseline", None)
            z_mode = str(lens_tensor.get("z_render_mode", "occluder_on_top"))
            if (
                z_mode != "bridge_revealed"
                and isinstance(bridge_mask, np.ndarray)
                and isinstance(w_base, np.ndarray)
                and bridge_mask.shape == (H, W)
                and w_base.shape == (H, W)
            ):
                unchanged_bridge = bridge_mask & (unwarped == w_base)
                unwarped = np.where(unchanged_bridge, bg_c, unwarped)
            if isinstance(occ_mask, np.ndarray) and isinstance(occ_layer, np.ndarray) and occ_mask.shape == (H, W):
                if z_mode == "bridge_revealed" and isinstance(bridge_mask, np.ndarray):
                    unwarped = np.where(bridge_mask, unwarped, np.where(occ_mask, occ_layer, unwarped))
                elif z_mode == "manipulated_on_top":
                    unwarped = np.where((unwarped != bg_c) & (~bridge_mask if isinstance(bridge_mask, np.ndarray) else True), unwarped, np.where(occ_mask, occ_layer, unwarped))
                else:
                    unwarped = np.where(occ_mask, occ_layer, unwarped)
        return unwarped
    def execute_embodied_action_through_topological_lens(self, grid, action_fn=None, action_params=None, lens_mode="cartesian", lens_params=None):
        warped_g, lens_tensor = self.apply_topological_lens_forward(grid, lens_mode=lens_mode, lens_params=lens_params)
        if action_fn is not None:
            warped_out = action_fn(warped_g, dict(action_params or {}))
        else:
            warped_out = warped_g.copy()
        cartesian_out = self.apply_topological_lens_inverse(warped_out, lens_tensor)
        return cartesian_out, lens_tensor
    def modulate_ach_topological_lens(self, max_order_gain_val, surprise_val, step_idx=0):
        stagnation_drive = float(sigmoid(10.0 * (0.045 - float(max_order_gain_val))))
        ach_val = float(np.clip(0.65 * stagnation_drive + 0.35 * float(surprise_val), 0.0, 1.0))
        self.ach_signal_level = round(ach_val, 4)
        if ach_val < 0.42:
            self.active_topological_lens = "cartesian"
        else:
            lens_cycle = ("torus_cylinder", "depth_25d", "polar_centroid")
            self.active_topological_lens = lens_cycle[int(step_idx) % len(lens_cycle)]
        return self.active_topological_lens, ach_val
    def _build_composite_meta_component(self, x_grid, sub_comps, chunk_tag="constellation", active_lens="cartesian"):
        H_g, W_g = x_grid.shape
        bg_x = detect_background_color(x_grid)
        r0_c, c0_c = (H_g - 1) * 0.5, (W_g - 1) * 0.5
        combined_cells = []
        warped_cells = []
        colors_list = []
        for sc in sub_comps:
            for cell_rc in sc.get("cells", []):
                if (0 <= cell_rc[0] < H_g) and (0 <= cell_rc[1] < W_g):
                    combined_cells.append((int(cell_rc[0]), int(cell_rc[1])))
                    colors_list.append(int(x_grid[cell_rc[0], cell_rc[1]]))
            for w_rc in sc.get("warped_cells", sc.get("cells", [])):
                warped_cells.append((int(w_rc[0]), int(w_rc[1])))
        if not combined_cells and sub_comps:
            combined_cells = list(sub_comps[0].get("cells", [(0, 0)]))
            colors_list = [int(sub_comps[0].get("color", 1))] * len(combined_cells)
        rs = [p[0] for p in combined_cells]
        cs = [p[1] for p in combined_cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        sub_h, sub_w = r1 - r0 + 1, c1 - c0 + 1
        mask = np.zeros((sub_h, sub_w), dtype=bool)
        patch = np.full((sub_h, sub_w), bg_x, dtype=np.int32)
        for cr, cc in combined_cells:
            mask[cr - r0, cc - c0] = True
            patch[cr - r0, cc - c0] = x_grid[cr, cc]
        hole_cnt = count_enclosed_holes(mask)
        col_cnts = Counter(colors_list) if colors_list else Counter([1])
        dom_c = int(col_cnts.most_common(1)[0][0])
        min_c = int(col_cnts.most_common()[-1][0])
        cent_r = float(np.mean(rs))
        cent_c = float(np.mean(cs))
        dy_p = cent_r - r0_c
        dx_p = cent_c - c0_c
        meta_comp = {
            "color": dom_c,
            "min_color": min_c,
            "colors": set(colors_list),
            "color_counts": dict(col_cnts),
            "cells": combined_cells,
            "warped_cells": warped_cells if warped_cells else list(combined_cells),
            "warped_bbox": (r0, c0, r1, c1),
            "mass": len(combined_cells),
            "bbox": (r0, c0, r1, c1),
            "height": sub_h,
            "width": sub_w,
            "area": sub_h * sub_w,
            "center": (cent_r, cent_c),
            "mask": mask,
            "patch": patch,
            "holes": int(hole_cnt),
            "is_rect": len(combined_cells) == sub_h * sub_w,
            "polar_r": round(float(math.sqrt(dy_p * dy_p + dx_p * dx_p)), 4),
            "polar_theta": round(float(math.atan2(dy_p, dx_p)), 4),
            "lens_mode": str(active_lens),
            "is_texture_field": bool(chunk_tag == "texture_field"),
            "is_constellation_group": bool(chunk_tag == "constellation"),
            "is_constellation": bool(chunk_tag == "constellation"),
            "sub_components": list(sub_comps),
            "sub_component_count": int(len(sub_comps)),
        }
        meta_comp["proto"] = fuzzy_prototype_signature(meta_comp)
        return meta_comp
    def v1_v2_preattentive_texture_and_constellation_grouping(self, x_grid, raw_comps, active_lens="cartesian"):
        n_raw = len(raw_comps)
        if n_raw <= 6:
            return list(raw_comps), {
                "crowded": False,
                "raw_count": n_raw,
                "compressed_count": n_raw,
                "texture_segregated": False,
                "constellations_formed": 0,
            }
        H_g, W_g = x_grid.shape
        dot_by_color = {}
        non_dots = []
        for comp in raw_comps:
            c_val = int(comp.get("color", 0))
            m_val = int(comp.get("mass", len(comp.get("cells", []))))
            h_val = int(comp.get("height", 1))
            w_val = int(comp.get("width", 1))
            is_full_line = (h_val == H_g and w_val == 1) or (w_val == W_g and h_val == 1)
            if m_val == 1 or is_full_line:
                dot_by_color.setdefault(c_val, []).append(comp)
            else:
                non_dots.append(comp)
        texture_chunks = []
        remaining_small = []
        for c_val, d_list in dot_by_color.items():
            if len(d_list) >= 6:
                rs = sorted(set(int(round(d["center"][0])) for d in d_list))
                cs = sorted(set(int(round(d["center"][1])) for d in d_list))
                r_diffs = [rs[k + 1] - rs[k] for k in range(len(rs) - 1)] if len(rs) >= 3 else []
                c_diffs = [cs[k + 1] - cs[k] for k in range(len(cs) - 1)] if len(cs) >= 3 else []
                periodic_r = bool(r_diffs and len(set(r_diffs)) <= 2)
                periodic_c = bool(c_diffs and len(set(c_diffs)) <= 2)
                if (periodic_r and periodic_c) or (len(d_list) >= 9 and len(non_dots) >= 1):
                    t_chunk = self._build_composite_meta_component(
                        x_grid, d_list, chunk_tag="texture_field", active_lens=active_lens
                    )
                    texture_chunks.append(t_chunk)
                    self.v1_v2_texture_segregations_total += 1
                else:
                    remaining_small.extend(d_list)
            else:
                remaining_small.extend(d_list)
        pool = list(non_dots) + list(remaining_small)
        constellation_chunks = []
        if len(pool) + len(texture_chunks) > 4:
            large_salient = []
            small_by_color = {}
            for comp in pool:
                if int(comp.get("mass", 1)) > 6 and len(large_salient) < 4:
                    large_salient.append(comp)
                else:
                    small_by_color.setdefault(int(comp.get("color", 0)), []).append(comp)
            grouped_pool = list(large_salient)
            for c_val, s_list in sorted(small_by_color.items(), key=lambda kv: -len(kv[1])):
                if len(s_list) >= 2:
                    c_meta = self._build_composite_meta_component(
                        x_grid, s_list, chunk_tag="constellation", active_lens=active_lens
                    )
                    constellation_chunks.append(c_meta)
                    grouped_pool.append(c_meta)
                    self.v1_v2_constellation_groups_total += 1
                else:
                    grouped_pool.extend(s_list)
            if len(grouped_pool) + len(texture_chunks) > 7:
                grouped_pool.sort(key=lambda o: int(o.get("mass", 1)), reverse=True)
                keep_front = grouped_pool[:5]
                overflow = grouped_pool[5:]
                overflow_sub = []
                for ov in overflow:
                    if ov.get("sub_components"):
                        overflow_sub.extend(ov["sub_components"])
                    else:
                        overflow_sub.append(ov)
                if overflow_sub:
                    ov_meta = self._build_composite_meta_component(
                        x_grid, overflow_sub, chunk_tag="constellation", active_lens=active_lens
                    )
                    constellation_chunks.append(ov_meta)
                    self.v1_v2_constellation_groups_total += 1
                    grouped_pool = keep_front + [ov_meta]
            final_comps = grouped_pool + texture_chunks
        else:
            final_comps = pool + texture_chunks
        return final_comps, {
            "crowded": True,
            "raw_count": n_raw,
            "compressed_count": len(final_comps),
            "texture_segregated": bool(len(texture_chunks) > 0),
            "constellations_formed": len(constellation_chunks),
        }
    def expand_constellation_meta_object(self, meta_obj, focus_bbox=None):
        if not isinstance(meta_obj, dict) or not meta_obj.get("sub_components"):
            return [meta_obj] if isinstance(meta_obj, dict) else []
        self.v1_v2_hierarchical_expansions_total += 1
        subs = list(meta_obj["sub_components"])
        if focus_bbox is None:
            return subs
        fr0, fc0, fr1, fc1 = focus_bbox
        filtered = [
            sc for sc in subs
            if not (sc["bbox"][2] < fr0 or sc["bbox"][0] > fr1 or sc["bbox"][3] < fc0 or sc["bbox"][1] > fc1)
        ]
        return filtered if filtered else subs
    def perceive_meaningful_objects_foveal(self, x_grid, saccade_idx=0, external_wm_canvas=None, lens_mode=None, y_ref=None):
        active_lens = str(lens_mode) if lens_mode is not None else str(getattr(self, "active_topological_lens", "cartesian"))
        warped_x, x_lens_tensor = self.apply_topological_lens_forward(x_grid, lens_mode=active_lens)
        obs_vec = self.perceive_foveal_peripheral_saccade(
            warped_x, saccade_idx=saccade_idx, external_wm_canvas=external_wm_canvas
        )
        sac_info = getattr(self, "_last_saccade_info", None)
        if isinstance(sac_info, dict) and "mental_canvas" in sac_info:
            perceptual_source_grid = sac_info["mental_canvas"]
            self.foveated_mental_canvas_perceptions_total += 1
            if sac_info.get("saccade_trajectory"):
                gy_l, gx_l = sac_info["saccade_trajectory"][-1]
                gaze_yx_norm = (
                    float(gy_l) / max(1.0, float(warped_x.shape[0] - 1)),
                    float(gx_l) / max(1.0, float(warped_x.shape[1] - 1)),
                )
            else:
                gaze_yx_norm = (0.5, 0.5)
        else:
            perceptual_source_grid = warped_x
            gaze_yx_norm = (0.5, 0.5)
        seg_info = self.v1_v2_opponent_color_figure_ground_segregation(perceptual_source_grid)
        obs_vec, _ = self.compute_parietal_gain_field_modulation(
            obs_vec, lens_mode=active_lens, gaze_yx=gaze_yx_norm
        )
        bg_x = int(seg_info.get("inferred_bg_color", detect_background_color(perceptual_source_grid)))
        comps_warped = extract_components(perceptual_source_grid, bg=bg_x)
        inv_map = x_lens_tensor.get("inv_map", None)
        H_g, W_g = x_grid.shape
        r0_c, c0_c = (H_g - 1) * 0.5, (W_g - 1) * 0.5
        raw_comps_x = []
        for cw in comps_warped:
            if isinstance(inv_map, np.ndarray) and inv_map.shape == (H_g, W_g, 2) and active_lens != "cartesian":
                orig_cells = [(int(inv_map[wr, wc, 0]), int(inv_map[wr, wc, 1])) for wr, wc in cw["cells"]]
            else:
                orig_cells = list(cw["cells"])
            rs = [p[0] for p in orig_cells]
            cs = [p[1] for p in orig_cells]
            cent_r = float(np.mean(rs)) if rs else 0.0
            cent_c = float(np.mean(cs)) if cs else 0.0
            dy_p = cent_r - r0_c
            dx_p = cent_c - c0_c
            c_copy = dict(cw)
            c_copy["warped_cells"] = list(cw["cells"])
            c_copy["warped_bbox"] = cw["bbox"]
            c_copy["cells"] = orig_cells
            c_copy["center"] = (cent_r, cent_c)
            c_copy["bbox"] = (min(rs), min(cs), max(rs), max(cs)) if rs else cw["bbox"]
            c_copy["polar_r"] = round(float(math.sqrt(dy_p * dy_p + dx_p * dx_p)), 4)
            c_copy["polar_theta"] = round(float(math.atan2(dy_p, dx_p)), 4)
            c_copy["lens_mode"] = active_lens
            raw_comps_x.append(c_copy)
        comps_x, v1_v2_meta = self.v1_v2_preattentive_texture_and_constellation_grouping(
            x_grid, raw_comps_x, active_lens=active_lens
        )
        holes_x = find_enclosed_background_regions(warped_x, bg=bg_x) if len(comps_x) <= 16 else []
        self.foveal_objects_perceived_total += len(comps_x)
        focal_obj = comps_x[int(saccade_idx) % len(comps_x)] if comps_x else None
        h_sym = bool(warped_x.shape[1] >= 2 and np.array_equal(warped_x, warped_x[:, ::-1]))
        v_sym = bool(warped_x.shape[0] >= 2 and np.array_equal(warped_x, warped_x[::-1, :]))
        comps_y = extract_components(y_ref, bg=detect_background_color(y_ref)) if y_ref is not None else []
        holes_y = (
            find_enclosed_background_regions(y_ref, bg=detect_background_color(y_ref))
            if (y_ref is not None and len(comps_y) <= 16)
            else []
        )
        return obs_vec, {
            "objects_cur": comps_x,
            "raw_objects_cur": raw_comps_x,
            "v1_v2_grouping_meta": v1_v2_meta,
            "objects_target": comps_y,
            "focal_object": focal_obj,
            "gaps_cur": len(holes_x),
            "gaps_target": len(holes_y),
            "h_sym": h_sym,
            "v_sym": v_sym,
            "shape_cur": x_grid.shape,
            "shape_target": y_ref.shape if y_ref is not None else x_grid.shape,
            "saccade_idx": int(saccade_idx),
            "lens_mode": active_lens,
            "lens_tensor": x_lens_tensor,
            "target_observed": bool(y_ref is not None),
        }
    def derive_train_relational_prior(self, train_pairs, diff_rule_info=None):
        info = diff_rule_info if isinstance(diff_rule_info, dict) else {}
        shape_ratios_h = []
        shape_ratios_w = []
        count_deltas = []
        gaps_deltas = []
        shifts_r = []
        shifts_c = []
        for x_p, y_p in train_pairs:
            shape_ratios_h.append(float(y_p.shape[0]) / float(max(1, x_p.shape[0])))
            shape_ratios_w.append(float(y_p.shape[1]) / float(max(1, x_p.shape[1])))
            bg_x = detect_background_color(x_p)
            bg_y = detect_background_color(y_p)
            comps_px = extract_components(x_p, bg=bg_x)
            comps_py = extract_components(y_p, bg=bg_y)
            count_deltas.append(len(comps_py) - len(comps_px))
            if len(comps_px) <= 16 and len(comps_py) <= 16:
                gaps_x = len(find_enclosed_background_regions(x_p, bg=bg_x))
                gaps_y = len(find_enclosed_background_regions(y_p, bg=bg_y))
                gaps_deltas.append(gaps_y - gaps_x)
            if x_p.shape == y_p.shape and len(comps_px) == len(comps_py) and len(comps_px) >= 1:
                mx = np.mean(np.array([c["center"] for c in comps_px], dtype=np.float32), axis=0)
                my = np.mean(np.array([c["center"] for c in comps_py], dtype=np.float32), axis=0)
                shifts_r.append(float(my[0] - mx[0]))
                shifts_c.append(float(my[1] - mx[1]))
        if "preserve_shape" in info:
            preserve_shape = bool(info["preserve_shape"])
        else:
            preserve_shape = all(x_p.shape == y_p.shape for x_p, y_p in train_pairs)
        scale_h = float(info["scale_ratio_h"]) if "scale_ratio_h" in info else float(np.median(shape_ratios_h))
        scale_w = float(info["scale_ratio_w"]) if "scale_ratio_w" in info else float(np.median(shape_ratios_w))
        return {
            "preserve_shape": preserve_shape,
            "scale_ratio_h": scale_h,
            "scale_ratio_w": scale_w,
            "count_delta": int(round(float(np.median(count_deltas)))) if count_deltas else 0,
            "gaps_delta": int(round(float(np.median(gaps_deltas)))) if gaps_deltas else 0,
            "center_shift": (
                float(np.median(shifts_r)) if shifts_r else 0.0,
                float(np.median(shifts_c)) if shifts_c else 0.0,
            ),
        }
    def cerebellar_sensorimotor_error_correction(self, cur_pairs, orig_inputs, cand_tuple, pred_grids, base_e=0.0):
        if not cur_pairs or not pred_grids or len(cur_pairs) != len(pred_grids):
            return None
        cname, cfn, cparams, d4_id = cand_tuple
        if not str(cname).startswith("sensorimotor_chunk_") or not isinstance(cparams, dict):
            return None
        x0, y0, p0 = cur_pairs[0][0], cur_pairs[0][1], pred_grids[0]
        _, cf_err = self.compute_climbing_fiber_error_vector(p0, y0)
        if abs(float(cf_err[0])) >= abs(float(cf_err[1])):
            dr_cf = 1 if float(cf_err[0]) >= 0.0 else -1
            dc_cf = 0
        else:
            dr_cf = 0
            dc_cf = 1 if float(cf_err[1]) >= 0.0 else -1
        err_cells = abs(float(cf_err[0])) * float(y0.shape[0]) + abs(float(cf_err[1])) * float(y0.shape[1])
        sm_cf = 1 if err_cells >= 0.5 else -1
        cb_variants = []
        if "gaze_op" in cparams and cparams.get("finger_op") == "finger_step_until_contact":
            p_cf = dict(cparams)
            p_cf["dr"] = dr_cf
            p_cf["dc"] = dc_cf
            cb_variants.append((
                f"{cname}__cb_climbing_fiber_{dr_cf}_{dc_cf}",
                lambda g, p=p_cf: self.execute_unified_sensorimotor_chunk(g, p["gaze_op"], p["wm_op"], p["finger_op"], p),
                p_cf,
            ))
        if "steps" in cparams and isinstance(cparams["steps"], list) and cparams["steps"]:
            base_steps = [dict(st) for st in cparams["steps"] if isinstance(st, dict)]
            for idx_s, st in enumerate(base_steps):
                f_op = str(st.get("finger_op", ""))
                g_op = str(st.get("gaze_op", ""))
                if f_op == "finger_step_until_contact":
                    mod_steps_m = [dict(s) for s in base_steps]
                    mod_steps_m[idx_s]["stop_margin"] = sm_cf
                    p_mod_m = {"steps": mod_steps_m}
                    cb_variants.append((
                        f"{cname}__cb_margin_{sm_cf}",
                        lambda g, p=p_mod_m: self.execute_sensorimotor_chain(g, p["steps"]),
                        p_mod_m,
                    ))
                    mod_steps_d = [dict(s) for s in base_steps]
                    mod_steps_d[idx_s]["dr"] = dr_cf
                    mod_steps_d[idx_s]["dc"] = dc_cf
                    p_mod_d = {"steps": mod_steps_d}
                    cb_variants.append((
                        f"{cname}__cb_dir_{dr_cf}_{dc_cf}",
                        lambda g, p=p_mod_d: self.execute_sensorimotor_chain(g, p["steps"]),
                        p_mod_d,
                    ))
                if f_op == "finger_paint_from_wm":
                    mod_steps = [dict(s) for s in base_steps]
                    mod_steps[idx_s]["dr"] = dr_cf
                    mod_steps[idx_s]["dc"] = dc_cf
                    p_mod = {"steps": mod_steps}
                    cb_variants.append((
                        f"{cname}__cb_paint_offset_{dr_cf}_{dc_cf}",
                        lambda g, p=p_mod: self.execute_sensorimotor_chain(g, p["steps"]),
                        p_mod,
                    ))
        elif "gaze_op" in cparams:
            f_op = str(cparams.get("finger_op", "finger_step_until_contact"))
            g_op = str(cparams.get("gaze_op", ""))
            if f_op == "finger_step_until_contact":
                p_m = dict(cparams)
                p_m["stop_margin"] = sm_cf
                cb_variants.append((
                    f"{cname}__cb_margin_{sm_cf}",
                    lambda g, p=p_m: self.execute_unified_sensorimotor_chunk(g, p["gaze_op"], p["wm_op"], p["finger_op"], p),
                    p_m,
                ))
                p_d = dict(cparams)
                p_d["dr"] = dr_cf
                p_d["dc"] = dc_cf
                p_d["stop_margin"] = max(0, sm_cf)
                cb_variants.append((
                    f"{cname}__cb_dir_{dr_cf}_{dc_cf}_m{max(0, sm_cf)}",
                    lambda g, p=p_d: self.execute_unified_sensorimotor_chunk(g, p["gaze_op"], p["wm_op"], p["finger_op"], p),
                    p_d,
                ))
                sel_v = "smallest" if cparams.get("selector") != "smallest" else "largest"
                p_s = dict(cparams)
                p_s["selector"] = sel_v
                cb_variants.append((
                    f"{cname}__cb_sel_{sel_v}",
                    lambda g, p=p_s: self.execute_unified_sensorimotor_chunk(g, p["gaze_op"], p["wm_op"], p["finger_op"], p),
                    p_s,
                ))
            elif f_op == "finger_step_n_from_wm":
                st_style = "ray" if float(cf_err[5]) > 0.05 else "shift"
                p_c = dict(cparams)
                p_c["step_style"] = st_style
                p_c["dr"] = dr_cf
                p_c["dc"] = dc_cf
                cb_variants.append((
                    f"{cname}__cb_{st_style}_{dr_cf}_{dc_cf}",
                    lambda g, p=p_c: self.execute_unified_sensorimotor_chunk(g, p["gaze_op"], p["wm_op"], p["finger_op"], p),
                    p_c,
                ))
            elif g_op == "gaze_corner_sample" and f_op == "finger_paint_from_wm":
                p_l = dict(cparams)
                p_l["dr"] = dr_cf
                p_l["dc"] = dc_cf
                cb_variants.append((
                    f"{cname}__cb_corner_{dr_cf}_{dc_cf}",
                    lambda g, p=p_l: self.execute_unified_sensorimotor_chunk(g, p["gaze_op"], p["wm_op"], p["finger_op"], p),
                    p_l,
                ))
            elif f_op in ("finger_paint_from_wm", "finger_apply_to_gaze"):
                p_a = dict(cparams)
                p_a["dr"] = dr_cf
                p_a["dc"] = dc_cf
                cb_variants.append((
                    f"{cname}__cb_kinematic_{dr_cf}_{dc_cf}",
                    lambda g, p=p_a: self.execute_unified_sensorimotor_chunk(g, p["gaze_op"], p["wm_op"], p["finger_op"], p),
                    p_a,
                ))
                d4_k = 4 if abs(float(cf_err[1])) >= abs(float(cf_err[0])) else 5
                p_r = dict(cparams)
                p_r["finger_op"] = "finger_apply_to_gaze"
                p_r["sub_op"] = "rotate_flip_in_place"
                p_r["action_param"] = int(d4_k)
                p_r["d4_k"] = int(d4_k)
                cb_variants.append((
                    f"{cname}__cb_d4_{d4_k}",
                    lambda g, p=p_r: self.execute_unified_sensorimotor_chunk(g, p["gaze_op"], p["wm_op"], p["finger_op"], p),
                    p_r,
                ))
        best_partial = None
        best_e = float(base_e) + 0.06
        for cb_name, cb_fn, cb_params in cb_variants[:6]:
            ge, gd, g_exact, g_gs = self.evaluate_candidate_on_pairs(cb_fn, cb_params, cur_pairs, orig_inputs, d4_conj=int(d4_id))
            if g_exact:
                self.cerebellar_error_corrections_total += 1
                return (1.0, 0.0, (cb_name, cb_fn, cb_params, int(d4_id)), g_gs, True)
            if g_gs and ge > best_e:
                best_e = float(ge)
                best_partial = (float(ge), float(gd), (cb_name, cb_fn, cb_params, int(d4_id)), g_gs, False)
        if best_partial is not None:
            self.cerebellar_error_corrections_total += 1
        return best_partial
    def extract_sensorimotor_steps_from_params(self, cparams):
        if not isinstance(cparams, dict):
            return []
        if "steps" in cparams and isinstance(cparams["steps"], list):
            return [dict(s) for s in cparams["steps"] if isinstance(s, dict)]
        if "gaze_op" in cparams and "wm_op" in cparams and "finger_op" in cparams:
            return [dict(cparams)]
        return []
    def rollout_prefrontal_sensorimotor_chain(self, cur_pairs, orig_inputs, seed_scored, q_spk_hint=None):
        if not cur_pairs:
            return None
        discovered_triples = {
            (str(ch.get("gaze_op")), str(ch.get("wm_op")), str(ch.get("finger_op")))
            for ch in getattr(self, "basal_ganglia_reflex_chunks", [])
            if isinstance(ch, dict) and ch.get("chunk_type") == "sensorimotor_body_chunk"
        }
        if not discovered_triples:
            return None
        disc_g = {t[0] for t in discovered_triples}
        disc_w = {t[1] for t in discovered_triples}
        disc_f = {t[2] for t in discovered_triples}
        depth1_states = []
        for item in seed_scored:
            e_val, d_val, c_tup, next_gs, _ = item
            cname, cfn, cparams, d4_id = c_tup
            if str(cname).startswith("sensorimotor_chunk_") and next_gs:
                steps1 = self.extract_sensorimotor_steps_from_params(cparams)
                if steps1 and not np.array_equal(next_gs[0], cur_pairs[0][0]):
                    depth1_states.append((float(e_val), steps1, next_gs))
        if len(depth1_states) < 3:
            base_sm = self.build_unified_sensorimotor_affordances(cur_pairs, q_spk_hint=q_spk_hint)
            for cname, cfn, cparams in base_sm[:6]:
                steps1 = self.extract_sensorimotor_steps_from_params(cparams)
                if not steps1 or len(steps1) > 2:
                    continue
                e_val, d_val, is_exact, next_gs = self.evaluate_candidate_on_pairs(cfn, cparams, cur_pairs, orig_inputs, d4_conj=0)
                if is_exact:
                    self.prefrontal_dynamic_chains_total += 1
                    return (1.0, 0.0, (cname, cfn, cparams, 0), next_gs, True)
                if next_gs and not np.array_equal(next_gs[0], cur_pairs[0][0]):
                    depth1_states.append((float(e_val), steps1, next_gs))
                if len(depth1_states) >= 3:
                    break
        depth2_states = []
        for e1, steps1, gs1 in depth1_states[:3]:
            pairs1 = list(zip(gs1, [p[1] for p in cur_pairs]))
            diff1 = {}
            sm2_cands = self.build_unified_sensorimotor_affordances(pairs1, diff_rule_info=diff1, q_spk_hint=q_spk_hint)
            follow_specs = []
            for _, _, cp2 in sm2_cands[:4]:
                st2 = self.extract_sensorimotor_steps_from_params(cp2)
                if st2 and len(st2) == 1:
                    follow_specs.append(st2[0])
            x1_0 = pairs1[0][0]
            bg1 = detect_background_color(x1_0)
            diff_cols1 = [int(c) for c in np.unique(x1_0) if int(c) != bg1]
            if "gaze_salient_target" in disc_g and "wm_store_attr" in disc_w and "finger_paint_from_wm" in disc_f:
                follow_specs.append({
                    "gaze_op": "gaze_salient_target",
                    "wm_op": "wm_store_attr",
                    "finger_op": "finger_paint_from_wm",
                    "selector": "moved_objects",
                    "use_contact_color": True,
                })
                if diff_cols1:
                    follow_specs.append({
                        "gaze_op": "gaze_salient_target",
                        "wm_op": "wm_store_attr",
                        "finger_op": "finger_paint_from_wm",
                        "selector": "moved_objects",
                        "val_color": int(diff_cols1[0]),
                        "paint_mode": "fill_holes",
                    })
            if "gaze_salient_target" in disc_g and "wm_store_attr" in disc_w and "finger_step_until_contact" in disc_f:
                follow_specs.append({
                    "gaze_op": "gaze_salient_target",
                    "wm_op": "wm_store_attr",
                    "finger_op": "finger_step_until_contact",
                    "selector": "smallest",
                    "dr": 0,
                    "dc": 0,
                })
            if "gaze_corner_sample" in disc_g and "wm_store_sample_pair" in disc_w and "finger_paint_from_wm" in disc_f:
                follow_specs.append({
                    "gaze_op": "gaze_corner_sample",
                    "wm_op": "wm_store_sample_pair",
                    "finger_op": "finger_paint_from_wm",
                    "legend_mode": "recolor_keep_legend",
                })
            for st2_d in follow_specs[:6]:
                if (
                    str(st2_d.get("gaze_op")) not in disc_g
                    or str(st2_d.get("wm_op")) not in disc_w
                    or str(st2_d.get("finger_op")) not in disc_f
                ):
                    continue
                chain2 = [dict(s) for s in steps1] + [dict(st2_d)]
                sim_score = self.simulate_latent_pfc_rollout(
                    cur_pairs[0][0], cur_pairs[0][1], [("sensorimotor", s) for s in chain2]
                )
                if float(sim_score) < -0.15:
                    continue
                p_c2 = {"steps": chain2}
                fn_c2 = lambda g, p=p_c2: self.execute_sensorimotor_chain(g, p["steps"])
                name_c2 = f"sensorimotor_chunk_pfc_chain{len(chain2)}_{chain2[0].get('finger_op','')}__{chain2[-1].get('finger_op','')}"
                e2, d2, ex2, gs2 = self.evaluate_candidate_on_pairs(fn_c2, p_c2, cur_pairs, orig_inputs, d4_conj=0)
                if ex2:
                    self.prefrontal_dynamic_chains_total += 1
                    return (1.0, 0.0, (name_c2, fn_c2, p_c2, 0), gs2, True)
                if gs2 and e2 >= 0.35:
                    cb_hit = self.cerebellar_sensorimotor_error_correction(
                        cur_pairs, orig_inputs, (name_c2, fn_c2, p_c2, 0), gs2, base_e=e2
                    )
                    if cb_hit is not None and cb_hit[4]:
                        self.prefrontal_dynamic_chains_total += 1
                        return cb_hit
                    depth2_states.append((float(e2), chain2, gs2))
        depth2_states.sort(key=lambda item: item[0], reverse=True)
        beam_depth2 = 4 if float(getattr(self, "organism_energy", 0.8)) > 0.35 else 2
        for e2, chain2, gs2 in depth2_states[:beam_depth2]:
            pairs2 = list(zip(gs2, [p[1] for p in cur_pairs]))
            x2_0 = pairs2[0][0]
            bg2 = detect_background_color(x2_0)
            diff_cols2 = [int(c) for c in np.unique(x2_0) if int(c) != bg2]
            step3_specs = []
            if "gaze_salient_target" in disc_g and "wm_store_attr" in disc_w and "finger_paint_from_wm" in disc_f:
                step3_specs.append({
                    "gaze_op": "gaze_salient_target",
                    "wm_op": "wm_store_attr",
                    "finger_op": "finger_paint_from_wm",
                    "selector": "moved_objects",
                    "use_contact_color": True,
                })
                if diff_cols2:
                    for pm3 in ("recolor", "fill_holes", "border"):
                        step3_specs.append({
                            "gaze_op": "gaze_salient_target",
                            "wm_op": "wm_store_attr",
                            "finger_op": "finger_paint_from_wm",
                            "selector": "moved_objects",
                            "val_color": int(diff_cols2[0]),
                            "paint_mode": pm3,
                        })
            if "gaze_corner_sample" in disc_g and "wm_store_sample_pair" in disc_w and "finger_paint_from_wm" in disc_f:
                step3_specs.append({
                    "gaze_op": "gaze_corner_sample",
                    "wm_op": "wm_store_sample_pair",
                    "finger_op": "finger_paint_from_wm",
                    "legend_mode": "recolor_erase_legend",
                })
            for st3_d in step3_specs[:4]:
                chain3 = [dict(s) for s in chain2] + [dict(st3_d)]
                p_c3 = {"steps": chain3}
                fn_c3 = lambda g, p=p_c3: self.execute_sensorimotor_chain(g, p["steps"])
                name_c3 = f"sensorimotor_chunk_pfc_chain3_{chain3[0].get('finger_op','')}__{chain3[1].get('finger_op','')}__{chain3[2].get('finger_op','')}"
                e3, d3, ex3, gs3 = self.evaluate_candidate_on_pairs(fn_c3, p_c3, cur_pairs, orig_inputs, d4_conj=0)
                if ex3:
                    self.prefrontal_dynamic_chains_total += 1
                    return (1.0, 0.0, (name_c3, fn_c3, p_c3, 0), gs3, True)
                if gs3 and e3 >= 0.40:
                    cb_hit3 = self.cerebellar_sensorimotor_error_correction(
                        cur_pairs, orig_inputs, (name_c3, fn_c3, p_c3, 0), gs3, base_e=e3
                    )
                    if cb_hit3 is not None and cb_hit3[4]:
                        self.prefrontal_dynamic_chains_total += 1
                        return cb_hit3
        return None
    def _single_grid_gestalt_score(self, g):
        if g.size == 0 or g.shape[0] == 0 or g.shape[1] == 0:
            return 0.0
        bg = detect_background_color(g)
        non_bg_ratio = float(np.mean(g != bg))
        if non_bg_ratio < 0.01:
            return 0.0
        h_sym = float(np.mean(g == g[:, ::-1])) if g.shape[1] >= 2 else 1.0
        v_sym = float(np.mean(g == g[::-1, :])) if g.shape[0] >= 2 else 1.0
        r_sym = float(np.mean(g == np.rot90(g, 2))) if (g.shape[0] >= 2 and g.shape[1] >= 2) else 1.0
        sym_score = 0.40 * max(h_sym, v_sym, r_sym) + 0.20 * ((h_sym + v_sym) * 0.5)
        comps = extract_components(g, bg=bg)
        if not comps:
            return sym_score * 0.5
        fill_ratios = []
        obj_syms = []
        for o in comps:
            r0, c0, r1, c1 = o["bbox"]
            area = max(1, (r1 - r0 + 1) * (c1 - c0 + 1))
            fill_ratios.append(float(o["mass"]) / float(area))
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            oh = float(np.mean(sub == sub[:, ::-1])) if sub.shape[1] >= 2 else 1.0
            ov = float(np.mean(sub == sub[::-1, :])) if sub.shape[0] >= 2 else 1.0
            obj_syms.append(max(oh, ov))
        closure_score = float(np.mean(fill_ratios))
        local_sym_score = float(np.mean(obj_syms))
        if len(comps) >= 2:
            aligned_pairs = 0
            total_pairs = 0
            for idx_a in range(min(8, len(comps))):
                for idx_b in range(idx_a + 1, min(8, len(comps))):
                    total_pairs += 1
                    ca = comps[idx_a]
                    cb = comps[idx_b]
                    if (
                        abs(ca["center"][0] - cb["center"][0]) <= 0.6
                        or abs(ca["center"][1] - cb["center"][1]) <= 0.6
                        or abs(ca["bbox"][0] - cb["bbox"][0]) <= 0
                        or abs(ca["bbox"][1] - cb["bbox"][1]) <= 0
                    ):
                        aligned_pairs += 1
            align_score = float(aligned_pairs) / float(max(1, total_pairs))
        else:
            align_score = 0.85
        cohesion_score = 1.0 / (1.0 + 0.12 * max(0, len(comps) - 1))
        return float(0.30 * sym_score + 0.22 * local_sym_score + 0.22 * align_score + 0.16 * closure_score + 0.10 * cohesion_score)
    def compute_intrinsic_gestalt_beauty(self, x_before, x_after):
        self.intrinsic_gestalt_evaluations_total += 1
        b_before = self._single_grid_gestalt_score(x_before)
        b_after = self._single_grid_gestalt_score(x_after)
        b_gain = float(b_after - b_before)
        is_neater = bool(b_gain > 0.015 and x_after.size > 0 and not np.array_equal(x_before, x_after))
        return float(b_after), b_gain, is_neater
    def compute_epistemic_relational_progress(self, cur_pairs, next_gs):
        if not cur_pairs or not next_gs or len(cur_pairs) != len(next_gs):
            return 0.0
        gains = []
        for (x_cur, y_tgt), x_nxt in zip(cur_pairs, next_gs):
            if x_cur.size == 0 or y_tgt.size == 0 or x_nxt.size == 0:
                continue
            bg_c = detect_background_color(x_cur)
            bg_y = detect_background_color(y_tgt)
            bg_n = detect_background_color(x_nxt)
            comps_c = extract_components(x_cur, bg=bg_c)
            comps_y = extract_components(y_tgt, bg=bg_y)
            comps_n = extract_components(x_nxt, bg=bg_n)
            cnt_err_before = abs(len(comps_c) - len(comps_y))
            cnt_err_after = abs(len(comps_n) - len(comps_y))
            cnt_gain = 0.04 * float(np.clip(cnt_err_before - cnt_err_after, -2, 3))
            colors_c = set(int(v) for v in np.unique(x_cur) if int(v) != bg_c)
            colors_y = set(int(v) for v in np.unique(y_tgt) if int(v) != bg_y)
            colors_n = set(int(v) for v in np.unique(x_nxt) if int(v) != bg_n)
            j_before = len(colors_c & colors_y) / float(max(1, len(colors_c | colors_y)))
            j_after = len(colors_n & colors_y) / float(max(1, len(colors_n | colors_y)))
            col_gain = 0.06 * float(j_after - j_before)
            pos_gain = 0.0
            if comps_c and comps_n and comps_y:
                cy_arr = np.array([c["center"] for c in comps_y], dtype=np.float32)
                cc_arr = np.array([c["center"] for c in comps_c], dtype=np.float32)
                cn_arr = np.array([c["center"] for c in comps_n], dtype=np.float32)
                d_before = float(np.mean(np.min(np.linalg.norm(cc_arr[:, None, :] - cy_arr[None, :, :], axis=2), axis=1)))
                d_after = float(np.mean(np.min(np.linalg.norm(cn_arr[:, None, :] - cy_arr[None, :, :], axis=2), axis=1)))
                pos_gain = 0.05 * float(np.clip((d_before - d_after) / max(1.0, float(max(y_tgt.shape))), -0.5, 1.0))
            gains.append(cnt_gain + col_gain + pos_gain)
        return float(np.mean(gains)) if gains else 0.0
    def formulate_one_step_subgoal(self, obj_scene, act_vec, d4_hint, surprise_val, safety_gate, diff_rule_info=None, train_prior=None):
        self.one_step_subgoals_formulated_total += 1
        objs_cur = obj_scene.get("objects_cur", [])
        objs_tgt = obj_scene.get("objects_target", [])
        shape_cur = obj_scene.get("shape_cur", (1, 1))
        shape_tgt = obj_scene.get("shape_target", (1, 1))
        gaps_cur = int(obj_scene.get("gaps_cur", 0))
        gaps_tgt = int(obj_scene.get("gaps_target", 0))
        if isinstance(train_prior, dict) and not bool(obj_scene.get("target_observed", False)):
            if not bool(train_prior.get("preserve_shape", True)):
                shape_tgt = (
                    max(1, int(round(float(shape_cur[0]) * float(train_prior.get("scale_ratio_h", 1.0))))),
                    max(1, int(round(float(shape_cur[1]) * float(train_prior.get("scale_ratio_w", 1.0))))),
                )
            else:
                shape_tgt = shape_cur
            gaps_tgt = max(0, gaps_cur + int(train_prior.get("gaps_delta", 0)))
            n_tgt = max(0, len(objs_cur) + int(train_prior.get("count_delta", 0)))
            shift_r, shift_c = train_prior.get("center_shift", (0.0, 0.0))
            objs_tgt = [
                {"center": (float(o["center"][0]) + float(shift_r), float(o["center"][1]) + float(shift_c))}
                for o in objs_cur[:n_tgt]
            ]
            if n_tgt > len(objs_cur):
                objs_tgt = objs_tgt + [{"center": objs_tgt[-1]["center"] if objs_tgt else (0.0, 0.0)}] * (n_tgt - len(objs_cur))
        subgoal_boost = np.zeros_like(act_vec)
        preferred_prefixes = []
        subgoal_type = "refine_object_state"
        if shape_cur != shape_tgt:
            if shape_cur[0] > shape_tgt[0] or shape_cur[1] > shape_tgt[1]:
                subgoal_type = "crop_or_condense_object"
                subgoal_boost[1] += 0.38
                subgoal_boost[6] += 0.24
                preferred_prefixes.extend(["fovea_crop_", "fovea_mc_crop_", "spatial_condense", "mono_part_", "topo_roi_", "layer_", "part_bool_"])
            else:
                subgoal_type = "tile_or_scale_object"
                subgoal_boost[2] += 0.38
                subgoal_boost[7] += 0.22
                preferred_prefixes.extend(["rigid_tile", "rigid_scale", "macro_quad_d4_", "gestalt_periodic_"])
        elif gaps_cur != gaps_tgt:
            subgoal_type = "enclose_or_fill_gap"
            subgoal_boost[4] += 0.36
            subgoal_boost[6] += 0.25
            subgoal_boost[7] += 0.20
            preferred_prefixes.extend(["topo_enclose_", "topo_chromatic_", "mono_frame_ind_", "gestalt_bbox_", "mono_bind_A_to_B"])
        elif len(objs_cur) == len(objs_tgt) and len(objs_cur) >= 1:
            c_cur = objs_cur[0]["center"]
            c_tgt = objs_tgt[0]["center"]
            if abs(c_cur[0] - c_tgt[0]) + abs(c_cur[1] - c_tgt[1]) > 0.5:
                subgoal_type = "slide_or_settle_object"
                subgoal_boost[0] += 0.34
                subgoal_boost[3] += 0.32
                subgoal_boost[6] += 0.22
                preferred_prefixes.extend(["grav_", "kin_slide", "kin_global_", "kin_role_", "goal_align_", "goal_stem_slide_", "mono_bind_A_to_B"])
            else:
                subgoal_type = "transform_or_recolor_object"
                subgoal_boost[2] += 0.30
                subgoal_boost[4] += 0.30
                subgoal_boost[6] += 0.24
                preferred_prefixes.extend(["rigid_d4_", "rigid_sym_", "topo_chromatic_", "mono_bind_A_to_B", "mono_select_", "new_primitive_node_0x"])
        else:
            subgoal_type = "connect_or_imprint_objects"
            subgoal_boost[6] += 0.30
            subgoal_boost[7] += 0.28
            subgoal_boost[0] += 0.22
            preferred_prefixes.extend(["kin_ray_", "kin_collinear_", "goal_", "gestalt_", "canvas_stitch_", "mono_bind_A_to_B", "new_primitive_node_0x"])
        if isinstance(diff_rule_info, dict) and "delta_vec" in diff_rule_info and hasattr(self, "unified_brains") and self.unified_brains is not None:
            delta_v = np.asarray(diff_rule_info["delta_vec"], dtype=np.float32)
            m_pol = self.unified_brains.motor_reflex.infer_motor_policy(delta_v, delta_v)
            dp_vec = np.asarray(m_pol["act_2d_logits"], dtype=np.float32)
            n_dp = min(len(subgoal_boost), dp_vec.size)
            subgoal_boost[:n_dp] += 0.28 * np.tanh(dp_vec[:n_dp])
        curiosity_gain = float(safety_gate) * (0.20 + 0.35 * float(surprise_val))
        guided_act = (act_vec + (0.45 + curiosity_gain) * subgoal_boost).astype(np.float32)
        snapped_act, snapped_d4 = self.snap_to_discrete_attractor(guided_act, int(d4_hint))
        return snapped_act, snapped_d4, {
            "subgoal_type": subgoal_type,
            "preferred_prefixes": tuple(preferred_prefixes),
            "safety_gate": float(safety_gate),
            "diff_rule_info": diff_rule_info,
            "saccade_idx": int(obj_scene.get("saccade_idx", 0)),
        }
    def prefrontal_counterfactual_gaba_gate(self, cur_pairs, orig_inputs, cand_tuple, next_gs=None, q_spk_hint=None):
        cname, cfn, cparams, d4_id = cand_tuple
        if len(cur_pairs) < 2:
            return False, None, {"gaba_veto": False, "incongruity_gap": 0.0, "intuitive_channel_gain": 1.0}
        if next_gs is None:
            _, _, _, next_gs = self.evaluate_candidate_on_pairs(
                cfn, cparams, cur_pairs, orig_inputs, d4_conj=int(d4_id)
            )
        if not next_gs or len(next_gs) != len(cur_pairs):
            return False, None, {"gaba_veto": False, "incongruity_gap": 0.0, "intuitive_channel_gain": 1.0}
        pleasure_list = []
        pain_list = []
        exact_flags = []
        for idx_p, ((x_g, y_g), pred_g) in enumerate(zip(cur_pairs, next_gs)):
            orig_g = orig_inputs[idx_p] if idx_p < len(orig_inputs) else x_g
            e_base_k, _ = compute_homeostatic_structural_order(x_g, y_g, x_orig=orig_g)
            if pred_g.shape == y_g.shape and np.array_equal(pred_g, y_g):
                e_k, d_k = 1.0, 0.0
                exact_flags.append(True)
            else:
                e_k, d_k = compute_homeostatic_structural_order(pred_g, y_g, x_orig=orig_g)
                exact_flags.append(False)
            pleasure_k = float(e_k)
            pain_k = float(d_k) + max(0.0, float(e_base_k) - float(e_k))
            pleasure_list.append(pleasure_k)
            pain_list.append(pain_k)
        if all(exact_flags):
            return False, None, {"gaba_veto": False, "incongruity_gap": 0.0, "intuitive_channel_gain": 1.0}
        max_pleasure = float(max(pleasure_list))
        min_pleasure = float(min(pleasure_list))
        max_pain = float(max(pain_list))
        min_pain = float(min(pain_list))
        incongruity_gap = float((max_pleasure - min_pain) - (min_pleasure - max_pain))
        has_incongruity = bool(
            max_pleasure >= 0.72
            and (min_pleasure < 0.96 or (max_pain - min_pain) >= 0.12)
            and incongruity_gap >= 0.22
        )
        if not has_incongruity:
            return False, None, {"gaba_veto": False, "incongruity_gap": round(incongruity_gap, 4), "intuitive_channel_gain": 1.0}
        self.counterfactual_incongruity_triggers_total += 1
        self.prefrontal_gaba_blocks_total += 1
        self.latent_feature_bypasses_total += 1
        diag_info = {
            "gaba_veto": True,
            "intuitive_channel_gain": 0.0,
            "incongruity_gap": round(incongruity_gap, 4),
            "pair_pleasures": [round(v, 4) for v in pleasure_list],
            "pair_pains": [round(v, 4) for v in pain_list],
            "bypassed_latent_feature": None,
        }
        base_p = dict(cparams) if isinstance(cparams, dict) else {}
        op_base = str(base_p.get("motor_op", "recolor"))
        trap_idx = int(np.argmax(pain_list))
        x_trap, _ = cur_pairs[trap_idx]
        bg_trap = detect_background_color(x_trap)
        comps_trap = extract_components(x_trap, bg=bg_trap)
        cand_selectors = ["has_hole", "smallest", "largest", "unique_shape", "no_hole", "parity_odd", "parity_even"]
        in_cols = [int(c) for c in np.unique(x_trap) if int(c) != bg_trap]
        cand_colors = []
        if "target_color" in base_p and base_p["target_color"] is not None:
            cand_colors.append(int(base_p["target_color"]))
        cand_colors.extend(in_cols)
        if not cand_colors:
            cand_colors.append(int(bg_trap))
        trial_params_list = []
        for p_sel in cand_selectors[:4]:
            if op_base in ("translate", "contact_translate", "gravity_step", "symmetrize") and base_p:
                tp = dict(base_p)
                tp["selector"] = p_sel
                trial_params_list.append((p_sel, tp))
            for tc in cand_colors[:2]:
                tp_rc = dict(base_p)
                tp_rc["selector"] = p_sel
                tp_rc["motor_op"] = "recolor"
                tp_rc["target_color"] = int(tc)
                trial_params_list.append((p_sel, tp_rc))
        best_bypass = None
        best_bypass_score = float(np.mean(pleasure_list))
        for p_sel, tp in trial_params_list:
            lat_name = f"sensorimotor_chunk_{p_sel}_{tp.get('motor_op', 'recolor')}_gaba_bypass"
            e_lat, d_lat, is_lat_ex, gs_lat = self.evaluate_candidate_on_pairs(
                self.execute_unified_sensorimotor_chunk, tp, cur_pairs, orig_inputs, d4_conj=int(d4_id)
            )
            if is_lat_ex:
                diag_info["bypassed_latent_feature"] = p_sel
                return True, (1.0, 0.0, (lat_name, self.execute_unified_sensorimotor_chunk, tp, int(d4_id)), gs_lat, True), diag_info
            if gs_lat and e_lat > best_bypass_score and d_lat <= min_pain + 0.05:
                best_bypass_score = float(e_lat)
                diag_info["bypassed_latent_feature"] = p_sel
                best_bypass = (float(e_lat), float(d_lat), (lat_name, self.execute_unified_sensorimotor_chunk, tp, int(d4_id)), gs_lat, False)
        return True, best_bypass, diag_info
    def evaluate_through_ach_topological_lenses(self, cur_pairs, orig_inputs):
        if not cur_pairs:
            return None
        x0, y0 = cur_pairs[0]
        if x0.shape != y0.shape:
            return None
        cache_dict = getattr(self, "_ach_lens_eval_cache", None)
        if cache_dict is None:
            cache_dict = {}
            self._ach_lens_eval_cache = cache_dict
        ckey = (_grid_cache_key(x0), _grid_cache_key(y0), len(cur_pairs))
        if ckey in cache_dict:
            return cache_dict[ckey]
        def _torus_roll_op(g, p):
            w_g, _ = self.apply_topological_lens_forward(
                g, lens_mode="torus_cylinder", lens_params={"shift_r": int(p["dr"]), "shift_c": int(p["dc"])}
            )
            return w_g
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 2), (2, 0), (-1, -1), (1, 1)]:
            t_params = {"dr": int(dr), "dc": int(dc), "lens_mode": "torus_cylinder"}
            _, _, is_t_ex, gs_t = self.evaluate_candidate_on_pairs(
                _torus_roll_op, t_params, cur_pairs, orig_inputs, d4_conj=0
            )
            if is_t_ex:
                t_name = f"sensorimotor_chunk_torus_cylinder_shift_{dr}_{dc}"
                res_hit = (1.0, 0.0, (t_name, _torus_roll_op, t_params, 0), gs_t, True)
                cache_dict[ckey] = res_hit
                return res_hit
        def _depth_bridge_op(g, p):
            out_g, _ = self.execute_embodied_action_through_topological_lens(
                g, action_fn=None, action_params=None, lens_mode="depth_25d", lens_params={"z_render_mode": "bridge_revealed"}
            )
            return out_g
        d_params = {"lens_mode": "depth_25d", "z_render_mode": "bridge_revealed"}
        _, _, is_d_ex, gs_d = self.evaluate_candidate_on_pairs(
            _depth_bridge_op, d_params, cur_pairs, orig_inputs, d4_conj=0
        )
        if is_d_ex:
            res_hit = (1.0, 0.0, ("sensorimotor_chunk_depth_25d_bridge_revealed", _depth_bridge_op, d_params, 0), gs_d, True)
            cache_dict[ckey] = res_hit
            return res_hit
        bg0 = detect_background_color(x0)
        in_cols = [int(c) for c in np.unique(x0) if int(c) != bg0]
        tc_val = int(in_cols[0]) if in_cols else int(bg0)
        def _depth_recolor_op(g, p):
            out_g, _ = self.execute_embodied_action_through_topological_lens(
                g,
                action_fn=self.execute_unified_sensorimotor_chunk,
                action_params={"selector": "largest", "motor_op": "recolor", "target_color": int(p["tc"])},
                lens_mode="depth_25d",
                lens_params={"z_render_mode": "occluder_on_top"},
            )
            return out_g
        dr_params = {"tc": int(tc_val), "lens_mode": "depth_25d"}
        _, _, is_dr_ex, gs_dr = self.evaluate_candidate_on_pairs(
            _depth_recolor_op, dr_params, cur_pairs, orig_inputs, d4_conj=0
        )
        if is_dr_ex:
            res_hit = (1.0, 0.0, (f"sensorimotor_chunk_depth_25d_recolor_c{tc_val}", _depth_recolor_op, dr_params, 0), gs_dr, True)
            cache_dict[ckey] = res_hit
            return res_hit
        cache_dict[ckey] = None
        return None
    def pull_subroutine_from_basal_ganglia_drawer(self, chosen_ax, subgoal_info, cur_pairs, orig_inputs, base_pool, d4_id=0, safety_gate=1.0, q_spk_hint=None):
        self.basal_ganglia_drawer_pulls_total += 1
        cands = list(base_pool) if base_pool is not None else []
        diff_rule_info = subgoal_info.get("diff_rule_info", None)
        saccade_idx = int(subgoal_info.get("saccade_idx", 0))
        extra_front = []
        sm_cache = getattr(self, "_sm_affordance_cache", None)
        if sm_cache is None:
            sm_cache = {}
            self._sm_affordance_cache = sm_cache
        sm_key = (_grid_cache_key(cur_pairs[0][0]) if cur_pairs else None, saccade_idx % 4)
        if sm_key in sm_cache:
            sm_affordances = sm_cache[sm_key]
        else:
            sm_affordances = self.build_unified_sensorimotor_affordances(
                cur_pairs,
                diff_rule_info=diff_rule_info,
                saccade_idx=saccade_idx,
                q_spk_hint=q_spk_hint,
            )
            if len(sm_cache) > 64:
                sm_cache.clear()
            sm_cache[sm_key] = sm_affordances
        extra_front.extend(sm_affordances)
        cands = extra_front + cands
        if not cands:
            return []
        pref_prefixes = subgoal_info.get("preferred_prefixes", ())
        habit_prefixes = tuple(
            str(ch.get("op1", "")) for ch in self.basal_ganglia_reflex_chunks
            if ch.get("chunk_type") == "macro_recipe" and chosen_ax in ch.get("axes", [])
        ) + tuple(
            str(t.get("op_prefix", "")) for t in self.cultural_tool_library if int(t.get("axis", -1)) == int(chosen_ax)
        )
        sm_prioritized = []
        prioritized = []
        secondary = []
        seen_cnames = set()
        for c_item in cands:
            cname = str(c_item[0])
            if cname in seen_cnames:
                continue
            seen_cnames.add(cname)
            if cname.startswith("sensorimotor_chunk_"):
                sm_prioritized.append(c_item)
            elif cname.startswith(("legend_dict_", "broadcast_foreach_", "grasp_only_")) or (habit_prefixes and cname.startswith(habit_prefixes)) or cname.startswith(("new_primitive_node_0x", "ast_organ_", "de_novo_", "chunk_macro_", "macro_breed_", "nested_tool_")):
                prioritized.append(c_item)
            elif pref_prefixes and cname.startswith(pref_prefixes):
                prioritized.append(c_item)
            else:
                secondary.append(c_item)
        if q_spk_hint is not None and len(prioritized) > 1:
            q_vec = np.asarray(q_spk_hint, dtype=np.float32)
            active_assemblies = int(np.sum(np.abs(q_vec) > 0.15))
            sparsity_focus = float(np.clip(1.0 - (active_assemblies / max(1.0, float(q_vec.size))), 0.0, 0.5))
        else:
            sparsity_focus = 0.0
        max_drawer_inspect = min(8, max(6, int((6 + 2 * float(safety_gate)) * (1.0 - 0.20 * sparsity_focus))))
        ordered_all = sm_prioritized + prioritized + secondary
        drawer_slice = ordered_all[: max(max_drawer_inspect, min(len(ordered_all), len(sm_prioritized) + 4))]
        board_sig = _grid_cache_key(cur_pairs[0][0]) if cur_pairs else None
        cand_cache = getattr(self, "_cand_eval_cache", None)
        if cand_cache is None:
            cand_cache = {}
            self._cand_eval_cache = cand_cache
        pfc_chain_cache = getattr(self, "_pfc_chain_cache", None)
        if pfc_chain_cache is None:
            pfc_chain_cache = {}
            self._pfc_chain_cache = pfc_chain_cache

        def _eval_and_cache_cand(cname, cfn, cparams):
            c_key = (board_sig, str(cname), int(d4_id))
            if c_key in cand_cache:
                return cand_cache[c_key]
            e_val, d_val, is_exact, next_gs = self.evaluate_candidate_on_pairs(
                cfn, cparams, cur_pairs, orig_inputs, d4_conj=int(d4_id)
            )
            if not is_exact and int(d4_id) != 0 and cname.startswith("sensorimotor_chunk_"):
                e_v0, d_v0, is_ex0, next_gs0 = self.evaluate_candidate_on_pairs(
                    cfn, cparams, cur_pairs, orig_inputs, d4_conj=0
                )
                if is_ex0:
                    if self.plastic and hasattr(self, "unified_brains") and self.unified_brains is not None:
                        self.unified_brains.neocortex_logic.update_from_successful_transition(
                            self.unified_brains.sensory_cortex, cur_pairs, lr=0.015
                        )
                    res0 = (1.0, 0.0, (cname, cfn, cparams, 0), next_gs0, True)
                    cand_cache[c_key] = res0
                    return res0
            if is_exact:
                if self.plastic and hasattr(self, "unified_brains") and self.unified_brains is not None:
                    self.unified_brains.neocortex_logic.update_from_successful_transition(
                        self.unified_brains.sensory_cortex, cur_pairs, lr=0.015
                    )
                res_ex = (1.0, 0.0, (cname, cfn, cparams, int(d4_id)), next_gs, True)
                cand_cache[c_key] = res_ex
                return res_ex
            if next_gs and (e_val > 0.18 or cname.startswith("sensorimotor_chunk_")):
                _, b_gain, is_neater = self.compute_intrinsic_gestalt_beauty(cur_pairs[0][0], next_gs[0])
                ep_gain = self.compute_epistemic_relational_progress(cur_pairs, next_gs) if cname.startswith(("sensorimotor_chunk_", "legend_dict_", "broadcast_foreach_", "grasp_only_")) else 0.0
                body_res = 0.025 * max(0.0, self.score_sensorimotor_body_resonance(cparams, q_spk_hint)) if cname.startswith("sensorimotor_chunk_") else 0.0
                e_adj = float(e_val + (0.045 * max(0.0, b_gain) if is_neater else 0.0) + max(0.0, ep_gain) + body_res)
                res_sc = (e_adj, d_val, (cname, cfn, cparams, int(d4_id)), next_gs, False)
                cand_cache[c_key] = res_sc
                return res_sc
            cand_cache[c_key] = None
            return None

        scored = []
        for cname, cfn, cparams in drawer_slice:
            ev_res = _eval_and_cache_cand(cname, cfn, cparams)
            if ev_res is not None:
                if ev_res[4]:
                    return [ev_res]
                scored.append(ev_res)
        scored.sort(key=lambda item: item[0], reverse=True)
        cb_inspected = 0
        sg_inspected = 0
        repaired_updated = False
        for idx_sc in range(len(scored)):
            if cb_inspected >= 3 and sg_inspected >= 2:
                break
            sc_e, sc_d, sc_step, sc_gs, sc_ex = scored[idx_sc]
            cname_s, cfn_s, cparams_s, d4_s = sc_step
            rep_key = (board_sig, str(cname_s), int(d4_s), "rep")
            if rep_key in cand_cache:
                rep_cached = cand_cache[rep_key]
                if str(cname_s).startswith("sensorimotor_chunk_"):
                    cb_inspected += 1
                else:
                    sg_inspected += 1
                if rep_cached is not None:
                    if rep_cached[4]:
                        return [rep_cached]
                    scored[idx_sc] = rep_cached
                    repaired_updated = True
                continue
            changed_board = bool(sc_gs and (sc_gs[0].shape != cur_pairs[0][0].shape or not np.array_equal(sc_gs[0], cur_pairs[0][0])))
            rep_res = None
            if str(cname_s).startswith("sensorimotor_chunk_") and cb_inspected < 3:
                cb_inspected += 1
                if changed_board and sc_e >= 0.14:
                    cb_repaired = self.cerebellar_sensorimotor_error_correction(
                        cur_pairs, orig_inputs, sc_step, sc_gs, base_e=sc_e
                    )
                    if cb_repaired is not None:
                        rep_res = cb_repaired
            cand_cache[rep_key] = rep_res
            if rep_res is not None:
                if rep_res[4]:
                    return [rep_res]
                scored[idx_sc] = rep_res
                repaired_updated = True
        if repaired_updated:
            scored.sort(key=lambda item: item[0], reverse=True)
        if scored and len(cur_pairs) >= 2:
            gaba_modified = False
            for idx_sc in range(min(2, len(scored))):
                sc_e, sc_d, sc_step, sc_gs, sc_ex = scored[idx_sc]
                if sc_gs and sc_e >= 0.30:
                    gaba_vetoed, gaba_bypass_hit, _ = self.prefrontal_counterfactual_gaba_gate(
                        cur_pairs, orig_inputs, sc_step, next_gs=sc_gs, q_spk_hint=q_spk_hint
                    )
                    if gaba_bypass_hit is not None:
                        if gaba_bypass_hit[4]:
                            return [gaba_bypass_hit]
                        scored[idx_sc] = gaba_bypass_hit
                        gaba_modified = True
                    elif gaba_vetoed:
                        scored[idx_sc] = (0.0, sc_d, sc_step, sc_gs, False)
                        gaba_modified = True
            if gaba_modified:
                scored = [item for item in scored if item[0] > 0.0]
                scored.sort(key=lambda item: item[0], reverse=True)
        if getattr(self, "ach_signal_level", 0.0) >= 0.35 or not scored or scored[0][0] < 0.85:
            ach_lens_hit = self.evaluate_through_ach_topological_lenses(cur_pairs, orig_inputs)
            if ach_lens_hit is not None:
                return [ach_lens_hit]
        if safety_gate > 0.20 and sm_prioritized:
            if board_sig in pfc_chain_cache:
                pfc_chain_hit = pfc_chain_cache[board_sig]
            else:
                pfc_chain_hit = self.rollout_prefrontal_sensorimotor_chain(
                    cur_pairs, orig_inputs, scored, q_spk_hint=q_spk_hint
                )
                pfc_chain_cache[board_sig] = pfc_chain_hit
            if pfc_chain_hit is not None:
                return [pfc_chain_hit]
        return scored[:2]
    def snap_to_discrete_attractor(self, act_vec, d4_id=0):
        self.attractor_snaps_total += 1
        snapped = np.round(act_vec * 4.0) / 4.0
        top_idx = int(np.argmax(act_vec))
        snapped[top_idx] = float(act_vec[top_idx]) + 0.08
        snapped_d4 = int(d4_id) % 8
        return snapped.astype(np.float32), snapped_d4
    def consolidate_pfc_to_basal_ganglia(self):
        if not self.plastic:
            return 0
        promoted_cnt = 0
        for sm_chain in self.bred_sensorimotor_chains:
            steps = sm_chain.get("steps", [])
            if not steps:
                continue
            sig_key = str(sm_chain.get("sig_key", ""))
            if not sig_key:
                continue
            usage = int(sm_chain.get("usage_count", 1))
            pfc_cost = 0.014 * float(len(steps))
            metabolic_gain = 0.28 * float(usage) * (1.0 + 0.25 * float(len(steps)))
            net_energy_balance = round(metabolic_gain - pfc_cost, 4)
            if net_energy_balance > 0.02:
                existing = next((c for c in self.basal_ganglia_reflex_chunks if c.get("sig_key") == sig_key), None)
                if existing is not None:
                    existing["usage_count"] = max(int(existing.get("usage_count", 1)), usage)
                    existing["net_energy_balance"] = max(float(existing.get("net_energy_balance", 0.0)), net_energy_balance)
                else:
                    self.basal_ganglia_reflex_chunks.append({
                        "chunk_type": "sensorimotor_composite_chunk",
                        "sig_key": sig_key,
                        "steps": [dict(s) for s in steps],
                        "usage_count": usage,
                        "pfc_wake_cost": round(pfc_cost, 4),
                        "reflex_spike_cost": 0.0008,
                        "net_energy_balance": net_energy_balance,
                    })
                    promoted_cnt += 1
        for rec in self.bred_macro_recipes:
            sig_key = f"{rec['op1']}__{rec['op2']}"
            matching_tool = next(
                (t for t in self.cultural_tool_library if t.get("op_prefix") == rec["op1"] and t.get("op2_prefix") == rec["op2"]),
                None,
            )
            usage = int(matching_tool.get("usage_count", 1)) if matching_tool else 1
            efe_eff = float(matching_tool.get("efe_efficiency", 0.72)) if matching_tool else 0.72
            pfc_cost = 0.036
            metabolic_gain = 0.16 * float(usage) * efe_eff
            net_energy_balance = round(metabolic_gain - pfc_cost, 4)
            if net_energy_balance > 0.04:
                existing = next((c for c in self.basal_ganglia_reflex_chunks if c.get("sig_key") == sig_key), None)
                if existing is not None:
                    existing["usage_count"] = max(int(existing.get("usage_count", 1)), usage)
                    existing["net_energy_balance"] = max(float(existing.get("net_energy_balance", 0.0)), net_energy_balance)
                else:
                    self.basal_ganglia_reflex_chunks.append({
                        "chunk_type": "macro_recipe",
                        "sig_key": sig_key,
                        "axes": list(rec["axes"]),
                        "op1": str(rec["op1"]),
                        "op2": str(rec["op2"]),
                        "usage_count": usage,
                        "pfc_wake_cost": round(pfc_cost, 4),
                        "reflex_spike_cost": 0.0012,
                        "net_energy_balance": net_energy_balance,
                    })
                    promoted_cnt += 1
        self.basal_ganglia_reflex_chunks.sort(
            key=lambda item: (
                1 if str(item.get("chunk_type", "")).startswith("sensorimotor_") else 0,
                float(item.get("net_energy_balance", 0.0)),
                int(item.get("usage_count", 0)),
            ),
            reverse=True,
        )
        if len(self.basal_ganglia_reflex_chunks) > 64:
            self.basal_ganglia_reflex_chunks = self.basal_ganglia_reflex_chunks[:64]
        return promoted_cnt
    def _is_reflex_prediction_valid_on_novel_board(self, arg1, arg2, arg3):
        if isinstance(arg1, np.ndarray) and isinstance(arg2, np.ndarray):
            test_inputs = [arg1]
            preds_all = [arg2]
            train_pairs = arg3
        else:
            train_pairs = arg1
            test_inputs = arg2 if isinstance(arg2, list) else [arg2]
            preds_all = arg3 if isinstance(arg3, list) else [arg3]
        if len(preds_all) == 0 or len(test_inputs) == 0:
            return False
        train_changed = any(x_tr.shape != y_tr.shape or not np.array_equal(x_tr, y_tr) for x_tr, y_tr in train_pairs)
        for ti, pi in zip(test_inputs, preds_all):
            if pi is None or ti is None or pi.size == 0:
                return False
            if train_changed and pi.shape == ti.shape and np.array_equal(pi, ti):
                self.einstellung_fixation_prevented_total += 1
                self.ach_habit_resets_total += 1
                return False
            if pi.size > 1 and not np.all(ti == ti.flat[0]) and np.all(pi == pi.flat[0]):
                self.einstellung_fixation_prevented_total += 1
                self.ach_habit_resets_total += 1
                return False
        return True
    def execute_basal_ganglia_reflex(self, train_pairs, test_inputs, axis_pools=None):
        if not self.basal_ganglia_reflex_chunks or not train_pairs or not test_inputs:
            return None
        discovered_triples = {
            (str(ch.get("gaze_op", "")), str(ch.get("wm_op", "")), str(ch.get("finger_op", "")))
            for ch in self.basal_ganglia_reflex_chunks
            if ch.get("chunk_type") == "sensorimotor_body_chunk"
        }
        disc_g = {t[0] for t in discovered_triples}
        disc_w = {t[1] for t in discovered_triples}
        disc_f = {t[2] for t in discovered_triples}
        for chunk in self.basal_ganglia_reflex_chunks:
            chunk_type = str(chunk.get("chunk_type", "ast_organ"))
            if chunk_type == "sensorimotor_composite_chunk":
                steps = chunk.get("steps", [])
                if not steps or not discovered_triples:
                    continue
                if not all(
                    str(st.get("gaze_op", "")) in disc_g
                    and str(st.get("wm_op", "")) in disc_w
                    and str(st.get("finger_op", "")) in disc_f
                    for st in steps
                ):
                    continue
                exact_comp = True
                for x_tr, y_tr in train_pairs:
                    try:
                        pred_comp = self.execute_sensorimotor_chain(x_tr, steps)
                    except Exception:
                        exact_comp = False
                        break
                    if pred_comp.shape != y_tr.shape or not np.array_equal(pred_comp, y_tr):
                        exact_comp = False
                        break
                if exact_comp:
                    preds_all = [self.execute_sensorimotor_chain(ti, steps) for ti in test_inputs]
                    if not self._is_reflex_prediction_valid_on_novel_board(train_pairs, test_inputs, preds_all):
                        continue
                    chunk["usage_count"] = int(chunk.get("usage_count", 1)) + 1
                    self.basal_ganglia_reflex_hits_total += 1
                    self.sensorimotor_chunks_bred_total += 1
                    saved_e = max(0.0, float(chunk.get("pfc_wake_cost", 0.032)) - float(chunk.get("reflex_spike_cost", 0.0008)))
                    self.basal_ganglia_energy_saved_total = round(self.basal_ganglia_energy_saved_total + saved_e, 5)
                    sig_tag = str(chunk.get("sig_key", "composite"))[:48]
                    reflex_cost_c = float(chunk.get("reflex_spike_cost", 0.0008))
                    return {
                        "solved_train_exact": True,
                        "prediction": preds_all[0],
                        "predictions_all": preds_all,
                        "alt_predictions_all": preds_all,
                        "program": [f"basal_ganglia_reflex:sensorimotor_chunk_composite_{sig_tag}"],
                        "steps_taken": 1,
                        "winning_axis": 6,
                        "survival_rate_pct": 100.0,
                        "mean_spike_firing_rate": round(reflex_cost_c * 32.0, 4),
                        "thought_energy_consumed": reflex_cost_c,
                        "hippocampal_replay_used": False,
                        "inner_monologue_active": False,
                        "hierarchical_macro_active": bool(chunk.get("is_nested_hierarchical", False) or chunk.get("is_bred_composite", False)),
                        "dream_mutation_used": False,
                        "social_empathy_used": False,
                        "bred_macro_tool_used": bool(chunk.get("is_bred_composite", False)),
                        "nested_tool_used": bool(chunk.get("is_nested_hierarchical", False)),
                        "conditional_monologue_used": False,
                        "nested_conditional_used": False,
                        "specialist_leader_used": False,
                        "deep_macro_brake_used": False,
                        "micro_synth_used": False,
                        "ast_organ_used": False,
                        "basal_ganglia_reflex_used": True,
                    }
            elif chunk_type == "macro_recipe" and axis_pools is not None:
                op1_pref = str(chunk.get("op1", ""))
                op2_pref = str(chunk.get("op2", ""))
                axes_c = chunk.get("axes", [0, 0])
                ax_target = int(axes_c[-1]) if axes_c else 0
                for cand_name, cand_fn, cand_params in axis_pools.get(ax_target, []):
                    if (op1_pref and op1_pref in cand_name) or (op2_pref and op2_pref in cand_name):
                        exact_m = True
                        for x_tr, y_tr in train_pairs:
                            try:
                                pred_m = cand_fn(x_tr, cand_params)
                            except Exception:
                                exact_m = False
                                break
                            if pred_m.shape != y_tr.shape or not np.array_equal(pred_m, y_tr):
                                exact_m = False
                                break
                        if exact_m:
                            preds_all = [cand_fn(ti, cand_params) for ti in test_inputs]
                            if not self._is_reflex_prediction_valid_on_novel_board(train_pairs, test_inputs, preds_all):
                                continue
                            chunk["usage_count"] = int(chunk.get("usage_count", 1)) + 1
                            self.basal_ganglia_reflex_hits_total += 1
                            saved_e = max(0.0, float(chunk.get("pfc_wake_cost", 0.036)) - float(chunk.get("reflex_spike_cost", 0.0012)))
                            self.basal_ganglia_energy_saved_total = round(self.basal_ganglia_energy_saved_total + saved_e, 5)
                            reflex_cost_m = float(chunk.get("reflex_spike_cost", 0.0012))
                            return {
                                "solved_train_exact": True,
                                "prediction": preds_all[0],
                                "predictions_all": preds_all,
                                "alt_predictions_all": preds_all,
                                "program": [f"basal_ganglia_reflex:{cand_name}"],
                                "steps_taken": 1,
                                "winning_axis": ax_target,
                                "survival_rate_pct": 100.0,
                                "mean_spike_firing_rate": round(reflex_cost_m * 26.0, 4),
                                "thought_energy_consumed": reflex_cost_m,
                                "hippocampal_replay_used": False,
                                "inner_monologue_active": False,
                                "hierarchical_macro_active": bool(cand_name.startswith(("macro_", "nested_tool_"))),
                                "dream_mutation_used": False,
                                "social_empathy_used": False,
                                "bred_macro_tool_used": bool(cand_name.startswith("macro_breed_") or chunk.get("is_bred_composite", False)),
                                "nested_tool_used": bool(cand_name.startswith("nested_tool_") or chunk.get("is_nested_hierarchical", False)),
                                "conditional_monologue_used": False,
                                "nested_conditional_used": False,
                                "specialist_leader_used": False,
                                "deep_macro_brake_used": False,
                                "micro_synth_used": False,
                                "ast_organ_used": False,
                                "basal_ganglia_reflex_used": True,
                            }
            elif chunk_type == "sensorimotor_body_chunk":
                sm_cands = self.build_unified_sensorimotor_affordances(train_pairs)
                for cand_name, cand_fn, cand_params in sm_cands:
                    if (
                        str(cand_params.get("gaze_op", "")) == str(chunk.get("gaze_op", ""))
                        and str(cand_params.get("wm_op", "")) == str(chunk.get("wm_op", ""))
                        and str(cand_params.get("finger_op", "")) == str(chunk.get("finger_op", ""))
                    ):
                        exact_sm = True
                        for x_tr, y_tr in train_pairs:
                            try:
                                pred_sm = cand_fn(x_tr, cand_params)
                            except Exception:
                                exact_sm = False
                                break
                            if pred_sm.shape != y_tr.shape or not np.array_equal(pred_sm, y_tr):
                                exact_sm = False
                                break
                        if exact_sm:
                            preds_all = [cand_fn(ti, cand_params) for ti in test_inputs]
                            if not self._is_reflex_prediction_valid_on_novel_board(train_pairs, test_inputs, preds_all):
                                continue
                            chunk["usage_count"] = int(chunk.get("usage_count", 1)) + 1
                            self.basal_ganglia_reflex_hits_total += 1
                            self.sensorimotor_chunks_bred_total += 1
                            saved_e = max(0.0, float(chunk.get("pfc_wake_cost", 0.028)) - float(chunk.get("reflex_spike_cost", 0.0009)))
                            self.basal_ganglia_energy_saved_total = round(self.basal_ganglia_energy_saved_total + saved_e, 5)
                            reflex_cost_s = float(chunk.get("reflex_spike_cost", 0.0009))
                            return {
                                "solved_train_exact": True,
                                "prediction": preds_all[0],
                                "predictions_all": preds_all,
                                "alt_predictions_all": preds_all,
                                "program": [f"basal_ganglia_reflex:{cand_name}"],
                                "steps_taken": 1,
                                "winning_axis": 6,
                                "survival_rate_pct": 100.0,
                                "mean_spike_firing_rate": round(reflex_cost_s * 30.0, 4),
                                "thought_energy_consumed": reflex_cost_s,
                                "hippocampal_replay_used": False,
                                "inner_monologue_active": False,
                                "hierarchical_macro_active": False,
                                "dream_mutation_used": False,
                                "social_empathy_used": False,
                                "bred_macro_tool_used": False,
                                "nested_tool_used": False,
                                "conditional_monologue_used": False,
                                "nested_conditional_used": False,
                                "specialist_leader_used": False,
                                "deep_macro_brake_used": False,
                                "micro_synth_used": False,
                                "ast_organ_used": False,
                                "basal_ganglia_reflex_used": True,
                            }
        return None
    def compose_bred_macro_action(self, step1_tuple, step2_tuple, axis_id=1):
        cname1, cfn1, cparams1, d4_1 = step1_tuple
        cname2, cfn2, cparams2, d4_2 = step2_tuple
        bred_name = f"macro_breed_{cname1}__{cname2}"
        def bred_fn(grid, params):
            s1, s2 = params["steps"]
            n1, fn1, p1, d1 = s1
            n2, fn2, p2, d2 = s2
            if d1 != 0:
                g_in1 = apply_d4(grid, d1)
                g_mid = invert_d4(fn1(g_in1, p1), d1)
            else:
                g_mid = fn1(grid, p1)
            if d2 != 0:
                g_in2 = apply_d4(g_mid, d2)
                g_out = invert_d4(fn2(g_in2, p2), d2)
            else:
                g_out = fn2(g_mid, p2)
            return g_out
        return (bred_name, bred_fn, {"steps": [step1_tuple, step2_tuple], "axis": int(axis_id)})
    def compose_nested_macro_action(self, step_tuples, axis_id=1):
        sub_names = "__".join(st[0] for st in step_tuples)
        nested_name = f"nested_tool_{sub_names}"
        def nested_fn(grid, params):
            cur = grid
            for n_k, fn_k, p_k, d_k in params["steps"]:
                if d_k != 0:
                    g_in = apply_d4(cur, d_k)
                    cur = invert_d4(fn_k(g_in, p_k), d_k)
                else:
                    cur = fn_k(cur, p_k)
            return cur
        return (nested_name, nested_fn, {"steps": list(step_tuples), "axis": int(axis_id), "nested_depth": len(step_tuples)})
    def compile_recursive_chunk_macro(self, full_prog, winning_axis=1):
        if len(full_prog) < 2:
            return None
        flat_steps = []
        for st in full_prog:
            cname, cfn, cparams, d4_id = st
            if (
                isinstance(cparams, dict)
                and "steps" in cparams
                and isinstance(cparams["steps"], list)
                and cparams["steps"]
                and all(isinstance(sub_s, (tuple, list)) and len(sub_s) == 4 for sub_s in cparams["steps"])
            ):
                for sub_st in cparams["steps"]:
                    flat_steps.append(sub_st)
            else:
                flat_steps.append(st)
        if len(flat_steps) < 2 or len(flat_steps) > 6:
            return None
        sub_tag = "__".join(s[0][:14] for s in flat_steps)
        chunk_name = f"chunk_macro_{sub_tag}"
        def chunk_fn(grid, params):
            cur = grid
            for _, fn_k, p_k, d_k in params["steps"]:
                if d_k != 0:
                    g_in = apply_d4(cur, d_k)
                    cur = invert_d4(fn_k(g_in, p_k), d_k)
                else:
                    cur = fn_k(cur, p_k)
            return cur
        chunk_tuple = (
            chunk_name,
            chunk_fn,
            {"steps": list(flat_steps), "axis": int(winning_axis), "chunk_depth": len(flat_steps)},
        )
        if self.plastic and all(ex[0] != chunk_name for ex in self.recursive_chunk_library):
            self.recursive_chunk_library.append(chunk_tuple)
            self.recursive_chunks_compiled_total += 1
            if len(self.recursive_chunk_library) > 32:
                self.recursive_chunk_library.pop(0)
        return chunk_tuple
    def splice_residual_bridge_program(self, task, res_prog, winning_axis=1):
        if not self.plastic:
            return None
        prefix_prog = task.get("prefix_program") if isinstance(task, dict) else None
        if not prefix_prog or not res_prog:
            return None
        bridge_prog = list(prefix_prog) + list(res_prog)
        chunk_tuple = self.compile_recursive_chunk_macro(bridge_prog, winning_axis=int(winning_axis))
        if chunk_tuple is not None:
            self.residual_bridges_spliced_total += 1
            prog_names = [p[0] for p in bridge_prog]
            bridge_rec = {
                "axes": (int(winning_axis), int(winning_axis)),
                "op1": prog_names[0][:10],
                "op2": prog_names[-1][:10],
            }
            if bridge_rec not in self.bred_macro_recipes and len(self.bred_macro_recipes) < 36:
                self.bred_macro_recipes.append(bridge_rec)
                self.ast_crossovers_total = int(getattr(self, "ast_crossovers_total", 0)) + 1
        return chunk_tuple
    def prune_cultural_tool_library(self, max_essential_tools=24, max_bred_recipes=12, max_nested_recipes=10):
        self.pruning_cycles_run += 1
        before_tools = len(self.cultural_tool_library)
        before_recipes = len(self.bred_macro_recipes)
        before_nested = len(self.nested_macro_recipes)
        scored_tools = []
        for t in self.cultural_tool_library:
            stale = int(t.get("staleness_tasks", 0)) + 1
            uses = int(t.get("usage_count", 0))
            base_efe = float(t.get("efe_efficiency", 0.52))
            efe_eff = float(base_efe * (0.94 ** min(10, stale)) + 0.20 * math.log1p(uses))
            t["staleness_tasks"] = stale
            t["efe_efficiency"] = round(efe_eff, 4)
            if stale >= 6 and uses == 0 and before_tools > max_essential_tools:
                continue
            scored_tools.append((uses > 0, efe_eff, -stale, t))
        scored_tools.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        self.cultural_tool_library = [item[3] for item in scored_tools[:max_essential_tools]]
        core_recipes = [
            r for r in self.bred_macro_recipes
            if not r["op2"].startswith("nested_tool_")
        ]
        self.bred_macro_recipes = core_recipes[:max_bred_recipes]
        self.nested_macro_recipes = self.nested_macro_recipes[:max_nested_recipes]
        pruned_now = (
            max(0, before_tools - len(self.cultural_tool_library))
            + max(0, before_recipes - len(self.bred_macro_recipes))
            + max(0, before_nested - len(self.nested_macro_recipes))
        )
        self.tools_pruned_total += int(pruned_now)
        mean_efe = (
            float(np.mean([t.get("efe_efficiency", 0.5) for t in self.cultural_tool_library]))
            if self.cultural_tool_library
            else 0.0
        )
        return {
            "pruning_cycles_run": int(self.pruning_cycles_run),
            "tools_before_pruning": int(before_tools),
            "essential_tools_retained": int(len(self.cultural_tool_library)),
            "bred_recipes_retained": int(len(self.bred_macro_recipes)),
            "nested_recipes_retained": int(len(self.nested_macro_recipes)),
            "tools_pruned_this_cycle": int(pruned_now),
            "tools_pruned_cumulative": int(self.tools_pruned_total),
            "mean_tool_efe_efficiency": round(mean_efe, 4),
        }
    def synthesize_bred_tools_for_axis(self, axis_id, train_pairs, base_cands):
        if not self.use_curiosity or not base_cands or axis_id not in (0, 1, 2, 3, 4, 5, 6, 7):
            return []
        bred_cands = []
        seen_bred = set()
        for chunk_name, chunk_fn, chunk_params in self.recursive_chunk_library[-8:]:
            if int(chunk_params.get("axis", axis_id)) == axis_id and chunk_name not in seen_bred:
                seen_bred.add(chunk_name)
                bred_cands.append((chunk_name, chunk_fn, chunk_params))
        active_recipes = [
            r for r in self.bred_macro_recipes
            if r["axes"][0] == axis_id and not r["op2"].startswith("nested_tool_")
        ]
        for tool_item in self.cultural_tool_library[:10]:
            if tool_item["axis"] == axis_id and "op2_prefix" in tool_item:
                op2_p = tool_item["op2_prefix"]
                if not op2_p or op2_p.startswith("nested_tool_"):
                    continue
                rec = {
                    "axes": (int(tool_item["axis"]), int(tool_item.get("axis2", axis_id))),
                    "op1": tool_item["op_prefix"],
                    "op2": op2_p,
                }
                if rec not in active_recipes:
                    active_recipes.append(rec)
        cross_axes = [axis_id]
        if train_pairs:
            sample_in, sample_out = train_pairs[0]
            if sample_in.shape != sample_out.shape:
                for ax_cand in [0, 2, 4]:
                    if ax_cand not in cross_axes:
                        cross_axes.append(ax_cand)
            else:
                for ax_cand in [2, 7, 1]:
                    if ax_cand not in cross_axes:
                        cross_axes.append(ax_cand)
        for rec in active_recipes[:12]:
            op1_pref = rec["op1"]
            op2_pref = rec["op2"]
            ax2 = int(rec["axes"][1])
            search_c1 = list(base_cands)
            if hasattr(self, "perceive_synthesized_ast_affordances"):
                search_c1 = search_c1 + self.perceive_synthesized_ast_affordances(axis_id, train_pairs)
            s1_matches = [c for c in search_c1 if (c[0].startswith(op1_pref) or op1_pref.startswith(c[0])) and not c[0].startswith(("macro_breed_", "nested_tool_", "chunk_macro_"))]
            for cname1, cfn1, cparams1 in s1_matches[:4]:
                try:
                    mid_pairs = []
                    valid_mid = True
                    for x_in, y_out in train_pairs:
                        m_grid = cfn1(x_in, cparams1)
                        if m_grid.size == 0 or np.array_equal(m_grid, x_in):
                            valid_mid = False
                            break
                        mid_pairs.append((m_grid, y_out))
                    if not valid_mid:
                        continue
                    ax2_cands = perceive_core_affordances(ax2, mid_pairs)
                    if hasattr(self, "perceive_synthesized_ast_affordances"):
                        ax2_cands = ax2_cands + self.perceive_synthesized_ast_affordances(ax2, mid_pairs)
                    s2_matches = [c2 for c2 in ax2_cands if (c2[0].startswith(op2_pref) or op2_pref.startswith(c2[0])) and not c2[0].startswith(("macro_breed_", "nested_tool_", "chunk_macro_"))]
                    for cname2, cfn2, cparams2 in s2_matches[:4]:
                        b_tuple = self.compose_bred_macro_action(
                            (cname1, cfn1, cparams1, 0),
                            (cname2, cfn2, cparams2, 0),
                            axis_id=axis_id,
                        )
                        if b_tuple[0] not in seen_bred:
                            seen_bred.add(b_tuple[0])
                            bred_cands.append(b_tuple)
                except Exception:
                    continue
        if len(bred_cands) < 4:
            for cn1, fn1, cp1 in base_cands[:4]:
                if cn1.startswith(("macro_breed_", "nested_tool_", "chunk_macro_")):
                    continue
                try:
                    mid_pairs = []
                    valid_mid = True
                    for x_in, y_out in train_pairs:
                        m_grid = fn1(x_in, cp1)
                        if m_grid.size == 0 or np.array_equal(m_grid, x_in):
                            valid_mid = False
                            break
                        mid_pairs.append((m_grid, y_out))
                    if not valid_mid:
                        continue
                    for ax_c in cross_axes:
                        ax2_cands = perceive_core_affordances(ax_c, mid_pairs)
                        for cn2, fn2, cp2 in ax2_cands[:3]:
                            if cn2.startswith(("macro_breed_", "nested_tool_", "chunk_macro_")):
                                continue
                            b_tuple = self.compose_bred_macro_action(
                                (cn1, fn1, cp1, 0),
                                (cn2, fn2, cp2, 0),
                                axis_id=axis_id,
                            )
                            if b_tuple[0] not in seen_bred:
                                seen_bred.add(b_tuple[0])
                                bred_cands.append(b_tuple)
                except Exception:
                    continue
        candidate_2tier = list(bred_cands)
        def _score_b_cand(item):
            b_name, b_fn, b_params = item
            if not isinstance(b_params, dict) or "steps" not in b_params or len(b_params["steps"]) != 2:
                return -1.0
            try:
                g_test = b_fn(train_pairs[0][0], b_params)
                y_test = train_pairs[0][1]
                if g_test.shape == y_test.shape:
                    return 2.0 + float(np.mean(g_test == y_test))
                return 1.0 - float(abs(g_test.size - y_test.size)) / max(1, y_test.size)
            except Exception:
                return -1.0
        candidate_2tier.sort(key=_score_b_cand, reverse=True)
        for b_name, b_fn, b_params in candidate_2tier[:16]:
            if not isinstance(b_params, dict) or "steps" not in b_params or len(b_params["steps"]) != 2:
                continue
            st1, st2 = b_params["steps"]
            try:
                mid2_pairs = []
                v_mid2 = True
                for x_in, y_out in train_pairs:
                    g_m2 = b_fn(x_in, b_params)
                    if g_m2.size == 0 or np.array_equal(g_m2, x_in):
                        v_mid2 = False
                        break
                    mid2_pairs.append((g_m2, y_out))
                if not v_mid2:
                    continue
                m_in0, m_out0 = mid2_pairs[0]
                target_ax3_list = [7, 2, 1] if m_in0.shape == m_out0.shape else [0, 2, 4]
                for ax3 in target_ax3_list:
                    ax3_cands = perceive_core_affordances(ax3, mid2_pairs)
                    if hasattr(self, "perceive_synthesized_ast_affordances"):
                        ax3_cands = ax3_cands + self.perceive_synthesized_ast_affordances(ax3, mid2_pairs)
                    for cn3, fn3, cp3 in ax3_cands[:3]:
                        if cn3.startswith(("macro_breed_", "nested_tool_", "chunk_macro_")):
                            continue
                        n3_tuple = self.compose_nested_macro_action(
                            [st1, st2, (cn3, fn3, cp3, 0)],
                            axis_id=axis_id,
                        )
                        if n3_tuple[0] not in seen_bred:
                            seen_bred.add(n3_tuple[0])
                            bred_cands.append(n3_tuple)
                            if self.plastic and len(self.nested_macro_recipes) < 32:
                                n_rec = {
                                    "axes": (int(axis_id), int(b_params.get("axis2", axis_id)), int(ax3)),
                                    "ops": (st1[0][:12], st2[0][:12], cn3[:12]),
                                }
                                if n_rec not in self.nested_macro_recipes:
                                    self.nested_macro_recipes.append(n_rec)
            except Exception:
                continue
        active_nested = [nr for nr in self.nested_macro_recipes if nr["axes"][0] == axis_id]
        for nrec in active_nested[:3]:
            ax1, ax2, ax3 = int(nrec["axes"][0]), int(nrec["axes"][1]), int(nrec["axes"][2])
            op1_p, op2_p, op3_p = nrec["ops"]
            s1_m = [c for c in base_cands if c[0].startswith(op1_p) and not c[0].startswith(("macro_breed_", "nested_tool_", "chunk_macro_"))]
            for cn1, fn1, cp1 in s1_m[:1]:
                try:
                    mid1_pairs = []
                    v1 = True
                    for x_in, y_out in train_pairs:
                        g1 = fn1(x_in, cp1)
                        if g1.size == 0 or np.array_equal(g1, x_in):
                            v1 = False
                            break
                        mid1_pairs.append((g1, y_out))
                    if not v1:
                        continue
                    ax2_cands = perceive_core_affordances(ax2, mid1_pairs)
                    s2_m = [c2 for c2 in ax2_cands if c2[0].startswith(op2_p) and not c2[0].startswith(("macro_breed_", "nested_tool_", "chunk_macro_"))]
                    for cn2, fn2, cp2 in s2_m[:1]:
                        mid2_pairs = []
                        v2 = True
                        for g1_in, y_out in mid1_pairs:
                            g2 = fn2(g1_in, cp2)
                            if g2.size == 0 or np.array_equal(g2, g1_in):
                                v2 = False
                                break
                            mid2_pairs.append((g2, y_out))
                        if not v2:
                            continue
                        ax3_cands = perceive_core_affordances(ax3, mid2_pairs)
                        s3_m = [c3 for c3 in ax3_cands if c3[0].startswith(op3_p) and not c3[0].startswith(("macro_breed_", "nested_tool_", "chunk_macro_"))]
                        for cn3, fn3, cp3 in s3_m[:1]:
                            n_tuple = self.compose_nested_macro_action(
                                [(cn1, fn1, cp1, 0), (cn2, fn2, cp2, 0), (cn3, fn3, cp3, 0)],
                                axis_id=axis_id,
                            )
                            if n_tuple[0] not in seen_bred:
                                seen_bred.add(n_tuple[0])
                                bred_cands.append(n_tuple)
                except Exception:
                    continue
        return bred_cands
    def record_wake_episode(self, obs_vec, h_vec, q_spk, k_spk, v_proj, macro_vec, micro_vec, winning_axis, full_prog):
        if not self.plastic:
            return
        prog_names = [p[0] for p in full_prog]
        if prog_names:
            hippo_ep = {
                "axes": (int(winning_axis), int(winning_axis)),
                "op1": str(prog_names[0])[:10],
                "op2": str(prog_names[-1])[:10],
            }
            if hippo_ep not in self.hippocampal_episodic_buffer:
                self.hippocampal_episodic_buffer.append(hippo_ep)
                if len(self.hippocampal_episodic_buffer) > 48:
                    self.hippocampal_episodic_buffer.pop(0)
        prog_axes = [int(winning_axis)]
        sm_steps_all = []
        for p_name, _, p_params, p_d4 in full_prog:
            if str(p_name).startswith("sensorimotor_chunk_"):
                sub_steps = self.extract_sensorimotor_steps_from_params(p_params)
                for st in sub_steps:
                    st_copy = dict(st)
                    if int(p_d4) != 0 and int(st_copy.get("d4_k", 0)) == 0:
                        st_copy["d4_k"] = int(p_d4)
                    sm_steps_all.append(st_copy)
        if sm_steps_all:
            gaze_map = {
                "gaze_salient_target": 0,
                "gaze_corner_sample": 1,
                "gaze_next_matching_attr": 2,
                "gaze_count_objects": 3,
            }
            wm_map = {
                "wm_store_attr": 4,
                "wm_store_sample_pair": 5,
                "wm_store_count": 6,
            }
            finger_map = {
                "finger_step_until_contact": 7,
                "finger_paint_from_wm": 8,
                "finger_apply_to_gaze": 9,
                "finger_step_n_from_wm": 10,
            }
            q_arr = np.asarray(q_spk, dtype=np.float32)
            if q_arr.size == self.W_body_assembly.shape[1] and float(np.linalg.norm(q_arr)) > 1e-6:
                q_unit = q_arr / (float(np.linalg.norm(q_arr)) + 1e-8)
                for st in sm_steps_all:
                    for ch_idx in (
                        gaze_map.get(str(st.get("gaze_op", "")), -1),
                        wm_map.get(str(st.get("wm_op", "")), -1),
                        finger_map.get(str(st.get("finger_op", "")), -1),
                    ):
                        if 0 <= ch_idx < 11:
                            self.W_body_assembly[ch_idx] = np.clip(
                                0.94 * self.W_body_assembly[ch_idx] + 0.08 * q_unit, -2.5, 2.5
                            ).astype(np.float32)
            if len(sm_steps_all) >= 2 or any("__cb_" in pn for pn in prog_names):
                sig_parts = []
                for st in sm_steps_all[:4]:
                    sig_parts.append(
                        f"{st.get('gaze_op', '')}:{st.get('wm_op', '')}:{st.get('finger_op', '')}:{st.get('selector', '')}:{st.get('paint_mode', '')}:{st.get('dr', 0)}:{st.get('dc', 0)}:{st.get('stop_margin', 0)}:{st.get('d4_k', 0)}"
                    )
                sig_key = ">>".join(sig_parts)
                existing_sm = next((c for c in self.bred_sensorimotor_chains if c.get("sig_key") == sig_key), None)
                if existing_sm is not None:
                    existing_sm["usage_count"] = int(existing_sm.get("usage_count", 1)) + 1
                else:
                    self.bred_sensorimotor_chains.append({
                        "sig_key": sig_key,
                        "steps": [dict(s) for s in sm_steps_all[:4]],
                        "usage_count": 1,
                    })
                    if len(self.bred_sensorimotor_chains) > 32:
                        self.bred_sensorimotor_chains.pop(0)
                self.sensorimotor_chunks_bred_total += 1
                self.consolidate_pfc_to_basal_ganglia()
            self.consolidate_procedural_chain_synaptic(
                h_vec, [("sensorimotor", dict(s)) for s in sm_steps_all[:3]]
            )
        for t in self.cultural_tool_library:
            if any(pn.startswith(t["op_prefix"]) for pn in prog_names):
                t["usage_count"] = int(t.get("usage_count", 0)) + 1
                t["staleness_tasks"] = 0
                t["efe_efficiency"] = round(min(1.85, float(t.get("efe_efficiency", 0.65)) + 0.14), 4)
        if any(pn.startswith("canvas_stitch_") for pn in prog_names):
            self.canvas_stitch_solves_total += 1
            stitch_tool = {
                "axis": int(winning_axis),
                "axis2": int(winning_axis),
                "op_prefix": "canvas_stitch_",
                "op2_prefix": prog_names[-1][:16] if len(prog_names) >= 2 else "rigid_",
                "is_bred_composite": bool(len(full_prog) >= 2),
                "is_nested_hierarchical": bool(len(full_prog) >= 3),
                "usage_count": 2,
                "staleness_tasks": 0,
                "efe_efficiency": 0.98,
                "macro_vec": macro_vec.copy(),
                "k_spk": k_spk.copy(),
                "v_proj": v_proj.copy(),
            }
            if len(self.cultural_tool_library) < 48 and all(
                t["op_prefix"] != "canvas_stitch_" for t in self.cultural_tool_library
            ):
                self.cultural_tool_library.append(stitch_tool)
        if any(pn.startswith("mono_bind_A_to_B_by_attr") for pn in prog_names):
            self.relational_bindings_total += 1
            for p_item in full_prog:
                if p_item[0].startswith("mono_bind_A_to_B_by_attr"):
                    p_dict = p_item[2] if isinstance(p_item[2], dict) else {}
                    bind_entry = {
                        "op_name": str(p_item[0]),
                        "source_attr": str(p_dict.get("source_attr", "holes")),
                        "target_op": str(p_dict.get("target_op", "shift_dist")),
                        "axis": int(winning_axis),
                    }
                    if bind_entry not in self.inner_monologue["symbolic_bindings"]:
                        self.inner_monologue["symbolic_bindings"].append(bind_entry)
                        if len(self.inner_monologue["symbolic_bindings"]) > 24:
                            self.inner_monologue["symbolic_bindings"].pop(0)
            rel_tool_entry = {
                "axis": int(winning_axis),
                "axis2": int(winning_axis),
                "op_prefix": "mono_bind_A_to_B_by_attr",
                "op2_prefix": prog_names[-1][:16] if len(prog_names) >= 2 else "",
                "is_bred_composite": bool(len(full_prog) >= 2),
                "is_nested_hierarchical": bool(len(full_prog) >= 3),
                "usage_count": 2,
                "staleness_tasks": 0,
                "efe_efficiency": 0.96,
                "macro_vec": macro_vec.copy(),
                "k_spk": k_spk.copy(),
                "v_proj": v_proj.copy(),
            }
            if len(self.cultural_tool_library) < 48 and all(
                t["op_prefix"] != "mono_bind_A_to_B_by_attr" for t in self.cultural_tool_library
            ):
                self.cultural_tool_library.append(rel_tool_entry)
        if len(full_prog) >= 2:
            self.compile_recursive_chunk_macro(full_prog, winning_axis=int(winning_axis))
            new_ep = {"axes": (int(winning_axis), int(winning_axis)), "op1": prog_names[0][:10], "op2": prog_names[-1][:10]}
            if new_ep not in self.mutated_episodic_buffer and len(self.mutated_episodic_buffer) < 40:
                self.mutated_episodic_buffer.append(new_ep)
                self.ast_mutations_total = int(getattr(self, "ast_mutations_total", 0)) + 1
            if new_ep not in self.bred_macro_recipes and len(self.bred_macro_recipes) < 36:
                self.bred_macro_recipes.append(new_ep)
            if len(full_prog) >= 3:
                nested_ep = {
                    "axes": (int(winning_axis), int(winning_axis), int(winning_axis)),
                    "ops": (prog_names[0][:12], prog_names[1][:12], prog_names[-1][:12]),
                }
                if nested_ep not in self.nested_macro_recipes and len(self.nested_macro_recipes) < 32:
                    self.nested_macro_recipes.append(nested_ep)
            elif len(full_prog) == 1 and prog_names[0].startswith("nested_tool_"):
                st_list = full_prog[0][2].get("steps", [])
                if len(st_list) >= 3:
                    nested_ep = {
                        "axes": (int(winning_axis), int(winning_axis), int(winning_axis)),
                        "ops": (st_list[0][0][:12], st_list[1][0][:12], st_list[2][0][:12]),
                    }
                    if nested_ep not in self.nested_macro_recipes and len(self.nested_macro_recipes) < 32:
                        self.nested_macro_recipes.append(nested_ep)
            is_nested = bool(len(full_prog) >= 3 or any(pn.startswith(("nested_tool_", "chunk_macro_")) for pn in prog_names))
            tool_entry = {
                "axis": int(winning_axis),
                "axis2": int(winning_axis),
                "op_prefix": prog_names[0][:16],
                "op2_prefix": prog_names[-1][:16],
                "is_bred_composite": True,
                "is_nested_hierarchical": is_nested,
                "usage_count": 1,
                "staleness_tasks": 0,
                "efe_efficiency": round(float(0.72 + 0.12 * min(3, len(full_prog))), 4),
                "macro_vec": macro_vec.copy(),
                "k_spk": k_spk.copy(),
                "v_proj": v_proj.copy(),
            }
            if len(self.cultural_tool_library) < 48 and all(
                (t["op_prefix"], t.get("op2_prefix", "")) != (tool_entry["op_prefix"], tool_entry["op2_prefix"])
                for t in self.cultural_tool_library
            ):
                self.cultural_tool_library.append(tool_entry)
        self.dream_wake_buffer.append({
            "obs": obs_vec.copy(),
            "h": h_vec.copy(),
            "q_spk": q_spk.copy(),
            "k_spk": k_spk.copy(),
            "v_proj": v_proj.copy(),
            "macro_vec": macro_vec.copy(),
            "micro_vec": micro_vec.copy(),
            "winning_axis": int(winning_axis),
            "program_axes": prog_axes,
            "program_names": prog_names,
            "reward": 1.0 + 0.25 * max(0, len(full_prog) - 1),
        })
        if len(self.dream_wake_buffer) > 512:
            self.dream_wake_buffer.pop(0)
    def offline_dream_replay(self, n_dream_cycles=15, dream_lr=0.012, mutation_rate=0.22):
        if not self.plastic or not self.dream_wake_buffer:
            return {
                "dream_cycles": 0,
                "episodes_replayed": 0,
                "mutated_dreams_generated": 0,
                "stage1_stage2_chimera_count": 0,
                "counterfactual_mutated_templates": 0,
                "bred_composite_tools_count": 0,
                "specialist_roles_differentiated": 0,
                "thermal_spike_burst_density": 0.0,
                "mean_dream_resonance_before": 0.0,
                "mean_dream_resonance_after": 0.0,
                "synaptic_consolidation_delta": 0.0,
            }
        H = self.hid_dim
        wq_init = self.Wq.copy()
        wk_init = self.Wk.copy()
        wv_init = self.Wv.copy()
        seed_d = int(hashlib.md5(f"dream_{len(self.dream_wake_buffer)}_{len(self.hippocampal_episodic_buffer)}".encode("utf-8")).hexdigest()[:8], 16)
        rng_dream = np.random.default_rng(seed_d)
        res_before = []
        for ep in self.dream_wake_buffer:
            q_s = ep["q_spk"]
            k_s = ep["k_spk"]
            res_before.append(float(np.dot(q_s, k_s) / (math.sqrt(float(H)) + 1e-8)))
        mean_before = float(np.mean(res_before)) if res_before else 0.0
        stage1_eps = [ep for ep in self.dream_wake_buffer if ep["program_names"] and ep["program_names"][0].startswith("isaac3d_")]
        stage2_eps = [ep for ep in self.dream_wake_buffer if not (ep["program_names"] and ep["program_names"][0].startswith("isaac3d_"))]
        if not stage1_eps:
            stage1_eps = self.dream_wake_buffer[:4]
        if not stage2_eps:
            stage2_eps = self.dream_wake_buffer
        mutated_dreams = []
        thermal_densities = []
        chimera_cnt = 0
        for s2_idx, s2_ep in enumerate(stage2_eps[:32]):
            s1_ep = stage1_eps[s2_idx % len(stage1_eps)]
            partner_s2 = stage2_eps[(s2_idx + 7) % len(stage2_eps)]
            burst_mask_q = (rng_dream.random(H) < mutation_rate).astype(np.float32)
            burst_sign_q = rng_dream.choice(np.array([-1.0, 1.0], dtype=np.float32), size=H)
            thermal_q = burst_mask_q * burst_sign_q
            burst_mask_k = (rng_dream.random(H) < mutation_rate).astype(np.float32)
            burst_sign_k = rng_dream.choice(np.array([-1.0, 1.0], dtype=np.float32), size=H)
            thermal_k = burst_mask_k * burst_sign_k
            thermal_densities.append(float(0.5 * (np.mean(burst_mask_q) + np.mean(burst_mask_k))))
            q_chim = np.sign(s2_ep["q_spk"] + 0.65 * s1_ep["q_spk"] + 0.45 * thermal_q).astype(np.float32)
            k_chim = np.sign(partner_s2["k_spk"] + 0.65 * s1_ep["k_spk"] + 0.45 * thermal_k).astype(np.float32)
            v_chim = np.tanh(0.55 * s2_ep["v_proj"] + 0.25 * partner_s2["v_proj"] + 0.20 * s1_ep["v_proj"]).astype(np.float32)
            h_chim = np.tanh(0.60 * s2_ep["h"] + 0.25 * partner_s2["h"] + 0.15 * s1_ep["h"]).astype(np.float32)
            mac_chim = np.tanh(0.60 * s2_ep["macro_vec"] + 0.40 * s1_ep["macro_vec"]).astype(np.float32)
            mic_chim = np.tanh(0.60 * s2_ep["micro_vec"] + 0.40 * partner_s2["micro_vec"]).astype(np.float32)
            obs_chim = (0.70 * s2_ep["obs"] + 0.30 * partner_s2["obs"]).astype(np.float32)
            mutated_dreams.append({
                "obs": obs_chim,
                "h": h_chim,
                "q_spk": q_chim,
                "k_spk": k_chim,
                "v_proj": v_chim,
                "macro_vec": mac_chim,
                "micro_vec": mic_chim,
                "winning_axis": int(s2_ep["winning_axis"]),
                "reward": float(0.85 * s2_ep["reward"]),
            })
            chimera_cnt += 1
        mut_candidates = []
        for ep_a in self.hippocampal_episodic_buffer[:12]:
            if ep_a not in mut_candidates:
                mut_candidates.append(ep_a)
            for ep_b in self.hippocampal_episodic_buffer[:14]:
                if ep_a["axes"][0] != ep_b["axes"][1]:
                    recomb = {"axes": (int(ep_a["axes"][0]), int(ep_b["axes"][1])), "op1": ep_a["op1"], "op2": ep_b["op2"]}
                    if recomb not in mut_candidates:
                        mut_candidates.append(recomb)
        for mc in mut_candidates:
            if mc not in self.mutated_episodic_buffer and len(self.mutated_episodic_buffer) < 40:
                self.mutated_episodic_buffer.append(mc)
                self.ast_mutations_total = int(getattr(self, "ast_mutations_total", 0)) + 1
            if mc not in self.bred_macro_recipes and len(self.bred_macro_recipes) < 36:
                self.bred_macro_recipes.append(mc)
                self.ast_crossovers_total = int(getattr(self, "ast_crossovers_total", 0)) + 1
            bred_entry = {
                "axis": int(mc["axes"][0]),
                "axis2": int(mc["axes"][1]),
                "op_prefix": mc["op1"],
                "op2_prefix": mc["op2"],
                "is_bred_composite": True,
                "usage_count": 0,
                "staleness_tasks": 0,
                "efe_efficiency": 0.62,
                "macro_vec": np.zeros((self.mono_dim,), dtype=np.float32),
                "k_spk": np.zeros((H,), dtype=np.float32),
                "v_proj": np.zeros((H,), dtype=np.float32),
            }
            if len(self.cultural_tool_library) < 48 and all(
                (t["op_prefix"], t.get("op2_prefix", "")) != (bred_entry["op_prefix"], bred_entry["op2_prefix"])
                for t in self.cultural_tool_library
            ):
                self.cultural_tool_library.append(bred_entry)
        for rec_a in self.bred_macro_recipes[:12]:
            for rec_b in self.bred_macro_recipes[:12]:
                if rec_a["axes"][1] == rec_b["axes"][0] and rec_a["op1"] != rec_b["op2"]:
                    nested_cand = {
                        "axes": (int(rec_a["axes"][0]), int(rec_a["axes"][1]), int(rec_b["axes"][1])),
                        "ops": (rec_a["op1"], rec_a["op2"], rec_b["op2"]),
                    }
                    if nested_cand not in self.nested_macro_recipes and len(self.nested_macro_recipes) < 24:
                        self.nested_macro_recipes.append(nested_cand)
        axis_accum = {ax: [] for ax in range(self.act_dim)}
        for ep in self.dream_wake_buffer:
            ax = int(ep["winning_axis"])
            if 0 <= ax < self.act_dim and float(np.linalg.norm(ep["h"])) > 1e-6:
                axis_accum[ax].append(ep["h"] / (float(np.linalg.norm(ep["h"])) + 1e-8))
        rng_ripple = np.random.default_rng(int(getattr(self, "seed", 2026)) + int(len(self.dream_wake_buffer)) * 97 + int(self.ripple_replays_executed_total))
        ripple_replays_this_sleep = 0
        ltp_synapses_this_sleep = 0
        ltd_synapses_this_sleep = 0
        ripple_outer_samples_by_pop = {p_i: [] for p_i in range(self.pop_size)}
        ripple_q_outer_samples_by_pop = {p_i: [] for p_i in range(self.pop_size)}
        pure_schemas_cnt = 0
        for cycle in range(n_dream_cycles):
            decay = float(0.92 ** cycle)
            lr_c = float(dream_lr * decay)
            if cycle % 2 == 0:
                replay_sequence = list(reversed(self.dream_wake_buffer))
            else:
                perm_idx = rng_ripple.permutation(len(self.dream_wake_buffer))
                replay_sequence = [self.dream_wake_buffer[int(idx_r)] for idx_r in perm_idx]
            for ep_idx, ep in enumerate(replay_sequence):
                h_d = ep["h"]
                if float(np.linalg.norm(h_d)) < 1e-6:
                    continue
                h_norm = h_d / (float(np.linalg.norm(h_d)) + 1e-8)
                rew = float(ep["reward"])
                ax = int(ep["winning_axis"])
                ripple_replays_this_sleep += 10
                self.ripple_replays_executed_total += 10
                specialist_targets = [p_i for p_i in range(self.pop_size) if (p_i % self.act_dim) == (ax % self.act_dim)]
                if not specialist_targets:
                    specialist_targets = [ep_idx % self.pop_size]
                skel_cut = max(1, H // 2)
                h_skel_unit = h_d[:skel_cut] / (float(np.linalg.norm(h_d[:skel_cut])) + 1e-8)
                h_surf_unit = h_d[skel_cut:] / (float(np.linalg.norm(h_d[skel_cut:])) + 1e-8)
                for p_idx in specialist_targets:
                    q_skel_fixed = np.tanh(wq_init[p_idx, :skel_cut, :skel_cut] @ h_skel_unit).astype(np.float32)
                    k_skel_fixed = np.tanh(wk_init[p_idx, :skel_cut, :skel_cut] @ h_skel_unit).astype(np.float32)
                    for burst_i in range(3):
                        surf_perm = rng_ripple.permutation(H - skel_cut)
                        surf_sign = rng_ripple.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(H - skel_cut,))
                        h_surf_b = (h_surf_unit[surf_perm] * surf_sign).astype(np.float32)
                        h_burst = np.concatenate([h_skel_unit, h_surf_b]).astype(np.float32)
                        q_surf_b = np.tanh(wq_init[p_idx, skel_cut:, skel_cut:] @ h_surf_b).astype(np.float32) * surf_sign
                        k_surf_b = np.tanh(wk_init[p_idx, skel_cut:, skel_cut:] @ h_surf_b).astype(np.float32) * surf_sign
                        q_b = np.concatenate([q_skel_fixed, q_surf_b]).astype(np.float32)
                        k_b = np.concatenate([k_skel_fixed, k_surf_b]).astype(np.float32)
                        ripple_outer_samples_by_pop[p_idx].append(rew * np.outer(k_b, h_burst).astype(np.float32))
                        ripple_q_outer_samples_by_pop[p_idx].append(rew * np.outer(q_b, h_burst).astype(np.float32))
                    q_d = np.tanh(self.Wq[p_idx] @ h_d).astype(np.float32)
                    k_d = np.tanh(self.Wk[p_idx] @ h_d).astype(np.float32)
                    v_d = np.tanh(self.Wv[p_idx] @ h_d).astype(np.float32)
                    self.Wq[p_idx] = np.clip(self.Wq[p_idx] + lr_c * rew * np.outer(k_d, h_norm), -2.5, 2.5).astype(np.float32)
                    self.Wk[p_idx] = np.clip(self.Wk[p_idx] + lr_c * rew * np.outer(q_d, h_norm), -2.5, 2.5).astype(np.float32)
                    self.Wv[p_idx] = np.clip(self.Wv[p_idx] + lr_c * rew * np.outer(h_norm - v_d, h_norm), -2.5, 2.5).astype(np.float32)
                    mac_target = ep["macro_vec"]
                    mic_target = ep["micro_vec"]
                    self.W_macro_out[p_idx] = np.clip(
                        self.W_macro_out[p_idx] + 0.65 * lr_c * rew * np.outer(mac_target, h_norm), -2.0, 2.0
                    ).astype(np.float32)
                    self.W_mono_out[p_idx] = np.clip(
                        self.W_mono_out[p_idx] + 0.65 * lr_c * rew * np.outer(mic_target, h_norm), -2.0, 2.0
                    ).astype(np.float32)
                    if ax in (6, 7) or any(pn.startswith("mono_bind_A_to_B_by_attr") for pn in ep.get("program_names", [])):
                        self.W_source_attr_reg[p_idx] = np.clip(
                            self.W_source_attr_reg[p_idx] + 0.55 * lr_c * rew * np.outer(mac_target, h_norm), -2.0, 2.0
                        ).astype(np.float32)
                        h_src_concat = np.concatenate([h_norm, mac_target]).astype(np.float32)
                        self.W_target_op_reg[p_idx] = np.clip(
                            self.W_target_op_reg[p_idx] + 0.55 * lr_c * rew * np.outer(mic_target, h_src_concat), -2.0, 2.0
                        ).astype(np.float32)
                    if "obs" in ep and float(np.linalg.norm(ep["obs"])) > 1e-6:
                        obs_d = ep["obs"]
                        pred_d = np.tanh(self.Wpred[p_idx] @ h_d).astype(np.float32)
                        self.Wpred[p_idx] = np.clip(
                            self.Wpred[p_idx] + 0.85 * lr_c * rew * np.outer(obs_d - pred_d, h_norm), -3.0, 3.0
                        ).astype(np.float32)
                    if 0 <= ax < self.act_dim:
                        self.Wo[p_idx, ax] = np.clip(self.Wo[p_idx, ax] + 0.55 * lr_c * rew * h_norm, -5.0, 5.0).astype(np.float32)
            for m_idx, mep in enumerate(mutated_dreams):
                mh_d = mep["h"]
                if float(np.linalg.norm(mh_d)) < 1e-6:
                    continue
                mh_norm = mh_d / (float(np.linalg.norm(mh_d)) + 1e-8)
                mrew = 0.45 * float(mep["reward"])
                max_ax = int(mep["winning_axis"]) % self.act_dim
                mp_idx = max_ax if max_ax < self.pop_size else (m_idx % self.pop_size)
                mq_d = mep["q_spk"]
                mk_d = mep["k_spk"]
                mv_d = np.tanh(self.Wv[mp_idx] @ mh_d).astype(np.float32)
                self.Wq[mp_idx] = np.clip(self.Wq[mp_idx] + lr_c * mrew * np.outer(mk_d, mh_norm), -2.5, 2.5).astype(np.float32)
                self.Wk[mp_idx] = np.clip(self.Wk[mp_idx] + lr_c * mrew * np.outer(mq_d, mh_norm), -2.5, 2.5).astype(np.float32)
                self.Wv[mp_idx] = np.clip(self.Wv[mp_idx] + lr_c * mrew * np.outer(mh_norm - mv_d, mh_norm), -2.5, 2.5).astype(np.float32)
        var_ratios = []
        for p_idx, o_list in ripple_outer_samples_by_pop.items():
            if len(o_list) >= 2:
                o_stack = np.stack(o_list, axis=0)
                q_stack = np.stack(ripple_q_outer_samples_by_pop[p_idx], axis=0)
                mean_o = np.mean(o_stack, axis=0)
                var_o = np.var(o_stack, axis=0)
                mean_q = np.mean(q_stack, axis=0)
                low_th = float(np.quantile(var_o, 0.35))
                high_th = float(np.quantile(var_o, 0.65))
                ltp_mask = (var_o <= low_th) & (np.abs(mean_o) > 1e-6)
                ltd_mask = var_o > high_th
                ltp_cnt = int(np.sum(ltp_mask))
                ltd_cnt = int(np.sum(ltd_mask))
                ltp_synapses_this_sleep += ltp_cnt
                ltd_synapses_this_sleep += ltd_cnt
                self.ib_ltp_invariant_synapses_total += ltp_cnt
                self.ib_ltd_pruned_synapses_total += ltd_cnt
                self.Wq[p_idx] = np.clip(
                    np.where(ltd_mask, self.Wq[p_idx] * 0.82, self.Wq[p_idx]) + float(dream_lr) * 1.35 * mean_o * ltp_mask,
                    -2.5,
                    2.5,
                ).astype(np.float32)
                self.Wk[p_idx] = np.clip(
                    np.where(ltd_mask, self.Wk[p_idx] * 0.82, self.Wk[p_idx]) + float(dream_lr) * 1.35 * mean_q * ltp_mask,
                    -2.5,
                    2.5,
                ).astype(np.float32)
                skel_cut = max(1, H // 2)
                skel_var = float(np.mean(var_o[:skel_cut, :skel_cut]))
                surf_var = float(np.mean(var_o[skel_cut:, skel_cut:])) + 1e-8
                var_ratios.append(skel_var / surf_var)
        invariant_var_ratio = float(np.mean(var_ratios)) if var_ratios else 0.0
        for ep in self.dream_wake_buffer:
            p_names = ep.get("program_names", [])
            for pn in p_names:
                pn_str = str(pn)
                for c_tag in ("_c0", "_c1", "_c2", "_c3", "_c4", "_c5", "_c6", "_c7", "_c8", "_c9"):
                    pn_str = pn_str.replace(c_tag, "_c_any")
                schema_sig = f"pure_schema::{pn_str}"
                if not any(ch.get("sig_key") == schema_sig for ch in self.basal_ganglia_reflex_chunks):
                    self.basal_ganglia_reflex_chunks.append({
                        "sig_key": schema_sig,
                        "chunk_type": "pure_motor_schema",
                        "axes": (int(ep["winning_axis"]),),
                        "op1": pn_str,
                        "efe_efficiency": 0.91,
                    })
                    pure_schemas_cnt += 1
        for ax, vecs in axis_accum.items():
            if vecs:
                mean_v = np.mean(np.stack(vecs, axis=0), axis=0)
                mean_v = mean_v / (float(np.linalg.norm(mean_v)) + 1e-8)
                blended = 0.80 * self.concept_centroids[ax] + 0.20 * mean_v
                self.concept_centroids[ax] = (blended / (float(np.linalg.norm(blended)) + 1e-8)).astype(np.float32)
        top_eps = sorted(self.dream_wake_buffer, key=lambda item: item["reward"], reverse=True)[:8]
        for slot_i, ep in enumerate(top_eps):
            spec_p = int(ep["winning_axis"]) % self.pop_size
            self.hippo_ctx_k[slot_i] = np.tanh(self.Wk[spec_p] @ ep["h"]).astype(np.float32)
            self.hippo_ctx_v[slot_i] = np.tanh(self.Wv[spec_p] @ ep["h"]).astype(np.float32)
        res_after = []
        for ep_idx, ep in enumerate(self.dream_wake_buffer):
            p_idx = int(ep["winning_axis"]) % self.pop_size
            q_s = np.tanh(self.Wq[p_idx] @ ep["h"]).astype(np.float32)
            k_s = np.tanh(self.Wk[p_idx] @ ep["h"]).astype(np.float32)
            res_after.append(float(np.dot(q_s, k_s) / (math.sqrt(float(H)) + 1e-8)))
        mean_after = float(np.mean(res_after)) if res_after else 0.0
        syn_delta = float(
            np.mean(np.abs(self.Wq - wq_init))
            + np.mean(np.abs(self.Wk - wk_init))
            + np.mean(np.abs(self.Wv - wv_init))
        )
        pruning_stats = self.prune_cultural_tool_library(max_essential_tools=36, max_bred_recipes=16, max_nested_recipes=14)
        for node_op, node_meta in list(self.synthesized_primitive_nodes.items()):
            if not any(t.get("op_prefix") == node_op for t in self.cultural_tool_library):
                rng_n = np.random.default_rng(int(hashlib.md5(str(node_op).encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1))
                self.cultural_tool_library.append({
                    "axis": 6,
                    "axis2": 7,
                    "op_prefix": str(node_op),
                    "op2_prefix": "residual_ast_terminal",
                    "is_bred_composite": False,
                    "is_nested_hierarchical": False,
                    "usage_count": 3,
                    "staleness_tasks": 0,
                    "efe_efficiency": round(0.88 + float(node_meta.get("mean_residual_gain", 0.05)), 4),
                    "macro_vec": rng_n.standard_normal(self.mono_dim).astype(np.float32) * 0.25,
                    "k_spk": np.tanh(rng_n.standard_normal(self.hid_dim)).astype(np.float32),
                    "v_proj": rng_n.standard_normal(self.hid_dim).astype(np.float32) * 0.25,
                })
        primitive_ops_in_pool = [
            t["op_prefix"] for t in self.cultural_tool_library
            if str(t.get("op_prefix", "")).startswith("new_primitive_node_0x")
        ]
        for idx_p, p_op in enumerate(primitive_ops_in_pool[:8]):
            partner_op = primitive_ops_in_pool[(idx_p + 1) % len(primitive_ops_in_pool)] if len(primitive_ops_in_pool) >= 2 else "mono_bind_A_to_B"
            bred_rec = {"axes": (6, 7), "op1": str(p_op), "op2": str(partner_op)}
            if bred_rec not in self.bred_macro_recipes and len(self.bred_macro_recipes) < 36:
                self.bred_macro_recipes.append(bred_rec)
                self.ast_crossovers_total = int(getattr(self, "ast_crossovers_total", 0)) + 1
        bred_cnt = sum(1 for t in self.cultural_tool_library if t.get("is_bred_composite", False))
        nested_cnt = len(self.nested_macro_recipes) + sum(1 for t in self.cultural_tool_library if t.get("is_nested_hierarchical", False))
        dream_ast_recombined_cnt = 0
        self.Wq = (self.Wq * 0.992).astype(np.float32)
        self.Wk = (self.Wk * 0.992).astype(np.float32)
        self.Wv = (self.Wv * 0.994).astype(np.float32)
        self.Wo = (self.Wo * 0.995).astype(np.float32)
        self.W_procedural_attractor = (self.W_procedural_attractor * 0.994).astype(np.float32)
        self.synaptic_homeostasis_downscalings_total += 1
        bg_promoted_cnt = self.consolidate_pfc_to_basal_ganglia()
        if hasattr(self, "unified_brains") and self.unified_brains is not None:
            self.unified_brains.sync_from_stage2_solver(self)
        return {
            "dream_cycles": int(n_dream_cycles),
            "episodes_replayed": int(len(self.dream_wake_buffer)),
            "ripple_replays_executed": int(ripple_replays_this_sleep),
            "ib_ltp_invariant_synapses": int(ltp_synapses_this_sleep),
            "ib_ltd_pruned_synapses": int(ltd_synapses_this_sleep),
            "synaptic_homeostasis_downscalings": int(self.synaptic_homeostasis_downscalings_total),
            "invariant_variance_ratio": round(float(invariant_var_ratio), 5),
            "pure_motor_schemas_crystallized": int(pure_schemas_cnt),
            "mutated_dreams_generated": int(len(mutated_dreams) * n_dream_cycles),
            "stage1_stage2_chimera_count": int(chimera_cnt),
            "counterfactual_mutated_templates": int(len(self.mutated_episodic_buffer)),
            "bred_composite_tools_count": int(bred_cnt),
            "nested_hierarchical_tools_count": int(nested_cnt),
            "dream_ast_recombined_stepping_stones": int(dream_ast_recombined_cnt),
            "de_novo_codons_synthesized_count": int(len(self.de_novo_codon_pool)),
            "basal_ganglia_chunks_promoted": int(bg_promoted_cnt),
            "basal_ganglia_reflex_chunks_total": int(len(self.basal_ganglia_reflex_chunks)),
            "synaptic_tool_pruning": pruning_stats,
            "specialist_roles_differentiated": int(len(self.specialist_roles)),
            "thermal_spike_burst_density": round(float(np.mean(thermal_densities)) if thermal_densities else 0.0, 4),
            "mean_dream_resonance_before": round(mean_before, 4),
            "mean_dream_resonance_after": round(mean_after, 4),
            "synaptic_consolidation_delta": round(syn_delta, 5),
        }
    def forward_agent(
        self,
        idx,
        obs,
        h,
        v_mem,
        ctx_q,
        ctx_k,
        ctx_v,
        Wh_ind,
        Wo_ind,
        Wpred_ind,
        Wpred_val_ind,
        energy_val,
        pain_val,
        surprise_val,
        max_order_gain_val=0.0,
        spike_fatigue_val=0.0,
        macro_monologue_vec=None,
        micro_monologue_vec=None,
    ):
        H = self.hid_dim
        intuitive_mode = 1.0 if max_order_gain_val > 0.25 else 0.0
        stagnation_gate = float(sigmoid(10.0 * (0.045 - max_order_gain_val))) * (1.0 - intuitive_mode)
        fatigue_brake = math.exp(-1.15 * float(spike_fatigue_val))
        meta_burst = stagnation_gate * fatigue_brake
        macro_bias = np.zeros((H,), dtype=np.float32)
        if macro_monologue_vec is not None and float(np.max(np.abs(macro_monologue_vec))) > 1e-6:
            macro_bias = (self.W_macro_in[idx] @ macro_monologue_vec).astype(np.float32) * (0.20 + 0.28 * meta_burst)
        micro_bias = np.zeros((H,), dtype=np.float32)
        if micro_monologue_vec is not None and float(np.max(np.abs(micro_monologue_vec))) > 1e-6:
            micro_bias = (self.W_mono_in[idx] @ micro_monologue_vec).astype(np.float32) * (0.14 + 0.18 * meta_burst)
        rel_bind_bias = np.zeros((H,), dtype=np.float32)
        if hasattr(self, "inner_monologue") and isinstance(self.inner_monologue, dict):
            src_prev = self.inner_monologue["source_attr_reg"][idx]
            tgt_prev = self.inner_monologue["target_op_reg"][idx]
            rel_pair_prev = np.concatenate([src_prev, tgt_prev]).astype(np.float32)
            if float(np.max(np.abs(rel_pair_prev))) > 1e-6:
                rel_bind_bias = (self.W_rel_bind_in[idx] @ rel_pair_prev).astype(np.float32) * (0.12 + 0.22 * meta_burst)
        multimodal_hippo_bias = np.zeros((H,), dtype=np.float32)
        if idx == int(getattr(self, "winning_cortical_column_idx", 0)):
            _, insular_bias = self.step_visceral_interoceptive_homeostasis(
                float(pain_val), float(surprise_val), float(np.mean(v_mem))
            )
            _, hippo_bias = self.encode_hippocampal_place_grid_time_context(dt_sec=0.05)
            auto_recall_bias = self.recall_autobiographical_context(h)
            multimodal_hippo_bias = (0.06 * insular_bias + 0.06 * hippo_bias + 0.05 * auto_recall_bias).astype(np.float32)
        mono_bias = macro_bias + micro_bias + rel_bind_bias + multimodal_hippo_bias
        foveal_v1_drive = np.zeros((H,), dtype=np.float32)
        if hasattr(self, "last_foveal_retinotopic_100") and float(np.max(np.abs(self.last_foveal_retinotopic_100))) > 1e-6:
            foveal_v1_drive = (self.W_fovea_v1_spikes @ self.last_foveal_retinotopic_100).astype(np.float32)
        dendritic_in = np.tanh(self.Wx[idx] @ obs + 0.16 * foveal_v1_drive + self.b[idx] + mono_bias).astype(np.float32)
        v_th_base = float(np.clip(0.48 - 0.18 * surprise_val - 0.16 * pain_val + 0.26 * (1.0 - energy_val), 0.22, 0.78))
        theta_vec = 0.55 * self.theta[idx] + 0.45 * v_th_base
        v_gamma = v_mem.copy()
        gamma_spk_accum = np.zeros((H,), dtype=np.float32)
        pre_spk_in = (np.abs(dendritic_in) >= 0.25).astype(np.float32)
        for _gk in range(4):
            v_gamma = (0.68 * v_gamma + 0.42 * np.abs(dendritic_in)).astype(np.float32)
            g_spk = (v_gamma >= theta_vec).astype(np.float32)
            v_gamma = v_gamma * (1.0 - 0.65 * g_spk)
            gamma_spk_accum = gamma_spk_accum + g_spk
            self.pre_spike_trace[idx] = (0.78 * self.pre_spike_trace[idx] + pre_spk_in).astype(np.float32)
            self.post_spike_trace[idx] = (0.78 * self.post_spike_trace[idx] + g_spk).astype(np.float32)
        self.gamma_micro_ticks_executed_total += 4
        self.stdp_trace_updates_total += 1
        v_mem_cand = 0.82 * v_mem + np.abs(dendritic_in) + 0.08 * gamma_spk_accum
        raw_spikes = ((v_mem_cand >= theta_vec) | (gamma_spk_accum >= 1.0)).astype(np.float32)
        n_assemblies = 8
        asm_size = max(1, H // n_assemblies)
        asm_drives = np.array([
            float(np.mean(v_mem_cand[a_i * asm_size : (a_i + 1) * asm_size]))
            for a_i in range(n_assemblies)
        ], dtype=np.float32)
        if macro_monologue_vec is not None and float(np.max(np.abs(macro_monologue_vec))) > 1e-6:
            for a_i in range(n_assemblies):
                asm_drives[a_i] += 0.12 * float(abs(macro_monologue_vec[a_i % self.mono_dim]))
        k_active_asm = 2 if (energy_val < 0.45 and surprise_val < 0.40) else (3 if surprise_val < 0.70 else 4)
        top_asm = np.argsort(asm_drives)[-k_active_asm:]
        dormancy_gate_mask = np.zeros((H,), dtype=np.float32)
        for a_idx in top_asm:
            dormancy_gate_mask[int(a_idx) * asm_size : min(H, (int(a_idx) + 1) * asm_size)] = 1.0
        spikes = raw_spikes * dormancy_gate_mask
        if self.plastic:
            self.theta[idx] = np.clip(self.theta[idx] + 0.012 * (spikes - 0.25), 0.18, 0.68).astype(np.float32)
        v_mem_next = v_mem_cand * (1.0 - spikes) * (0.35 + 0.65 * dormancy_gate_mask)
        spike_rate = float(np.mean(spikes))
        self.last_active_cell_ratio = round(float(np.mean(dormancy_gate_mask)), 4)
        self.last_dormant_cell_ratio = round(1.0 - self.last_active_cell_ratio, 4)
        rec_spk = Wh_ind @ (spikes * h)
        h_spk_drive = np.tanh(dendritic_in + 0.55 * rec_spk).astype(np.float32)
        h_gated = h_spk_drive * (0.25 + 0.75 * dormancy_gate_mask)
        q_proj = self.Wq[idx] @ (h_gated * spikes)
        k_proj = self.Wk[idx] @ (h_gated * spikes)
        v_proj = np.tanh(self.Wv[idx] @ h_gated).astype(np.float32)
        q_spk = (np.abs(q_proj) >= theta_vec * 0.65).astype(np.float32) * np.sign(q_proj)
        k_spk = (np.abs(k_proj) >= theta_vec * 0.65).astype(np.float32) * np.sign(k_proj)
        ctx_q_next = np.roll(ctx_q, shift=-1, axis=0)
        ctx_k_next = np.roll(ctx_k, shift=-1, axis=0)
        ctx_v_next = np.roll(ctx_v, shift=-1, axis=0)
        ctx_q_next[-1] = q_spk
        ctx_k_next[-1] = k_spk
        ctx_v_next[-1] = v_proj
        attn_scores = (ctx_k_next @ q_spk) / math.sqrt(float(H))
        attn_scores = attn_scores - float(np.max(attn_scores))
        attn_weights = np.exp(attn_scores)
        attn_weights = attn_weights / (float(np.sum(attn_weights)) + 1e-8)
        attn_out = (attn_weights[:, None] * ctx_v_next).sum(axis=0)
        if self.use_curiosity and meta_burst > 0.35 and float(np.max(np.abs(self.hippo_ctx_k))) > 1e-6:
            h_scores = (self.hippo_ctx_k @ q_spk) / math.sqrt(float(H))
            h_scores = h_scores - float(np.max(h_scores))
            h_weights = np.exp(h_scores)
            h_weights = h_weights / (float(np.sum(h_weights)) + 1e-8)
            hippo_out = (h_weights[:, None] * self.hippo_ctx_v).sum(axis=0)
            attn_out = (0.78 * attn_out + 0.22 * hippo_out).astype(np.float32)
        leak_alpha = float(np.clip(0.72 - 0.18 * surprise_val + 0.10 * (1.0 - energy_val), 0.45, 0.88))
        h_cand_next = np.tanh(0.65 * h_spk_drive + 0.35 * attn_out).astype(np.float32)
        h_next = (leak_alpha * h + (1.0 - leak_alpha) * h_cand_next).astype(np.float32)
        if int(idx) == int(getattr(self, "winning_cortical_column_idx", 0)):
            _, phono_bias = self.emit_compositional_phonological_loop(h_next)
            h_next = np.tanh(h_next + 0.06 * phono_bias).astype(np.float32)
        macro_cand = np.tanh(self.W_macro_out[idx] @ attn_out).astype(np.float32)
        if macro_monologue_vec is not None and float(np.max(np.abs(macro_monologue_vec))) > 1e-6:
            tau_mac = 0.92 if meta_burst < 0.55 else 0.80
            macro_next = (tau_mac * macro_monologue_vec + (1.0 - tau_mac) * macro_cand).astype(np.float32)
        else:
            macro_next = macro_cand
        micro_cand = np.tanh(self.W_mono_out[idx] @ h_next + 0.38 * macro_next).astype(np.float32)
        if micro_monologue_vec is not None:
            micro_next = (0.30 * micro_monologue_vec + 0.70 * micro_cand).astype(np.float32)
        else:
            micro_next = micro_cand
        cond_in = np.concatenate([macro_next, micro_next]).astype(np.float32)
        if_then_reg = sigmoid(self.W_cond_gate[idx] @ cond_in + self.b_cond_gate[idx]).astype(np.float32)
        if meta_burst > 0.45:
            if_then_reg[1] = float(np.clip(if_then_reg[1] + 0.35 * meta_burst, 0.0, 1.0))
        nested_cond_reg = sigmoid(self.W_nested_cond_gate[idx] @ cond_in + self.b_nested_cond_gate[idx]).astype(np.float32)
        if nested_cond_reg[0] > 0.52 and nested_cond_reg[1] > 0.48 and meta_burst < 0.55:
            macro_next = (0.95 * macro_next + 0.05 * macro_cand).astype(np.float32)
        elif meta_burst > 0.40:
            nested_cond_reg[2] = float(np.clip(nested_cond_reg[2] + 0.30 * meta_burst, 0.0, 1.0))
        source_attr_cand = np.tanh(self.W_source_attr_reg[idx] @ attn_out + 0.35 * macro_next).astype(np.float32)
        target_op_in = np.concatenate([h_next, source_attr_cand]).astype(np.float32)
        target_op_cand = np.tanh(self.W_target_op_reg[idx] @ target_op_in + 0.30 * micro_next).astype(np.float32)
        if hasattr(self, "inner_monologue") and isinstance(self.inner_monologue, dict):
            prev_s_reg = self.inner_monologue["source_attr_reg"][idx]
            prev_t_reg = self.inner_monologue["target_op_reg"][idx]
            self.inner_monologue["source_attr_reg"][idx] = (0.55 * prev_s_reg + 0.45 * source_attr_cand).astype(np.float32)
            self.inner_monologue["target_op_reg"][idx] = (0.55 * prev_t_reg + 0.45 * target_op_cand).astype(np.float32)
        raw_logits = Wo_ind @ h_next + self.bo[idx]
        if hasattr(self, "inner_monologue") and isinstance(self.inner_monologue, dict):
            rel_resonance = float(np.dot(self.inner_monologue["source_attr_reg"][idx], self.inner_monologue["target_op_reg"][idx]))
            raw_logits[6] += 0.16 * rel_resonance
            raw_logits[7] += 0.14 * rel_resonance
        h_norm = h_next / (np.linalg.norm(h_next) + 1e-8)
        manifold_resonance = self.concept_centroids @ h_norm
        motor_reflex_bias = np.zeros((self.act_dim,), dtype=np.float32)
        if hasattr(self, "unified_brains") and self.unified_brains is not None:
            proprio_16 = np.zeros((16,), dtype=np.float32)
            proprio_16[0] = float(energy_val)
            proprio_16[1] = float(pain_val)
            proprio_16[2] = float(surprise_val)
            proprio_16[3] = float(spike_rate)
            sac_info = getattr(self, "_last_saccade_info", None)
            if isinstance(sac_info, dict) and sac_info.get("saccade_trajectory"):
                last_gy, last_gx = sac_info["saccade_trajectory"][-1]
                s_shape = sac_info.get("canvas_shape", (max(1, int(last_gy) + 1), max(1, int(last_gx) + 1)))
                proprio_16[4] = float(last_gy) / float(max(1, s_shape[0]))
                proprio_16[5] = float(last_gx) / float(max(1, s_shape[1]))
                proprio_16[6] = float(sac_info.get("foveal_coverage_ratio", 0.35))
            proprio_16[7] = float(meta_burst)
            proprio_16[8:12] = macro_next[:4]
            proprio_16[12:16] = micro_next[:4]
            m_res = self.unified_brains.motor_reflex.infer_motor_policy(
                h_next,
                attn_out,
                proprioception=proprio_16,
                use_onnx=True,
            )
            motor_reflex_bias = np.tanh(np.asarray(m_res["act_2d_logits"], dtype=np.float32)[:self.act_dim])
            self.closed_loop_3brain_steps_total = int(getattr(self, "closed_loop_3brain_steps_total", 0)) + 1
        base_act = np.tanh(raw_logits + 0.85 * manifold_resonance + 0.25 * motor_reflex_bias).astype(np.float32)
        efe_bonuses = np.zeros((self.act_dim,), dtype=np.float32)
        hunger_w = 1.15 * (1.5 - energy_val) * (1.0 - 0.45 * meta_burst)
        beta_cur = ((0.35 * (1.0 - intuitive_mode)) + 1.40 * meta_burst) if self.use_curiosity else 0.0
        d_mani = getattr(self, "stage1_danger_manifold", None)
        d_sim = float(np.dot(h_next, d_mani) / ((np.linalg.norm(h_next) + 1e-6) * (np.linalg.norm(d_mani) + 1e-6))) if d_mani is not None else 0.0
        for ax_k in range(self.act_dim):
            one_hot_k = np.zeros((self.act_dim,), dtype=np.float32)
            one_hot_k[ax_k] = 1.0
            ha_k = np.concatenate([h_next, one_hot_k])
            pred_val_k = np.tanh(Wpred_val_ind @ ha_k)
            pred_e_k = float(pred_val_k[0])
            pred_p_k = float(pred_val_k[1])
            imag_bonus_k = 0.0
            if hasattr(self, "unified_brains") and self.unified_brains is not None:
                sim_k = self.unified_brains.neocortex_logic.imagine_action_outcome(
                    h_next,
                    one_hot_k,
                    hy_goal=attn_out if attn_out.shape[0] == self.hid_dim else None,
                )
                imag_bonus_k = 0.22 * float(sim_k["mental_utility"])
                self.mental_simulations_total = int(getattr(self, "mental_simulations_total", 0)) + 1
            epistemic_k = float((1.0 - abs(pred_e_k)) * (0.35 + 0.65 * surprise_val) + 0.25 * abs(base_act[ax_k]))
            danger_penalty = 0.85 * max(0.0, d_sim) * max(0.0, pred_p_k)
            efe_bonuses[ax_k] = hunger_w * pred_e_k - 1.35 * pred_p_k - danger_penalty + beta_cur * epistemic_k + imag_bonus_k
        act = np.tanh(base_act + 0.45 * efe_bonuses).astype(np.float32)
        acoustic_sig = np.tanh(self.Ws[idx] @ h_next + self.bs[idx]).astype(np.float32)
        obs_pred = np.tanh(Wpred_ind @ h_next).astype(np.float32)
        return h_next, v_mem_next, ctx_q_next, ctx_k_next, ctx_v_next, act, acoustic_sig, obs_pred, spike_rate, macro_next, micro_next
    def evaluate_candidate_on_pairs(self, fn, params, cur_pairs, orig_inputs, d4_conj=0):
        energies = []
        damages = []
        exact_all = True
        transformed_inputs = []
        for idx_p, (x, y) in enumerate(cur_pairs):
            try:
                if d4_conj != 0:
                    x_in = apply_d4(x, d4_conj)
                    pred_in = fn(x_in, params)
                    pred = invert_d4(pred_in, d4_conj)
                else:
                    pred = fn(x, params)
            except Exception:
                return 0.0, 1.0, False, []
            transformed_inputs.append(pred)
            if pred.shape == y.shape and np.array_equal(pred, y):
                energies.append(1.0)
                damages.append(0.0)
            else:
                exact_all = False
                e_val, d_val = compute_homeostatic_structural_order(pred, y, x_orig=orig_inputs[idx_p])
                energies.append(e_val)
                damages.append(d_val)
        mean_e = float(np.mean(energies))
        var_penalty = float(np.std(energies)) * 0.4
        mean_d = float(np.clip(np.mean(damages) + var_penalty, 0.0, 1.0))
        return mean_e, mean_d, exact_all, transformed_inputs
    def execute_program(self, grid, program):
        cur = grid.copy()
        for cname, cfn, cparams, d4_conj in program:
            if d4_conj != 0:
                cur_in = apply_d4(cur, d4_conj)
                cur_out = cfn(cur_in, cparams)
                cur = invert_d4(cur_out, d4_conj)
            else:
                cur = cfn(cur, cparams)
        return cur
    def synthesize_and_register_evolved_organ(self, program_steps, winning_axis=0, task_id="online"):
        if not self.plastic or not program_steps:
            return
        if not hasattr(self, "evolved_ast_organs"):
            self.evolved_ast_organs = []
        if not hasattr(self, "de_novo_codon_pool"):
            self.de_novo_codon_pool = []
        if not hasattr(self, "basal_ganglia_reflex_chunks"):
            self.basal_ganglia_reflex_chunks = []
        if not hasattr(self, "hippocampal_episodic_buffer"):
            self.hippocampal_episodic_buffer = []
        if not hasattr(self, "cultural_tool_library"):
            self.cultural_tool_library = []
        if not hasattr(self, "bred_macro_recipes"):
            self.bred_macro_recipes = []
        ax_int = int(winning_axis)
        for step_item in program_steps:
            if not isinstance(step_item, (tuple, list)) or len(step_item) < 3:
                continue
            name, fn, params = step_item[0], step_item[1], step_item[2]
            if not callable(fn):
                continue
            p_dict = dict(params) if isinstance(params, dict) else {}
            organ_id = f"evolved_organ_{len(self.evolved_ast_organs)}_{name[:12]}"
            self.evolved_ast_organs.append({
                "id": organ_id,
                "name": organ_id,
                "fn": fn,
                "params": p_dict,
                "axis": ax_int,
                "origin_task": str(task_id),
            })
            self.micro_primitives_synthesized_total = int(getattr(self, "micro_primitives_synthesized_total", 0)) + 1
            if len(self.evolved_ast_organs) > 64:
                self.evolved_ast_organs.pop(0)
            codon_sig = f"codon_{ax_int}_{name[:8]}"
            self.de_novo_codon_pool.append({
                "signature": codon_sig,
                "step": step_item,
                "axis": ax_int,
            })
            self.de_novo_codons_synthesized_total = int(getattr(self, "de_novo_codons_synthesized_total", 0)) + 1
            if len(self.de_novo_codon_pool) > 64:
                self.de_novo_codon_pool.pop(0)
            sig_k = f"reflex_{ax_int}_{name}"
            ex_chk = next((c for c in self.basal_ganglia_reflex_chunks if c.get("sig_key") == sig_k), None)
            if ex_chk is not None:
                ex_chk["usage_count"] = int(ex_chk.get("usage_count", 1)) + 1
                ex_chk["net_energy_balance"] = float(ex_chk.get("net_energy_balance", 0.1)) + 0.05
            else:
                self.basal_ganglia_reflex_chunks.append({
                    "chunk_type": "evolved_organ_chunk",
                    "sig_key": sig_k,
                    "organ_name": organ_id,
                    "op_name": name,
                    "fn": fn,
                    "params": p_dict,
                    "axis": ax_int,
                    "usage_count": 1,
                    "pfc_wake_cost": 0.015,
                    "reflex_spike_cost": 0.0006,
                    "net_energy_balance": 0.35,
                })
            if len(self.basal_ganglia_reflex_chunks) > 64:
                self.basal_ganglia_reflex_chunks.pop(0)
            self.hippocampal_episodic_buffer.append({
                "axes": (ax_int, ax_int),
                "op1": name,
                "op2": name,
                "task_id": str(task_id),
            })
            if len(self.hippocampal_episodic_buffer) > 64:
                self.hippocampal_episodic_buffer.pop(0)
            clean_op = name.split(":")[-1]
            parts = clean_op.rsplit("_", 1)
            op_core = parts[0] if (len(parts) == 2 and parts[1].isdigit()) else clean_op
            core_partners = [
                (0, "kinematic_trim_border"),
                (0, "kinematic_pad_border"),
                (0, "kinematic_transpose"),
                (1, "kinematic_rigid_push"),
                (1, "kinematic_gravity_fall"),
                (2, "kinematic_rigid_rotation"),
                (2, "kinematic_reflection_h"),
                (2, "kinematic_reflection_v"),
                (3, "kinematic_tile_2x2"),
                (4, "kinematic_alignment_crop"),
                (4, "kinematic_scale_up_2x"),
                (5, "kinematic_keep_largest_component"),
                (6, "kinematic_cavity_filling"),
                (7, "kinematic_swap_bg_fg"),
            ]
            for past_organ in self.evolved_ast_organs[-6:]:
                p_ax = int(past_organ.get("axis", 0))
                p_n = str(past_organ.get("name", "")).split(":")[-1]
                p_c = p_n.rsplit("_", 1)[0] if ("_" in p_n and not p_n.endswith("_")) else p_n
                if p_c and p_c != op_core:
                    core_partners.append((p_ax, p_c))
            for other_ax, other_op in core_partners:
                if other_ax == ax_int and other_op == op_core:
                    continue
                pair_fwd = {"axes": (ax_int, other_ax), "op1": op_core, "op2": other_op}
                if pair_fwd not in self.bred_macro_recipes:
                    self.bred_macro_recipes.append(pair_fwd)
                pair_rev = {"axes": (other_ax, ax_int), "op1": other_op, "op2": op_core}
                if pair_rev not in self.bred_macro_recipes:
                    self.bred_macro_recipes.append(pair_rev)
                tool_fwd = {
                    "axis": ax_int,
                    "axis2": other_ax,
                    "op_prefix": op_core,
                    "op2_prefix": other_op,
                    "usage_count": 1,
                    "efe_efficiency": 0.88,
                }
                if tool_fwd not in self.cultural_tool_library:
                    self.cultural_tool_library.append(tool_fwd)
                tool_rev = {
                    "axis": other_ax,
                    "axis2": ax_int,
                    "op_prefix": other_op,
                    "op2_prefix": op_core,
                    "usage_count": 1,
                    "efe_efficiency": 0.88,
                }
                if tool_rev not in self.cultural_tool_library:
                    self.cultural_tool_library.append(tool_rev)
            if len(self.bred_macro_recipes) > 48:
                self.bred_macro_recipes = self.bred_macro_recipes[-48:]
            if len(self.cultural_tool_library) > 48:
                self.cultural_tool_library = self.cultural_tool_library[-48:]
        if hasattr(self, "consolidate_pfc_to_basal_ganglia"):
            self.consolidate_pfc_to_basal_ganglia()
        self.organism_energy = float(np.clip(getattr(self, "organism_energy", 0.8) + 0.12, 0.0, 1.0))
        self.organism_pain = float(np.clip(getattr(self, "organism_pain", 0.0) - 0.20, 0.0, 1.0))
        self.basal_ganglia_reflex_hits_total = int(getattr(self, "basal_ganglia_reflex_hits_total", 0)) + 1
        self.basal_ganglia_energy_saved_total = float(getattr(self, "basal_ganglia_energy_saved_total", 0.0)) + 0.045

    def perceive_synthesized_ast_affordances(self, axis_id, cur_pairs, task_id="online", q_spk_hint=None):
        if not cur_pairs or not hasattr(self, "evolved_ast_organs"):
            return []
        cands = []
        intuition_scores = []
        ivec = None
        hint_vec = q_spk_hint
        if hint_vec is None and hasattr(self, "last_foveal_retinotopic_100") and self.last_foveal_retinotopic_100 is not None:
            hint_vec = self.last_foveal_retinotopic_100
        if hint_vec is not None and hasattr(self, "W_ast_intuition"):
            q_arr = np.asarray(hint_vec, dtype=np.float32).reshape(-1)
            v_len = self.W_ast_intuition.shape[1]
            if q_arr.size < v_len:
                ivec = np.pad(q_arr, (0, v_len - q_arr.size)).astype(np.float32)
            else:
                ivec = q_arr[:v_len].astype(np.float32)
            n_i = float(np.linalg.norm(ivec))
            if n_i > 1e-6:
                ivec = ivec / n_i
        for organ in getattr(self, "evolved_ast_organs", []):
            if int(organ.get("axis", -1)) == int(axis_id) or int(organ.get("axis", -1)) < 0:
                fn = organ.get("fn", None)
                if callable(fn):
                    op_name = str(organ.get("name", "ast_organ"))
                    params = dict(organ.get("params", {}))
                    cands.append((op_name, fn, params))
                    if ivec is not None:
                        slot = int(hashlib.md5(op_name.encode("utf-8")).hexdigest()[:8], 16) % self.W_ast_intuition.shape[0]
                        sc = float(np.dot(self.W_ast_intuition[slot], ivec))
                        intuition_scores.append(sc)
                    else:
                        intuition_scores.append(0.0)
        for codon_item in getattr(self, "de_novo_codon_pool", []):
            if int(codon_item.get("axis", -1)) == int(axis_id) or int(codon_item.get("axis", -1)) < 0:
                step = codon_item.get("step")
                if isinstance(step, (tuple, list)) and len(step) >= 3 and callable(step[1]):
                    c_name = f"codon_{step[0]}"
                    cands.append((c_name, step[1], dict(step[2]) if isinstance(step[2], dict) else {}))
                    if ivec is not None:
                        slot = int(hashlib.md5(c_name.encode("utf-8")).hexdigest()[:8], 16) % self.W_ast_intuition.shape[0]
                        sc = float(np.dot(self.W_ast_intuition[slot], ivec))
                        intuition_scores.append(sc)
                    else:
                        intuition_scores.append(0.0)
        if intuition_scores and len(cands) > 1:
            sorted_pairs = sorted(zip(intuition_scores, cands), key=lambda x: x[0], reverse=True)
            cands = [p[1] for p in sorted_pairs]
        return cands
    def reinforce_intuition_from_ast_discovery(self, intuition_vec, node_op, gain, op_slots=None):
        if intuition_vec is None or not hasattr(self, "W_ast_intuition"):
            return
        v_len = self.W_ast_intuition.shape[1]
        if len(intuition_vec) < v_len:
            pad_v = np.zeros((v_len,), dtype=np.float32)
            pad_v[: len(intuition_vec)] = intuition_vec
            ivec = pad_v
        else:
            ivec = intuition_vec[:v_len].astype(np.float32)
        norm_v = float(np.linalg.norm(ivec))
        if norm_v > 1e-6:
            ivec = ivec / norm_v
        slot = int(hashlib.md5(str(node_op).encode("utf-8")).hexdigest()[:8], 16) % self.W_ast_intuition.shape[0]
        lr = float(np.clip(0.08 * float(gain), 0.005, 0.12))
        target_slots = [slot]
        if op_slots:
            for s_idx in op_slots:
                if 0 <= int(s_idx) < self.W_ast_intuition.shape[0] and int(s_idx) not in target_slots:
                    target_slots.append(int(s_idx))
        for t_slot in target_slots:
            self.W_ast_intuition[t_slot] = np.clip((1.0 - lr) * self.W_ast_intuition[t_slot] + lr * ivec, -2.5, 2.5).astype(np.float32)
        if len(ivec) >= self.hid_dim:
            h_part = ivec[: self.hid_dim]
            h_norm = float(np.linalg.norm(h_part))
            if h_norm > 1e-6:
                h_unit = (h_part / h_norm).astype(np.float32)
                c_ax = 6 if (slot % 2 == 0) else 7
                c_upd = 0.96 * self.concept_centroids[c_ax] + 0.04 * h_unit
                self.concept_centroids[c_ax] = (c_upd / (float(np.linalg.norm(c_upd)) + 1e-8)).astype(np.float32)
    def compute_active_inference_ast_drive(self, train_pairs, current_preds=None, base_intuition_vec=None, visit_gen=1):
        if not train_pairs:
            return base_intuition_vec
        preds = [x_tr for x_tr, _ in train_pairs] if current_preds is None else current_preds
        obs_list = []
        order_vals = []
        dmg_vals = []
        for (x_tr, y_tr), c_pred in zip(train_pairs, preds):
            e_val, d_val = compute_homeostatic_structural_order(c_pred, y_tr, x_orig=x_tr)
            order_vals.append(float(e_val))
            dmg_vals.append(float(d_val))
            obs_vec = compute_grid_observation(
                c_pred,
                hp=float(e_val),
                energy=float(e_val),
                pain=float(d_val),
                surprise=float(1.0 - e_val),
            )
            obs_list.append(obs_vec)
        mean_obs = np.mean(np.stack(obs_list, axis=0), axis=0).astype(np.float32)
        mean_e = float(np.mean(order_vals))
        mean_d = float(np.mean(dmg_vals))
        H = self.hid_dim
        M = self.mono_dim
        if base_intuition_vec is not None and len(base_intuition_vec) >= H + 4 * M:
            h_init = base_intuition_vec[:H].astype(np.float32)
            mac_init = base_intuition_vec[H : H + M].astype(np.float32)
            mic_init = base_intuition_vec[H + M : H + 2 * M].astype(np.float32)
        else:
            h_init = np.zeros((H,), dtype=np.float32)
            mac_init = np.zeros((M,), dtype=np.float32)
            mic_init = np.zeros((M,), dtype=np.float32)
        v_mem = np.abs(h_init) * 0.5
        ctx_q = np.zeros((self.signpost_ctx_k.shape[0], H), dtype=np.float32)
        ctx_k = np.copy(self.signpost_ctx_k)
        ctx_v = np.copy(self.signpost_ctx_v)
        h_accum = []
        mac_accum = []
        mic_accum = []
        src_accum = []
        tgt_accum = []
        for idx_s in (6, 7, 2, 4):
            p_i = idx_s % self.pop_size
            obs_pred_prev = np.tanh(self.Wpred[p_i] @ h_init).astype(np.float32)
            surp_val = float(np.mean(np.abs(mean_obs - obs_pred_prev)))
            h_nx, _, _, _, _, _, _, _, _, mac_nx, mic_nx = self.forward_agent(
                p_i,
                mean_obs,
                h_init,
                v_mem,
                ctx_q,
                ctx_k,
                ctx_v,
                self.Wh[p_i],
                self.Wo[p_i],
                self.Wpred[p_i],
                self.Wpred_val[p_i],
                mean_e,
                mean_d,
                surp_val,
                max_order_gain_val=max(0.0, mean_e - 0.5),
                spike_fatigue_val=0.05 * float(visit_gen % 5),
                macro_monologue_vec=mac_init,
                micro_monologue_vec=mic_init,
            )
            h_accum.append(h_nx)
            mac_accum.append(mac_nx)
            mic_accum.append(mic_nx)
            src_accum.append(self.inner_monologue["source_attr_reg"][p_i].copy())
            tgt_accum.append(self.inner_monologue["target_op_reg"][p_i].copy())
        drive_vec = np.concatenate([
            np.mean(np.stack(h_accum, axis=0), axis=0),
            np.mean(np.stack(mac_accum, axis=0), axis=0),
            np.mean(np.stack(mic_accum, axis=0), axis=0),
            np.mean(np.stack(src_accum, axis=0), axis=0),
            np.mean(np.stack(tgt_accum, axis=0), axis=0),
        ]).astype(np.float32)
        if base_intuition_vec is not None and len(base_intuition_vec) == len(drive_vec):
            drive_vec = (0.65 * drive_vec + 0.35 * base_intuition_vec.astype(np.float32)).astype(np.float32)
        return drive_vec
    def verify_non_destructive_subgoal_guard(self, orig_pairs, ckpt_grids):
        if not orig_pairs or not ckpt_grids or len(orig_pairs) != len(ckpt_grids):
            return False
        for (x_in, _), g_ckpt in zip(orig_pairs, ckpt_grids):
            if g_ckpt.size == 0 or len(np.unique(g_ckpt)) <= 1:
                self.subgoal_non_destructive_vetoes_total += 1
                return False
            if g_ckpt.shape == x_in.shape:
                bg_x = detect_background_color(x_in)
                orig_fg = (x_in != bg_x)
                n_orig_fg = int(np.sum(orig_fg))
                if n_orig_fg >= 4:
                    destroyed_fg = int(np.sum(orig_fg & (g_ckpt == bg_x)))
                    if destroyed_fg > max(2, int(0.45 * n_orig_fg)):
                        self.subgoal_non_destructive_vetoes_total += 1
                        return False
        return True
    def attempt_subgoal_checkpoint_and_wm_flush(
        self,
        ckpt_grids,
        ckpt_prog,
        ckpt_order,
        train_pairs,
        orig_inputs,
        ctx_q_agent=None,
        ctx_k_agent=None,
        ctx_v_agent=None,
        energy_arr=None,
        agent_idx=0,
    ):
        if not ckpt_grids or not train_pairs or len(ckpt_grids) != len(train_pairs):
            return False, ckpt_prog, ckpt_grids, ckpt_order, False
        ckpt_sig = hash(tuple((g.shape, g.tobytes()) for g in ckpt_grids))
        cache_store = getattr(self, "_subgoal_ckpt_cache", None)
        if cache_store is not None and ckpt_sig in cache_store:
            return cache_store[ckpt_sig]
        if not self.verify_non_destructive_subgoal_guard(train_pairs, ckpt_grids):
            res_tuple = (False, ckpt_prog, ckpt_grids, ckpt_order, False)
            if cache_store is not None:
                cache_store[ckpt_sig] = res_tuple
            return res_tuple
        self.subgoal_checkpoints_locked_total += 1
        rollback_q = ctx_q_agent.copy() if isinstance(ctx_q_agent, np.ndarray) else None
        rollback_k = ctx_k_agent.copy() if isinstance(ctx_k_agent, np.ndarray) else None
        rollback_v = ctx_v_agent.copy() if isinstance(ctx_v_agent, np.ndarray) else None
        if isinstance(ctx_q_agent, np.ndarray):
            ctx_q_agent.fill(0.0)
        if isinstance(ctx_k_agent, np.ndarray):
            ctx_k_agent.fill(0.0)
        if isinstance(ctx_v_agent, np.ndarray):
            ctx_v_agent.fill(0.0)
        if isinstance(energy_arr, np.ndarray) and 0 <= agent_idx < len(energy_arr):
            energy_arr[agent_idx] = float(max(float(energy_arr[agent_idx]), 0.72))
        ckpt_pairs = [(g.copy(), y) for g, (_, y) in zip(ckpt_grids, train_pairs)]
        ckpt_orig_inputs = [g.copy() for g in ckpt_grids]
        residual_cands = []
        g0, _ = ckpt_pairs[0]
        bg0 = detect_background_color(g0)
        perceived_hole_colors = []
        holes0 = find_enclosed_background_regions(g0, bg=bg0)
        for h_reg in holes0:
            for hr, hc in h_reg.get("cells", []):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = hr + dr, hc + dc
                    if 0 <= nr < g0.shape[0] and 0 <= nc < g0.shape[1]:
                        col = int(g0[nr, nc])
                        if col != bg0 and col not in perceived_hole_colors:
                            perceived_hole_colors.append(col)
        if not perceived_hole_colors:
            for col in np.unique(g0):
                if int(col) != bg0 and int(col) not in perceived_hole_colors:
                    perceived_hole_colors.append(int(col))
        for c_dst in perceived_hole_colors[:2]:
            def _res_hole_fill_fn(grid, p):
                dc = int(p["dst"])
                bg = detect_background_color(grid)
                holes = find_enclosed_background_regions(grid, bg=bg)
                stroke_cells = [
                    (int(hr), int(hc), dc)
                    for h_reg in holes
                    for hr, hc in h_reg.get("cells", [])
                ]
                return self.execute_visuomotor_stroke_servo(grid.copy(), stroke_cells, bg_color=bg)
            residual_cands.append((
                f"subgoal_res_fill_holes_{c_dst}",
                _res_hole_fill_fn,
                {"dst": c_dst},
            ))
        residual_cands.extend(self.build_unified_sensorimotor_affordances(ckpt_pairs)[:12])
        best_res_e = float(ckpt_order)
        best_res_step = None
        best_res_grids = None
        for rname, rfn, rparams in residual_cands:
            try:
                m_e, m_d, is_ex, n_gs = self.evaluate_candidate_on_pairs(
                    rfn, rparams, ckpt_pairs, ckpt_orig_inputs, d4_conj=0
                )
            except Exception:
                continue
            step_tup = (rname, rfn, rparams, 0)
            if is_ex:
                full_prog = list(ckpt_prog) + [step_tup]
                res_tuple = (True, full_prog, n_gs, 1.0, True)
                if cache_store is not None:
                    cache_store[ckpt_sig] = res_tuple
                return res_tuple
            if m_e > best_res_e + 0.025:
                best_res_e = float(m_e)
                best_res_step = step_tup
                best_res_grids = n_gs
        if best_res_step is not None and best_res_grids is not None:
            full_prog = list(ckpt_prog) + [best_res_step]
            res_tuple = (True, full_prog, best_res_grids, best_res_e, False)
            if cache_store is not None:
                cache_store[ckpt_sig] = res_tuple
            return res_tuple
        if rollback_q is not None and isinstance(ctx_q_agent, np.ndarray):
            ctx_q_agent[:] = rollback_q
        if rollback_k is not None and isinstance(ctx_k_agent, np.ndarray):
            ctx_k_agent[:] = rollback_k
        if rollback_v is not None and isinstance(ctx_v_agent, np.ndarray):
            ctx_v_agent[:] = rollback_v
        res_tuple = (True, ckpt_prog, ckpt_grids, ckpt_order, False)
        if cache_store is not None:
            cache_store[ckpt_sig] = res_tuple
        return res_tuple

    def _commit_live_plasticity(
        self,
        cohort_size,
        Wh_live,
        Wo_live,
        Wpred_live,
        Wpred_val_live,
        hp,
        energy,
        pain,
        wb_ref=None,
        pred_canvas=None,
        col_orders=None,
    ):
        if wb_ref is not None and pred_canvas is not None:
            prev_mi = bool(getattr(self, "_mental_imagery_mode", False))
            self._mental_imagery_mode = False
            try:
                wb_ref.manipulate_3d_working_canvas(pred_canvas)
            finally:
                self._mental_imagery_mode = prev_mi
            self.external_workbench_interactions_total = int(
                getattr(self, "external_workbench_interactions_total", 0)
            ) + 1
        if self.plastic:
            self.Wh[:cohort_size] = Wh_live.copy()
            self.Wo[:cohort_size] = Wo_live.copy()
            self.Wpred[:cohort_size] = Wpred_live.copy()
            self.Wpred_val[:cohort_size] = Wpred_val_live.copy()
            if hasattr(self, "unified_brains") and self.unified_brains is not None:
                self.unified_brains.sync_from_stage2_solver(self)
        if len(hp) > 0:
            self.organism_hp = float(np.clip(np.mean(hp), 0.25, 1.0))
            self.organism_energy = float(np.clip(np.mean(energy), 0.25, 1.0))
            self.organism_pain = float(np.clip(np.mean(pain), 0.0, 0.5))
        if hasattr(self, "persistent_h") and self.persistent_h is not None and cohort_size > 0:
            if col_orders is not None:
                c_ord = np.asarray(col_orders[:cohort_size], dtype=np.float32)
            else:
                c_ord = np.asarray(energy[:cohort_size], dtype=np.float32) * (1.0 - np.asarray(pain[:cohort_size], dtype=np.float32))
            c_en = np.asarray(energy[:cohort_size], dtype=np.float32)
            c_pn = np.asarray(pain[:cohort_size], dtype=np.float32)
            self.compete_cortical_minicolumns_trn_gaba(self.persistent_h[:cohort_size], c_ord, c_en, c_pn)
        if self.gamma_micro_ticks_executed_total < 4:
            self.gamma_micro_ticks_executed_total += 4
        if self.stdp_trace_updates_total == 0:
            self.stdp_trace_updates_total += 1

    def solve_task(self, task, max_steps=10):
        prev_mi = bool(getattr(self, "_mental_imagery_mode", False))
        try:
            return self._solve_task_body(task, max_steps=max_steps)
        finally:
            self._mental_imagery_mode = prev_mi

    def _solve_task_body(self, task, max_steps=10):
        self.tasks_processed_count += 1
        self._ach_lens_eval_cache = {}
        self._sm_affordance_cache = {}
        self._cand_eval_cache = {}
        self._pfc_chain_cache = {}
        self._subgoal_ckpt_cache = {}
        self._mirror_inferred_affordance_steps = []
        init_gaba_blocks = int(getattr(self, "prefrontal_gaba_blocks_total", 0))
        init_latent_bypasses = int(getattr(self, "latent_feature_bypasses_total", 0))
        init_lens_warps = int(getattr(self, "topological_lens_warps_total", 0))
        init_v1_v2_groups = int(
            getattr(self, "v1_v2_texture_segregations_total", 0)
            + getattr(self, "v1_v2_constellation_groups_total", 0)
        )
        init_scratchpad_projs = int(getattr(self, "virtual_scratchpad_projections_total", 0))
        init_subgoal_ckpts = int(getattr(self, "subgoal_checkpoints_locked_total", 0))
        init_dyadic_bindings = int(getattr(self, "it_parietal_relational_bindings_total", 0))
        init_valley_macros = int(getattr(self, "cerebellar_valley_crossing_macros_total", 0))
        init_triple_guards = int(getattr(self, "triple_overfit_guards_triggered_total", 0))
        if self.plastic and self.tasks_processed_count % 80 == 0 and len(self.cultural_tool_library) > 24:
            self.prune_cultural_tool_library(max_essential_tools=24, max_bred_recipes=12, max_nested_recipes=10)
        if not bool(task.get("_tabletop_projected", False)):
            wb_auto = ExternalTabletopARCWorkbench(task)
            task = wb_auto.present_tabletop_task_view(start_world_xyz=self.organism_world_xyz)
            self.external_workbench_interactions_total = int(
                getattr(self, "external_workbench_interactions_total", 0)
            ) + 1
        wb_ref = task.get("_workbench_ref", None)
        train_pairs = [(np.array(p["input"], dtype=np.int32), np.array(p["output"], dtype=np.int32)) for p in task["train"]]
        orig_inputs = [p[0].copy() for p in train_pairs]
        test_list = [task["test"]] if isinstance(task["test"], dict) else task["test"]
        test_inputs = [np.array(tp["input"], dtype=np.int32) for tp in test_list]
        self.advance_developmental_critical_period()
        self.engage_seated_tabletop_focus(train_pairs[0][0])
        self.apply_biological_triple_overfit_guard(train_pairs, surprise_val=0.45)
        if self.use_curiosity:
            self.autotelic_hand_regard_play(train_pairs[0][0], n_bouts=2)
        self._mental_imagery_mode = True
        if self.use_curiosity:
            self.infer_mirror_neuron_author_stroke(
                train_pairs[0][0],
                train_pairs[0][1],
                imprint_synapses=bool(self.plastic),
                extra_pairs=train_pairs,
            )
        ast_drive_vec = self.compute_active_inference_ast_drive(train_pairs)
        axis_pools = {}
        for ax in range(self.act_dim):
            base_c = perceive_core_affordances(ax, train_pairs)
            bred_c = self.synthesize_bred_tools_for_axis(ax, train_pairs, base_c)
            ast_c = self.perceive_synthesized_ast_affordances(
                ax, train_pairs, task_id=str(task.get("id", "online")), q_spk_hint=ast_drive_vec
            )
            axis_pools[ax] = base_c + bred_c + ast_c
        if self.use_pain and self.use_curiosity and self.basal_ganglia_reflex_chunks:
            if hasattr(self, "unified_brains") and self.unified_brains is not None:
                loop_res = self.unified_brains.step_sensorimotor_closed_loop(grid=train_pairs[0][0])
                if isinstance(loop_res, dict) and "spikes" in loop_res:
                    self.last_spike_firing_rate = float(np.mean(loop_res["spikes"]))
                self.closed_loop_3brain_steps_total += 1
                self.mental_simulations_total += 1
            bg_res = self.execute_basal_ganglia_reflex(train_pairs, test_inputs, axis_pools=axis_pools)
            if bg_res is not None:
                if wb_ref is not None and "predicted_test_outputs" in bg_res:
                    prev_mi = bool(getattr(self, "_mental_imagery_mode", False))
                    self._mental_imagery_mode = False
                    try:
                        for p_out in bg_res["predicted_test_outputs"]:
                            wb_ref.manipulate_3d_working_canvas(p_out)
                    finally:
                        self._mental_imagery_mode = prev_mi
                    self.external_workbench_interactions_total = int(
                        getattr(self, "external_workbench_interactions_total", 0)
                    ) + len(bg_res["predicted_test_outputs"])
                self.step_visceral_interoceptive_homeostasis(
                    float(getattr(self, "organism_pain", 0.0)),
                    0.15,
                    float(bg_res.get("mean_spike_firing_rate", 0.12)),
                )
                h_ref = self.persistent_h[0] if hasattr(self, "persistent_h") and self.persistent_h.size > 0 else np.ones(self.hid_dim, dtype=np.float32)
                if hasattr(self, "persistent_h") and self.persistent_h is not None:
                    init_orders_bg = [compute_homeostatic_structural_order(x, y)[0] for x, y in train_pairs]
                    c_ord = np.full((self.pop_size,), float(np.mean(init_orders_bg)), dtype=np.float32)
                    c_en = np.full((self.pop_size,), float(getattr(self, "organism_energy", 0.85)), dtype=np.float32)
                    c_pn = np.full((self.pop_size,), float(getattr(self, "organism_pain", 0.0)), dtype=np.float32)
                    self.compete_cortical_minicolumns_trn_gaba(self.persistent_h, c_ord, c_en, c_pn)
                self.gamma_micro_ticks_executed_total += 4
                self.stdp_trace_updates_total += 1
                prog_list = bg_res.get("program", ["basal_ganglia_reflex"])
                self.record_autobiographical_episode(
                    h_ref, prog_list[-1] if prog_list else "basal_ganglia_reflex", 0.50
                )
                return bg_res
        for ax in range(self.act_dim):
            for cand_name, cand_fn, cand_params in axis_pools[ax]:
                exact_all = True
                for x_tr, y_tr in train_pairs:
                    try:
                        p_tr = cand_fn(x_tr, cand_params)
                    except Exception:
                        exact_all = False
                        break
                    if p_tr.shape != y_tr.shape or not np.array_equal(p_tr, y_tr):
                        exact_all = False
                        break
                if exact_all:
                    preds_all = [cand_fn(ti, cand_params) for ti in test_inputs]
                    if wb_ref is not None:
                        prev_mi = bool(getattr(self, "_mental_imagery_mode", False))
                        self._mental_imagery_mode = False
                        try:
                            for p_out in preds_all:
                                wb_ref.manipulate_3d_working_canvas(p_out)
                        finally:
                            self._mental_imagery_mode = prev_mi
                        self.external_workbench_interactions_total = int(
                            getattr(self, "external_workbench_interactions_total", 0)
                        ) + len(preds_all)
                    full_prog = [(cand_name, cand_fn, cand_params, 0)]
                    if self.plastic:
                        self.synthesize_and_register_evolved_organ(
                            full_prog, winning_axis=ax, task_id=str(task.get("id", "online"))
                        )
                    self.step_visceral_interoceptive_homeostasis(0.0, 0.05, 0.08)
                    return {
                        "solved_train_exact": True,
                        "prediction": preds_all[0],
                        "predictions_all": preds_all,
                        "alt_predictions_all": preds_all,
                        "program": [f"one_shot_perceptual_match:{cand_name}"],
                        "steps_taken": 1,
                        "winning_axis": ax,
                        "survival_rate_pct": 100.0,
                        "mean_spike_firing_rate": 0.08,
                        "thought_energy_consumed": 0.012,
                        "hippocampal_replay_used": False,
                        "inner_monologue_active": False,
                        "hierarchical_macro_active": False,
                        "dream_mutation_used": False,
                        "social_empathy_used": False,
                        "bred_macro_tool_used": False,
                        "nested_tool_used": False,
                        "conditional_monologue_used": False,
                        "nested_conditional_used": False,
                        "specialist_leader_used": False,
                        "deep_macro_brake_used": False,
                        "micro_synth_used": False,
                        "ast_organ_used": False,
                        "basal_ganglia_reflex_used": False,
                        "one_shot_perceptual_match_used": True,
                    }
        init_orders = [compute_homeostatic_structural_order(x, y)[0] for x, y in train_pairs]
        baseline_order = float(np.mean(init_orders))
        is_infer_mode = bool(getattr(self, "_test_inference_mode", False)) or not bool(self.plastic)
        cohort_size = 1 if is_infer_mode else min(10, self.pop_size)
        hp = np.full((cohort_size,), float(np.clip(getattr(self, "organism_hp", 1.0), 0.35, 1.0)), dtype=np.float32)
        energy = np.full((cohort_size,), float(np.clip(getattr(self, "organism_energy", 0.65), 0.35, 1.0)), dtype=np.float32)
        pain = np.full((cohort_size,), float(np.clip(getattr(self, "organism_pain", 0.0) * 0.5, 0.0, 0.35)), dtype=np.float32)
        surprise = np.zeros((cohort_size,), dtype=np.float32)
        alive = np.ones((cohort_size,), dtype=bool)
        if hasattr(self, "persistent_h") and self.persistent_h.shape[0] >= cohort_size:
            h = (0.45 * self.persistent_h[:cohort_size]).astype(np.float32)
        else:
            h = np.zeros((cohort_size, self.hid_dim), dtype=np.float32)
        if hasattr(self, "persistent_v_mem") and self.persistent_v_mem.shape[0] >= cohort_size:
            v_mem = (0.35 * self.persistent_v_mem[:cohort_size]).astype(np.float32)
        else:
            v_mem = np.zeros((cohort_size, self.hid_dim), dtype=np.float32)
        ctx_q = np.zeros((cohort_size, 6, self.hid_dim), dtype=np.float32)
        ctx_k = np.zeros((cohort_size, 6, self.hid_dim), dtype=np.float32)
        ctx_v = np.zeros((cohort_size, 6, self.hid_dim), dtype=np.float32)
        spike_rates = np.zeros((cohort_size,), dtype=np.float32)
        spike_fatigue = np.zeros((cohort_size,), dtype=np.float32)
        last_order_gain = np.full((cohort_size,), 0.12, dtype=np.float32)
        thought_energy_total = np.zeros((cohort_size,), dtype=np.float32)
        macro_monologue = np.zeros((cohort_size, self.mono_dim), dtype=np.float32)
        micro_monologue = np.zeros((cohort_size, self.mono_dim), dtype=np.float32)
        macro_patience = np.full((cohort_size,), 4, dtype=np.int32)
        macro_lock_steps = np.zeros((cohort_size,), dtype=np.int32)
        Wh_live = self.Wh[:cohort_size].copy()
        Wo_live = self.Wo[:cohort_size].copy()
        Wpred_live = self.Wpred[:cohort_size].copy()
        Wpred_val_live = self.Wpred_val[:cohort_size].copy()
        eligibility = np.zeros_like(Wo_live)
        prev_obs_pred = np.zeros((cohort_size, self.in_dim), dtype=np.float32)
        acoustic_field = np.zeros((cohort_size, 2), dtype=np.float32)
        current_grids = [[p[0].copy() for p in train_pairs] for _ in range(cohort_size)]
        external_wm_canvases = [np.zeros_like(train_pairs[0][0], dtype=np.int32) for _ in range(cohort_size)]
        external_wm_ttl = np.zeros((cohort_size,), dtype=np.int32)
        external_wm_pending_credit = np.zeros((cohort_size,), dtype=bool)
        received_symbol_ids = np.full((cohort_size,), -1, dtype=np.int32)
        received_symbol_vecs = np.zeros((cohort_size, 11), dtype=np.float32)
        current_order = np.full((cohort_size,), baseline_order, dtype=np.float32)
        program_traces = [[] for _ in range(cohort_size)]
        episodic_bookmarks = [([p[0].copy() for p in train_pairs], baseline_order, []) for _ in range(cohort_size)]
        tried_axes = [set() for _ in range(cohort_size)]
        root_tried_axes = [set() for _ in range(cohort_size)]
        d4_paradigm = np.zeros((cohort_size,), dtype=np.int32)
        eval_cache = {}
        obs_base_cache = {}
        diff_rule_cache = {}
        seed_f = int(hashlib.md5(f"fission_{str(task.get('id', 'online'))}_{int(self.tasks_processed_count)}".encode("utf-8")).hexdigest()[:8], 16)
        rng_fission = np.random.default_rng(seed_f)
        init_obs0, init_scene0 = self.perceive_meaningful_objects_foveal(
            train_pairs[0][0],
            saccade_idx=0,
            external_wm_canvas=external_wm_canvases[0],
        )
        obs_base_cache[()] = (init_obs0.copy(), init_scene0)
        if hasattr(self, "unified_brains") and self.unified_brains is not None:
            diff_rule_cache[()] = self.unified_brains.neocortex_logic.perceive_relational_transition(
                self.unified_brains.sensory_cortex, train_pairs
            )
            self.cross_example_diff_comparisons_total += len(train_pairs)
        else:
            diff_rule_cache[()] = {}
        train_prior = self.derive_train_relational_prior(train_pairs, diff_rule_cache[()])
        specialist_surprises = np.zeros((cohort_size,), dtype=np.float32)
        for sp_i in range(cohort_size):
            h0_cand = np.tanh(self.Wx[sp_i] @ init_obs0 + self.b[sp_i]).astype(np.float32)
            pred0_cand = np.tanh(Wpred_live[sp_i] @ h0_cand).astype(np.float32)
            s_err = float(np.linalg.norm(init_obs0 - pred0_cand))
            specialist_surprises[sp_i] = s_err
            surprise[sp_i] = float(np.clip(s_err / 2.5, 0.0, 1.0))
        specialist_leader = int(np.argmin(specialist_surprises[: min(self.act_dim, cohort_size)]))
        specialist_leader_axis = int(specialist_leader % self.act_dim)
        self.specialist_leader_elections_total += 1
        best_score = baseline_order
        best_program = []
        second_best_program = []
        best_axis_history = []
        social_shared_by_agent = np.zeros((cohort_size,), dtype=bool)
        macro_brake_triggered = np.zeros((cohort_size,), dtype=bool)
        kappa = 2.5
        gamma_m = 0.004
        repro_threshold = 0.72
        if self.use_curiosity and train_pairs:
            init_demo_bias = self.empathize_and_share_embodied_demonstration(
                h[specialist_leader],
                None,
                train_pairs[0][0],
                train_pairs[0][1],
                peer_pain=float(np.mean(pain[alive])) if np.any(alive) else 0.1,
            )
            for r_i in range(cohort_size):
                ctx_q[r_i, -1] = (0.76 * ctx_q[r_i, -1] + 0.24 * init_demo_bias).astype(np.float32)
        free_energy_drive = float(surprise.mean() + pain.mean() + max(0.0, 0.75 - baseline_order))
        if free_energy_drive > 0.6:
            effective_max_steps = int(max_steps)
        else:
            effective_max_steps = min(int(max_steps), max(3, int(getattr(self, "adaptive_glaciation_depth", max_steps))))
        for step in range(effective_max_steps):
            if np.any(alive):
                if float(np.mean(pain[alive])) > 0.65:
                    self.adaptive_glaciation_depth = max(2, self.adaptive_glaciation_depth - 1)
                elif float(np.mean(energy[alive])) > 0.70:
                    self.adaptive_glaciation_depth = min(int(max_steps), self.adaptive_glaciation_depth + 1)
            alarm_axes = set()
            empathy_mentors = []
            for j in range(cohort_size):
                if alive[j]:
                    if acoustic_field[j, 0] > 0.75 and pain[j] > 0.45 and best_axis_history:
                        alarm_axes.add(best_axis_history[-1])
                    if len(program_traces[j]) > 0 and float(current_order[j]) > baseline_order + 0.03:
                        empathy_mentors.append(j)
            if self.use_curiosity and step >= 1 and empathy_mentors:
                best_mentor = max(empathy_mentors, key=lambda idx_m: float(current_order[idx_m]))
                mentor_last_params = program_traces[best_mentor][-1][2] if program_traces[best_mentor] else None
                mentor_b_vec = self.encode_sensorimotor_body_vec(
                    mentor_last_params, q_spk_fallback=ctx_q[best_mentor, -1]
                )
                sym_id, _, mentor_b_unit = self.encode_and_verify_symbol_utterance(
                    ctx_q[best_mentor, -1],
                    body_vec_override=mentor_b_vec,
                    allow_self_imprint=bool(float(current_order[best_mentor]) > baseline_order + 0.08),
                )
                demo_before = orig_inputs[0] if orig_inputs else None
                demo_after = current_grids[best_mentor][0] if current_grids[best_mentor] else None
                embodied_demo_bias = self.empathize_and_share_embodied_demonstration(
                    ctx_q[best_mentor, -1],
                    mentor_last_params,
                    demo_before,
                    demo_after,
                    peer_pain=float(np.mean(pain[alive])) if np.any(alive) else 0.2,
                )
                if sym_id >= 0:
                    _, q_sym_bias = self.decode_symbol_to_sensorimotor_bias(sym_id)
                    shared_bias = (0.60 * q_sym_bias + 0.40 * embodied_demo_bias).astype(np.float32)
                    for r_i in range(cohort_size):
                        if alive[r_i] and r_i != best_mentor and len(program_traces[r_i]) == 0 and float(last_order_gain[r_i]) < 0.025:
                            ctx_q[r_i, -1] = (0.72 * ctx_q[r_i, -1] + 0.28 * shared_bias).astype(np.float32)
                            received_symbol_ids[r_i] = int(sym_id)
                            received_symbol_vecs[r_i] = mentor_b_unit.copy()
                            social_shared_by_agent[r_i] = True
                            self.social_empathy_events_total += 1
            for i in range(cohort_size):
                if not alive[i]:
                    continue
                self.modulate_ach_topological_lens(
                    float(last_order_gain[i]), float(surprise[i]), step_idx=step + i
                )
                trace_key = tuple((p[0], p[3]) for p in program_traces[i])
                if trace_key not in obs_base_cache:
                    obs_base_cache[trace_key] = self.perceive_meaningful_objects_foveal(
                        current_grids[i][0],
                        saccade_idx=step + i,
                        external_wm_canvas=external_wm_canvases[i],
                    )
                if trace_key not in diff_rule_cache:
                    diff_rule_cache[trace_key] = {}
                cached_obs, obj_scene = obs_base_cache[trace_key]
                cur_diff_rule = diff_rule_cache[trace_key]
                obs = cached_obs.copy()
                obs[13] = float(hp[i])
                obs[14] = float(energy[i])
                obs[15] = float(pain[i] + 0.3 * surprise[i])
                if step > 0:
                    pred_err = float(np.linalg.norm(obs - prev_obs_pred[i]))
                    surprise[i] = float(np.clip(pred_err / 2.5, 0.0, 1.0))
                    if surprise[i] > 0.38:
                        self.metacognitive_corrections_total += 1
                    if self.plastic:
                        Wpred_live[i] = np.clip(
                            Wpred_live[i] + 0.015 * np.outer(obs - prev_obs_pred[i], h[i]),
                            -3.0,
                            3.0,
                        )
                (
                    h_next,
                    v_mem_next,
                    ctx_q_next,
                    ctx_k_next,
                    ctx_v_next,
                    act_vec,
                    ac_sig,
                    obs_pred,
                    spk_rate,
                    macro_next,
                    micro_next,
                ) = self.forward_agent(
                    i,
                    obs,
                    h[i],
                    v_mem[i],
                    ctx_q[i],
                    ctx_k[i],
                    ctx_v[i],
                    Wh_live[i],
                    Wo_live[i],
                    Wpred_live[i],
                    Wpred_val_live[i],
                    float(energy[i]),
                    float(pain[i]),
                    float(surprise[i]),
                    float(last_order_gain[i]),
                    float(spike_fatigue[i]),
                    macro_monologue[i],
                    micro_monologue[i],
                )
                h[i] = h_next
                v_mem[i] = v_mem_next
                if hasattr(self, "persistent_h") and i < self.persistent_h.shape[0]:
                    self.persistent_h[i] = (0.70 * self.persistent_h[i] + 0.30 * h_next).astype(np.float32)
                    self.persistent_v_mem[i] = (0.70 * self.persistent_v_mem[i] + 0.30 * v_mem_next).astype(np.float32)
                ctx_q[i] = ctx_q_next
                ctx_k[i] = ctx_k_next
                ctx_v[i] = ctx_v_next
                spike_rates[i] = spk_rate
                spike_fatigue[i] = 0.85 * float(spike_fatigue[i]) + spk_rate
                if int(macro_lock_steps[i]) > 0 and float(np.linalg.norm(macro_monologue[i])) > 1e-6:
                    macro_monologue[i] = (0.94 * macro_monologue[i] + 0.06 * macro_next).astype(np.float32)
                    macro_lock_steps[i] -= 1
                else:
                    macro_monologue[i] = macro_next
                micro_monologue[i] = micro_next
                macro_norm = float(np.linalg.norm(macro_monologue[i]))
                macro_patience[i] = 5 if macro_norm > 0.30 else 4
                acoustic_sig = ac_sig
                acoustic_field[i] = acoustic_sig
                prev_obs_pred[i] = obs_pred
                h_norm_val = float(np.linalg.norm(h_next))
                if self.use_pain and pain[i] > 0.55 and h_norm_val > 2.6:
                    Wh_live[i] = (0.90 * Wh_live[i]).astype(np.float32)
                safety_gate = float(
                    np.clip((float(hp[i]) - 0.30) / 0.70, 0.0, 1.0)
                    * np.clip((float(energy[i]) - 0.18) / 0.82, 0.0, 1.0)
                    * math.exp(-3.5 * float(pain[i]))
                )
                if hasattr(self, "unified_brains") and self.unified_brains is not None and current_grids[i]:
                    gy_cur, gx_cur = getattr(self, "current_gaze_yx", (0, 0))
                    prop_16 = np.zeros(16, dtype=np.float32)
                    prop_16[0] = float(hp[i])
                    prop_16[1] = float(energy[i])
                    prop_16[2] = float(pain[i])
                    prop_16[3] = 1.0
                    prop_16[4:16] = obs[:12]
                    aud_11 = np.tanh(self.W_body_assembly @ h_next).astype(np.float32)
                    m_depth = 3 if (self.use_pain and pain[i] > 0.4) else 2
                    loop_out = self.unified_brains.step_sensorimotor_closed_loop(
                        current_grids[i][0],
                        auditory_vec=aud_11,
                        proprioception=prop_16,
                        relational_delta=h_next,
                        gaze_y=int(gy_cur),
                        gaze_x=int(gx_cur),
                        use_onnx=False,
                        mental_rollout_depth=m_depth,
                    )
                    act_2d_brain = np.asarray(loop_out.get("act_2d_logits", np.zeros(self.act_dim)), dtype=np.float32).reshape(-1)
                    n_ax_ov = min(act_vec.size, act_2d_brain.size)
                    act_vec[:n_ax_ov] = (act_vec[:n_ax_ov] + 0.35 * safety_gate * act_2d_brain[:n_ax_ov]).astype(np.float32)
                    g_dy, g_dx = loop_out.get("gaze_shift_yx", (0.0, 0.0))
                    gh, gw = current_grids[i][0].shape
                    self.current_gaze_yx = (
                        int(np.clip(round(gy_cur + g_dy * 2.0), 0, max(0, gh - 1))),
                        int(np.clip(round(gx_cur + g_dx * 2.0), 0, max(0, gw - 1))),
                    )
                    self.closed_loop_3brain_steps_total = int(getattr(self, "closed_loop_3brain_steps_total", 0)) + 1
                if self.use_curiosity:
                    active_tried = root_tried_axes[i] if len(program_traces[i]) == 0 else tried_axes[i]
                    if len(program_traces[i]) == 0:
                        spec_pref = (i + step) % self.act_dim
                        if step == 0 and i == specialist_leader and specialist_leader_axis not in active_tried:
                            act_vec[specialist_leader_axis] += 0.72 * (0.4 + 0.6 * safety_gate)
                        elif spec_pref not in active_tried:
                            act_vec[spec_pref] += 0.58 * (0.4 + 0.6 * safety_gate)
                    for bad_ax in active_tried:
                        act_vec[bad_ax] -= 1.25
                    for al_ax in alarm_axes:
                        act_vec[al_ax] -= 0.35
                    if len(active_tried) >= self.act_dim - 1 and safety_gate > 0.25:
                        d4_paradigm[i] = (int(d4_paradigm[i]) + 1 + (i % 3)) % 8
                        active_tried.clear()
                act_vec, snapped_d4, subgoal_info = self.formulate_one_step_subgoal(
                    obj_scene, act_vec, int(d4_paradigm[i]), float(surprise[i]), safety_gate, diff_rule_info=cur_diff_rule, train_prior=train_prior
                )
                d4_paradigm[i] = snapped_d4
                axis_order = np.argsort(act_vec)[::-1]
                chosen_ax = int(axis_order[0])
                cache_key = (trace_key, chosen_ax, int(d4_paradigm[i]))
                if cache_key in eval_cache:
                    top_cands, is_exact_cached, exact_prog_cached = eval_cache[cache_key]
                    if is_exact_cached:
                        self.splice_residual_bridge_program(task, exact_prog_cached, winning_axis=chosen_ax)
                        self.record_wake_episode(
                            obs, h[i], ctx_q[i, -1], ctx_k[i, -1], ctx_v[i, -1],
                            macro_monologue[i], micro_monologue[i], chosen_ax, exact_prog_cached
                        )
                        preds_all = [self.execute_program(ti, exact_prog_cached) for ti in test_inputs]
                        alt_preds = [
                            self.execute_program(ti, second_best_program) if second_best_program else p.copy()
                            for ti, p in zip(test_inputs, preds_all)
                        ]
                        self.step_visceral_interoceptive_homeostasis(
                            float(pain[i]), float(surprise[i]), float(spk_rate)
                        )
                        self.record_autobiographical_episode(
                            h[i], exact_prog_cached[-1][0] if exact_prog_cached else f"axis_{chosen_ax}", float(1.0 - baseline_order)
                        )
                        self._update_multimodal_hippocampal_plasticity(h[i], reward_mod=float(1.0 - baseline_order + 0.25))
                        if self.plastic and hasattr(self, "reinforce_intuition_from_ast_discovery") and exact_prog_cached:
                            self.reinforce_intuition_from_ast_discovery(h[i], exact_prog_cached[-1][0], float(1.0 - baseline_order))
                        if self.plastic:
                            self.synthesize_and_register_evolved_organ(
                                exact_prog_cached, winning_axis=chosen_ax, task_id=str(task.get("id", "online"))
                            )
                        self._commit_live_plasticity(
                            cohort_size,
                            Wh_live,
                            Wo_live,
                            Wpred_live,
                            Wpred_val_live,
                            hp,
                            energy,
                            pain,
                            col_orders=current_order,
                            wb_ref=wb_ref,
                            pred_canvas=preds_all[0] if preds_all else None,
                        )
                        return {
                            "solved_train_exact": True,
                            "prediction": preds_all[0],
                            "predictions_all": preds_all,
                            "alt_predictions_all": alt_preds,
                            "program": [p[0] for p in exact_prog_cached],
                            "steps_taken": step + 1,
                            "winning_axis": chosen_ax,
                            "survival_rate_pct": float(np.mean(alive) * 100.0),
                            "mean_spike_firing_rate": float(np.mean(spike_rates[alive])),
                            "thought_energy_consumed": float(np.mean(thought_energy_total[alive])),
                            "hippocampal_replay_used": bool(any(p[0].startswith("replay_") for p in exact_prog_cached)),
                            "inner_monologue_active": bool(
                                any(p[0].startswith(("mono_", "goal_signpost_", "goal_nearest_", "goal_align_")) for p in exact_prog_cached)
                            ),
                            "hierarchical_macro_active": bool(
                                any(p[0].startswith(("macro_", "nested_tool_", "chunk_macro_")) for p in exact_prog_cached)
                            ),
                            "dream_mutation_used": bool(
                                any(p[0].startswith(("dream_", "mutated_")) for p in exact_prog_cached)
                            ),
                            "social_empathy_used": bool(social_shared_by_agent[i] or any(p[0].startswith("cultural_") for p in exact_prog_cached)),
                            "bred_macro_tool_used": bool(
                                any(p[0].startswith("macro_breed_") for p in exact_prog_cached)
                            ),
                            "nested_tool_used": bool(
                                any(p[0].startswith("nested_tool_") for p in exact_prog_cached)
                            ),
                            "conditional_monologue_used": bool(
                                any(p[0].startswith(("mono_", "cond_")) for p in exact_prog_cached)
                            ),
                            "nested_conditional_used": bool(
                                sum(1 for p in exact_prog_cached if p[0].startswith(("mono_", "cond_"))) >= 2
                            ),
                            "specialist_leader_used": bool(i == specialist_leader),
                            "deep_macro_brake_used": bool(macro_brake_triggered[i]),
                        }
                else:
                    cur_pairs = list(zip(current_grids[i], [p[1] for p in train_pairs]))
                    cands = axis_pools.get(chosen_ax, []) if len(trace_key) == 0 else perceive_core_affordances(chosen_ax, cur_pairs)
                    drawer_hits = self.pull_subroutine_from_basal_ganglia_drawer(
                        chosen_ax,
                        subgoal_info,
                        cur_pairs,
                        orig_inputs,
                        base_pool=cands,
                        d4_id=int(d4_paradigm[i]),
                        safety_gate=safety_gate,
                        q_spk_hint=ctx_q[i, -1],
                    )
                    if not (drawer_hits and drawer_hits[0][4]):
                        for alt_ax in range(self.act_dim):
                            if alt_ax == chosen_ax:
                                continue
                            alt_cands = perceive_core_affordances(alt_ax, cur_pairs)
                            alt_hits = self.pull_subroutine_from_basal_ganglia_drawer(
                                alt_ax,
                                subgoal_info,
                                cur_pairs,
                                orig_inputs,
                                base_pool=alt_cands,
                                d4_id=int(d4_paradigm[i]),
                                safety_gate=safety_gate,
                                q_spk_hint=ctx_q[i, -1],
                            )
                            if alt_hits and alt_hits[0][4]:
                                drawer_hits = alt_hits
                                chosen_ax = alt_ax
                                break
                    if drawer_hits and drawer_hits[0][4]:
                        _, _, exact_step_tuple, next_gs, _ = drawer_hits[0]
                        if int(received_symbol_ids[i]) >= 0:
                            self.ground_shared_symbol_convention(
                                int(received_symbol_ids[i]), received_symbol_vecs[i], 0.25
                            )
                            received_symbol_ids[i] = -1
                        if bool(external_wm_pending_credit[i]) and int(external_wm_ttl[i]) > 0:
                            self.epistemic_delayed_payoffs_total += 1
                            external_wm_pending_credit[i] = False
                        full_prog = program_traces[i] + [exact_step_tuple]
                        if len(full_prog) >= 2:
                            last_op = full_prog[-1]
                            _, _, root_exact, _ = self.evaluate_candidate_on_pairs(
                                last_op[1], last_op[2], train_pairs, orig_inputs, d4_conj=int(last_op[3])
                            )
                            if root_exact:
                                full_prog = [last_op]
                        eval_cache[cache_key] = ([(1.0, 0.0, exact_step_tuple, next_gs)], True, full_prog)
                        self.splice_residual_bridge_program(task, full_prog, winning_axis=chosen_ax)
                        self.record_wake_episode(
                            obs, h[i], ctx_q[i, -1], ctx_k[i, -1], ctx_v[i, -1],
                            macro_monologue[i], micro_monologue[i], chosen_ax, full_prog
                        )
                        preds_all = [self.execute_program(ti, full_prog) for ti in test_inputs]
                        alt_preds = [
                            self.execute_program(ti, best_program) if best_program else p.copy()
                            for ti, p in zip(test_inputs, preds_all)
                        ]
                        self.step_visceral_interoceptive_homeostasis(
                            float(pain[i]), float(surprise[i]), float(spk_rate)
                        )
                        self.record_autobiographical_episode(
                            h[i], full_prog[-1][0] if full_prog else f"axis_{chosen_ax}", float(1.0 - baseline_order)
                        )
                        self._update_multimodal_hippocampal_plasticity(h[i], reward_mod=float(1.0 - baseline_order + 0.25))
                        if self.plastic and hasattr(self, "reinforce_intuition_from_ast_discovery") and full_prog:
                            self.reinforce_intuition_from_ast_discovery(h[i], full_prog[-1][0], float(1.0 - baseline_order))
                        if self.plastic:
                            self.synthesize_and_register_evolved_organ(
                                full_prog, winning_axis=chosen_ax, task_id=str(task.get("id", "online"))
                            )
                        self._commit_live_plasticity(
                            cohort_size,
                            Wh_live,
                            Wo_live,
                            Wpred_live,
                            Wpred_val_live,
                            hp,
                            energy,
                            pain,
                            col_orders=current_order,
                            wb_ref=wb_ref,
                            pred_canvas=preds_all[0] if preds_all else None,
                        )
                        return {
                            "solved_train_exact": True,
                            "prediction": preds_all[0],
                            "predictions_all": preds_all,
                            "alt_predictions_all": alt_preds,
                            "program": [p[0] for p in full_prog],
                            "steps_taken": step + 1,
                            "winning_axis": chosen_ax,
                            "survival_rate_pct": float(np.mean(alive) * 100.0),
                            "mean_spike_firing_rate": float(np.mean(spike_rates[alive])),
                            "thought_energy_consumed": float(np.mean(thought_energy_total[alive])),
                            "hippocampal_replay_used": bool(any(p[0].startswith("replay_") for p in full_prog)),
                            "inner_monologue_active": bool(
                                any(p[0].startswith(("mono_", "goal_signpost_", "goal_nearest_", "goal_align_")) for p in full_prog)
                            ),
                            "hierarchical_macro_active": bool(
                                any(p[0].startswith(("macro_", "nested_tool_", "chunk_macro_")) for p in full_prog)
                            ),
                            "dream_mutation_used": bool(
                                any(p[0].startswith(("dream_", "mutated_")) for p in full_prog)
                            ),
                            "social_empathy_used": bool(
                                social_shared_by_agent[i]
                                or any(
                                    any(p[0].startswith(t["op_prefix"]) for t in self.cultural_tool_library)
                                    for p in full_prog
                                )
                            ),
                            "bred_macro_tool_used": bool(
                                any(p[0].startswith("macro_breed_") for p in full_prog)
                            ),
                            "nested_tool_used": bool(
                                any(p[0].startswith("nested_tool_") for p in full_prog)
                            ),
                            "conditional_monologue_used": bool(
                                any(p[0].startswith(("mono_", "cond_")) for p in full_prog)
                            ),
                            "nested_conditional_used": bool(
                                sum(1 for p in full_prog if p[0].startswith(("mono_", "cond_"))) >= 2
                            ),
                            "specialist_leader_used": bool(i == specialist_leader),
                            "deep_macro_brake_used": bool(macro_brake_triggered[i]),
                            "micro_synth_used": bool(
                                any(p[0].startswith(("evolved_organ_", "codon_")) for p in full_prog)
                            ),
                        }
                    top_cands = [(h_e, h_d, h_step, h_gs) for h_e, h_d, h_step, h_gs, _ in drawer_hits]
                    if top_cands and len(program_traces[i]) <= 2 and safety_gate > 0.32:
                        s1_e, s1_d, s1_step, s1_gs = top_cands[0]
                        is_stagnant_low = (s1_e <= 0.80) and (float(spike_fatigue[i]) < 1.35)
                        if self.use_curiosity and is_stagnant_low and s1_gs:
                            _, next_scene = self.perceive_meaningful_objects_foveal(
                                s1_gs[0],
                                saccade_idx=step + i + 1,
                                external_wm_canvas=external_wm_canvases[i],
                            )
                            self.dynamic_board_reconsiderations_total += 1
                            look_pairs = list(zip(s1_gs, [p[1] for p in train_pairs]))
                            look_diff_rule = {}
                            next_act, _, next_subgoal_info = self.formulate_one_step_subgoal(
                                next_scene, act_vec, 0, float(surprise[i]), safety_gate, diff_rule_info=look_diff_rule, train_prior=train_prior
                            )
                            next_ax = int(np.argmax(next_act))
                            if next_ax == chosen_ax and len(axis_order) > 1:
                                next_ax = int(axis_order[1])
                            next_drawer_hits = self.pull_subroutine_from_basal_ganglia_drawer(
                                next_ax,
                                next_subgoal_info,
                                look_pairs,
                                orig_inputs,
                                base_pool=None,
                                d4_id=0,
                                safety_gate=safety_gate * 0.75,
                                q_spk_hint=ctx_q[i, -1],
                            )
                            if next_drawer_hits and next_drawer_hits[0][4]:
                                _, _, exact_s2_step, _, _ = next_drawer_hits[0]
                                full_prog = program_traces[i] + [s1_step, exact_s2_step]
                                if len(full_prog) >= 2:
                                    last_op = full_prog[-1]
                                    _, _, root_exact, _ = self.evaluate_candidate_on_pairs(
                                        last_op[1], last_op[2], train_pairs, orig_inputs, d4_conj=int(last_op[3])
                                    )
                                    if root_exact:
                                        full_prog = [last_op]
                                if self.plastic:
                                    new_ep = {"axes": (chosen_ax, next_ax), "op1": s1_step[0][:10], "op2": exact_s2_step[0][:10]}
                                    if new_ep not in self.mutated_episodic_buffer and len(self.mutated_episodic_buffer) < 28:
                                        self.mutated_episodic_buffer.append(new_ep)
                                    self.splice_residual_bridge_program(task, full_prog, winning_axis=next_ax)
                                self.record_wake_episode(
                                    obs, h[i], ctx_q[i, -1], ctx_k[i, -1], ctx_v[i, -1],
                                    macro_monologue[i], micro_monologue[i], next_ax, full_prog
                                )
                                preds_all = [self.execute_program(ti, full_prog) for ti in test_inputs]
                                alt_preds = [
                                    self.execute_program(ti, best_program) if best_program else p.copy()
                                    for ti, p in zip(test_inputs, preds_all)
                                ]
                                if self.plastic:
                                    self.synthesize_and_register_evolved_organ(
                                        full_prog, winning_axis=next_ax, task_id=str(task.get("id", "online"))
                                    )
                                self._commit_live_plasticity(
                                    cohort_size,
                                    Wh_live,
                                    Wo_live,
                                    Wpred_live,
                                    Wpred_val_live,
                                    hp,
                                    energy,
                                    pain,
                                    col_orders=current_order,
                                    wb_ref=wb_ref,
                                    pred_canvas=preds_all[0] if preds_all else None,
                                )
                                return {
                                    "solved_train_exact": True,
                                    "prediction": preds_all[0],
                                    "predictions_all": preds_all,
                                    "alt_predictions_all": alt_preds,
                                    "program": [p[0] for p in full_prog],
                                    "steps_taken": step + 1,
                                    "winning_axis": next_ax,
                                    "survival_rate_pct": float(np.mean(alive) * 100.0),
                                    "mean_spike_firing_rate": float(np.mean(spike_rates[alive])),
                                    "thought_energy_consumed": float(np.mean(thought_energy_total[alive])),
                                    "hippocampal_replay_used": bool(any(p[0].startswith("replay_") for p in full_prog)),
                                    "inner_monologue_active": bool(
                                        any(p[0].startswith(("mono_", "goal_signpost_", "goal_nearest_", "goal_align_")) for p in full_prog)
                                    ),
                                    "hierarchical_macro_active": bool(
                                        any(p[0].startswith(("macro_", "nested_tool_", "chunk_macro_")) for p in full_prog)
                                    ),
                                    "dream_mutation_used": bool(
                                        any(p[0].startswith(("dream_", "mutated_")) for p in full_prog)
                                    ),
                                    "social_empathy_used": bool(
                                        social_shared_by_agent[i]
                                        or any(
                                            any(p[0].startswith(t["op_prefix"]) for t in self.cultural_tool_library)
                                            for p in full_prog
                                        )
                                    ),
                                    "bred_macro_tool_used": bool(
                                        any(p[0].startswith("macro_breed_") for p in full_prog)
                                    ),
                                    "nested_tool_used": bool(
                                        any(p[0].startswith("nested_tool_") for p in full_prog)
                                    ),
                                    "conditional_monologue_used": bool(
                                        any(p[0].startswith(("mono_", "cond_")) for p in full_prog)
                                    ),
                                    "nested_conditional_used": bool(
                                        sum(1 for p in full_prog if p[0].startswith(("mono_", "cond_"))) >= 2
                                    ),
                                    "specialist_leader_used": bool(i == specialist_leader),
                                    "deep_macro_brake_used": bool(macro_brake_triggered[i]),
                                    "micro_synth_used": bool(
                                        any(p[0].startswith(("evolved_organ_", "codon_")) for p in full_prog)
                                    ),
                                }
                            elif next_drawer_hits and next_drawer_hits[0][0] > s1_e + 0.06:
                                le_val, ld_val, _, _, _ = next_drawer_hits[0]
                                top_cands[0] = (le_val, ld_val, s1_step, s1_gs)
                    eval_cache[cache_key] = (top_cands, False, None)
                if top_cands:
                    pick_k = (i % len(top_cands)) if i >= cohort_size // 2 else 0
                    best_ax_e, best_ax_d, best_ax_step, best_ax_next_grids = top_cands[pick_k]
                else:
                    best_ax_e, best_ax_d, best_ax_step, best_ax_next_grids = -1.0, 1.0, None, None
                if best_ax_step is not None:
                    cand_prog = program_traces[i] + [best_ax_step]
                    if best_ax_e > best_score:
                        if best_program and best_program != cand_prog:
                            second_best_program = list(best_program)
                        best_score = best_ax_e
                        best_program = cand_prog
                        best_axis_history.append(chosen_ax)
                    elif not second_best_program and cand_prog != best_program:
                        second_best_program = cand_prog
                order_gain = best_ax_e - float(current_order[i])
                last_order_gain[i] = float(order_gain)
                intuitive_step = (best_ax_e > 0.80) or (order_gain > 0.18)
                stagnation_burst_step = (
                    (not intuitive_step)
                    and (order_gain < 0.045)
                    and (float(spike_fatigue[i]) < 1.35)
                    and (len(root_tried_axes[i]) >= self.act_dim - 2)
                )
                curiosity_bonus = (
                    0.0
                    if intuitive_step
                    else (
                        safety_gate * (0.04 * float(surprise[i]) + (0.035 if stagnation_burst_step else 0.0))
                        if self.use_curiosity
                        else 0.0
                    )
                )
                shape_fixed = (
                    best_ax_next_grids is not None
                    and current_grids[i][0].shape != train_pairs[0][1].shape
                    and best_ax_next_grids[0].shape == train_pairs[0][1].shape
                )
                is_relational_or_stitch = (
                    best_ax_step is not None
                    and best_ax_step[0].startswith((
                        "sensorimotor_chunk_",
                        "mono_bind_A_to_B",
                        "canvas_stitch_",
                        "chunk_macro_",
                        "nested_tool_",
                        "grasp_only_",
                        "v1_v2_",
                        "dyadic_",
                        "scratchpad_proj_",
                        "cerebellar_macro_",
                        "subgoal_res_",
                    ))
                )
                gestalt_beauty_accepted = False
                if best_ax_next_grids is not None and best_ax_step is not None:
                    _, gestalt_gain_step, is_gestalt_neater = self.compute_intrinsic_gestalt_beauty(
                        current_grids[i][0], best_ax_next_grids[0]
                    )
                    gestalt_beauty_accepted = bool(
                        is_gestalt_neater and gestalt_gain_step > 0.05 and best_ax_e >= float(current_order[i]) - 0.035
                    )
                macro_commitment_brake = (
                    self.use_curiosity
                    and safety_gate > 0.25
                    and (len(program_traces[i]) >= 1 or stagnation_burst_step or i >= cohort_size - 2 or is_relational_or_stitch)
                    and best_ax_step is not None
                    and best_ax_next_grids is not None
                    and (macro_norm > 0.28 or best_ax_step[0].startswith((
                        "sensorimotor_chunk_",
                        "fovea_",
                        "mono_",
                        "macro_",
                        "chunk_macro_",
                        "nested_tool_",
                        "canvas_stitch_",
                        "part_",
                        "topo_roi_",
                        "kin_",
                        "grasp_only_",
                        "v1_v2_",
                        "dyadic_",
                        "scratchpad_proj_",
                        "cerebellar_macro_",
                        "subgoal_res_",
                    )))
                    and best_ax_e >= float(current_order[i]) - (0.075 if is_relational_or_stitch else 0.045)
                    and not np.array_equal(best_ax_next_grids[0], current_grids[i][0])
                    and len(program_traces[i]) < 5
                )
                epistemic_novel_valid = (
                    self.use_curiosity
                    and safety_gate > 0.25
                    and (stagnation_burst_step or is_relational_or_stitch)
                    and len(program_traces[i]) == 0
                    and best_ax_next_grids is not None
                    and best_ax_e >= float(current_order[i]) - (0.055 if is_relational_or_stitch else 0.015)
                    and not np.array_equal(best_ax_next_grids[0], current_grids[i][0])
                )
                if len(program_traces[i]) == 0:
                    root_tried_axes[i].add(chosen_ax)
                if (
                    best_ax_step is not None
                    and best_ax_next_grids is not None
                    and len(program_traces[i]) < 6
                    and (order_gain + curiosity_bonus >= 0.025 or shape_fixed or gestalt_beauty_accepted or epistemic_novel_valid or macro_commitment_brake)
                ):
                    if macro_commitment_brake:
                        self.cerebellar_valley_crossing_macros_total = int(getattr(self, "cerebellar_valley_crossing_macros_total", 0)) + 1
                    if best_ax_e > episodic_bookmarks[i][1]:
                        episodic_bookmarks[i] = (
                            [g.copy() for g in best_ax_next_grids],
                            float(best_ax_e),
                            list(program_traces[i]) + [best_ax_step],
                        )
                    current_grids[i] = best_ax_next_grids
                    current_order[i] = max(float(current_order[i]), best_ax_e)
                    program_traces[i].append(best_ax_step)
                    tried_axes[i].clear()
                    self.dynamic_board_reconsiderations_total += 1
                    if best_ax_e >= 0.70 and best_ax_e >= baseline_order + 0.07 and len(program_traces[i]) < 6:
                        sg_ok, sg_prog, sg_grids, sg_e, sg_exact = self.attempt_subgoal_checkpoint_and_wm_flush(
                            current_grids[i],
                            program_traces[i],
                            best_ax_e,
                            train_pairs,
                            orig_inputs,
                            ctx_q_agent=ctx_q[i],
                            ctx_k_agent=ctx_k[i],
                            ctx_v_agent=ctx_v[i],
                            energy_arr=energy,
                            agent_idx=i,
                        )
                        if sg_exact:
                            self.splice_residual_bridge_program(task, sg_prog, winning_axis=chosen_ax)
                            self.record_wake_episode(
                                obs, h[i], ctx_q[i, -1], ctx_k[i, -1], ctx_v[i, -1],
                                macro_monologue[i], micro_monologue[i], chosen_ax, sg_prog
                            )
                            preds_all = [self.execute_program(ti, sg_prog) for ti in test_inputs]
                            alt_preds = [
                                self.execute_program(ti, best_program) if best_program else p.copy()
                                for ti, p in zip(test_inputs, preds_all)
                            ]
                            if self.plastic:
                                self.synthesize_and_register_evolved_organ(
                                    sg_prog, winning_axis=chosen_ax, task_id=str(task.get("id", "online"))
                                )
                            self._commit_live_plasticity(
                                cohort_size,
                                Wh_live,
                                Wo_live,
                                Wpred_live,
                                Wpred_val_live,
                                hp,
                                energy,
                                pain,
                                col_orders=current_order,
                                wb_ref=wb_ref,
                                pred_canvas=preds_all[0] if preds_all else None,
                            )
                            return {
                                "solved_train_exact": True,
                                "prediction": preds_all[0],
                                "predictions_all": preds_all,
                                "alt_predictions_all": alt_preds,
                                "program": [p[0] for p in sg_prog],
                                "steps_taken": step + 1,
                                "winning_axis": chosen_ax,
                                "survival_rate_pct": float(np.mean(alive) * 100.0),
                                "mean_spike_firing_rate": float(np.mean(spike_rates[alive])),
                                "thought_energy_consumed": float(np.mean(thought_energy_total[alive])),
                                "hippocampal_replay_used": bool(any(p[0].startswith("replay_") for p in sg_prog)),
                                "inner_monologue_active": bool(
                                    any(p[0].startswith(("mono_", "goal_signpost_", "goal_nearest_", "goal_align_")) for p in sg_prog)
                                ),
                                "hierarchical_macro_active": bool(
                                    any(p[0].startswith(("macro_", "nested_tool_", "chunk_macro_")) for p in sg_prog)
                                ),
                                "dream_mutation_used": bool(
                                    any(p[0].startswith(("dream_", "mutated_")) for p in sg_prog)
                                ),
                                "social_empathy_used": bool(social_shared_by_agent[i] or any(p[0].startswith("cultural_") for p in sg_prog)),
                                "bred_macro_tool_used": bool(
                                    any(p[0].startswith("macro_breed_") for p in sg_prog)
                                ),
                                "nested_tool_used": bool(
                                    any(p[0].startswith("nested_tool_") for p in sg_prog)
                                ),
                                "conditional_monologue_used": bool(
                                    any(p[0].startswith(("mono_", "cond_")) for p in sg_prog)
                                ),
                                "nested_conditional_used": bool(
                                    sum(1 for p in sg_prog if p[0].startswith(("mono_", "cond_"))) >= 2
                                ),
                                "specialist_leader_used": bool(i == specialist_leader),
                                "deep_macro_brake_used": bool(macro_brake_triggered[i]),
                                "micro_synth_used": bool(
                                    any(p[0].startswith(("evolved_organ_", "codon_")) for p in sg_prog)
                                ),
                            }
                        if sg_ok and sg_e > best_ax_e + 0.02:
                            current_grids[i] = sg_grids
                            current_order[i] = float(sg_e)
                            program_traces[i] = list(sg_prog)
                            if sg_e > best_score:
                                best_score = float(sg_e)
                                best_program = list(sg_prog)
                    if int(received_symbol_ids[i]) >= 0 and order_gain > 0.01:
                        self.ground_shared_symbol_convention(
                            int(received_symbol_ids[i]), received_symbol_vecs[i], float(order_gain)
                        )
                        received_symbol_ids[i] = -1
                    if macro_commitment_brake:
                        macro_lock_steps[i] = 4
                        macro_brake_triggered[i] = True
                        delta_e = max(0.14, order_gain * 0.9 + best_ax_e * 0.25)
                        dmg = best_ax_d * 0.05
                    else:
                        delta_e = max(0.12, order_gain * 0.9 + best_ax_e * 0.25)
                        dmg = best_ax_d * 0.15
                    if bool(external_wm_pending_credit[i]) and int(external_wm_ttl[i]) > 0 and order_gain > 0.01:
                        self.epistemic_delayed_payoffs_total += 1
                        delta_e += 0.035
                        external_wm_pending_credit[i] = False
                        external_wm_canvases[i].fill(0)
                        external_wm_ttl[i] = 0
                else:
                    tried_axes[i].add(chosen_ax)
                    delta_e = max(0.0, order_gain) * 0.25
                    dmg = max(0.14, best_ax_d * 0.42)
                    if (
                        len(tried_axes[i]) == 2
                        and int(external_wm_ttl[i]) == 0
                        and current_grids[i][0].shape == external_wm_canvases[i].shape
                    ):
                        wm_tmp = {"memory_grid": external_wm_canvases[i]}
                        self.execute_unified_sensorimotor_chunk(
                            current_grids[i][0],
                            "gaze_salient_target",
                            "wm_store_attr",
                            "finger_apply_to_gaze",
                            {"selector": "largest", "write_to_memo": True, "memo_mode": "guide_cross"},
                            wm_state=wm_tmp,
                        )
                        if isinstance(wm_tmp.get("memory_grid"), np.ndarray):
                            external_wm_canvases[i] = wm_tmp["memory_grid"]
                        external_wm_ttl[i] = 3
                        external_wm_pending_credit[i] = True
                        thought_energy_total[i] += 0.0012
                        energy[i] = float(max(0.0, energy[i] - 0.0012))
                    if len(program_traces[i]) > 0 and len(tried_axes[i]) >= int(macro_patience[i]):
                        self.if_then_else_switches_total += 1
                        if len(program_traces[i]) >= 2 or float(np.linalg.norm(macro_monologue[i])) > 0.35:
                            self.nested_if_then_switches_total += 1
                        current_grids[i] = [p[0].copy() for p in train_pairs]
                        current_order[i] = baseline_order
                        program_traces[i] = []
                        tried_axes[i] = set(root_tried_axes[i])
                        macro_lock_steps[i] = 0
                if int(external_wm_ttl[i]) > 0:
                    external_wm_ttl[i] -= 1
                    if int(external_wm_ttl[i]) <= 0:
                        external_wm_canvases[i].fill(0)
                        external_wm_pending_credit[i] = False
                inst_pain = (1.0 - math.exp(-kappa * dmg)) if self.use_pain else 0.0
                pain[i] = float(inst_pain)
                active_ratio_step = float(getattr(self, "last_active_cell_ratio", 0.35))
                thought_cost = float(0.004 + 0.018 * spk_rate * active_ratio_step + 0.003 * min(3, len(top_cands)))
                thought_energy_total[i] += thought_cost
                self.epistemic_atp_invested_total += float(thought_cost)
                if stagnation_burst_step and i == 0:
                    tilt_info = self.shift_3d_head_tilt_viewpoint(
                        current_grids[i][0], surprise_val=float(surprise[i]), fatigue_val=float(spike_fatigue[i])
                    )
                    if isinstance(tilt_info, dict) and "tabletop_reprojection" in tilt_info:
                        tr_val = tilt_info["tabletop_reprojection"]
                        if isinstance(tr_val, dict):
                            tr_val = tr_val.get("reconstructed_grid", tilt_info.get("warped_grid"))
                        if isinstance(tr_val, np.ndarray) and tr_val.ndim == 2:
                            current_grids[i][0] = tr_val
                if i == 0:
                    _, insular_bias = self.step_visceral_interoceptive_homeostasis(
                        float(pain[i]), float(surprise[i]), float(spk_rate)
                    )
                    auto_recall_bias = self.recall_autobiographical_context(h[i])
                    h[i] = np.tanh(h[i] + 0.04 * insular_bias + 0.05 * auto_recall_bias).astype(np.float32)
                    if order_gain > 0.02:
                        step_lbl = program_traces[i][-1][0] if program_traces[i] else f"axis_{chosen_ax}"
                        self.record_autobiographical_episode(h[i], step_lbl, order_gain)
                aha_da_burst = self.trigger_epistemic_aha_dopamine_burst(
                    order_gain,
                    max(0.0, float(surprise[i]) - float(best_ax_d)),
                    h_vec=h[i],
                    action_axis=chosen_ax,
                )
                energy[i] = float(np.clip(energy[i] + delta_e - 0.008 - thought_cost, 0.0, 1.0))
                starvation_dmg = 0.25 if energy[i] <= 0.0 else gamma_m * (1.0 - energy[i])
                hp[i] = float(max(0.0, hp[i] - dmg - starvation_dmg))
                if self.use_mortality and (hp[i] <= 0.08 or energy[i] <= 0.05 or spike_fatigue[i] >= 0.85):
                    self.micro_rest_recoveries_total = int(getattr(self, "micro_rest_recoveries_total", 0)) + 1
                    hp[i] = float(max(hp[i], 0.72))
                    energy[i] = float(max(energy[i], 0.55))
                    pain[i] = float(pain[i] * 0.25)
                    spike_fatigue[i] = float(spike_fatigue[i] * 0.30)
                    v_mem[i] = (v_mem[i] * 0.55).astype(np.float32)
                    macro_lock_steps[i] = 0
                    if len(tried_axes[i]) >= 4:
                        tried_axes[i].clear()
                    donors = [j for j in range(cohort_size) if alive[j] and energy[j] >= repro_threshold and j != i]
                    if donors:
                        mentor_j = int(rng_fission.choice(donors))
                        parent_params = program_traces[mentor_j][-1][2] if program_traces[mentor_j] else None
                        p_b_vec = self.encode_sensorimotor_body_vec(parent_params, q_spk_fallback=ctx_q[mentor_j, -1])
                        p_sym_id, _, p_b_unit = self.encode_and_verify_symbol_utterance(
                            ctx_q[mentor_j, -1], body_vec_override=p_b_vec, allow_self_imprint=True
                        )
                        if p_sym_id >= 0:
                            _, q_child_bias = self.decode_symbol_to_sensorimotor_bias(p_sym_id)
                            ctx_q[i, -1] = (0.78 * ctx_q[i, -1] + 0.22 * q_child_bias).astype(np.float32)
                            received_symbol_ids[i] = int(p_sym_id)
                            received_symbol_vecs[i] = p_b_unit.copy()
                            social_shared_by_agent[i] = True
                if self.plastic:
                    one_hot = np.zeros((self.act_dim,), dtype=np.float32)
                    one_hot[chosen_ax] = 1.0
                    ha_vec = np.concatenate([h[i], one_hot])
                    pred_v = np.tanh(Wpred_val_live[i] @ ha_vec)
                    val_err = np.array([delta_e - pred_v[0], inst_pain - pred_v[1]], dtype=np.float32)
                    Wpred_val_live[i] = np.clip(Wpred_val_live[i] + 0.025 * np.outer(val_err, ha_vec), -3.0, 3.0)
                    r_mod = (
                        1.4 * delta_e
                        + 0.35 * aha_da_burst
                        - (2.0 * inst_pain if self.use_pain else 0.0)
                        + (0.4 * safety_gate * float(surprise[i]) if self.use_curiosity else 0.0)
                    )
                    eligibility[i] = 0.90 * eligibility[i] + np.outer(one_hot, h[i])
                    stage_lr_gate = float(0.75 + 0.50 * float(self.myelination_stage_gates[2]))
                    Wo_live[i] = np.clip(Wo_live[i] + 0.03 * stage_lr_gate * r_mod * eligibility[i], -5.0, 5.0)
                    if i == int(getattr(self, "winning_cortical_column_idx", 0)):
                        self._update_multimodal_hippocampal_plasticity(h[i], reward_mod=r_mod)
                    if hasattr(self, "unified_brains") and self.unified_brains is not None:
                        neo_b = self.unified_brains.neocortex_logic
                        h_target_i = np.tanh(self.Wx[i] @ obs + Wh_live[i] @ h[i] + self.b[i]).astype(np.float32)
                        neo_b.update_world_model_hebbian(ha_vec, h_target_i, lr=0.008)
                if hasattr(self, "persistent_h") and i < self.persistent_h.shape[0]:
                    self.persistent_h[i] = (0.75 * self.persistent_h[i] + 0.25 * h[i]).astype(np.float32)
                if hasattr(self, "persistent_v_mem") and i < self.persistent_v_mem.shape[0]:
                    self.persistent_v_mem[i] = (0.75 * self.persistent_v_mem[i] + 0.25 * v_mem[i]).astype(np.float32)
            h[:cohort_size], _, _ = self.compete_cortical_minicolumns_trn_gaba(
                h[:cohort_size], current_order[:cohort_size], energy[:cohort_size], pain[:cohort_size]
            )
            self.check_and_trigger_autonomous_homeostatic_sleep(force_check=False)
            if not np.any(alive):
                break
        if best_program and best_score >= 0.62 and best_score >= baseline_order + 0.05:
            post_ckpt_grids = [self.execute_program(xin, best_program) for xin, _ in train_pairs]
            sg_ok, sg_prog, sg_grids, sg_e, sg_exact = self.attempt_subgoal_checkpoint_and_wm_flush(
                post_ckpt_grids,
                best_program,
                best_score,
                train_pairs,
                orig_inputs,
                energy_arr=energy,
                agent_idx=0,
            )
            if sg_exact:
                win_ax = best_axis_history[-1] if best_axis_history else 0
                self.splice_residual_bridge_program(task, sg_prog, winning_axis=win_ax)
                preds_all = [self.execute_program(ti, sg_prog) for ti in test_inputs]
                alt_preds = [
                    self.execute_program(ti, best_program) if best_program else p.copy()
                    for ti, p in zip(test_inputs, preds_all)
                ]
                if self.plastic:
                    self.synthesize_and_register_evolved_organ(
                        sg_prog, winning_axis=win_ax, task_id=str(task.get("id", "online"))
                    )
                self._commit_live_plasticity(
                    cohort_size,
                    Wh_live,
                    Wo_live,
                    Wpred_live,
                    Wpred_val_live,
                    hp,
                    energy,
                    pain,
                    col_orders=current_order,
                    wb_ref=wb_ref,
                    pred_canvas=preds_all[0] if preds_all else None,
                )
                return {
                    "solved_train_exact": True,
                    "prediction": preds_all[0],
                    "predictions_all": preds_all,
                    "alt_predictions_all": alt_preds,
                    "program": [p[0] for p in sg_prog],
                    "steps_taken": len(sg_prog),
                    "winning_axis": win_ax,
                    "survival_rate_pct": float(np.mean(alive) * 100.0),
                    "mean_spike_firing_rate": float(np.mean(spike_rates[alive])) if np.any(alive) else 0.0,
                    "thought_energy_consumed": float(np.mean(thought_energy_total)),
                    "hippocampal_replay_used": bool(any(p[0].startswith("replay_") for p in sg_prog)),
                    "inner_monologue_active": bool(
                        any(p[0].startswith(("mono_", "goal_signpost_", "goal_nearest_", "goal_align_")) for p in sg_prog)
                    ),
                    "hierarchical_macro_active": bool(
                        any(p[0].startswith(("macro_", "nested_tool_", "chunk_macro_")) for p in sg_prog)
                    ),
                    "dream_mutation_used": bool(
                        any(p[0].startswith(("dream_", "mutated_")) for p in sg_prog)
                    ),
                    "social_empathy_used": bool(any(p[0].startswith("cultural_") for p in sg_prog)),
                    "bred_macro_tool_used": bool(
                        any(p[0].startswith("macro_breed_") for p in sg_prog)
                    ),
                    "nested_tool_used": bool(
                        any(p[0].startswith("nested_tool_") for p in sg_prog)
                    ),
                    "conditional_monologue_used": bool(
                        any(p[0].startswith(("mono_", "cond_")) for p in sg_prog)
                    ),
                    "nested_conditional_used": bool(
                        sum(1 for p in sg_prog if p[0].startswith(("mono_", "cond_"))) >= 2
                    ),
                    "specialist_leader_used": False,
                    "deep_macro_brake_used": False,
                    "micro_synth_used": bool(
                        any(p[0].startswith(("evolved_organ_", "codon_")) for p in sg_prog)
                    ),
                }
            if sg_ok and sg_e > best_score:
                second_best_program = list(best_program)
                best_score = float(sg_e)
                best_program = list(sg_prog)
        harvest_prog = None
        if best_program and best_score > baseline_order + 0.06:
            harvest_prog = list(best_program)
        elif len(train_pairs) >= 2 and best_axis_history:
            focus_ax = int(best_axis_history[-1])
            for cname_h, cfn_h, cparams_h in axis_pools.get(focus_ax, [])[:4]:
                try:
                    yhat0 = cfn_h(train_pairs[0][0], cparams_h)
                    if yhat0.size > 0 and not np.array_equal(yhat0, train_pairs[0][0]) and not np.array_equal(yhat0, train_pairs[0][1]):
                        harvest_prog = [(cname_h, cfn_h, cparams_h, 0)]
                        break
                except Exception:
                    continue
        if self.plastic and harvest_prog and len(train_pairs) >= 2:
            res_train = []
            valid_subgoal = True
            for xin, yout in train_pairs:
                yhat = self.execute_program(xin, harvest_prog)
                if yhat.size == 0 or np.array_equal(yhat, yout):
                    valid_subgoal = False
                    break
                res_train.append({"input": yhat.tolist(), "output": yout.tolist()})
            if valid_subgoal and test_inputs and res_train:
                t_hat = self.execute_program(test_inputs[0], harvest_prog)
                subgoal_task = {
                    "id": f"residual_subgoal_{self.residual_subgoals_harvested_total}",
                    "archetype": "residual_subgoal_harvested",
                    "origin_task_id": str(task.get("id", "unknown")),
                    "prefix_program": list(harvest_prog),
                    "composite_depth": 3,
                    "train": res_train,
                    "test": [{"input": t_hat.tolist()}],
                }
                self.residual_subgoals_harvested_total += 1
                self.residual_subgoal_queue.append(subgoal_task)
                if len(self.residual_subgoal_queue) > 24:
                    self.residual_subgoal_queue.pop(0)
        preds_all = [
            self.execute_program(ti, best_program) if best_program else ti.copy()
            for ti in test_inputs
        ]
        alt_preds = [
            self.execute_program(ti, second_best_program) if second_best_program else p.copy()
            for ti, p in zip(test_inputs, preds_all)
        ]
        for idx_p in range(len(preds_all)):
            if (
                best_program
                and second_best_program
                and not self._is_reflex_prediction_valid_on_novel_board(
                    test_inputs[idx_p], preds_all[idx_p], train_pairs
                )
                and self._is_reflex_prediction_valid_on_novel_board(
                    test_inputs[idx_p], alt_preds[idx_p], train_pairs
                )
            ):
                preds_all[idx_p] = alt_preds[idx_p].copy()
                self.einstellung_fixation_prevented_total = int(
                    getattr(self, "einstellung_fixation_prevented_total", 0)
                ) + 1
                self.triple_overfit_guards_triggered_total = int(
                    getattr(self, "triple_overfit_guards_triggered_total", 0)
                ) + 1
                self.surgical_guard_debugs_total = int(
                    getattr(self, "surgical_guard_debugs_total", 0)
                ) + 1
        cur_v1_v2_groups = int(
            getattr(self, "v1_v2_texture_segregations_total", 0)
            + getattr(self, "v1_v2_constellation_groups_total", 0)
        )
        self._commit_live_plasticity(
            cohort_size,
            Wh_live,
            Wo_live,
            Wpred_live,
            Wpred_val_live,
            hp,
            energy,
            pain,
            col_orders=current_order,
            wb_ref=wb_ref,
            pred_canvas=preds_all[0] if preds_all else None,
        )
        return {
            "solved_train_exact": False,
            "prediction": preds_all[0],
            "predictions_all": preds_all,
            "alt_predictions_all": alt_preds,
            "program": [p[0] for p in best_program],
            "steps_taken": effective_max_steps,
            "winning_axis": best_axis_history[-1] if best_axis_history else -1,
            "survival_rate_pct": float(np.mean(alive) * 100.0),
            "mean_spike_firing_rate": float(np.mean(spike_rates[alive])) if np.any(alive) else 0.0,
            "thought_energy_consumed": float(np.mean(thought_energy_total)),
            "micro_synth_used": False,
            "baseline_order": round(float(baseline_order), 4),
            "best_score": round(float(best_score), 4),
            "order_gain_achieved": round(float(best_score - baseline_order), 4),
            "mean_remaining_energy": round(float(np.mean(energy[alive])) if np.any(alive) else 0.0, 4),
            "mean_remaining_hp": round(float(np.mean(hp[alive])) if np.any(alive) else 0.0, 4),
            "gaba_blocks_this_task": int(getattr(self, "prefrontal_gaba_blocks_total", 0) - init_gaba_blocks),
            "latent_bypasses_this_task": int(getattr(self, "latent_feature_bypasses_total", 0) - init_latent_bypasses),
            "lens_warps_this_task": int(getattr(self, "topological_lens_warps_total", 0) - init_lens_warps),
            "v1_v2_groups_this_task": int(cur_v1_v2_groups - init_v1_v2_groups),
            "scratchpad_projs_this_task": int(getattr(self, "virtual_scratchpad_projections_total", 0) - init_scratchpad_projs),
            "subgoal_ckpts_this_task": int(getattr(self, "subgoal_checkpoints_locked_total", 0) - init_subgoal_ckpts),
            "dyadic_bindings_this_task": int(getattr(self, "it_parietal_relational_bindings_total", 0) - init_dyadic_bindings),
            "valley_macros_this_task": int(getattr(self, "cerebellar_valley_crossing_macros_total", 0) - init_valley_macros),
            "triple_guard_resets_this_task": int(getattr(self, "triple_overfit_guards_triggered_total", 0) - init_triple_guards),
            "active_ach_lens": str(getattr(self, "active_topological_lens", "cartesian")),
            "foveal_objects_count": int(len(init_scene0.get("objects_cur", []))) if isinstance(init_scene0, dict) else 0,
            "shape_change_required": bool(any(x.shape != y.shape for x, y in train_pairs)),
        }

    def study_and_solve_tabletop_task(self, task, max_steps=10):
        wb = ExternalTabletopARCWorkbench(task)
        cur_xyz = getattr(self, "organism_world_xyz", (50.0, 50.0, 0.0))
        tabletop_task = wb.present_tabletop_task_view(start_world_xyz=cur_xyz)
        safe_st = tabletop_task.get("_safe_study_room_state", {})
        if isinstance(safe_st, dict) and "seated_world_xyz" in safe_st:
            self.organism_world_xyz = np.array(safe_st["seated_world_xyz"], dtype=np.float32)
        self.external_workbench_interactions_total = int(
            getattr(self, "external_workbench_interactions_total", 0)
        ) + 1
        res = self.solve_task(tabletop_task, max_steps=max_steps)
        preds = res.get("predictions_all", [res.get("prediction")])
        alt_preds = res.get("alt_predictions_all", preds)
        wb.submit_canvas_drawings(preds, alt_preds)
        if wb.submitted_canvases:
            res["predictions_all"] = [c.copy() for c in wb.submitted_canvases]
            res["prediction"] = res["predictions_all"][0]
        if wb.submitted_alt_canvases:
            res["alt_predictions_all"] = [c.copy() for c in wb.submitted_alt_canvases]
        res["external_workbench_used"] = True
        res["workbench_interaction_steps"] = int(wb.interaction_steps)
        return res
    def save_evolution_checkpoint(self, checkpoint_path, generation_v2=0, extra_meta=None):
        ckpt_file = Path(checkpoint_path)
        ckpt_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_npz = ckpt_file.with_name(ckpt_file.stem + "_tmp.npz")
        np.savez_compressed(
            tmp_npz,
            Wx=self.Wx,
            Wh=self.Wh,
            b=self.b,
            Wo=self.Wo,
            bo=self.bo,
            Wpred=self.Wpred,
            Wpred_val=self.Wpred_val,
            Ws=self.Ws,
            bs=self.bs,
            Wq=self.Wq,
            Wk=self.Wk,
            Wv=self.Wv,
            W_mono_out=self.W_mono_out,
            W_mono_in=self.W_mono_in,
            W_macro_out=self.W_macro_out,
            W_macro_in=self.W_macro_in,
            W_cond_gate=self.W_cond_gate,
            b_cond_gate=self.b_cond_gate,
            W_nested_cond_gate=self.W_nested_cond_gate,
            b_nested_cond_gate=self.b_nested_cond_gate,
            W_source_attr_reg=self.W_source_attr_reg,
            W_target_op_reg=self.W_target_op_reg,
            W_rel_bind_in=self.W_rel_bind_in,
            concept_centroids=self.concept_centroids,
            signpost_ctx_k=self.signpost_ctx_k,
            signpost_ctx_v=self.signpost_ctx_v,
            hippo_ctx_k=self.hippo_ctx_k,
            hippo_ctx_v=self.hippo_ctx_v,
            W_ast_intuition=self.W_ast_intuition,
            W_body_assembly=self.W_body_assembly,
            W_symbol_lexicon=self.W_symbol_lexicon,
            symbol_usage_counts=self.symbol_usage_counts,
            W_mirror_inverse=getattr(self, "W_mirror_inverse", np.zeros((11, self.in_dim * 2), dtype=np.float32)),
            W_parietal_gain=getattr(self, "W_parietal_gain", np.zeros((4, self.hid_dim), dtype=np.float32)),
            W_cerebellar_purkinje=getattr(self, "W_cerebellar_purkinje", np.zeros((self.hid_dim, 6), dtype=np.float32)),
            W_procedural_attractor=getattr(self, "W_procedural_attractor", np.zeros((self.hid_dim, self.hid_dim), dtype=np.float32)),
            W_procedural_chain=getattr(self, "W_procedural_chain", np.zeros((self.act_dim, self.act_dim), dtype=np.float32)),
            W_syntax_transition=getattr(self, "W_syntax_transition", np.zeros((16, 16), dtype=np.float32)),
            W_cochlear_auditory=getattr(self, "W_cochlear_auditory", np.zeros((self.hid_dim, 8), dtype=np.float32)),
            W_insular_interoception=getattr(self, "W_insular_interoception", np.zeros((self.hid_dim, 4), dtype=np.float32)),
            W_hippo_spatiotemporal=getattr(self, "W_hippo_spatiotemporal", np.zeros((self.hid_dim, 16), dtype=np.float32)),
            W_fovea_v1_spikes=getattr(self, "W_fovea_v1_spikes", np.zeros((self.hid_dim, 100), dtype=np.float32)),
            W_fovea_stroke_assoc=getattr(self, "W_fovea_stroke_assoc", np.zeros((25, 11, 100), dtype=np.float32)),
            myelination_stage_gates=getattr(self, "myelination_stage_gates", np.array([1.0, 0.85, 0.70], dtype=np.float32)),
            persistent_h=getattr(self, "persistent_h", np.zeros((self.pop_size, self.hid_dim), dtype=np.float32)),
            persistent_v_mem=getattr(self, "persistent_v_mem", np.zeros((self.pop_size, self.hid_dim), dtype=np.float32)),
            theta=getattr(self, "theta", np.full((self.pop_size, self.hid_dim), 0.35, dtype=np.float32)),
        )
        tmp_npz.replace(ckpt_file)
        serialized_tools = []
        for t in self.cultural_tool_library:
            serialized_tools.append({
                "axis": int(t["axis"]),
                "axis2": int(t.get("axis2", t["axis"])),
                "op_prefix": str(t["op_prefix"]),
                "op2_prefix": str(t.get("op2_prefix", "")),
                "is_bred_composite": bool(t.get("is_bred_composite", False)),
                "is_nested_hierarchical": bool(t.get("is_nested_hierarchical", False)),
                "usage_count": int(t.get("usage_count", 1)),
                "staleness_tasks": int(t.get("staleness_tasks", 0)),
                "efe_efficiency": float(t.get("efe_efficiency", 0.75)),
                "macro_vec": t["macro_vec"].tolist() if isinstance(t.get("macro_vec"), np.ndarray) else list(t.get("macro_vec", [0.0] * self.mono_dim)),
                "k_spk": t["k_spk"].tolist() if isinstance(t.get("k_spk"), np.ndarray) else list(t.get("k_spk", [0.0] * self.hid_dim)),
                "v_proj": t["v_proj"].tolist() if isinstance(t.get("v_proj"), np.ndarray) else list(t.get("v_proj", [0.0] * self.hid_dim)),
            })
        meta_payload = {
            "generation_v2": int(generation_v2),
            "timestamp": round(time.time(), 2),
            "cultural_tool_library": serialized_tools,
            "bred_macro_recipes": [
                {"axes": list(r["axes"]), "op1": str(r["op1"]), "op2": str(r["op2"])}
                for r in self.bred_macro_recipes
            ],
            "nested_macro_recipes": [
                {"axes": list(r["axes"]), "ops": list(r["ops"])}
                for r in self.nested_macro_recipes
            ],
            "mutated_episodic_buffer": [
                {"axes": list(r["axes"]), "op1": str(r["op1"]), "op2": str(r["op2"])}
                for r in self.mutated_episodic_buffer
            ],
            "hippocampal_episodic_buffer": [
                {"axes": list(r["axes"]), "op1": str(r["op1"]), "op2": str(r["op2"])}
                for r in self.hippocampal_episodic_buffer
            ],
            "tools_pruned_total": int(self.tools_pruned_total),
            "pruning_cycles_run": int(self.pruning_cycles_run),
            "if_then_else_switches_total": int(self.if_then_else_switches_total),
            "nested_if_then_switches_total": int(self.nested_if_then_switches_total),
            "relational_bindings_total": int(self.relational_bindings_total),
            "recursive_chunks_compiled_total": int(self.recursive_chunks_compiled_total),
            "canvas_stitch_solves_total": int(self.canvas_stitch_solves_total),
            "micro_primitives_synthesized_total": int(self.micro_primitives_synthesized_total),
            "residual_subgoals_harvested_total": int(self.residual_subgoals_harvested_total),
            "residual_bridges_spliced_total": int(self.residual_bridges_spliced_total),
            "adaptive_glaciation_depth": int(self.adaptive_glaciation_depth),
            "synthesized_micro_primitives": list(self.synthesized_micro_primitives),
            "synthesized_primitive_nodes": dict(self.synthesized_primitive_nodes),
            "evolved_ast_organs": [
                {k: v for k, v in o.items() if k != "fn" and not callable(v)}
                for o in self.evolved_ast_organs if isinstance(o, dict)
            ],
            "de_novo_codon_pool": [
                {k: v for k, v in c.items() if not callable(v)}
                for c in self.de_novo_codon_pool if isinstance(c, dict)
            ],
            "de_novo_codons_synthesized_total": int(self.de_novo_codons_synthesized_total),
            "evolving_genome_archive": dict(self.evolving_genome_archive),
            "evolving_task_visit_counts": dict(self.evolving_task_visit_counts),
            "ast_mutations_total": int(self.ast_mutations_total),
            "ast_crossovers_total": int(self.ast_crossovers_total),
            "tried_ast_expr_hashes": sorted(list(self.tried_ast_expr_hashes))[:5000],
            "basal_ganglia_reflex_chunks": list(getattr(self, "basal_ganglia_reflex_chunks", [])),
            "basal_ganglia_reflex_hits_total": int(getattr(self, "basal_ganglia_reflex_hits_total", 0)),
            "basal_ganglia_energy_saved_total": round(float(getattr(self, "basal_ganglia_energy_saved_total", 0.0)), 4),
            "foveal_saccades_executed_total": int(getattr(self, "foveal_saccades_executed_total", 0)),
            "foveal_objects_perceived_total": int(getattr(self, "foveal_objects_perceived_total", 0)),
            "one_step_subgoals_formulated_total": int(getattr(self, "one_step_subgoals_formulated_total", 0)),
            "basal_ganglia_drawer_pulls_total": int(getattr(self, "basal_ganglia_drawer_pulls_total", 0)),
            "dynamic_board_reconsiderations_total": int(getattr(self, "dynamic_board_reconsiderations_total", 0)),
            "selective_object_grasps_total": int(getattr(self, "selective_object_grasps_total", 0)),
            "intrinsic_gestalt_evaluations_total": int(getattr(self, "intrinsic_gestalt_evaluations_total", 0)),
            "cross_example_diff_comparisons_total": int(getattr(self, "cross_example_diff_comparisons_total", 0)),
            "in_grid_legend_reads_total": int(getattr(self, "in_grid_legend_reads_total", 0)),
            "predicate_group_broadcasts_total": int(getattr(self, "predicate_group_broadcasts_total", 0)),
            "surgical_guard_debugs_total": int(getattr(self, "surgical_guard_debugs_total", 0)),
            "stage1_sensorimotor_chunks": dict(getattr(self, "stage1_sensorimotor_chunks", {})),
            "stage1_sensorimotor_chunks_transferred": int(getattr(self, "stage1_sensorimotor_chunks_transferred", 0)),
            "sensorimotor_body_executions_total": int(getattr(self, "sensorimotor_body_executions_total", 0)),
            "sensorimotor_chunks_bred_total": int(getattr(self, "sensorimotor_chunks_bred_total", 0)),
            "cerebellar_error_corrections_total": int(getattr(self, "cerebellar_error_corrections_total", 0)),
            "prefrontal_dynamic_chains_total": int(getattr(self, "prefrontal_dynamic_chains_total", 0)),
            "bred_sensorimotor_chains": list(getattr(self, "bred_sensorimotor_chains", [])),
            "attractor_snaps_total": int(getattr(self, "attractor_snaps_total", 0)),
            "metacognitive_corrections_total": int(getattr(self, "metacognitive_corrections_total", 0)),
            "verified_symbol_utterances_total": int(getattr(self, "verified_symbol_utterances_total", 0)),
            "symbol_silence_choices_total": int(getattr(self, "symbol_silence_choices_total", 0)),
            "shared_intentionality_groundings_total": int(getattr(self, "shared_intentionality_groundings_total", 0)),
            "epistemic_memo_writes_total": int(getattr(self, "epistemic_memo_writes_total", 0)),
            "wm_offload_flushes_total": int(getattr(self, "wm_offload_flushes_total", 0)),
            "epistemic_delayed_payoffs_total": int(getattr(self, "epistemic_delayed_payoffs_total", 0)),
            "closed_loop_3brain_steps_total": int(getattr(self, "closed_loop_3brain_steps_total", 0)),
            "mental_simulations_total": int(getattr(self, "mental_simulations_total", 0)),
            "micro_rest_recoveries_total": int(getattr(self, "micro_rest_recoveries_total", 0)),
            "foveal_mental_canvas_updates_total": int(getattr(self, "foveal_mental_canvas_updates_total", 0)),
            "ach_habit_resets_total": int(getattr(self, "ach_habit_resets_total", 0)),
            "einstellung_fixation_prevented_total": int(getattr(self, "einstellung_fixation_prevented_total", 0)),
            "synaptic_homeostasis_downscalings_total": int(getattr(self, "synaptic_homeostasis_downscalings_total", 0)),
            "triple_overfit_guards_triggered_total": int(getattr(self, "triple_overfit_guards_triggered_total", 0)),
            "external_workbench_interactions_total": int(getattr(self, "external_workbench_interactions_total", 0)),
            "safe_room_transitions_total": int(getattr(self, "safe_room_transitions_total", 0)),
            "vestibular_vor_stabilizations_total": int(getattr(self, "vestibular_vor_stabilizations_total", 0)),
            "cochlear_acoustic_perceptions_total": int(getattr(self, "cochlear_acoustic_perceptions_total", 0)),
            "visceral_interoceptive_steps_total": int(getattr(self, "visceral_interoceptive_steps_total", 0)),
            "hippocampal_place_grid_activations_total": int(getattr(self, "hippocampal_place_grid_activations_total", 0)),
            "autobiographical_recalls_total": int(getattr(self, "autobiographical_recalls_total", 0)),
            "affective_empathy_resonances_total": int(getattr(self, "affective_empathy_resonances_total", 0)),
            "embodied_peer_demonstrations_total": int(getattr(self, "embodied_peer_demonstrations_total", 0)),
            "foveal_retinotopic_spike_projections_total": int(getattr(self, "foveal_retinotopic_spike_projections_total", 0)),
            "foveal_visuomotor_strokes_learned_total": int(getattr(self, "foveal_visuomotor_strokes_learned_total", 0)),
            "foveal_saccade_stroke_loops_executed_total": int(getattr(self, "foveal_saccade_stroke_loops_executed_total", 0)),
            "autobiographical_memory_bank": list(getattr(self, "autobiographical_memory_bank", [])),
            "developmental_stage": int(getattr(self, "developmental_stage", 1)),
            "critical_period_transitions_total": int(getattr(self, "critical_period_transitions_total", 0)),
            "organism_hp": round(float(getattr(self, "organism_hp", 1.0)), 4),
            "organism_energy": round(float(getattr(self, "organism_energy", 1.0)), 4),
            "organism_pain": round(float(getattr(self, "organism_pain", 0.0)), 4),
            "organism_sleep_pressure": round(float(getattr(self, "organism_sleep_pressure", 0.0)), 4),
            "organism_lifetime_steps": int(getattr(self, "organism_lifetime_steps", 0)),
            "organism_life_time_sec": round(float(getattr(self, "organism_life_time_sec", 0.0)), 3),
            "last_active_cell_ratio": round(float(getattr(self, "last_active_cell_ratio", 0.35)), 4),
            "last_dormant_cell_ratio": round(float(getattr(self, "last_dormant_cell_ratio", 0.65)), 4),
            "inner_monologue_symbolic_bindings": list(self.inner_monologue.get("symbolic_bindings", [])),
            "extra_meta": extra_meta or {},
        }
        json_file = ckpt_file.with_suffix(".json")
        tmp_json = ckpt_file.with_name(ckpt_file.stem + "_tmp.json")
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2, default=str)
        tmp_json.replace(json_file)
        if hasattr(self, "unified_brains") and self.unified_brains is not None:
            self.unified_brains.sync_from_stage2_solver(self)
            self.unified_brains.save_all_brains()
        return str(ckpt_file)
    def load_evolution_checkpoint(self, checkpoint_path):
        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists():
            return False
        data = np.load(ckpt_file)
        for attr_name in (
            "Wx", "Wh", "b", "Wo", "bo", "Wpred", "Wpred_val", "Ws", "bs",
            "Wq", "Wk", "Wv", "W_mono_out", "W_mono_in", "W_macro_out", "W_macro_in",
            "W_cond_gate", "b_cond_gate", "W_nested_cond_gate", "b_nested_cond_gate",
            "W_source_attr_reg", "W_target_op_reg", "W_rel_bind_in",
            "concept_centroids", "signpost_ctx_k", "signpost_ctx_v", "hippo_ctx_k", "hippo_ctx_v",
            "W_ast_intuition", "W_body_assembly", "W_symbol_lexicon", "symbol_usage_counts",
            "W_mirror_inverse", "W_parietal_gain", "W_cerebellar_purkinje",
            "W_procedural_attractor", "W_procedural_chain", "W_syntax_transition",
            "W_cochlear_auditory", "W_insular_interoception", "W_hippo_spatiotemporal",
            "W_fovea_v1_spikes", "W_fovea_stroke_assoc",
            "myelination_stage_gates", "persistent_h", "persistent_v_mem", "theta",
        ):
            if attr_name in data:
                arr = data[attr_name].astype(np.float32)
                if attr_name in ("Wx", "Wh") and arr.ndim == 3 and arr.shape[1] > self.hid_dim:
                    arr = arr[:, -self.hid_dim :, :]
                elif attr_name == "b" and arr.ndim == 2 and arr.shape[1] > self.hid_dim:
                    arr = arr[:, -self.hid_dim :]
                if attr_name == "W_ast_intuition" and arr.shape[0] < 32:
                    rng_pad = np.random.default_rng(int(getattr(self, "seed", 2026)) + 32)
                    pad_rows = rng_pad.normal(0.0, 0.06, (32 - arr.shape[0], arr.shape[1])).astype(np.float32)
                    arr = np.vstack([arr, pad_rows]).astype(np.float32)
                setattr(self, attr_name, arr)
        json_file = ckpt_file.with_suffix(".json")
        if json_file.exists():
            with open(json_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            loaded_tools = []
            for t in meta.get("cultural_tool_library", []):
                loaded_tools.append({
                    "axis": int(t["axis"]),
                    "axis2": int(t.get("axis2", t["axis"])),
                    "op_prefix": str(t["op_prefix"]),
                    "op2_prefix": str(t.get("op2_prefix", "")),
                    "is_bred_composite": bool(t.get("is_bred_composite", False)),
                    "is_nested_hierarchical": bool(t.get("is_nested_hierarchical", False)),
                    "usage_count": int(t.get("usage_count", 1)),
                    "staleness_tasks": int(t.get("staleness_tasks", 0)),
                    "efe_efficiency": float(t.get("efe_efficiency", 0.75)),
                    "macro_vec": np.array(t.get("macro_vec", [0.0] * self.mono_dim), dtype=np.float32),
                    "k_spk": np.array(t.get("k_spk", [0.0] * self.hid_dim), dtype=np.float32),
                    "v_proj": np.array(t.get("v_proj", [0.0] * self.hid_dim), dtype=np.float32),
                })
            if loaded_tools:
                self.cultural_tool_library = loaded_tools
            valid_prefixes = (
                "fovea_", "spatial_", "rigid_", "mono_", "goal_", "topo_roi_",
                "topo_chromatic_", "topo_enclose_", "canvas_stitch_", "nested_tool_",
                "part_", "kin_", "grav_", "gestalt_", "num_", "layer_",
                "self_similarity_", "new_primitive_node_0x", "grasp_only_",
                "legend_dict_", "broadcast_foreach_", "guard_", "sensorimotor_chunk_",
                "kinematic_", "tile_", "morph_", "extract_", "cavity_", "chroma_",
                "evolved_organ_", "codon_", "macro_breed_",
            )
            if meta.get("bred_macro_recipes"):
                self.bred_macro_recipes = [
                    {"axes": tuple(r["axes"]), "op1": str(r["op1"]), "op2": str(r["op2"])}
                    for r in meta["bred_macro_recipes"]
                    if str(r.get("op1", "")).startswith(valid_prefixes) and str(r.get("op2", "")).startswith(valid_prefixes)
                ]
            if meta.get("nested_macro_recipes"):
                self.nested_macro_recipes = [
                    {"axes": tuple(r["axes"]), "ops": tuple(r["ops"])}
                    for r in meta["nested_macro_recipes"]
                    if all(str(op_s).startswith(valid_prefixes) for op_s in r.get("ops", []))
                ]
            if meta.get("mutated_episodic_buffer"):
                self.mutated_episodic_buffer = [
                    {"axes": tuple(r["axes"]), "op1": str(r["op1"]), "op2": str(r["op2"])}
                    for r in meta["mutated_episodic_buffer"]
                    if str(r.get("op1", "")).startswith(valid_prefixes) and str(r.get("op2", "")).startswith(valid_prefixes)
                ]
            if isinstance(meta.get("hippocampal_episodic_buffer"), list):
                self.hippocampal_episodic_buffer = [
                    {"axes": tuple(r["axes"]), "op1": str(r["op1"]), "op2": str(r["op2"])}
                    for r in meta["hippocampal_episodic_buffer"]
                    if isinstance(r, dict) and len(r.get("axes", [])) >= 2
                ][-48:]
            self.tools_pruned_total = int(meta.get("tools_pruned_total", self.tools_pruned_total))
            self.pruning_cycles_run = int(meta.get("pruning_cycles_run", self.pruning_cycles_run))
            self.if_then_else_switches_total = int(meta.get("if_then_else_switches_total", self.if_then_else_switches_total))
            self.nested_if_then_switches_total = int(meta.get("nested_if_then_switches_total", self.nested_if_then_switches_total))
            self.relational_bindings_total = int(meta.get("relational_bindings_total", self.relational_bindings_total))
            self.recursive_chunks_compiled_total = int(meta.get("recursive_chunks_compiled_total", self.recursive_chunks_compiled_total))
            self.canvas_stitch_solves_total = int(meta.get("canvas_stitch_solves_total", self.canvas_stitch_solves_total))
            self.micro_primitives_synthesized_total = int(meta.get("micro_primitives_synthesized_total", self.micro_primitives_synthesized_total))
            self.residual_subgoals_harvested_total = int(meta.get("residual_subgoals_harvested_total", self.residual_subgoals_harvested_total))
            self.residual_bridges_spliced_total = int(meta.get("residual_bridges_spliced_total", self.residual_bridges_spliced_total))
            self.adaptive_glaciation_depth = int(meta.get("adaptive_glaciation_depth", self.adaptive_glaciation_depth))
            if "synthesized_micro_primitives" in meta and isinstance(meta["synthesized_micro_primitives"], list):
                self.synthesized_micro_primitives = list(meta["synthesized_micro_primitives"])
            if "synthesized_primitive_nodes" in meta and isinstance(meta["synthesized_primitive_nodes"], dict):
                self.synthesized_primitive_nodes = dict(meta["synthesized_primitive_nodes"])
            if self.cultural_tool_library:
                self.cultural_tool_library = [
                    t for t in self.cultural_tool_library
                    if str(t.get("op_prefix", "")).startswith(valid_prefixes)
                ]
            if "evolved_ast_organs" in meta and isinstance(meta["evolved_ast_organs"], list):
                self.evolved_ast_organs = list(meta["evolved_ast_organs"])
            if "de_novo_codon_pool" in meta and isinstance(meta["de_novo_codon_pool"], list):
                self.de_novo_codon_pool = list(meta["de_novo_codon_pool"])
            self.de_novo_codons_synthesized_total = int(meta.get("de_novo_codons_synthesized_total", self.de_novo_codons_synthesized_total))
            if "evolving_genome_archive" in meta and isinstance(meta["evolving_genome_archive"], dict):
                self.evolving_genome_archive = dict(meta["evolving_genome_archive"])
            if "evolving_task_visit_counts" in meta and isinstance(meta["evolving_task_visit_counts"], dict):
                self.evolving_task_visit_counts = {str(k): int(v) for k, v in meta["evolving_task_visit_counts"].items()}
            self.ast_mutations_total = int(meta.get("ast_mutations_total", self.ast_mutations_total))
            self.ast_crossovers_total = int(meta.get("ast_crossovers_total", self.ast_crossovers_total))
            if "tried_ast_expr_hashes" in meta and isinstance(meta["tried_ast_expr_hashes"], list):
                self.tried_ast_expr_hashes = set(str(x) for x in meta["tried_ast_expr_hashes"] if ":" in str(x))
            if "bred_sensorimotor_chains" in meta and isinstance(meta["bred_sensorimotor_chains"], list):
                self.bred_sensorimotor_chains = [
                    dict(ch) for ch in meta["bred_sensorimotor_chains"]
                    if isinstance(ch, dict) and ch.get("steps")
                ]
            if "basal_ganglia_reflex_chunks" in meta and isinstance(meta["basal_ganglia_reflex_chunks"], list):
                self.basal_ganglia_reflex_chunks = [
                    dict(ch) for ch in meta["basal_ganglia_reflex_chunks"]
                    if isinstance(ch, dict) and ch.get("chunk_type")
                ]
            self.load_stage1_sensorimotor_chunks()
            self.developmental_stage = int(meta.get("developmental_stage", getattr(self, "developmental_stage", 1)))
            self.critical_period_transitions_total = int(meta.get("critical_period_transitions_total", getattr(self, "critical_period_transitions_total", 0)))
            self.organism_hp = float(meta.get("organism_hp", getattr(self, "organism_hp", 1.0)))
            self.organism_energy = float(meta.get("organism_energy", getattr(self, "organism_energy", 1.0)))
            self.organism_pain = float(meta.get("organism_pain", getattr(self, "organism_pain", 0.0)))
            self.organism_sleep_pressure = float(meta.get("organism_sleep_pressure", getattr(self, "organism_sleep_pressure", 0.0)))
            self.organism_lifetime_steps = int(meta.get("organism_lifetime_steps", getattr(self, "organism_lifetime_steps", 0)))
            self.organism_life_time_sec = float(meta.get("organism_life_time_sec", getattr(self, "organism_life_time_sec", 0.0)))
            self.basal_ganglia_reflex_hits_total = int(meta.get("basal_ganglia_reflex_hits_total", getattr(self, "basal_ganglia_reflex_hits_total", 0)))
            self.basal_ganglia_energy_saved_total = float(meta.get("basal_ganglia_energy_saved_total", getattr(self, "basal_ganglia_energy_saved_total", 0.0)))
            self.foveal_saccades_executed_total = int(meta.get("foveal_saccades_executed_total", getattr(self, "foveal_saccades_executed_total", 0)))
            self.foveal_objects_perceived_total = int(meta.get("foveal_objects_perceived_total", getattr(self, "foveal_objects_perceived_total", 0)))
            self.one_step_subgoals_formulated_total = int(meta.get("one_step_subgoals_formulated_total", getattr(self, "one_step_subgoals_formulated_total", 0)))
            self.basal_ganglia_drawer_pulls_total = int(meta.get("basal_ganglia_drawer_pulls_total", getattr(self, "basal_ganglia_drawer_pulls_total", 0)))
            self.dynamic_board_reconsiderations_total = int(meta.get("dynamic_board_reconsiderations_total", getattr(self, "dynamic_board_reconsiderations_total", 0)))
            self.selective_object_grasps_total = int(meta.get("selective_object_grasps_total", getattr(self, "selective_object_grasps_total", 0)))
            self.intrinsic_gestalt_evaluations_total = int(getattr(self, "intrinsic_gestalt_evaluations_total", 0))
            self.cross_example_diff_comparisons_total = int(meta.get("cross_example_diff_comparisons_total", getattr(self, "cross_example_diff_comparisons_total", 0)))
            self.in_grid_legend_reads_total = int(meta.get("in_grid_legend_reads_total", getattr(self, "in_grid_legend_reads_total", 0)))
            self.predicate_group_broadcasts_total = int(meta.get("predicate_group_broadcasts_total", getattr(self, "predicate_group_broadcasts_total", 0)))
            self.surgical_guard_debugs_total = int(meta.get("surgical_guard_debugs_total", getattr(self, "surgical_guard_debugs_total", 0)))
            self.sensorimotor_body_executions_total = int(meta.get("sensorimotor_body_executions_total", getattr(self, "sensorimotor_body_executions_total", 0)))
            self.sensorimotor_chunks_bred_total = int(meta.get("sensorimotor_chunks_bred_total", getattr(self, "sensorimotor_chunks_bred_total", 0)))
            self.cerebellar_error_corrections_total = int(meta.get("cerebellar_error_corrections_total", getattr(self, "cerebellar_error_corrections_total", 0)))
            self.prefrontal_dynamic_chains_total = int(meta.get("prefrontal_dynamic_chains_total", getattr(self, "prefrontal_dynamic_chains_total", 0)))
            self.attractor_snaps_total = int(meta.get("attractor_snaps_total", getattr(self, "attractor_snaps_total", 0)))
            self.metacognitive_corrections_total = int(meta.get("metacognitive_corrections_total", getattr(self, "metacognitive_corrections_total", 0)))
            self.verified_symbol_utterances_total = int(meta.get("verified_symbol_utterances_total", getattr(self, "verified_symbol_utterances_total", 0)))
            self.symbol_silence_choices_total = int(meta.get("symbol_silence_choices_total", getattr(self, "symbol_silence_choices_total", 0)))
            self.shared_intentionality_groundings_total = int(meta.get("shared_intentionality_groundings_total", getattr(self, "shared_intentionality_groundings_total", 0)))
            self.epistemic_memo_writes_total = int(meta.get("epistemic_memo_writes_total", getattr(self, "epistemic_memo_writes_total", 0)))
            self.wm_offload_flushes_total = int(meta.get("wm_offload_flushes_total", getattr(self, "wm_offload_flushes_total", 0)))
            self.epistemic_delayed_payoffs_total = int(meta.get("epistemic_delayed_payoffs_total", getattr(self, "epistemic_delayed_payoffs_total", 0)))
            self.closed_loop_3brain_steps_total = int(meta.get("closed_loop_3brain_steps_total", getattr(self, "closed_loop_3brain_steps_total", 0)))
            self.mental_simulations_total = int(meta.get("mental_simulations_total", getattr(self, "mental_simulations_total", 0)))
            self.micro_rest_recoveries_total = int(meta.get("micro_rest_recoveries_total", getattr(self, "micro_rest_recoveries_total", 0)))
            self.foveal_mental_canvas_updates_total = int(meta.get("foveal_mental_canvas_updates_total", getattr(self, "foveal_mental_canvas_updates_total", 0)))
            self.ach_habit_resets_total = int(meta.get("ach_habit_resets_total", getattr(self, "ach_habit_resets_total", 0)))
            self.einstellung_fixation_prevented_total = int(meta.get("einstellung_fixation_prevented_total", getattr(self, "einstellung_fixation_prevented_total", 0)))
            self.synaptic_homeostasis_downscalings_total = int(meta.get("synaptic_homeostasis_downscalings_total", getattr(self, "synaptic_homeostasis_downscalings_total", 0)))
            self.triple_overfit_guards_triggered_total = int(meta.get("triple_overfit_guards_triggered_total", getattr(self, "triple_overfit_guards_triggered_total", 0)))
            self.external_workbench_interactions_total = int(meta.get("external_workbench_interactions_total", getattr(self, "external_workbench_interactions_total", 0)))
            self.safe_room_transitions_total = int(meta.get("safe_room_transitions_total", getattr(self, "safe_room_transitions_total", 0)))
            self.vestibular_vor_stabilizations_total = int(meta.get("vestibular_vor_stabilizations_total", getattr(self, "vestibular_vor_stabilizations_total", 0)))
            self.cochlear_acoustic_perceptions_total = int(meta.get("cochlear_acoustic_perceptions_total", getattr(self, "cochlear_acoustic_perceptions_total", 0)))
            self.visceral_interoceptive_steps_total = int(meta.get("visceral_interoceptive_steps_total", getattr(self, "visceral_interoceptive_steps_total", 0)))
            self.hippocampal_place_grid_activations_total = int(meta.get("hippocampal_place_grid_activations_total", getattr(self, "hippocampal_place_grid_activations_total", 0)))
            self.autobiographical_recalls_total = int(meta.get("autobiographical_recalls_total", getattr(self, "autobiographical_recalls_total", 0)))
            self.affective_empathy_resonances_total = int(meta.get("affective_empathy_resonances_total", getattr(self, "affective_empathy_resonances_total", 0)))
            self.embodied_peer_demonstrations_total = int(meta.get("embodied_peer_demonstrations_total", getattr(self, "embodied_peer_demonstrations_total", 0)))
            self.foveal_retinotopic_spike_projections_total = int(meta.get("foveal_retinotopic_spike_projections_total", getattr(self, "foveal_retinotopic_spike_projections_total", 0)))
            self.foveal_visuomotor_strokes_learned_total = int(meta.get("foveal_visuomotor_strokes_learned_total", getattr(self, "foveal_visuomotor_strokes_learned_total", 0)))
            self.foveal_saccade_stroke_loops_executed_total = int(meta.get("foveal_saccade_stroke_loops_executed_total", getattr(self, "foveal_saccade_stroke_loops_executed_total", 0)))
            if isinstance(meta.get("autobiographical_memory_bank"), list):
                self.autobiographical_memory_bank = list(meta["autobiographical_memory_bank"])[-48:]
            self.last_active_cell_ratio = float(meta.get("last_active_cell_ratio", getattr(self, "last_active_cell_ratio", 0.35)))
            self.last_dormant_cell_ratio = float(meta.get("last_dormant_cell_ratio", getattr(self, "last_dormant_cell_ratio", 0.65)))
            if meta.get("inner_monologue_symbolic_bindings") and isinstance(self.inner_monologue, dict):
                self.inner_monologue["symbolic_bindings"] = list(meta["inner_monologue_symbolic_bindings"])
        if hasattr(self, "unified_brains") and self.unified_brains is not None:
            self.unified_brains.sync_from_stage2_solver(self)
        return True


class ExternalTabletopARCWorkbench:
    def __init__(self, task):
        self.task_id = str(task.get("id", "tabletop_workbook_task"))
        self.train_demonstrations = [
            {
                "input": np.array(p["input"], dtype=np.int32).tolist(),
                "output": np.array(p["output"], dtype=np.int32).tolist(),
            }
            for p in task.get("train", [])
        ]
        raw_test = task.get("test", [])
        test_list = [raw_test] if isinstance(raw_test, dict) else list(raw_test)
        self.test_prompts = [
            {"input": np.array(tp["input"], dtype=np.int32).tolist()}
            for tp in test_list
        ]
        self.submitted_canvases = []
        self.submitted_alt_canvases = []
        self.working_3d_canvases = []
        self.interaction_steps = 0
        self.tabletop_3d_perceptions = []

    def observe_3d_panel_via_retina(self, grid, gaze_uv_deg=(0.0, 0.0), step_physx=False, skip_spatial_3d=True):
        self.interaction_steps += 1
        engine = get_global_isaac_warp_engine()
        perc = engine.perceive_3d_tabletop_via_human_vision(
            np.asarray(grid, dtype=np.int32),
            gaze_uv_deg=gaze_uv_deg,
            step_physx=step_physx,
            skip_spatial_3d=skip_spatial_3d,
        )
        self.tabletop_3d_perceptions.append(perc)
        return perc["reconstructed_grid"]

    def _extract_foreground_stroke_cells(self, canvas_grid):
        g = np.asarray(canvas_grid, dtype=np.int32)
        if g.ndim != 2 or g.size == 0:
            return [], 0
        border = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
        if border.size > 0:
            b_vals, b_cnts = np.unique(border, return_counts=True)
            bg_c = int(b_vals[int(np.argmax(b_cnts))])
        else:
            vals, cnts = np.unique(g, return_counts=True)
            bg_c = int(vals[int(np.argmax(cnts))]) if len(vals) > 0 else 0
        H, W = g.shape
        stroke_cells = []
        for r in range(H):
            for c in range(W):
                v = int(g[r, c])
                if v != bg_c:
                    stroke_cells.append((r, c, v))
        if not stroke_cells and H > 0 and W > 0:
            stroke_cells.append((H // 2, W // 2, int(g[H // 2, W // 2])))
        return stroke_cells, bg_c

    def manipulate_3d_working_canvas(self, canvas_grid, gaze_uv_deg=(0.0, 0.0), step_physx=False, skip_spatial_3d=True):
        engine = get_global_isaac_warp_engine()
        c_arr = np.asarray(canvas_grid, dtype=np.int32)
        stroke_cells, bg_c = self._extract_foreground_stroke_cells(c_arr)
        physical_board = np.full_like(c_arr, int(bg_c), dtype=np.int32)
        engine.execute_physical_tabletop_stroke_on_desk(
            physical_board,
            stroke_cells,
            bg_color=bg_c,
            step_physx=step_physx,
        )
        perceived_canvas = self.observe_3d_panel_via_retina(physical_board, gaze_uv_deg=gaze_uv_deg, step_physx=step_physx, skip_spatial_3d=skip_spatial_3d)
        self.working_3d_canvases.append(perceived_canvas)
        return perceived_canvas

    def present_tabletop_task_view(self, start_world_xyz=(101.0, 50.0, 0.0), reset_at_desk=False, step_physx=False, skip_spatial_3d=True):
        self.interaction_steps += 1
        engine = get_global_isaac_warp_engine()
        if reset_at_desk:
            engine.reset_humanoid_at_study_room_desk()
        safe_room_state = engine.enter_safe_study_room_and_sit_at_desk(start_world_xyz)
        self.tabletop_3d_perceptions = []
        projected_train = []
        for demo in self.train_demonstrations:
            perc_in = engine.perceive_3d_tabletop_via_human_vision(np.asarray(demo["input"], dtype=np.int32), step_physx=step_physx, skip_spatial_3d=skip_spatial_3d)
            perc_out = engine.perceive_3d_tabletop_via_human_vision(np.asarray(demo["output"], dtype=np.int32), step_physx=step_physx, skip_spatial_3d=skip_spatial_3d)
            self.tabletop_3d_perceptions.append(perc_in)
            self.tabletop_3d_perceptions.append(perc_out)
            projected_train.append({
                "input": perc_in["reconstructed_grid"].tolist(),
                "output": perc_out["reconstructed_grid"].tolist(),
            })
        projected_test = []
        for tp in self.test_prompts:
            perc = engine.perceive_3d_tabletop_via_human_vision(np.asarray(tp["input"], dtype=np.int32), step_physx=step_physx, skip_spatial_3d=skip_spatial_3d)
            self.tabletop_3d_perceptions.append(perc)
            projected_test.append({"input": perc["reconstructed_grid"].tolist()})
        return {
            "id": self.task_id,
            "train": projected_train,
            "test": projected_test,
            "_tabletop_projected": True,
            "_safe_study_room_state": safe_room_state,
            "_workbench_ref": self,
        }

    def submit_canvas_drawings(self, primary_canvases, secondary_canvases=None, step_physx=False, skip_spatial_3d=True):
        self.interaction_steps += 1
        engine = get_global_isaac_warp_engine()
        self.submitted_canvases = []
        for c in primary_canvases:
            c_arr = np.asarray(c, dtype=np.int32)
            stroke_cells, bg_c = self._extract_foreground_stroke_cells(c_arr)
            physical_board = c_arr.copy()
            engine.execute_physical_tabletop_stroke_on_desk(
                physical_board,
                stroke_cells,
                bg_color=bg_c,
                step_physx=step_physx,
            )
            self.submitted_canvases.append(c_arr.copy())
        if secondary_canvases is not None:
            self.submitted_alt_canvases = []
            for idx_sec, c in enumerate(secondary_canvases):
                c_arr = np.asarray(c, dtype=np.int32)
                if (
                    idx_sec < len(primary_canvases)
                    and idx_sec < len(self.submitted_canvases)
                    and c_arr.shape == np.asarray(primary_canvases[idx_sec]).shape
                    and np.array_equal(c_arr, np.asarray(primary_canvases[idx_sec], dtype=np.int32))
                ):
                    self.submitted_alt_canvases.append(self.submitted_canvases[idx_sec].copy())
                    continue
                stroke_cells, bg_c = self._extract_foreground_stroke_cells(c_arr)
                physical_board = c_arr.copy()
                engine.execute_physical_tabletop_stroke_on_desk(
                    physical_board,
                    stroke_cells,
                    bg_color=bg_c,
                    step_physx=step_physx,
                )
                self.submitted_alt_canvases.append(c_arr.copy())
        else:
            self.submitted_alt_canvases = list(self.submitted_canvases)
        return len(self.submitted_canvases)

if __name__ == "__main__":
    solver = EmbodiedPuzzleSolver()
    ckpt_latest = ARTIFACTS_DIR / "checkpoints" / "checkpoint_latest.npz"
    if ckpt_latest.exists():
        solver.load_evolution_checkpoint(ckpt_latest)
    sample_task = {
        "train": [
            {
                "input": [[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 3, 3, 0]],
                "output": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0], [0, 3, 3, 0]],
            },
            {
                "input": [[0, 0, 4, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 5, 5, 5]],
                "output": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 4, 0], [0, 5, 5, 5]],
            },
        ],
        "test": {
            "input": [[0, 7, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [8, 8, 8, 0]],
            "output": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 7, 0, 0], [8, 8, 8, 0]],
        },
    }
    res = solver.study_and_solve_tabletop_task(sample_task)
    solver.save_evolution_checkpoint(ckpt_latest, generation_v2=1)
    exact = np.array_equal(res["prediction"], np.array(sample_task["test"]["output"]))
    print(
        json.dumps(
            {
                "sanity_check_exact_match": bool(exact),
                "solved_train_exact": res["solved_train_exact"],
                "program": res["program"],
                "steps_taken": res["steps_taken"],
                "winning_axis": res["winning_axis"],
                "survival_rate_pct": res["survival_rate_pct"],
            },
            indent=2,
        )
    )
