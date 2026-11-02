import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from isaac_warp_3d_engine import get_global_isaac_warp_engine

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX_RUNTIME = True
except Exception:
    HAS_ONNX_RUNTIME = False


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_UINT64 = 10
GGML_TYPE_F32 = 0
GGUF_DEFAULT_ALIGNMENT = 32


def _gguf_write_string(buf, s):
    b = s.encode("utf-8")
    buf.extend(struct.pack("<Q", len(b)))
    buf.extend(b)


def _gguf_read_string(data, offset):
    length = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    s = data[offset:offset + length].decode("utf-8")
    return s, offset + length


def _gguf_align_offset(offset, alignment=GGUF_DEFAULT_ALIGNMENT):
    rem = offset % alignment
    if rem == 0:
        return offset
    return offset + (alignment - rem)


class GGUFBinaryIO:
    @staticmethod
    def save_gguf(filepath, tensors_dict, metadata_dict=None):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        meta = {}
        meta["general.architecture"] = ("string", "spiking_neocortex")
        meta["general.name"] = ("string", "unified_neocortex_system2")
        meta["general.learning_rule"] = ("string", "three_factor_hebbian_predictive_coding_oja")
        meta["spiking_neocortex.embedding_length"] = ("uint32", 128)
        if metadata_dict:
            for k, v in metadata_dict.items():
                if isinstance(v, str):
                    meta[k] = ("string", v)
                elif isinstance(v, float):
                    meta[k] = ("float32", float(v))
                elif isinstance(v, int):
                    meta[k] = ("uint32", int(v))

        clean_tensors = {}
        for name, arr in tensors_dict.items():
            np_arr = np.asarray(arr, dtype=np.float32)
            if not np_arr.flags["C_CONTIGUOUS"]:
                np_arr = np.ascontiguousarray(np_arr, dtype=np.float32)
            clean_tensors[name] = np_arr

        buf = bytearray()
        buf.extend(GGUF_MAGIC)
        buf.extend(struct.pack("<I", GGUF_VERSION))
        buf.extend(struct.pack("<Q", len(clean_tensors)))
        buf.extend(struct.pack("<Q", len(meta)))

        for k, (vtype, val) in meta.items():
            _gguf_write_string(buf, k)
            if vtype == "string":
                buf.extend(struct.pack("<I", GGUF_TYPE_STRING))
                _gguf_write_string(buf, val)
            elif vtype == "uint32":
                buf.extend(struct.pack("<I", GGUF_TYPE_UINT32))
                buf.extend(struct.pack("<I", int(val)))
            elif vtype == "int32":
                buf.extend(struct.pack("<I", GGUF_TYPE_INT32))
                buf.extend(struct.pack("<i", int(val)))
            elif vtype == "float32":
                buf.extend(struct.pack("<I", GGUF_TYPE_FLOAT32))
                buf.extend(struct.pack("<f", float(val)))

        tensor_data_offsets = {}
        current_data_offset = 0
        for name, np_arr in clean_tensors.items():
            current_data_offset = _gguf_align_offset(current_data_offset, GGUF_DEFAULT_ALIGNMENT)
            tensor_data_offsets[name] = current_data_offset
            current_data_offset += np_arr.nbytes

        for name, np_arr in clean_tensors.items():
            _gguf_write_string(buf, name)
            dims = np_arr.shape
            n_dims = len(dims)
            buf.extend(struct.pack("<I", n_dims))
            for d in reversed(dims):
                buf.extend(struct.pack("<Q", int(d)))
            buf.extend(struct.pack("<I", GGML_TYPE_F32))
            buf.extend(struct.pack("<Q", int(tensor_data_offsets[name])))

        header_end = _gguf_align_offset(len(buf), GGUF_DEFAULT_ALIGNMENT)
        if header_end > len(buf):
            buf.extend(b"\x00" * (header_end - len(buf)))

        data_start = len(buf)
        for name, np_arr in clean_tensors.items():
            target_pos = data_start + tensor_data_offsets[name]
            if target_pos > len(buf):
                buf.extend(b"\x00" * (target_pos - len(buf)))
            buf.extend(np_arr.tobytes())

        with open(filepath, "wb") as f:
            f.write(buf)
        return True

    @staticmethod
    def load_gguf(filepath):
        with open(filepath, "rb") as f:
            data = f.read()

        if len(data) < 24 or data[:4] != GGUF_MAGIC:
            raise ValueError("Invalid GGUF file header")

        offset = 4
        version = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")

        tensor_count = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        kv_count = struct.unpack_from("<Q", data, offset)[0]
        offset += 8

        metadata = {}
        for _ in range(kv_count):
            key, offset = _gguf_read_string(data, offset)
            vtype = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            if vtype == GGUF_TYPE_STRING:
                val, offset = _gguf_read_string(data, offset)
            elif vtype == GGUF_TYPE_UINT32:
                val = struct.unpack_from("<I", data, offset)[0]
                offset += 4
            elif vtype == GGUF_TYPE_INT32:
                val = struct.unpack_from("<i", data, offset)[0]
                offset += 4
            elif vtype == GGUF_TYPE_FLOAT32:
                val = struct.unpack_from("<f", data, offset)[0]
                offset += 4
            elif vtype == GGUF_TYPE_UINT64:
                val = struct.unpack_from("<Q", data, offset)[0]
                offset += 8
            else:
                raise ValueError(f"Unsupported GGUF metadata type: {vtype}")
            metadata[key] = val

        tensor_infos = []
        for _ in range(tensor_count):
            name, offset = _gguf_read_string(data, offset)
            n_dims = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            rev_dims = []
            for _ in range(n_dims):
                d = struct.unpack_from("<Q", data, offset)[0]
                offset += 8
                rev_dims.append(int(d))
            dims = tuple(reversed(rev_dims))
            ggml_type = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            rel_offset = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            tensor_infos.append((name, dims, ggml_type, rel_offset))

        data_start = _gguf_align_offset(offset, GGUF_DEFAULT_ALIGNMENT)
        tensors = {}
        for name, dims, ggml_type, rel_offset in tensor_infos:
            if ggml_type != GGML_TYPE_F32:
                raise ValueError(f"Unsupported GGML tensor type: {ggml_type}")
            count = int(np.prod(dims)) if len(dims) > 0 else 1
            start = data_start + rel_offset
            arr = np.frombuffer(data, dtype=np.float32, count=count, offset=start).copy()
            tensors[name] = arr.reshape(dims)

        return tensors, metadata


class SensoryCortexModule(nn.Module):
    def __init__(self, hid_dim=128):
        super().__init__()
        self.hid_dim = int(hid_dim)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc_visual = nn.Linear(32 * 3 * 3, 64)
        self.fc_spatial = nn.Linear(16, 32)
        self.fc_auditory = nn.Linear(11, 32)
        self.W_q_sens = nn.Linear(64, 32, bias=False)
        self.W_k_sens = nn.Linear(64, 32, bias=False)
        self.W_v_sens = nn.Linear(64, 64, bias=False)
        self.fc_fusion = nn.Linear(128, self.hid_dim)
        self.fc_retinal16 = nn.Linear(self.hid_dim, 16)
        self.register_buffer("theta_sensory", torch.full((self.hid_dim,), 0.25, dtype=torch.float32))
        self.ort_session = None
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, visual_patch, spatial_stats, auditory_vec):
        x = F.gelu(self.conv1(visual_patch))
        x = F.gelu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (3, 3))
        x = x.reshape(x.size(0), -1)
        v_feat = torch.tanh(self.fc_visual(x))
        s_feat = torch.tanh(self.fc_spatial(spatial_stats))
        a_feat = torch.tanh(self.fc_auditory(auditory_vec))
        q_v = torch.tanh(self.W_q_sens(v_feat))
        sa_cat = torch.cat([s_feat, a_feat], dim=-1)
        k_sa = torch.tanh(self.W_k_sens(sa_cat))
        v_sa = torch.tanh(self.W_v_sens(sa_cat))
        q_spk = q_v * torch.sigmoid(8.0 * (torch.abs(q_v) - 0.20))
        k_spk = k_sa * torch.sigmoid(8.0 * (torch.abs(k_sa) - 0.20))
        attn_gate = torch.sigmoid(torch.sum(q_spk * k_spk, dim=-1, keepdim=True) * 0.1767767)
        v_mod = torch.tanh(v_feat + 0.35 * attn_gate * v_sa)
        cat_feat = torch.cat([v_mod, s_feat, a_feat], dim=-1)
        pre_sens = self.fc_fusion(cat_feat)
        spk_gate = torch.sigmoid(6.0 * (torch.abs(pre_sens) - self.theta_sensory))
        sensory_features = torch.tanh(pre_sens) * (0.35 + 0.65 * spk_gate)
        retinal_16 = torch.sigmoid(self.fc_retinal16(sensory_features))
        return sensory_features, retinal_16

    @staticmethod
    def extract_grid_patch_and_stats(grid, gaze_y=None, gaze_x=None, bg_color=None):
        g = np.asarray(grid, dtype=np.int32)
        if g.ndim != 2 or g.size == 0:
            g = np.zeros((1, 1), dtype=np.int32)
        H, W = g.shape
        if bg_color is None:
            u_vals, u_cnts = np.unique(g, return_counts=True)
            bg = int(u_vals[int(np.argmax(u_cnts))]) if u_vals.size > 0 else 0
        else:
            bg = int(bg_color)
        if gaze_y is None:
            gaze_y = H // 2
        if gaze_x is None:
            gaze_x = W // 2
        gy = int(np.clip(gaze_y, 0, max(0, H - 1)))
        gx = int(np.clip(gaze_x, 0, max(0, W - 1)))

        patch = np.zeros((9, 9), dtype=np.float32)
        foveal_vals = []
        parafoveal_vals = []
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                ny = gy + dy
                nx = gx + dx
                ecc = max(abs(dy), abs(dx))
                if ecc <= 2:
                    if 0 <= ny < H and 0 <= nx < W:
                        val = float(g[ny, nx]) / 9.0
                        patch[dy + 4, dx + 4] = val
                        foveal_vals.append(float(g[ny, nx]))
                    else:
                        foveal_vals.append(float(bg))
                else:
                    blur_sum = 0.0
                    blur_cnt = 0.0
                    for by in range(-1, 2):
                        for bx in range(-1, 2):
                            py = ny + by
                            px = nx + bx
                            if 0 <= py < H and 0 <= px < W:
                                blur_sum += float(g[py, px]) / 9.0
                                blur_cnt += 1.0
                    b_val = blur_sum / max(1.0, blur_cnt)
                    patch[dy + 4, dx + 4] = b_val
                    parafoveal_vals.append(b_val)

        stats = np.zeros(16, dtype=np.float32)
        dirs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        max_r = max(H, W, 4)
        for d_idx, (vy, vx) in enumerate(dirs):
            sal_acc = 0.0
            w_acc = 0.0
            for r in range(3, max_r + 1):
                for ortho in (-1, 0, 1):
                    ry = gy + vy * r + (-vx) * ortho
                    rx = gx + vx * r + vy * ortho
                    if 0 <= ry < H and 0 <= rx < W:
                        decay = 1.0 / (1.0 + 0.35 * float(r))
                        is_fg = 1.0 if g[ry, rx] != bg else 0.0
                        sal_acc += decay * (0.65 * is_fg + 0.35 * (float(g[ry, rx]) / 9.0))
                        w_acc += decay
            stats[d_idx] = float(sal_acc / max(1e-5, w_acc)) if w_acc > 0.0 else 0.0

        fov_arr = np.asarray(foveal_vals, dtype=np.float32).reshape(5, 5)
        fov_fg = (fov_arr != float(bg)).astype(np.float32)
        stats[8] = float(np.mean(fov_fg))
        stats[9] = float(np.std(fov_arr / 9.0))
        stats[10] = float(np.mean(fov_arr == np.flip(fov_arr, axis=0)))
        stats[11] = float(np.mean(fov_arr == np.flip(fov_arr, axis=1)))
        inner_mean = float(np.mean(patch[2:7, 2:7]))
        outer_mean = float(np.mean(parafoveal_vals)) if parafoveal_vals else 0.0
        stats[12] = float(np.clip(abs(inner_mean - outer_mean), 0.0, 1.0))
        surround_ring = float(np.mean(fov_fg[0, :]) + np.mean(fov_fg[-1, :]) + np.mean(fov_fg[:, 0]) + np.mean(fov_fg[:, -1])) * 0.25
        center_empty = 1.0 if fov_arr[2, 2] == float(bg) else 0.0
        stats[13] = float(surround_ring * (0.5 + 0.5 * center_empty))
        stats[14] = float(gy) / float(max(1, H - 1))
        stats[15] = float(gx) / float(max(1, W - 1))

        return patch.reshape(1, 1, 9, 9), stats.reshape(1, 16)

    def _encode_single_fixation(self, grid, auditory_vec=None, gaze_y=None, gaze_x=None, use_onnx=True):
        patch_np, stats_np = self.extract_grid_patch_and_stats(grid, gaze_y=gaze_y, gaze_x=gaze_x)
        if auditory_vec is None:
            aud_np = np.zeros((1, 11), dtype=np.float32)
        else:
            aud_np = np.asarray(auditory_vec, dtype=np.float32).reshape(1, 11)

        if use_onnx and self.ort_session is not None:
            try:
                outputs = self.ort_session.run(
                    None,
                    {
                        "visual_patch": patch_np,
                        "spatial_stats": stats_np,
                        "auditory_vec": aud_np,
                    },
                )
                return outputs[0][0].astype(np.float64), outputs[1][0].astype(np.float64), stats_np[0].astype(np.float64)
            except Exception:
                pass

        with torch.no_grad():
            vp_t = torch.from_numpy(patch_np)
            ss_t = torch.from_numpy(stats_np)
            av_t = torch.from_numpy(aud_np)
            feat_t, ret_t = self.forward(vp_t, ss_t, av_t)
            return (
                feat_t[0].cpu().numpy().astype(np.float64),
                ret_t[0].cpu().numpy().astype(np.float64),
                stats_np[0].astype(np.float64),
            )

    OPPONENT_COLOR_LUT = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.32, -0.35, -0.82],
            [0.48, 0.88, 0.18],
            [0.58, -0.78, 0.35],
            [0.88, 0.12, 0.90],
            [0.50, 0.00, 0.00],
            [0.54, 0.76, -0.58],
            [0.68, 0.62, 0.64],
            [0.74, -0.56, -0.48],
            [0.36, 0.65, -0.15],
        ],
        dtype=np.float64,
    )

    def compute_v1_v2_opponent_border_ownership(self, grid):
        g = np.asarray(grid, dtype=np.int32)
        if g.ndim != 2 or g.size == 0:
            g = np.zeros((1, 1), dtype=np.int32)
        H, W = g.shape
        g_clip = np.clip(g, 0, 9)
        opp_map = self.OPPONENT_COLOR_LUT[g_clip]

        pad_opp = np.pad(opp_map, ((1, 1), (1, 1), (0, 0)), mode="edge")
        surround = (
            pad_opp[:-2, 1:-1]
            + pad_opp[2:, 1:-1]
            + pad_opp[1:-1, :-2]
            + pad_opp[1:-1, 2:]
            + pad_opp[:-2, :-2]
            + pad_opp[:-2, 2:]
            + pad_opp[2:, :-2]
            + pad_opp[2:, 2:]
        ) / 8.0
        dog_contrast = np.linalg.norm(opp_map - surround, axis=-1)

        border_mask = np.zeros((H, W), dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True

        ground_likelihood = np.zeros(10, dtype=np.float64)
        figure_ownership = np.zeros((H, W), dtype=np.float64)
        total_px = float(max(1, H * W))
        border_px = float(max(1, int(np.sum(border_mask))))

        for c in range(10):
            mask_c = g_clip == c
            cnt = int(np.sum(mask_c))
            if cnt == 0:
                ground_likelihood[c] = -1e9
                continue
            area_frac = float(cnt) / total_px
            border_frac = float(np.sum(mask_c & border_mask)) / border_px
            mean_contrast = float(np.mean(dog_contrast[mask_c]))
            lum_val = float(self.OPPONENT_COLOR_LUT[c, 0])
            dark_prior = 0.25 if c == 0 else (0.05 * (1.0 - lum_val))
            ground_likelihood[c] = (
                0.45 * border_frac
                + 0.40 * area_frac
                - 0.30 * mean_contrast
                + dark_prior
            )

        bg_color = int(np.argmax(ground_likelihood))
        bg_opp = self.OPPONENT_COLOR_LUT[bg_color]
        chrom_dist = np.linalg.norm(opp_map - bg_opp[None, None, :], axis=-1)
        figure_ownership = np.tanh(1.4 * chrom_dist + 0.8 * dog_contrast)
        figure_mask = (g_clip != bg_color) & (figure_ownership > 0.08)

        return {
            "opponent_map": opp_map,
            "dog_contrast": dog_contrast,
            "figure_ownership": figure_ownership,
            "figure_mask": figure_mask,
            "background_color": bg_color,
            "inferred_bg_color": bg_color,
            "ground_likelihood": ground_likelihood,
        }

    def saccade_and_integrate_mental_canvas(self, grid, auditory_vec=None, max_saccades=4, motor_shift_hint=None, use_onnx=True):
        g = np.asarray(grid, dtype=np.int32)
        if g.ndim != 2 or g.size == 0:
            g = np.zeros((1, 1), dtype=np.int32)
        H, W = g.shape
        gy = H // 2
        gx = W // 2

        v1_seg = self.compute_v1_v2_opponent_border_ownership(g)
        bg_col = int(v1_seg["background_color"])
        salience_map = v1_seg["figure_ownership"].copy()

        mental_canvas = np.full((H, W), bg_col, dtype=np.int32)
        foveated_mask = np.zeros((H, W), dtype=bool)
        ior_map = np.zeros((H, W), dtype=np.float64)
        saccade_trajectory = []

        integrated_feat = np.zeros(self.hid_dim, dtype=np.float64)
        integrated_ret = np.zeros(16, dtype=np.float64)
        dirs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        adaptive_steps = max(4, min(10, int(max_saccades)))

        for step_idx in range(adaptive_steps):
            gy = int(np.clip(gy, 0, max(0, H - 1)))
            gx = int(np.clip(gx, 0, max(0, W - 1)))
            saccade_trajectory.append((gy, gx))

            periph_r = max(4, min(10, max(H, W) // 2))
            py_min = max(0, gy - periph_r)
            py_max = min(H, gy + periph_r + 1)
            px_min = max(0, gx - periph_r)
            px_max = min(W, gx + periph_r + 1)
            for py in range(py_min, py_max, 2):
                for px in range(px_min, px_max, 2):
                    ecc = max(abs(py - gy), abs(px - gx))
                    if ecc <= 2:
                        continue
                    pool_span = 2 if ecc <= 6 else 3
                    y_end = min(H, py + pool_span)
                    x_end = min(W, px + pool_span)
                    sub = g[py:y_end, px:x_end]
                    sub_foveated = foveated_mask[py:y_end, px:x_end]
                    if np.all(sub_foveated):
                        continue
                    non_bg = sub[sub != bg_col]
                    if non_bg.size > 0:
                        vals, cnts = np.unique(non_bg, return_counts=True)
                        dom_c = int(vals[int(np.argmax(cnts))])
                        pooled_patch = np.where(sub != bg_col, dom_c, bg_col)
                        mental_canvas[py:y_end, px:x_end] = np.where(
                            sub_foveated,
                            mental_canvas[py:y_end, px:x_end],
                            pooled_patch,
                        )

            y0 = max(0, gy - 2)
            y1 = min(H, gy + 3)
            x0 = max(0, gx - 2)
            x1 = min(W, gx + 3)
            mental_canvas[y0:y1, x0:x1] = g[y0:y1, x0:x1]
            foveated_mask[y0:y1, x0:x1] = True
            ior_map[y0:y1, x0:x1] += 1.0

            feat_s, ret_s, stats_s = self._encode_single_fixation(
                g, auditory_vec=auditory_vec, gaze_y=gy, gaze_x=gx, use_onnx=use_onnx
            )
            if step_idx == 0:
                integrated_feat = feat_s.copy()
                integrated_ret = ret_s.copy()
            else:
                alpha = 1.0 / float(step_idx + 1)
                integrated_feat = np.tanh((1.0 - alpha) * integrated_feat + alpha * feat_s)
                integrated_ret = np.clip((1.0 - alpha) * integrated_ret + alpha * ret_s, 0.0, 1.0)

            if step_idx + 1 < adaptive_steps:
                periph_sal = stats_s[:8]
                best_score = -1e9
                next_gy, next_gx = gy, gx
                step_radius = max(2, min(4, max(H, W) // 3))
                for d_i, (vy, vx) in enumerate(dirs):
                    cand_y = int(np.clip(gy + vy * step_radius, 0, max(0, H - 1)))
                    cand_x = int(np.clip(gx + vx * step_radius, 0, max(0, W - 1)))
                    ior_pen = float(ior_map[cand_y, cand_x]) * 0.45
                    hint_bonus = 0.0
                    if motor_shift_hint is not None and len(motor_shift_hint) >= 2:
                        hint_bonus = 0.15 * float(vy * motor_shift_hint[0] + vx * motor_shift_hint[1])
                    score = float(periph_sal[d_i]) + 0.35 * float(salience_map[cand_y, cand_x]) + hint_bonus - ior_pen
                    if score > best_score:
                        best_score = score
                        next_gy, next_gx = cand_y, cand_x
                gy, gx = next_gy, next_gx

        return {
            "sensory_features": integrated_feat,
            "retinal_16": integrated_ret,
            "mental_canvas": mental_canvas,
            "foveated_mask": foveated_mask,
            "saccade_trajectory": saccade_trajectory,
            "foveal_coverage_ratio": float(np.mean(foveated_mask)),
            "background_color": bg_col,
            "figure_ownership": v1_seg["figure_ownership"],
        }

    def encode_retinotopic_foveal_patch_spikes(self, grid, gaze_y, gaze_x, bg_color=None):
        g = np.asarray(grid, dtype=np.int32)
        if g.ndim != 2 or g.size == 0:
            return np.zeros((100,), dtype=np.float32)
        h, w = g.shape
        gy = int(np.clip(int(gaze_y), 0, h - 1))
        gx = int(np.clip(int(gaze_x), 0, w - 1))
        if bg_color is None:
            v1_seg = self.compute_v1_v2_opponent_border_ownership(g)
            bg = int(v1_seg["background_color"])
        else:
            bg = int(bg_color)
        center_col = int(np.clip(g[gy, gx], 0, 9))
        spikes = np.zeros((100,), dtype=np.float32)
        idx = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ry = gy + dy
                rx = gx + dx
                if 0 <= ry < h and 0 <= rx < w:
                    c_val = int(np.clip(g[ry, rx], 0, 9))
                    fg_sp = 1.0 if c_val != bg else 0.0
                    if c_val == center_col and c_val != bg:
                        same_sp = 1.0
                    elif c_val != bg:
                        same_sp = -1.0
                    else:
                        same_sp = 0.0
                    opp = self.OPPONENT_COLOR_LUT[c_val]
                    spikes[idx] = float(fg_sp)
                    spikes[idx + 1] = float(same_sp)
                    spikes[idx + 2] = float(opp[0])
                    spikes[idx + 3] = float(0.5 * (opp[1] + opp[2]))
                else:
                    spikes[idx] = -0.5
                    spikes[idx + 1] = 0.0
                    spikes[idx + 2] = 0.0
                    spikes[idx + 3] = 0.0
                idx += 4
        return spikes

    def encode_grid(self, grid, auditory_vec=None, gaze_y=None, gaze_x=None, use_onnx=True):
        if gaze_y is not None or gaze_x is not None:
            feat, ret, _ = self._encode_single_fixation(
                grid, auditory_vec=auditory_vec, gaze_y=gaze_y, gaze_x=gaze_x, use_onnx=use_onnx
            )
            return feat, ret
        sac_info = self.saccade_and_integrate_mental_canvas(
            grid, auditory_vec=auditory_vec, max_saccades=3, use_onnx=use_onnx
        )
        return sac_info["sensory_features"], sac_info["retinal_16"]

    def export_onnx(self, onnx_path):
        os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
        self.eval()
        dev = next(self.parameters()).device
        dummy_patch = torch.zeros(1, 1, 9, 9, dtype=torch.float32, device=dev)
        dummy_stats = torch.zeros(1, 16, dtype=torch.float32, device=dev)
        dummy_aud = torch.zeros(1, 11, dtype=torch.float32, device=dev)
        torch.onnx.export(
            self,
            (dummy_patch, dummy_stats, dummy_aud),
            onnx_path,
            input_names=["visual_patch", "spatial_stats", "auditory_vec"],
            output_names=["sensory_features", "retinal_16"],
            dynamic_axes={
                "visual_patch": {0: "batch_size"},
                "spatial_stats": {0: "batch_size"},
                "auditory_vec": {0: "batch_size"},
                "sensory_features": {0: "batch_size"},
                "retinal_16": {0: "batch_size"},
            },
            opset_version=17,
            dynamo=False,
        )
        if HAS_ONNX_RUNTIME:
            self.load_onnx_session(onnx_path)
        return True

    def load_onnx_session(self, onnx_path):
        if HAS_ONNX_RUNTIME and os.path.exists(onnx_path):
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 1
            sess_opts.inter_op_num_threads = 1
            avail = ort.get_available_providers()
            provs = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]
            self.ort_session = ort.InferenceSession(
                onnx_path, sess_options=sess_opts, providers=provs
            )
            return True
        return False


class NeocortexSystem2Module(nn.Module):
    DRAWER_NAMES = (
        "topic_0",
        "topic_1",
        "topic_2",
        "topic_3",
        "topic_4",
        "topic_5",
    )

    def __init__(self, hid_dim=128, seed=42):
        super().__init__()
        self.hid_dim = int(hid_dim)
        rng = np.random.default_rng(seed)
        scale_in = 1.0 / np.sqrt(27.0)
        scale_h = 1.0 / np.sqrt(float(self.hid_dim))

        self.W_in = nn.Parameter(torch.tensor(rng.normal(0.0, scale_in, (self.hid_dim, 27)), dtype=torch.float32))
        self.Wh = nn.Parameter(torch.tensor(rng.normal(0.0, scale_h * 0.5, (self.hid_dim, self.hid_dim)), dtype=torch.float32))
        self.Wq = nn.Parameter(torch.tensor(rng.normal(0.0, scale_h, (self.hid_dim, self.hid_dim)), dtype=torch.float32))
        self.Wk = nn.Parameter(torch.tensor(rng.normal(0.0, scale_h, (self.hid_dim, self.hid_dim)), dtype=torch.float32))
        self.Wv = nn.Parameter(torch.tensor(rng.normal(0.0, scale_h, (self.hid_dim, self.hid_dim)), dtype=torch.float32))
        self.W_pred = nn.Parameter(torch.tensor(rng.normal(0.0, scale_h, (16, self.hid_dim)), dtype=torch.float32))
        self.W_sound_pred = nn.Parameter(torch.tensor(rng.normal(0.0, scale_h, (11, self.hid_dim)), dtype=torch.float32))
        self.W_symbol_emit = nn.Parameter(torch.tensor(rng.normal(0.0, scale_h, (8, self.hid_dim)), dtype=torch.float32))
        self.W_symbol_lexicon = nn.Parameter(torch.tensor(rng.normal(0.0, 0.15, (16, 11)), dtype=torch.float32))
        self.W_body_assembly = nn.Parameter(torch.tensor(rng.normal(0.0, scale_h, (11, self.hid_dim)), dtype=torch.float32))
        self.W_macro_out = nn.Parameter(torch.tensor(rng.normal(0.0, scale_h * 0.4, (4, self.hid_dim)), dtype=torch.float32))
        self.W_macro_in = nn.Parameter(torch.tensor(rng.normal(0.0, 0.20, (self.hid_dim, 4)), dtype=torch.float32))
        self.W_mono_out = nn.Parameter(torch.tensor(rng.normal(0.0, scale_h * 0.4, (4, self.hid_dim)), dtype=torch.float32))
        self.W_mono_in = nn.Parameter(torch.tensor(rng.normal(0.0, 0.22, (self.hid_dim, 4)), dtype=torch.float32))
        self.W_rel_delta = nn.Parameter(torch.tensor(rng.normal(0.0, 1.0 / np.sqrt(384.0), (self.hid_dim, self.hid_dim * 3)), dtype=torch.float32))
        self.W_drawer_gate = nn.Parameter(torch.tensor(rng.normal(0.0, scale_h, (len(self.DRAWER_NAMES), self.hid_dim)), dtype=torch.float32))
        self.W_world_forward = nn.Parameter(torch.tensor(rng.normal(0.0, scale_h, (self.hid_dim, self.hid_dim + 8)), dtype=torch.float32))
        self.W_aux_concepts = nn.Parameter(torch.tensor(rng.normal(0.0, scale_h, (6, self.hid_dim)), dtype=torch.float32))
        self.W_parietal_gain = nn.Parameter(torch.tensor(rng.normal(0.0, scale_h, (4, self.hid_dim)), dtype=torch.float32))
        self.W_procedural_attractor = nn.Parameter(torch.tensor(rng.normal(0.0, scale_h, (16, self.hid_dim)), dtype=torch.float32))
        self.W_procedural_chain = nn.Parameter(torch.tensor(rng.normal(0.0, 0.15, (16, 33)), dtype=torch.float32))
        self.W_syntax_transition = nn.Parameter(torch.tensor(rng.normal(0.0, 0.20, (8, 8)), dtype=torch.float32))
        self.W_mirror_inverse = nn.Parameter(torch.tensor(rng.normal(0.0, 0.15, (11, 16)), dtype=torch.float32))
        self.W_fovea_v1_spikes = nn.Parameter(torch.tensor(rng.normal(0.0, 0.08, (self.hid_dim, 100)), dtype=torch.float32))
        self.W_fovea_stroke_assoc = nn.Parameter(torch.tensor(rng.normal(0.0, 0.02, (25, 11, 100)), dtype=torch.float32))
        self.myelination_stage_gates = nn.Parameter(torch.tensor([1.0, 0.65, 0.40], dtype=torch.float32))
        self.theta = nn.Parameter(torch.full((self.hid_dim,), 0.25, dtype=torch.float32))

    def forward(
        self,
        neo_input,
        h_prev=None,
        foveal_spikes=None,
        modulatory_macro=None,
        modulatory_mono=None,
        developmental_stage=0,
    ):
        if neo_input.ndim == 1:
            neo_in = neo_input.unsqueeze(0)
        else:
            neo_in = neo_input
        idx_stage = max(0, min(int(developmental_stage), len(self.myelination_stage_gates) - 1))
        gate_m = self.myelination_stage_gates[idx_stage]
        if neo_in.shape[1] == 27:
            h_in = neo_in @ self.W_in.T
        else:
            h_in = neo_in[:, :self.hid_dim]
        if foveal_spikes is not None:
            if foveal_spikes.ndim == 1:
                f_spk = foveal_spikes.unsqueeze(0)
            else:
                f_spk = foveal_spikes
            foveal_drive = torch.tanh(f_spk @ self.W_fovea_v1_spikes.T)
            h_in = h_in + 0.35 * foveal_drive
        if h_prev is not None:
            if h_prev.ndim == 1:
                h_p = h_prev.unsqueeze(0)
            else:
                h_p = h_prev
            rec_drive = torch.tanh(h_p @ self.Wh.T)
            proc_attractor = torch.tanh(h_p @ self.W_procedural_attractor.T)
            chain_slot = torch.tanh(proc_attractor @ self.W_procedural_chain)
            chain_fb = torch.tanh(chain_slot @ self.W_procedural_chain.T @ self.W_procedural_attractor)
            h_next = torch.tanh(gate_m * (0.60 * h_in + 0.25 * rec_drive + 0.15 * chain_fb))
        else:
            h_next = torch.tanh(gate_m * h_in)
        if modulatory_macro is not None:
            if modulatory_macro.ndim == 1:
                m_mac = modulatory_macro.unsqueeze(0)
            else:
                m_mac = modulatory_macro
            mac_mod = torch.tanh(m_mac @ self.W_macro_in.T)
            h_next = h_next + 0.10 * mac_mod
        if modulatory_mono is not None:
            if modulatory_mono.ndim == 1:
                m_mono = modulatory_mono.unsqueeze(0)
            else:
                m_mono = modulatory_mono
            mono_mod = torch.tanh(m_mono @ self.W_mono_in.T)
            h_next = h_next + 0.10 * mono_mod
        q_neo = torch.tanh(h_next @ self.Wq.T)
        k_neo = torch.tanh(h_next @ self.Wk.T)
        v_neo = torch.tanh(h_next @ self.Wv.T)
        q_spk = q_neo * torch.sigmoid(8.0 * (torch.abs(q_neo) - self.theta))
        k_spk = k_neo * torch.sigmoid(8.0 * (torch.abs(k_neo) - self.theta))
        attn_sc = torch.sigmoid(torch.sum(q_spk * k_spk, dim=-1, keepdim=True) * (1.0 / np.sqrt(self.hid_dim)))
        h_attn = torch.tanh(h_next + 0.35 * attn_sc * v_neo)
        spk_gate_neo = torch.sigmoid(6.0 * (torch.abs(h_attn) - self.theta))
        h_final = h_attn * (0.35 + 0.65 * spk_gate_neo)
        with torch.no_grad():
            self.theta.add_(0.01 * (spk_gate_neo.mean(dim=0) - 0.35)).clamp_(0.10, 0.60)
        macro_out = torch.tanh(h_final @ self.W_macro_out.T)
        mono_out = torch.tanh(h_final @ self.W_mono_out.T)
        sound_pred = torch.tanh(h_final @ self.W_sound_pred.T)
        return h_final, macro_out, mono_out, sound_pred

    def infer_body_assembly(self, h_state):
        with torch.no_grad():
            dev = self.W_body_assembly.device
            ht = torch.as_tensor(h_state, dtype=torch.float32, device=dev).reshape(-1, self.hid_dim)
            return torch.tanh(ht @ self.W_body_assembly.T).squeeze(0).cpu().numpy().astype(np.float32)

    def project_3d_physics_to_2d_tabletop(
        self,
        grid,
        head_center_xyz=(0.0, -0.25, 2.65),
        head_shift_delta_xyz=(0.16, 0.05, 0.0),
        gaze_uv_deg=(0.0, 0.0),
    ):
        engine = get_global_isaac_warp_engine()
        perc = engine.perceive_3d_tabletop_via_human_vision(
            grid,
            head_center_xyz=head_center_xyz,
            head_shift_delta_xyz=head_shift_delta_xyz,
            gaze_uv_deg=gaze_uv_deg,
        )
        contact_support_vec = perc["contact_support_vec"]
        height_map = perc["height_map"]
        with torch.no_grad():
            w_par = self.W_parietal_gain.detach().cpu().numpy().astype(np.float32)
        par_proj = np.tanh(w_par[:, :4] @ contact_support_vec).astype(np.float32)
        world_bias = np.tanh(w_par.T[:16, :4] @ (contact_support_vec + 0.35 * par_proj)).astype(np.float32)
        tabletop_vec = perc["tabletop_affordance_vec"].copy()
        tabletop_vec[4:8] = 0.5 * tabletop_vec[4:8] + 0.5 * par_proj
        tabletop_vec[11:16] = 0.5 * tabletop_vec[11:16] + 0.5 * world_bias[:5]
        perc["tabletop_affordance_vec"] = tabletop_vec
        perc["height_map"] = height_map
        return perc

    def infer_mirror_neuron_author_stroke(self, x_grid, y_grid, update_weights=True, lr=0.025):
        gx = np.asarray(x_grid, dtype=np.int32)
        gy = np.asarray(y_grid, dtype=np.int32)
        if gx.ndim != 2 or gx.size == 0:
            gx = np.zeros((3, 3), dtype=np.int32)
        if gy.ndim != 2 or gy.size == 0:
            gy = np.zeros((3, 3), dtype=np.int32)
        px = self.project_3d_physics_to_2d_tabletop(gx)
        py = self.project_3d_physics_to_2d_tabletop(gy)
        hx, wx = gx.shape
        hy, wy = gy.shape
        hm_x = px["height_map_cuda"]
        hm_y = py["height_map_cuda"]
        fgx = hm_x > 0.05
        fgy = hm_y > 0.05
        if bool(torch.any(fgx).item()):
            yx_idx = torch.nonzero(fgx, as_tuple=False).to(torch.float32)
            cy_x = float(torch.mean(yx_idx[:, 0]).item()) / float(max(1, hx - 1))
            cx_x = float(torch.mean(yx_idx[:, 1]).item()) / float(max(1, wx - 1))
        else:
            cy_x, cx_x = 0.5, 0.5
        if bool(torch.any(fgy).item()):
            yy_idx = torch.nonzero(fgy, as_tuple=False).to(torch.float32)
            cy_y = float(torch.mean(yy_idx[:, 0]).item()) / float(max(1, hy - 1))
            cx_y = float(torch.mean(yy_idx[:, 1]).item()) / float(max(1, wy - 1))
        else:
            cy_y, cx_y = 0.5, 0.5
        dy_shift = float(np.clip(cy_y - cy_x, -1.0, 1.0))
        dx_shift = float(np.clip(cx_y - cx_x, -1.0, 1.0))
        area_x = float(torch.mean(fgx.to(torch.float32)).item())
        area_y = float(torch.mean(fgy.to(torch.float32)).item())
        area_delta = float(np.clip(area_y - area_x, -1.0, 1.0))
        scale_h = float(np.clip(float(hy) / float(max(1, hx)) - 1.0, -1.0, 1.0))
        scale_w = float(np.clip(float(wy) / float(max(1, wx)) - 1.0, -1.0, 1.0))
        aff_delta = py["tabletop_affordance_vec"] - px["tabletop_affordance_vec"]
        elev_delta = float(np.clip(float(torch.mean(hm_y).item() - torch.mean(hm_x).item()), -1.0, 1.0))
        disp_delta = float(np.clip(aff_delta[7], -1.0, 1.0))
        flow_delta = float(np.clip(aff_delta[8], -1.0, 1.0))
        tj_delta = float(np.clip(aff_delta[9], -1.0, 1.0))
        col_x = px["left_retinal_screen"]["color_idx"]
        col_y = py["left_retinal_screen"]["color_idx"]
        ux = set(int(v) for v in torch.unique(col_x).tolist() if v > 0)
        uy = set(int(v) for v in torch.unique(col_y).tolist() if v > 0)
        new_color_sig = 1.0 if len(uy - ux) > 0 else 0.0
        color_count_delta = float(np.clip((len(uy) - len(ux)) / 5.0, -1.0, 1.0))
        same_shape_sig = 1.0 if gx.shape == gy.shape else -1.0
        ret_diff_density = float(torch.mean((col_x != col_y).to(torch.float32)).item()) if col_x.shape == col_y.shape else 1.0
        vis_delta_16 = np.array(
            [
                dy_shift,
                dx_shift,
                area_delta,
                scale_h,
                scale_w,
                elev_delta,
                disp_delta,
                flow_delta,
                tj_delta,
                new_color_sig,
                color_count_delta,
                same_shape_sig,
                ret_diff_density,
                cy_y - 0.5,
                cx_y - 0.5,
                area_y,
            ],
            dtype=np.float32,
        )
        with torch.no_grad():
            dev = self.W_mirror_inverse.device
            vd_t = torch.tensor(vis_delta_16, dtype=torch.float32, device=dev)
            stroke_t = torch.tanh(torch.mv(self.W_mirror_inverse, vd_t))
            target_vec11 = torch.zeros(11, dtype=torch.float32, device=dev)
            target_vec11[:min(11, vd_t.shape[0])] = vd_t[:min(11, vd_t.shape[0])]
            target_t = torch.tanh(target_vec11)
            if update_weights:
                err_m = (target_t - stroke_t) * (1.0 - stroke_t * stroke_t)
                self.W_mirror_inverse.add_(float(lr) * torch.outer(err_m, vd_t) - 0.0005 * self.W_mirror_inverse)
                self.W_mirror_inverse.clamp_(-2.5, 2.5)
                stroke_t = torch.tanh(torch.mv(self.W_mirror_inverse, vd_t))
            author_stroke_11 = stroke_t.detach().cpu().numpy().astype(np.float32)
            target_stroke = target_t.detach().cpu().numpy().astype(np.float32)
        return {
            "visual_delta_16": vis_delta_16,
            "author_stroke_11": author_stroke_11,
            "target_stroke_11": target_stroke,
        }

    def imprint_foveal_stroke_association(self, s_pre_100, k_idx, ch_idx, lr=0.045):
        k_i = int(np.clip(int(k_idx), 0, 24))
        c_i = int(np.clip(int(ch_idx), 0, 10))
        with torch.no_grad():
            dev = self.W_fovea_stroke_assoc.device
            s_t = torch.as_tensor(s_pre_100, dtype=torch.float32, device=dev).reshape(100)
            s_norm = float(torch.linalg.norm(s_t).item())
            if s_norm < 1e-6:
                return
            s_unit = s_t / s_norm
            w_slice = self.W_fovea_stroke_assoc[k_i]
            sims = torch.mv(w_slice, s_unit)
            for other_c in range(11):
                if other_c == c_i:
                    continue
                sim_v = float(sims[other_c].item())
                if sim_v > 0.0:
                    w_slice[other_c].sub_(float(lr) * 0.35 * sim_v * s_unit)
            w_win = w_slice[c_i]
            w_win.mul_(1.0 - 0.015 * float(lr)).add_(float(lr) * s_unit)
            w_win_norm = float(torch.linalg.norm(w_win).item())
            if w_win_norm > 1.5:
                w_win.mul_(1.5 / w_win_norm)
            w_slice.clamp_(-2.5, 2.5)

    def readout_foveal_stroke_association(self, s_cur_100):
        with torch.no_grad():
            dev = self.W_fovea_stroke_assoc.device
            s_t = torch.as_tensor(s_cur_100, dtype=torch.float32, device=dev).reshape(100)
            s_norm = float(torch.linalg.norm(s_t).item())
            if s_norm < 1e-6:
                return np.zeros((25, 11), dtype=np.float32)
            s_unit = s_t / s_norm
            raw_drive = torch.matmul(self.W_fovea_stroke_assoc, s_unit)
            top2_vals, _ = torch.topk(raw_drive, k=2, dim=1)
            margin = (top2_vals[:, 0] - top2_vals[:, 1]).unsqueeze(1)
            inhibited_drive = raw_drive * torch.clamp(margin / 0.12, 0.25, 1.5)
            return inhibited_drive.detach().cpu().numpy().astype(np.float32)

    def emit_compositional_phonological_loop(self, h_vec):
        h_np = np.asarray(h_vec, dtype=np.float64).reshape(self.hid_dim)
        with torch.no_grad():
            w_emit = self.W_symbol_emit.detach().cpu().numpy().astype(np.float64)
            w_syn = self.W_syntax_transition.detach().cpu().numpy().astype(np.float64)
            w_lex = self.W_symbol_lexicon.detach().cpu().numpy().astype(np.float64)

        logits_subj = w_emit @ h_np
        subj_id = int(np.argmax(logits_subj))

        logits_rel = w_syn @ logits_subj + 0.35 * (w_emit @ np.roll(h_np, 16))
        logits_rel[subj_id] -= 1.25
        rel_id = int(np.argmax(logits_rel))

        logits_act = (w_syn @ logits_rel) + 0.35 * (w_emit @ np.roll(h_np, 32))
        logits_act[subj_id] -= 1.0
        logits_act[rel_id] -= 1.0
        act_id = int(np.argmax(logits_act))

        v_subj = w_lex[subj_id % w_lex.shape[0]]
        v_rel = w_lex[(rel_id + 4) % w_lex.shape[0]]
        v_act = w_lex[(act_id + 8) % w_lex.shape[0]]
        composed_aud_11 = np.tanh(0.45 * v_subj + 0.35 * v_rel + 0.40 * v_act)
        return {
            "phrase_ids": (subj_id, rel_id, act_id),
            "composed_auditory_11": composed_aud_11,
        }

    def encode_grid_state(self, sensory_module, grid, auditory_vec=None):
        sens_feat, ret_16 = sensory_module.encode_grid(grid, auditory_vec=auditory_vec)
        aud_11 = np.zeros(11, dtype=np.float64) if auditory_vec is None else np.asarray(auditory_vec, dtype=np.float64).reshape(11)
        obs_27 = np.concatenate([ret_16, aud_11], axis=0)

        with torch.no_grad():
            w_in = self.W_in.detach().cpu().numpy().astype(np.float64)
            wq = self.Wq.detach().cpu().numpy().astype(np.float64)
            wk = self.Wk.detach().cpu().numpy().astype(np.float64)
            wv = self.Wv.detach().cpu().numpy().astype(np.float64)
            wh = self.Wh.detach().cpu().numpy().astype(np.float64)
            th = self.theta.detach().cpu().numpy().astype(np.float64)

        inj = np.tanh(w_in @ obs_27 + sens_feat)
        q = np.tanh(wq @ inj)
        k = np.tanh(wk @ inj)
        v = np.tanh(wv @ inj)
        attn_gain = float(np.dot(q, k)) / np.sqrt(float(self.hid_dim))
        drive = wh @ inj + np.tanh(attn_gain) * v
        v_mem = np.zeros(self.hid_dim, dtype=np.float64)
        spk_acc = np.zeros(self.hid_dim, dtype=np.float64)
        for _ in range(4):
            v_mem = 0.72 * v_mem + 0.42 * drive
            fired = (np.abs(v_mem) > th).astype(np.float64)
            spk_acc += fired * np.sign(v_mem)
            v_mem = v_mem * (1.0 - 0.65 * fired)
        ctx = np.tanh(0.65 * np.tanh(drive - th * 0.2) + 0.35 * (spk_acc / 4.0))
        return ctx

    def imagine_action_outcome(self, hx, act_2d_vec, hy_goal=None):
        hx_np = np.asarray(hx, dtype=np.float64).reshape(self.hid_dim)
        act_np = np.asarray(act_2d_vec, dtype=np.float64).ravel()
        if act_np.size < 8:
            act_pad = np.zeros(8, dtype=np.float64)
            act_pad[:act_np.size] = act_np
            act_np = act_pad
        else:
            act_np = act_np[:8]

        with torch.no_grad():
            w_world = self.W_world_forward.detach().cpu().numpy().astype(np.float64)
            w_pred = self.W_pred.detach().cpu().numpy().astype(np.float64)

        joint_in = np.concatenate([hx_np, np.tanh(act_np)], axis=0)
        imagined_hy = np.tanh(w_world @ joint_in)
        pred_retinal = 1.0 / (1.0 + np.exp(-np.clip(w_pred @ imagined_hy, -10.0, 10.0)))

        utility = 0.0
        if hy_goal is not None:
            goal_np = np.asarray(hy_goal, dtype=np.float64).reshape(self.hid_dim)
            n_im = float(np.linalg.norm(imagined_hy))
            n_gl = float(np.linalg.norm(goal_np))
            if n_im > 1e-8 and n_gl > 1e-8:
                utility = float(np.dot(imagined_hy / n_im, goal_np / n_gl))
            else:
                utility = -float(np.mean((imagined_hy - goal_np) ** 2))
        else:
            utility = float(np.mean( np.abs(imagined_hy - hx_np) ))

        return {
            "imagined_hy": imagined_hy,
            "predicted_retinal": pred_retinal,
            "mental_utility": utility,
        }

    def perceive_relational_transition(self, sensory_module, train_pairs):
        if not train_pairs:
            zero_delta = np.zeros(self.hid_dim, dtype=np.float64)
            default_scores = {name: 1.0 / len(self.DRAWER_NAMES) for name in self.DRAWER_NAMES}
            return {
                "delta_vec": zero_delta,
                "consistency": 0.0,
                "drawer_scores": default_scores,
                "ranked_drawers": list(self.DRAWER_NAMES),
                "preserve_shape": True,
                "scale_ratio_h": 1.0,
                "scale_ratio_w": 1.0,
                "latent_topic_features": np.zeros(6, dtype=np.float32),
            }

        with torch.no_grad():
            w_rel = self.W_rel_delta.detach().cpu().numpy().astype(np.float64)
            w_gate = self.W_drawer_gate.detach().cpu().numpy().astype(np.float64)
            w_aux = self.W_aux_concepts.detach().cpu().numpy().astype(np.float64)

        pair_deltas = []
        preserve_shape_all = True
        scale_ratios_h = []
        scale_ratios_w = []

        for x_in, y_out in train_pairs:
            gx = np.asarray(x_in, dtype=np.int32)
            gy = np.asarray(y_out, dtype=np.int32)
            hx = self.encode_grid_state(sensory_module, gx)
            hy = self.encode_grid_state(sensory_module, gy)
            diff_h = hy - hx
            prod_h = hy * hx
            joint = np.concatenate([hx, hy, diff_h + 0.5 * prod_h], axis=0)
            rel_vec = np.tanh(w_rel @ joint)
            pair_deltas.append(rel_vec)

            if gx.shape != gy.shape:
                preserve_shape_all = False
                scale_ratios_h.append(float(gy.shape[0]) / float(max(1, gx.shape[0])))
                scale_ratios_w.append(float(gy.shape[1]) / float(max(1, gx.shape[1])))
            else:
                scale_ratios_h.append(1.0)
                scale_ratios_w.append(1.0)

        delta_stack = np.stack(pair_deltas, axis=0)
        mean_delta = np.mean(delta_stack, axis=0)
        norm_mean = float(np.linalg.norm(mean_delta))
        if norm_mean > 1e-8:
            unit_mean = mean_delta / norm_mean
            cos_sims = []
            for d in pair_deltas:
                nd = float(np.linalg.norm(d))
                if nd > 1e-8:
                    cos_sims.append(float(np.dot(d / nd, unit_mean)))
            consistency = float(np.clip(np.mean(cos_sims), 0.0, 1.0)) if cos_sims else 0.5
        else:
            consistency = 0.0

        raw_logits = w_gate @ mean_delta
        total_logits = raw_logits - np.max(raw_logits)
        probs = np.exp(total_logits)
        probs = probs / float(np.sum(probs) + 1e-12)

        aux_logits = np.clip(w_aux @ mean_delta, -10.0, 10.0)
        aux_probs = 1.0 / (1.0 + np.exp(-aux_logits))
        scale_h = float(np.median(scale_ratios_h)) if scale_ratios_h else 1.0
        scale_w = float(np.median(scale_ratios_w)) if scale_ratios_w else 1.0

        drawer_scores = {
            self.DRAWER_NAMES[i]: float(probs[i])
            for i in range(len(self.DRAWER_NAMES))
        }
        ranked_indices = np.argsort(-probs)
        ranked_drawers = [self.DRAWER_NAMES[int(i)] for i in ranked_indices]

        return {
            "delta_vec": mean_delta,
            "consistency": consistency,
            "drawer_scores": drawer_scores,
            "ranked_drawers": ranked_drawers,
            "preserve_shape": preserve_shape_all,
            "scale_ratio_h": scale_h,
            "scale_ratio_w": scale_w,
            "latent_topic_features": aux_probs.astype(np.float32),
        }

    def update_from_successful_transition(self, sensory_module, train_pairs, winning_drawer_name=None, lr=0.015):
        if not train_pairs:
            return
        info = self.perceive_relational_transition(sensory_module, train_pairs)
        delta_vec = info["delta_vec"]

        with torch.no_grad():
            dev = self.W_drawer_gate.device
            d_t = torch.tensor(delta_vec, dtype=torch.float32, device=dev)
            logits = torch.mv(self.W_drawer_gate, d_t)
            probs = torch.softmax(logits, dim=0)
            if winning_drawer_name is not None and str(winning_drawer_name) in self.DRAWER_NAMES:
                win_idx = self.DRAWER_NAMES.index(str(winning_drawer_name))
            else:
                win_idx = int(torch.argmax(probs).item())
            target = torch.zeros_like(probs)
            target[win_idx] = 1.0
            err = target - probs
            oja_term = torch.sum(self.W_drawer_gate * self.W_drawer_gate, dim=1, keepdim=True)
            self.W_drawer_gate.add_(float(lr) * (torch.outer(err, d_t) - 0.01 * probs.unsqueeze(1) * self.W_drawer_gate * oja_term))
            self.W_drawer_gate.clamp_(-3.0, 3.0)

            aux_act = torch.tanh(torch.mv(self.W_aux_concepts, d_t))
            aux_norm = torch.sum(self.W_aux_concepts * self.W_aux_concepts, dim=1, keepdim=True)
            self.W_aux_concepts.add_(float(lr) * (torch.outer(aux_act, d_t) - 0.02 * aux_act.unsqueeze(1) * self.W_aux_concepts * aux_norm))
            self.W_aux_concepts.clamp_(-3.0, 3.0)

            for gx_raw, gy_raw in train_pairs:
                gx = np.asarray(gx_raw, dtype=np.int32)
                gy = np.asarray(gy_raw, dtype=np.int32)
                hx = self.encode_grid_state(sensory_module, gx)
                hy = self.encode_grid_state(sensory_module, gy)
                diff_h = hy - hx
                prod_h = hy * hx
                joint_rel = torch.tensor(
                    np.concatenate([hx, hy, diff_h + 0.5 * prod_h], axis=0),
                    dtype=torch.float32,
                    device=dev,
                )
                rel_out = torch.tanh(torch.mv(self.W_rel_delta, joint_rel))
                rel_hebb = torch.outer(rel_out * (1.0 - rel_out * rel_out), joint_rel)
                self.W_rel_delta.add_(float(lr) * 0.15 * rel_hebb - 0.0005 * self.W_rel_delta)
                self.W_rel_delta.clamp_(-3.0, 3.0)

                act_syn = np.zeros(8, dtype=np.float32)
                act_syn[win_idx % 8] = 1.0
                joint_t = torch.tensor(np.concatenate([hx, act_syn], axis=0), dtype=torch.float32, device=dev)
                hy_t = torch.tensor(hy, dtype=torch.float32, device=dev)
                pred_hy = torch.tanh(torch.mv(self.W_world_forward, joint_t))
                world_err = hy_t - pred_hy
                self.W_world_forward.add_(float(lr) * 0.5 * torch.outer(world_err, joint_t))
                self.W_world_forward.clamp_(-3.0, 3.0)

    def update_world_model_hebbian(self, ha_vec, h_target, lr=0.008):
        with torch.no_grad():
            dev = self.W_world_forward.device
            ha_t = torch.as_tensor(ha_vec, dtype=torch.float32, device=dev).reshape(-1)
            ht_t = torch.as_tensor(h_target, dtype=torch.float32, device=dev).reshape(self.hid_dim)
            if ha_t.numel() < self.hid_dim + 8:
                pad = torch.zeros(self.hid_dim + 8, dtype=torch.float32, device=dev)
                pad[: ha_t.numel()] = ha_t
                ha_t = pad
            else:
                ha_t = ha_t[: self.hid_dim + 8]
            pred_h = torch.tanh(torch.mv(self.W_world_forward, ha_t))
            err_wm = (ht_t - pred_h) * (1.0 - pred_h * pred_h)
            self.W_world_forward.add_(float(lr) * torch.outer(err_wm, ha_t) - 0.0005 * self.W_world_forward)
            self.W_world_forward.clamp_(-2.5, 2.5)

    def to_numpy_dict(self):
        out = {}
        with torch.no_grad():
            for name, param in self.named_parameters():
                out[name] = param.detach().cpu().numpy().astype(np.float32).copy()
            for name, buf in self.named_buffers():
                out[name] = buf.detach().cpu().numpy().astype(np.float32).copy()
        return out

    def load_from_numpy_dict(self, tensors_dict):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in tensors_dict:
                    arr = np.asarray(tensors_dict[name], dtype=np.float32)
                    if arr.shape == tuple(param.shape):
                        param.copy_(torch.from_numpy(arr))
            for name, buf in self.named_buffers():
                if name in tensors_dict:
                    arr = np.asarray(tensors_dict[name], dtype=np.float32)
                    if arr.shape == tuple(buf.shape):
                        buf.copy_(torch.from_numpy(arr))

    def export_gguf(self, gguf_path, extra_meta=None):
        tensors = self.to_numpy_dict()
        return GGUFBinaryIO.save_gguf(gguf_path, tensors, metadata_dict=extra_meta)

    def load_gguf(self, gguf_path):
        tensors, metadata = GGUFBinaryIO.load_gguf(gguf_path)
        self.load_from_numpy_dict(tensors)
        return metadata


class MotorReflexModule(nn.Module):
    def __init__(self, hid_dim=128, num_primitives=32):
        super().__init__()
        self.hid_dim = int(hid_dim)
        self.num_primitives = int(num_primitives)
        self.W_act_3d = nn.Linear(self.hid_dim, 5)
        self.W_act_2d = nn.Linear(self.hid_dim, 8)
        self.W_joint_21d = nn.Linear(self.hid_dim, 21)
        self.fc_proprio_83_to_16 = nn.Linear(83, 16)
        self.W_q_mot = nn.Linear(self.hid_dim, 32, bias=False)
        self.W_k_mot = nn.Linear(self.hid_dim, 32, bias=False)
        self.fc_reflex_hidden = nn.Linear(self.hid_dim * 2 + 16, self.hid_dim)
        self.fc_primitive_logits = nn.Linear(self.hid_dim, self.num_primitives)
        self.fc_stop_gate = nn.Linear(self.hid_dim, 1)
        self.register_buffer("theta_motor", torch.full((self.hid_dim,), 0.25, dtype=torch.float32))
        self.ort_session = None
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.8)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, neocortex_state, relational_delta, proprioception):
        q_m = torch.tanh(self.W_q_mot(neocortex_state))
        k_m = torch.tanh(self.W_k_mot(relational_delta))
        q_m_spk = q_m * torch.sigmoid(8.0 * (torch.abs(q_m) - 0.20))
        k_m_spk = k_m * torch.sigmoid(8.0 * (torch.abs(k_m) - 0.20))
        attn_m = torch.sigmoid(torch.sum(q_m_spk * k_m_spk, dim=-1, keepdim=True) * 0.1767767)
        joint = torch.cat([neocortex_state, relational_delta * (0.50 + 0.50 * attn_m), proprioception], dim=-1)
        pre_mot = self.fc_reflex_hidden(joint)
        mot_spk_gate = torch.sigmoid(6.0 * (torch.abs(pre_mot) - self.theta_motor))
        reflex_h = torch.tanh(pre_mot) * (0.35 + 0.65 * mot_spk_gate)
        mot_combined = 0.50 * neocortex_state + 0.50 * reflex_h
        act_3d_logits = self.W_act_3d(mot_combined)
        act_2d_logits = self.W_act_2d(mot_combined)
        joint_effort_21d = torch.tanh(self.W_joint_21d(mot_combined))
        primitive_logits = self.fc_primitive_logits(reflex_h)
        stop_prob = torch.sigmoid(self.fc_stop_gate(reflex_h))
        return act_3d_logits, act_2d_logits, primitive_logits, stop_prob, joint_effort_21d

    def infer_motor_policy(self, neocortex_state, relational_delta, proprioception=None, use_onnx=True):
        ns = np.asarray(neocortex_state, dtype=np.float32).reshape(1, self.hid_dim)
        rd = np.asarray(relational_delta, dtype=np.float32).reshape(1, self.hid_dim)
        if proprioception is None:
            pr = np.zeros((1, 16), dtype=np.float32)
        else:
            pr_raw = np.asarray(proprioception, dtype=np.float32).reshape(-1)
            if pr_raw.size == 83:
                with torch.no_grad():
                    pr_t83 = torch.from_numpy(pr_raw.reshape(1, 83))
                    pr = torch.tanh(self.fc_proprio_83_to_16(pr_t83)).cpu().numpy().astype(np.float32)
            else:
                pr = pr_raw[:16].reshape(1, 16)

        if use_onnx and self.ort_session is not None:
            try:
                outputs = self.ort_session.run(
                    None,
                    {
                        "neocortex_state": ns,
                        "relational_delta": rd,
                        "proprioception": pr,
                    },
                )
                return {
                    "act_3d_logits": outputs[0][0].astype(np.float64),
                    "act_2d_logits": outputs[1][0].astype(np.float64),
                    "primitive_logits": outputs[2][0].astype(np.float64),
                    "stop_prob": float(outputs[3][0, 0]),
                    "joint_effort_21d": outputs[4][0].astype(np.float64),
                }
            except Exception:
                pass

        with torch.no_grad():
            ns_t = torch.from_numpy(ns)
            rd_t = torch.from_numpy(rd)
            pr_t = torch.from_numpy(pr)
            a3, a2, pl, sp, j21 = self.forward(ns_t, rd_t, pr_t)
            return {
                "act_3d_logits": a3[0].cpu().numpy().astype(np.float64),
                "act_2d_logits": a2[0].cpu().numpy().astype(np.float64),
                "primitive_logits": pl[0].cpu().numpy().astype(np.float64),
                "stop_prob": float(sp[0, 0].item()),
                "joint_effort_21d": j21[0].cpu().numpy().astype(np.float64),
            }

    def export_onnx(self, onnx_path):
        os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
        self.eval()
        dev = next(self.parameters()).device
        dummy_ns = torch.zeros(1, self.hid_dim, dtype=torch.float32, device=dev)
        dummy_rd = torch.zeros(1, self.hid_dim, dtype=torch.float32, device=dev)
        dummy_pr = torch.zeros(1, 16, dtype=torch.float32, device=dev)
        torch.onnx.export(
            self,
            (dummy_ns, dummy_rd, dummy_pr),
            onnx_path,
            input_names=["neocortex_state", "relational_delta", "proprioception"],
            output_names=["act_3d_logits", "act_2d_logits", "primitive_logits", "stop_prob", "joint_effort_21d"],
            dynamic_axes={
                "neocortex_state": {0: "batch_size"},
                "relational_delta": {0: "batch_size"},
                "proprioception": {0: "batch_size"},
                "act_3d_logits": {0: "batch_size"},
                "act_2d_logits": {0: "batch_size"},
                "primitive_logits": {0: "batch_size"},
                "stop_prob": {0: "batch_size"},
                "joint_effort_21d": {0: "batch_size"},
            },
            opset_version=17,
            dynamo=False,
        )
        if HAS_ONNX_RUNTIME:
            self.load_onnx_session(onnx_path)
        return True

    def load_onnx_session(self, onnx_path):
        if HAS_ONNX_RUNTIME and os.path.exists(onnx_path):
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 1
            sess_opts.inter_op_num_threads = 1
            avail = ort.get_available_providers()
            provs = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]
            self.ort_session = ort.InferenceSession(
                onnx_path, sess_options=sess_opts, providers=provs
            )
            return True
        return False


class UnifiedBiologicalBrainSystem:
    def __init__(self, hid_dim=128, artifact_dir=None, seed=42):
        self.hid_dim = int(hid_dim)
        if artifact_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            artifact_dir = os.path.join(base_dir, "artifacts")
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)

        self.sensory_cortex = SensoryCortexModule(hid_dim=self.hid_dim)
        self.neocortex_logic = NeocortexSystem2Module(hid_dim=self.hid_dim, seed=seed)
        self.motor_reflex = MotorReflexModule(hid_dim=self.hid_dim, num_primitives=32)

    @property
    def paths(self):
        return {
            "sensory_pth": os.path.join(self.artifact_dir, "sensory_cortex.pth"),
            "sensory_onnx": os.path.join(self.artifact_dir, "sensory_cortex.onnx"),
            "neocortex_pth": os.path.join(self.artifact_dir, "neocortex_logic.pth"),
            "neocortex_gguf": os.path.join(self.artifact_dir, "neocortex_logic.gguf"),
            "motor_pth": os.path.join(self.artifact_dir, "motor_reflex.pth"),
            "motor_onnx": os.path.join(self.artifact_dir, "motor_reflex.onnx"),
        }

    def step_sensorimotor_closed_loop(
        self,
        grid,
        auditory_vec=None,
        proprioception=None,
        relational_delta=None,
        gaze_y=None,
        gaze_x=None,
        use_onnx=True,
        mental_rollout_depth=1,
    ):
        sens_feat, ret_16 = self.sensory_cortex.encode_grid(
            grid, auditory_vec=auditory_vec, gaze_y=gaze_y, gaze_x=gaze_x, use_onnx=use_onnx
        )
        aud_11 = (
            np.zeros(11, dtype=np.float64)
            if auditory_vec is None
            else np.asarray(auditory_vec, dtype=np.float64).reshape(11)
        )
        obs_27 = np.concatenate([ret_16, aud_11], axis=0)

        with torch.no_grad():
            neo = self.neocortex_logic
            dev = neo.W_in.device
            obs_t = torch.tensor(obs_27, dtype=torch.float32, device=dev).unsqueeze(0)
            h_p_t = getattr(self, "_last_neo_state_tensor", None)
            if not isinstance(h_p_t, torch.Tensor) or h_p_t.device != dev:
                h_p_t = None
            mono_p_t = getattr(self, "_last_mono_out_tensor", None)
            if not isinstance(mono_p_t, torch.Tensor) or mono_p_t.device != dev:
                mono_p_t = None
            h_next_t, macro_out_t, mono_out_t, snd_pred_t = neo(obs_t, h_prev=h_p_t, modulatory_mono=mono_p_t)
            self._last_neo_state_tensor = h_next_t.detach()
            self._last_mono_out_tensor = mono_out_t.detach()
            neo_state = h_next_t[0].cpu().numpy().astype(np.float64)
            w_body = neo.W_body_assembly.detach().cpu().numpy().astype(np.float64)
            body_11 = np.tanh(w_body @ neo_state)

        rel_d = (
            sens_feat
            if relational_delta is None
            else np.asarray(relational_delta, dtype=np.float64).reshape(self.hid_dim)
        )
        motor_out = self.motor_reflex.infer_motor_policy(
            neo_state, rel_d, proprioception=proprioception, use_onnx=use_onnx
        )
        prim_logits = motor_out["primitive_logits"].copy()
        act_2d = motor_out["act_2d_logits"].copy()
        imagination = self.neocortex_logic.imagine_action_outcome(
            neo_state, act_2d, hy_goal=np.tanh(neo_state + rel_d)
        )

        curr_hy = imagination["imagined_hy"]
        total_utility = float(imagination.get("mental_utility", 0.0))
        depth_runs = max(1, min(int(mental_rollout_depth), 4))
        for _ in range(depth_runs - 1):
            with torch.no_grad():
                pred_ret = imagination.get("predicted_retinal", ret_16)
                pred_obs27 = np.concatenate([pred_ret, aud_11], axis=0)
                pred_obs_t = torch.tensor(pred_obs27, dtype=torch.float32, device=dev).unsqueeze(0)
                h_next_sim_t, _, _, _ = neo(
                    pred_obs_t,
                    h_prev=torch.tensor(curr_hy, dtype=torch.float32, device=dev).unsqueeze(0),
                    modulatory_mono=mono_out_t,
                )
                sim_neo_state = h_next_sim_t[0].cpu().numpy().astype(np.float64)
            sim_motor = self.motor_reflex.infer_motor_policy(
                sim_neo_state, rel_d, proprioception=proprioception, use_onnx=use_onnx
            )
            sim_act2d = sim_motor["act_2d_logits"]
            sim_imag = self.neocortex_logic.imagine_action_outcome(
                sim_neo_state, sim_act2d, hy_goal=np.tanh(sim_neo_state + rel_d)
            )
            curr_hy = sim_imag["imagined_hy"]
            sim_u = float(sim_imag.get("mental_utility", 0.0))
            total_utility += sim_u
            if sim_u > 0.0:
                act_2d = (0.75 * act_2d + 0.25 * sim_act2d).astype(np.float64)

        gaze_dy = float(np.tanh(0.6 * prim_logits[0] + 0.4 * (body_11[1] - body_11[0])))
        gaze_dx = float(np.tanh(0.6 * prim_logits[1] + 0.4 * (body_11[3] - body_11[2])))

        return {
            "sensory_features": sens_feat,
            "retinal_16": ret_16,
            "neocortex_state": neo_state,
            "body_assembly_11": body_11,
            "act_3d_logits": motor_out["act_3d_logits"],
            "act_2d_logits": act_2d,
            "primitive_logits": prim_logits,
            "stop_prob": motor_out["stop_prob"],
            "joint_effort_21d": motor_out.get("joint_effort_21d"),
            "gaze_shift_yx": (gaze_dy, gaze_dx),
            "wm_gate_vec": body_11[4:7],
            "finger_op_vec": body_11[7:11],
            "imagined_hy": curr_hy,
            "predicted_retinal": imagination["predicted_retinal"],
            "mental_utility": total_utility / depth_runs,
            "macro_monologue_4": macro_out_t[0].detach().cpu().numpy().astype(np.float64),
            "monoamine_dynamics_4": mono_out_t[0].detach().cpu().numpy().astype(np.float64),
            "sound_pred_11": snd_pred_t[0].detach().cpu().numpy().astype(np.float64),
            "mental_rollout_depth_achieved": depth_runs,
        }

    def sync_from_stage1_brain(self, stage1_brain):
        def _to_2d_np(val):
            if isinstance(val, torch.Tensor):
                arr = val.detach().cpu().numpy().astype(np.float32)
            else:
                arr = np.asarray(val, dtype=np.float32)
            while arr.ndim > 2:
                arr = np.mean(arr, axis=0)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr

        def _fit_matrix(val, target_rows, target_cols):
            src = _to_2d_np(val)
            out = np.zeros((target_rows, target_cols), dtype=np.float32)
            r = min(src.shape[0], target_rows)
            c = min(src.shape[1], target_cols)
            out[:r, :c] = src[:r, :c]
            if target_rows > r and r > 0:
                reps_r = (target_rows + r - 1) // r
                tiled = np.tile(src, (reps_r, 1))[:target_rows, :c]
                out[:, :c] = tiled
            if target_cols > c and c > 0:
                reps_c = (target_cols + c - 1) // c
                tiled = np.tile(out[:, :c], (1, reps_c))[:, :target_cols]
                out = tiled
            return out

        def _fit_vector(val, target_len):
            if isinstance(val, torch.Tensor):
                src = val.detach().cpu().numpy().astype(np.float32)
            else:
                src = np.asarray(val, dtype=np.float32)
            if src.ndim > 1:
                src = np.mean(src, axis=0).ravel()
            else:
                src = src.ravel()
            if src.size == 0:
                return np.zeros(target_len, dtype=np.float32)
            reps = (target_len + src.size - 1) // src.size
            return np.tile(src, reps)[:target_len].astype(np.float32)

        def _get_attr_or_key(obj, name):
            if isinstance(obj, dict) or hasattr(obj, "files"):
                return obj[name] if name in obj else None
            return getattr(obj, name, None)

        with torch.no_grad():
            for attr, shape in [
                ("W_in", (self.hid_dim, 27)),
                ("Wh", (self.hid_dim, self.hid_dim)),
                ("Wq", (self.hid_dim, self.hid_dim)),
                ("Wk", (self.hid_dim, self.hid_dim)),
                ("Wv", (self.hid_dim, self.hid_dim)),
                ("W_pred", (16, self.hid_dim)),
                ("W_sound_pred", (11, self.hid_dim)),
                ("W_symbol_emit", (8, self.hid_dim)),
                ("W_symbol_lexicon", (16, 11)),
                ("W_body_assembly", (11, self.hid_dim)),
                ("W_macro_out", (4, self.hid_dim)),
                ("W_macro_in", (self.hid_dim, 4)),
                ("W_mono_out", (4, self.hid_dim)),
                ("W_mono_in", (self.hid_dim, 4)),
            ]:
                val = _get_attr_or_key(stage1_brain, attr)
                if val is None and attr == "W_in":
                    val = _get_attr_or_key(stage1_brain, "W_enc")
                if val is None and attr == "W_pred":
                    val = _get_attr_or_key(stage1_brain, "Wpred_retinal")
                if val is not None:
                    if attr == "Wh":
                        src_wh = _to_2d_np(val)
                        if src_wh.shape[0] > self.hid_dim:
                            val = src_wh[-self.hid_dim:, :]
                    fitted = _fit_matrix(val, shape[0], shape[1])
                    getattr(self.neocortex_logic, attr).copy_(torch.from_numpy(fitted))

            th_val = _get_attr_or_key(stage1_brain, "theta")
            if th_val is not None:
                fitted_th = _fit_vector(th_val, self.hid_dim)
                self.neocortex_logic.theta.copy_(torch.from_numpy(fitted_th))

            wo_val = _get_attr_or_key(stage1_brain, "Wo")
            if wo_val is None:
                wo_val = _get_attr_or_key(stage1_brain, "W_act_3d")
            if wo_val is not None:
                w3d = _fit_matrix(wo_val, 5, self.hid_dim)
                self.motor_reflex.W_act_3d.weight.copy_(torch.from_numpy(w3d))

            w2d_val = _get_attr_or_key(stage1_brain, "W_act_2d")
            if w2d_val is None and wo_val is not None:
                w2d_val = wo_val
            if w2d_val is not None:
                w2d = _fit_matrix(w2d_val, 8, self.hid_dim)
                self.motor_reflex.W_act_2d.weight.copy_(torch.from_numpy(w2d))

            w_ret_val = _get_attr_or_key(stage1_brain, "W_enc_retinal")
            if w_ret_val is not None:
                self.sensory_cortex.fc_retinal16.weight.copy_(
                    torch.from_numpy(_fit_matrix(_to_2d_np(w_ret_val).T, 16, self.hid_dim))
                )
            else:
                w_in_np = self.neocortex_logic.W_in.detach().cpu().numpy()
                self.sensory_cortex.fc_retinal16.weight.copy_(
                    torch.from_numpy(_fit_matrix(w_in_np[:, :16].T, 16, self.hid_dim))
                )

    def sync_to_stage1_brain(self, stage1_brain):
        with torch.no_grad():
            dev = getattr(stage1_brain, "Wq", torch.zeros(1)).device
            neo = self.neocortex_logic
            wq_t = neo.Wq.detach().to(dev)
            wk_t = neo.Wk.detach().to(dev)
            wv_t = neo.Wv.detach().to(dev)
            wh_t = neo.Wh.detach().to(dev)
            w_in_t = neo.W_in.detach().to(dev)
            w_pred_t = neo.W_pred.detach().to(dev)
            w_mac_o = neo.W_macro_out.detach().to(dev)
            w_mac_i = neo.W_macro_in.detach().to(dev)
            w_mon_o = neo.W_mono_out.detach().to(dev)
            w_mon_i = neo.W_mono_in.detach().to(dev)
            w_body_t = neo.W_body_assembly.detach().to(dev)
            w_sym_t = neo.W_symbol_lexicon.detach().to(dev)
            w_act3d_t = self.motor_reflex.W_act_3d.weight.detach().to(dev)
            w_ret_t = self.sensory_cortex.fc_retinal16.weight.detach().to(dev).T

            if hasattr(stage1_brain, "Wq") and stage1_brain.Wq.shape[1:] == wq_t.shape:
                stage1_brain.Wq.copy_(0.50 * stage1_brain.Wq + 0.50 * wq_t.unsqueeze(0))
            if hasattr(stage1_brain, "Wk") and stage1_brain.Wk.shape[1:] == wk_t.shape:
                stage1_brain.Wk.copy_(0.50 * stage1_brain.Wk + 0.50 * wk_t.unsqueeze(0))
            if hasattr(stage1_brain, "Wv") and stage1_brain.Wv.shape[1:] == wv_t.shape:
                stage1_brain.Wv.copy_(0.50 * stage1_brain.Wv + 0.50 * wv_t.unsqueeze(0))
            if hasattr(stage1_brain, "Wh"):
                if stage1_brain.Wh.shape[1:] == (self.hid_dim, self.hid_dim):
                    stage1_brain.Wh.copy_(0.50 * stage1_brain.Wh + 0.50 * wh_t.unsqueeze(0))
                elif stage1_brain.Wh.shape[1] > self.hid_dim and stage1_brain.Wh.shape[2] == self.hid_dim:
                    stage1_brain.Wh[:, -self.hid_dim:, :].copy_(
                        0.50 * stage1_brain.Wh[:, -self.hid_dim:, :] + 0.50 * wh_t.unsqueeze(0)
                    )
            if hasattr(stage1_brain, "W_enc") and stage1_brain.W_enc.shape[1] == self.hid_dim:
                c_in = min(stage1_brain.W_enc.shape[2], w_in_t.shape[1])
                stage1_brain.W_enc[:, :, :c_in].copy_(
                    0.50 * stage1_brain.W_enc[:, :, :c_in] + 0.50 * w_in_t[:, :c_in].unsqueeze(0)
                )
            if hasattr(stage1_brain, "W_enc_retinal") and stage1_brain.W_enc_retinal.shape[1:] == w_ret_t.shape:
                stage1_brain.W_enc_retinal.copy_(0.50 * stage1_brain.W_enc_retinal + 0.50 * w_ret_t.unsqueeze(0))
            if hasattr(stage1_brain, "Wx_retinal") and stage1_brain.Wx_retinal.shape[1:] == w_ret_t.shape:
                stage1_brain.Wx_retinal.copy_(0.50 * stage1_brain.Wx_retinal + 0.50 * w_ret_t.unsqueeze(0))
            if hasattr(stage1_brain, "Wpred_retinal") and stage1_brain.Wpred_retinal.shape[1:] == w_pred_t.shape:
                stage1_brain.Wpred_retinal.copy_(0.50 * stage1_brain.Wpred_retinal + 0.50 * w_pred_t.unsqueeze(0))
            if hasattr(stage1_brain, "Wo") and stage1_brain.Wo.shape[1:] == w_act3d_t.shape:
                stage1_brain.Wo.copy_(0.50 * stage1_brain.Wo + 0.50 * w_act3d_t.unsqueeze(0))
            if hasattr(stage1_brain, "W_macro_out") and stage1_brain.W_macro_out.shape[1:] == w_mac_o.shape:
                stage1_brain.W_macro_out.copy_(0.50 * stage1_brain.W_macro_out + 0.50 * w_mac_o.unsqueeze(0))
            if hasattr(stage1_brain, "W_macro_in") and stage1_brain.W_macro_in.shape[1:] == w_mac_i.shape:
                stage1_brain.W_macro_in.copy_(0.50 * stage1_brain.W_macro_in + 0.50 * w_mac_i.unsqueeze(0))
            if hasattr(stage1_brain, "W_mono_out") and stage1_brain.W_mono_out.shape[1:] == w_mon_o.shape:
                stage1_brain.W_mono_out.copy_(0.50 * stage1_brain.W_mono_out + 0.50 * w_mon_o.unsqueeze(0))
            if hasattr(stage1_brain, "W_mono_in") and stage1_brain.W_mono_in.shape[1:] == w_mon_i.shape:
                stage1_brain.W_mono_in.copy_(0.50 * stage1_brain.W_mono_in + 0.50 * w_mon_i.unsqueeze(0))
            if hasattr(stage1_brain, "W_body_assembly") and stage1_brain.W_body_assembly.shape[1:] == w_body_t.shape:
                stage1_brain.W_body_assembly.copy_(0.50 * stage1_brain.W_body_assembly + 0.50 * w_body_t.unsqueeze(0))
            if hasattr(stage1_brain, "W_symbol_lexicon") and stage1_brain.W_symbol_lexicon.shape[1:] == w_sym_t.shape:
                stage1_brain.W_symbol_lexicon.copy_(
                    F.normalize(0.50 * stage1_brain.W_symbol_lexicon + 0.50 * w_sym_t.unsqueeze(0), dim=-1, eps=1e-6)
                )
            if hasattr(stage1_brain, "theta"):
                th_neo = neo.theta.detach().to(dev)
                if stage1_brain.theta.ndim == 2:
                    stage1_brain.theta.copy_(0.50 * stage1_brain.theta + 0.50 * th_neo.unsqueeze(0))
                else:
                    stage1_brain.theta.copy_(0.50 * stage1_brain.theta + 0.50 * th_neo)

    def educate_three_brains_in_3d_arena(self, arena, stage1_brain, n_epochs=6, lr=0.008):
        self.sync_from_stage1_brain(stage1_brain)
        obs_all = arena.get_observations().detach().cpu().numpy().astype(np.float32)
        retinas_t, fov_sal_t, per_sal_t = arena.render_first_person_retinal_grids(retina_h=9, retina_w=9)
        ret_obs_16d_t = arena.extract_retinal_16d_observations(retinas_t, fov_sal_t, per_sal_t)
        retinas_np = retinas_t.detach().cpu().numpy().astype(np.int32)
        ret_16d_np = ret_obs_16d_t.detach().cpu().numpy().astype(np.float32)
        h_np = stage1_brain.h.detach().cpu().numpy().astype(np.float32)
        w_snd_np = self.neocortex_logic.W_sound_pred.detach().cpu().numpy().astype(np.float32)
        hippo_acts_np = stage1_brain.hippo_acts.mean(dim=1).detach().cpu().numpy().astype(np.float32)
        alive_np = arena.alive.detach().cpu().numpy()
        hp_np = arena.hp.detach().cpu().numpy().astype(np.float32)
        en_np = arena.energy.detach().cpu().numpy().astype(np.float32)
        pain_np = arena.pain.detach().cpu().numpy().astype(np.float32)
        prop_83_np = (
            arena.get_proprioception_83d().detach().cpu().numpy().astype(np.float32)
            if hasattr(arena, "get_proprioception_83d")
            else None
        )

        n_batch = min(32, int(arena.pos.shape[0]), retinas_np.shape[0])
        patches_list = []
        stats_list = []
        aud_list = []
        target_h_list = []
        target_ret_list = []
        prop_list = []
        prop83_list = []
        target_act3d_list = []
        target_act2d_list = []
        target_joint21_list = []
        target_prim_list = []
        target_stop_list = []

        for idx in range(n_batch):
            ret_grid_i = retinas_np[idx]
            p_np, s_np = self.sensory_cortex.extract_grid_patch_and_stats(ret_grid_i)
            patches_list.append(p_np[0])
            stats_list.append(s_np[0])
            aud_v = np.tanh(w_snd_np @ h_np[idx]).astype(np.float32)
            aud_list.append(aud_v)
            target_h_list.append(h_np[idx])
            target_ret_list.append(ret_16d_np[idx])

            pr_v = np.zeros(16, dtype=np.float32)
            pr_v[0] = hp_np[idx]
            pr_v[1] = en_np[idx]
            pr_v[2] = pain_np[idx]
            pr_v[3] = 1.0 if alive_np[idx] else 0.0
            pr_v[4:16] = obs_all[idx, 16:28]
            prop_list.append(pr_v)
            if prop_83_np is not None and idx < prop_83_np.shape[0]:
                prop83_list.append(prop_83_np[idx])

            a3d = hippo_acts_np[idx].copy()
            if np.linalg.norm(a3d) < 1e-5:
                a3d = np.tanh(stage1_brain.Wo[idx].detach().cpu().numpy() @ h_np[idx]).astype(np.float32)
            target_act3d_list.append(a3d)

            with torch.no_grad():
                h_t_i = torch.tensor(h_np[idx], dtype=torch.float32)
                ff_j21 = torch.tanh(self.motor_reflex.W_joint_21d(h_t_i)).detach().cpu().numpy().astype(np.float32)
            fb_torque_21 = np.zeros(21, dtype=np.float32)
            cpg_ref_21 = np.zeros(21, dtype=np.float32)
            upright_i = 1.0
            if hasattr(arena, "rl_env"):
                rl_e = arena.rl_env
                if hasattr(rl_e, "agent_cerebellar_joint_torques_21d") and idx < rl_e.agent_cerebellar_joint_torques_21d.shape[0]:
                    fb_torque_21 = rl_e.agent_cerebellar_joint_torques_21d[idx].detach().cpu().numpy().astype(np.float32)
                if hasattr(rl_e, "agent_cpg_target_joints_21d") and idx < rl_e.agent_cpg_target_joints_21d.shape[0]:
                    cpg_ref_21 = rl_e.agent_cpg_target_joints_21d[idx].detach().cpu().numpy().astype(np.float32)
                if hasattr(rl_e, "agent_torso_upright_cos") and idx < rl_e.agent_torso_upright_cos.shape[0]:
                    upright_i = float(rl_e.agent_torso_upright_cos[idx].item())
            postural_gate = float(np.clip((upright_i - 0.25) / 0.75, 0.15, 1.0))
            j21 = np.clip(
                postural_gate * (0.45 * ff_j21 + 0.55 * fb_torque_21) + (1.0 - postural_gate) * cpg_ref_21,
                -1.0,
                1.0,
            ).astype(np.float32)
            target_joint21_list.append(j21)

            a2d = np.zeros(8, dtype=np.float32)
            a2d[:5] = a3d
            a2d[5:8] = aud_v[:3]
            target_act2d_list.append(a2d)

            p_targ = np.zeros(32, dtype=np.float32)
            p_targ[:11] = aud_v
            p_targ[11:19] = a2d
            p_targ[19:32] = h_np[idx, :13]
            target_prim_list.append(p_targ)
            stop_val = 1.0 if (hp_np[idx] > 0.6 and pain_np[idx] < 0.1 and en_np[idx] > 0.5) else 0.0
            target_stop_list.append([stop_val])

        vp_batch = torch.tensor(np.stack(patches_list, axis=0), dtype=torch.float32)
        ss_batch = torch.tensor(np.stack(stats_list, axis=0), dtype=torch.float32)
        av_batch = torch.tensor(np.stack(aud_list, axis=0), dtype=torch.float32)
        th_batch = torch.tensor(np.stack(target_h_list, axis=0), dtype=torch.float32)
        tr_batch = torch.tensor(np.stack(target_ret_list, axis=0), dtype=torch.float32)
        pr_batch = torch.tensor(np.stack(prop_list, axis=0), dtype=torch.float32)
        pr83_batch = (
            torch.tensor(np.stack(prop83_list, axis=0), dtype=torch.float32)
            if len(prop83_list) == n_batch
            else None
        )
        ta3_batch = torch.tensor(np.stack(target_act3d_list, axis=0), dtype=torch.float32)
        ta2_batch = torch.tensor(np.stack(target_act2d_list, axis=0), dtype=torch.float32)
        tj21_batch = torch.tensor(np.stack(target_joint21_list, axis=0), dtype=torch.float32)
        tpr_batch = torch.tensor(np.stack(target_prim_list, axis=0), dtype=torch.float32)
        tst_batch = torch.tensor(np.stack(target_stop_list, axis=0), dtype=torch.float32)

        self.sensory_cortex.eval()
        self.motor_reflex.eval()
        self.neocortex_logic.eval()

        s_loss_init = 0.0
        s_loss_final = 0.0
        m_loss_init = 0.0
        m_loss_final = 0.0
        wm_loss_init = 0.0
        wm_loss_final = 0.0
        b_sz = float(max(1, vp_batch.size(0)))
        eta_local = float(lr) * 2.5
        oja_decay = 0.0005

        with torch.no_grad():
            for ep in range(max(1, int(n_epochs))):
                c1_out = F.gelu(self.sensory_cortex.conv1(vp_batch))
                c2_out = F.gelu(self.sensory_cortex.conv2(c1_out))
                vis_pool = F.adaptive_avg_pool2d(c2_out, (3, 3)).reshape(vp_batch.size(0), -1)
                v_feat = torch.tanh(self.sensory_cortex.fc_visual(vis_pool))
                s_feat_part = torch.tanh(self.sensory_cortex.fc_spatial(ss_batch))
                a_feat_part = torch.tanh(self.sensory_cortex.fc_auditory(av_batch))
                q_v = torch.tanh(self.sensory_cortex.W_q_sens(v_feat))
                sa_cat = torch.cat([s_feat_part, a_feat_part], dim=-1)
                k_sa = torch.tanh(self.sensory_cortex.W_k_sens(sa_cat))
                v_sa = torch.tanh(self.sensory_cortex.W_v_sens(sa_cat))
                q_spk = q_v * torch.sigmoid(8.0 * (torch.abs(q_v) - 0.20))
                k_spk = k_sa * torch.sigmoid(8.0 * (torch.abs(k_sa) - 0.20))
                attn_gate = torch.sigmoid(torch.sum(q_spk * k_spk, dim=-1, keepdim=True) * 0.1767767)
                v_mod = torch.tanh(v_feat + 0.35 * attn_gate * v_sa)
                cat_feat = torch.cat([v_mod, s_feat_part, a_feat_part], dim=-1)
                pre_sens = self.sensory_cortex.fc_fusion(cat_feat)
                spk_gate = torch.sigmoid(6.0 * (torch.abs(pre_sens) - self.sensory_cortex.theta_sensory))
                s_feat = torch.tanh(pre_sens) * (0.35 + 0.65 * spk_gate)
                s_ret = torch.sigmoid(self.sensory_cortex.fc_retinal16(s_feat))
                loss_s = F.mse_loss(s_feat, th_batch) + F.mse_loss(s_ret, tr_batch)

                loc_err_ret = (tr_batch - s_ret) * s_ret * (1.0 - s_ret)
                loc_err_feat = (th_batch - s_feat) * (1.0 - torch.tanh(pre_sens) ** 2)
                self.sensory_cortex.fc_retinal16.weight.add_(
                    eta_local * (loc_err_ret.T @ s_feat) / b_sz - oja_decay * self.sensory_cortex.fc_retinal16.weight
                )
                self.sensory_cortex.fc_retinal16.bias.add_(eta_local * loc_err_ret.mean(dim=0))
                self.sensory_cortex.fc_fusion.weight.add_(
                    eta_local * (loc_err_feat.T @ cat_feat) / b_sz - oja_decay * self.sensory_cortex.fc_fusion.weight
                )
                self.sensory_cortex.fc_fusion.bias.add_(eta_local * loc_err_feat.mean(dim=0))

                grad_cat = loc_err_feat @ self.sensory_cortex.fc_fusion.weight
                dim_v = v_feat.shape[1]
                dim_sp = s_feat_part.shape[1]
                dim_au = a_feat_part.shape[1]
                loc_err_v = grad_cat[:, :dim_v] * (1.0 - v_feat * v_feat)
                loc_err_sp = grad_cat[:, dim_v : dim_v + dim_sp] * (1.0 - s_feat_part * s_feat_part)
                loc_err_au = grad_cat[:, dim_v + dim_sp : dim_v + dim_sp + dim_au] * (1.0 - a_feat_part * a_feat_part)

                self.sensory_cortex.fc_visual.weight.add_(
                    eta_local * (loc_err_v.T @ vis_pool) / b_sz - oja_decay * self.sensory_cortex.fc_visual.weight
                )
                self.sensory_cortex.fc_visual.bias.add_(eta_local * loc_err_v.mean(dim=0))
                self.sensory_cortex.fc_spatial.weight.add_(
                    eta_local * (loc_err_sp.T @ ss_batch) / b_sz - oja_decay * self.sensory_cortex.fc_spatial.weight
                )
                self.sensory_cortex.fc_spatial.bias.add_(eta_local * loc_err_sp.mean(dim=0))
                self.sensory_cortex.fc_auditory.weight.add_(
                    eta_local * (loc_err_au.T @ av_batch) / b_sz - oja_decay * self.sensory_cortex.fc_auditory.weight
                )
                self.sensory_cortex.fc_auditory.bias.add_(eta_local * loc_err_au.mean(dim=0))
                self.sensory_cortex.W_v_sens.weight.add_(
                    eta_local * 0.35 * ((loc_err_v * attn_gate).T @ sa_cat) / b_sz - oja_decay * self.sensory_cortex.W_v_sens.weight
                )
                self.sensory_cortex.W_q_sens.weight.add_(
                    eta_local * 0.12 * (q_spk.T @ v_feat) / b_sz - oja_decay * self.sensory_cortex.W_q_sens.weight
                )
                self.sensory_cortex.W_k_sens.weight.add_(
                    eta_local * 0.12 * (k_spk.T @ sa_cat) / b_sz - oja_decay * self.sensory_cortex.W_k_sens.weight
                )
                pred_c1 = F.conv_transpose2d(c2_out.detach(), self.sensory_cortex.conv2.weight, padding=1)
                if pred_c1.shape != c1_out.shape:
                    pred_c1 = F.interpolate(pred_c1, size=(c1_out.shape[2], c1_out.shape[3]), mode="nearest")
                err_c1 = c1_out.detach() - pred_c1
                pred_vp = F.conv_transpose2d(c1_out.detach(), self.sensory_cortex.conv1.weight, padding=1)
                if pred_vp.shape != vp_batch.shape:
                    pred_vp = F.interpolate(pred_vp, size=(vp_batch.shape[2], vp_batch.shape[3]), mode="nearest")
                err_vp = vp_batch.detach() - pred_vp

                err_c1_pad = F.pad(err_c1, (1, 1, 1, 1))
                err_vp_pad = F.pad(err_vp, (1, 1, 1, 1))
                h2, w2 = c2_out.shape[2], c2_out.shape[3]
                h1, w1 = c1_out.shape[2], c1_out.shape[3]
                w2_delta = torch.zeros_like(self.sensory_cortex.conv2.weight)
                for dr in range(3):
                    for dc in range(3):
                        c1_shift = err_c1_pad[:, :, dr : dr + h2, dc : dc + w2]
                        w2_delta[:, :, dr, dc] = torch.einsum("bchw,bdhw->cd", c2_out, c1_shift) / (b_sz * h2 * w2)
                self.sensory_cortex.conv2.weight.add_(eta_local * 0.05 * w2_delta - oja_decay * self.sensory_cortex.conv2.weight)

                w1_delta = torch.zeros_like(self.sensory_cortex.conv1.weight)
                for dr in range(3):
                    for dc in range(3):
                        vp_shift = err_vp_pad[:, :, dr : dr + h1, dc : dc + w1]
                        w1_delta[:, :, dr, dc] = torch.einsum("bchw,bdhw->cd", c1_out, vp_shift) / (b_sz * h1 * w1)
                self.sensory_cortex.conv1.weight.add_(eta_local * 0.05 * w1_delta - oja_decay * self.sensory_cortex.conv1.weight)
                self.sensory_cortex.theta_sensory.copy_(
                    torch.clamp(self.sensory_cortex.theta_sensory + 0.01 * (spk_gate.mean(dim=0) - 0.35), 0.12, 0.55)
                )

                s_feat_det = s_feat.detach()
                a3_out = self.motor_reflex.W_act_3d(th_batch)
                a2_out = self.motor_reflex.W_act_2d(th_batch)
                j21_out = torch.tanh(self.motor_reflex.W_joint_21d(th_batch))
                if pr83_batch is not None:
                    pr16_proj = torch.tanh(self.motor_reflex.fc_proprio_83_to_16(pr83_batch))
                    pr_used = 0.50 * pr_batch + 0.50 * pr16_proj
                else:
                    pr16_proj = None
                    pr_used = pr_batch
                q_m = torch.tanh(self.motor_reflex.W_q_mot(th_batch))
                k_m = torch.tanh(self.motor_reflex.W_k_mot(s_feat_det))
                q_m_spk = q_m * torch.sigmoid(8.0 * (torch.abs(q_m) - 0.20))
                k_m_spk = k_m * torch.sigmoid(8.0 * (torch.abs(k_m) - 0.20))
                attn_m = torch.sigmoid(torch.sum(q_m_spk * k_m_spk, dim=-1, keepdim=True) * 0.1767767)
                joint_m = torch.cat([th_batch, s_feat_det * (0.50 + 0.50 * attn_m), pr_used], dim=-1)
                pre_mot = self.motor_reflex.fc_reflex_hidden(joint_m)
                mot_spk_gate = torch.sigmoid(6.0 * (torch.abs(pre_mot) - self.motor_reflex.theta_motor))
                reflex_h = torch.tanh(pre_mot) * (0.35 + 0.65 * mot_spk_gate)
                pr_out = self.motor_reflex.fc_primitive_logits(reflex_h)
                st_out = torch.sigmoid(self.motor_reflex.fc_stop_gate(reflex_h))

                loss_m = (
                    F.mse_loss(a3_out, ta3_batch)
                    + F.mse_loss(a2_out, ta2_batch)
                    + F.mse_loss(j21_out, tj21_batch)
                    + F.mse_loss(pr_out, tpr_batch)
                    + F.binary_cross_entropy(st_out.clamp(1e-5, 1.0 - 1e-5), tst_batch)
                )

                err_a3 = ta3_batch - a3_out
                err_a2 = ta2_batch - a2_out
                err_j21 = (tj21_batch - j21_out) * (1.0 - j21_out * j21_out)
                err_pr = tpr_batch - pr_out
                err_st = tst_batch - st_out

                self.motor_reflex.W_act_3d.weight.add_(
                    eta_local * (err_a3.T @ th_batch) / b_sz - oja_decay * self.motor_reflex.W_act_3d.weight
                )
                self.motor_reflex.W_act_3d.bias.add_(eta_local * err_a3.mean(dim=0))
                self.motor_reflex.W_act_2d.weight.add_(
                    eta_local * (err_a2.T @ th_batch) / b_sz - oja_decay * self.motor_reflex.W_act_2d.weight
                )
                self.motor_reflex.W_act_2d.bias.add_(eta_local * err_a2.mean(dim=0))
                self.motor_reflex.W_joint_21d.weight.add_(
                    eta_local * (err_j21.T @ th_batch) / b_sz - oja_decay * self.motor_reflex.W_joint_21d.weight
                )
                self.motor_reflex.W_joint_21d.bias.add_(eta_local * err_j21.mean(dim=0))
                if pr83_batch is not None and pr16_proj is not None:
                    err_pr83 = (pr_batch - pr16_proj) * (1.0 - pr16_proj * pr16_proj)
                    self.motor_reflex.fc_proprio_83_to_16.weight.add_(
                        eta_local * (err_pr83.T @ pr83_batch) / b_sz - oja_decay * self.motor_reflex.fc_proprio_83_to_16.weight
                    )
                    self.motor_reflex.fc_proprio_83_to_16.bias.add_(eta_local * err_pr83.mean(dim=0))
                self.motor_reflex.fc_primitive_logits.weight.add_(
                    eta_local * (err_pr.T @ reflex_h) / b_sz - oja_decay * self.motor_reflex.fc_primitive_logits.weight
                )
                self.motor_reflex.fc_primitive_logits.bias.add_(eta_local * err_pr.mean(dim=0))
                self.motor_reflex.fc_stop_gate.weight.add_(
                    eta_local * (err_st.T @ reflex_h) / b_sz - oja_decay * self.motor_reflex.fc_stop_gate.weight
                )
                self.motor_reflex.fc_stop_gate.bias.add_(eta_local * err_st.mean(dim=0))

                loc_err_rh = (th_batch[:, :reflex_h.shape[-1]] - reflex_h) * (1.0 - torch.tanh(pre_mot) ** 2)
                self.motor_reflex.fc_reflex_hidden.weight.add_(
                    eta_local * (loc_err_rh.T @ joint_m) / b_sz - oja_decay * self.motor_reflex.fc_reflex_hidden.weight
                )
                self.motor_reflex.fc_reflex_hidden.bias.add_(eta_local * loc_err_rh.mean(dim=0))
                self.motor_reflex.W_q_mot.weight.add_(
                    eta_local * 0.12 * (q_m_spk.T @ th_batch) / b_sz - oja_decay * self.motor_reflex.W_q_mot.weight
                )
                self.motor_reflex.W_k_mot.weight.add_(
                    eta_local * 0.12 * (k_m_spk.T @ s_feat_det) / b_sz - oja_decay * self.motor_reflex.W_k_mot.weight
                )
                self.motor_reflex.theta_motor.copy_(
                    torch.clamp(self.motor_reflex.theta_motor + 0.01 * (mot_spk_gate.mean(dim=0) - 0.35), 0.12, 0.55)
                )

                joint_wm = torch.cat([s_feat_det, ta2_batch], dim=-1)
                if b_sz >= 2:
                    wm_src = joint_wm[:-1]
                    wm_tgt = th_batch[1:]
                    wm_sz = b_sz - 1
                else:
                    prev_wm = getattr(self, "_prev_arena_joint_wm", None)
                    wm_src = prev_wm if (isinstance(prev_wm, torch.Tensor) and prev_wm.shape == joint_wm.shape) else joint_wm
                    wm_tgt = th_batch
                    wm_sz = b_sz
                pred_h_wm = torch.tanh(wm_src @ self.neocortex_logic.W_world_forward.T)
                loss_wm = F.mse_loss(pred_h_wm, wm_tgt)
                err_wm = (wm_tgt - pred_h_wm) * (1.0 - pred_h_wm * pred_h_wm)
                self.neocortex_logic.W_world_forward.add_(
                    eta_local * 2.0 * (err_wm.T @ wm_src) / wm_sz - oja_decay * self.neocortex_logic.W_world_forward
                )
                self.neocortex_logic.W_world_forward.clamp_(-3.0, 3.0)

                if ep == 0:
                    s_loss_init = float(loss_s.item())
                    m_loss_init = float(loss_m.item())
                    wm_loss_init = float(loss_wm.item())
                s_loss_final = float(loss_s.item())
                m_loss_final = float(loss_m.item())
                wm_loss_final = float(loss_wm.item())

            self._prev_arena_joint_wm = joint_wm[-1:].detach().clone()

            if hasattr(arena, "agent_joint_effort_21d"):
                j21_live = torch.tanh(self.motor_reflex.W_joint_21d(th_batch)).to(
                    device=arena.agent_joint_effort_21d.device, dtype=torch.float32
                )
                n_copy = min(int(j21_live.shape[0]), int(arena.agent_joint_effort_21d.shape[0]))
                arena.agent_joint_effort_21d[:n_copy] = j21_live[:n_copy]

        toy_transitions = 0
        prev_retinas = getattr(self, "_prev_arena_retinas_np", None)
        for idx in range(n_batch):
            if not bool(alive_np[idx]):
                continue
            cur_ret_full = retinas_np[idx].copy()
            vals_u, counts_u = np.unique(cur_ret_full, return_counts=True)
            bg_dom = int(vals_u[int(np.argmax(counts_u))]) if vals_u.size > 0 else 0
            cur_ret = cur_ret_full
            if int(np.sum(cur_ret != bg_dom)) < 2:
                continue
            if (
                isinstance(prev_retinas, np.ndarray)
                and prev_retinas.shape == retinas_np.shape
            ):
                prev_full = prev_retinas[idx].copy()
                p_vals, p_cnts = np.unique(prev_full, return_counts=True)
                p_bg = int(p_vals[int(np.argmax(p_cnts))]) if p_vals.size > 0 else 0
                prev_ret = prev_full
                if int(np.sum(prev_ret != p_bg)) >= 2 and not np.array_equal(prev_ret, cur_ret):
                    self.neocortex_logic.update_from_successful_transition(
                        self.sensory_cortex,
                        [(prev_ret, cur_ret)],
                        winning_drawer_name=None,
                        lr=0.02,
                    )
                    toy_transitions += 1
                    if toy_transitions >= 12:
                        break
        self._prev_arena_retinas_np = retinas_np.copy()

        return {
            "sensory_3d_loss_initial": round(s_loss_init, 6),
            "sensory_3d_loss_final": round(s_loss_final, 6),
            "motor_3d_loss_initial": round(m_loss_init, 6),
            "motor_3d_loss_final": round(m_loss_final, 6),
            "world_model_loss_initial": round(wm_loss_init, 6),
            "world_model_loss_final": round(wm_loss_final, 6),
            "closed_loop_3brain_steps": int(getattr(stage1_brain, "closed_loop_3brain_steps", 0)),
            "neocortex_3d_toy_transitions_educated": int(toy_transitions),
        }

    def sync_to_stage2_solver(self, solver):
        with torch.no_grad():
            neo = self.neocortex_logic
            wq_np = neo.Wq.detach().cpu().numpy().astype(np.float32)
            wk_np = neo.Wk.detach().cpu().numpy().astype(np.float32)
            wv_np = neo.Wv.detach().cpu().numpy().astype(np.float32)
            wh_np = neo.Wh.detach().cpu().numpy().astype(np.float32)
            w_mac_o = neo.W_macro_out.detach().cpu().numpy().astype(np.float32)
            w_mac_i = neo.W_macro_in.detach().cpu().numpy().astype(np.float32)
            w_mon_o = neo.W_mono_out.detach().cpu().numpy().astype(np.float32)
            w_mon_i = neo.W_mono_in.detach().cpu().numpy().astype(np.float32)
            w_pred_np = neo.W_pred.detach().cpu().numpy().astype(np.float32)
            w_act2d_np = self.motor_reflex.W_act_2d.weight.detach().cpu().numpy().astype(np.float32)
            w_ret_np = self.sensory_cortex.fc_retinal16.weight.detach().cpu().numpy().astype(np.float32).T

            if hasattr(solver, "Wq") and solver.Wq.ndim == 3:
                solver.Wq = (0.70 * solver.Wq + 0.30 * wq_np[None, :, :]).astype(np.float32)
            if hasattr(solver, "Wk") and solver.Wk.ndim == 3:
                solver.Wk = (0.70 * solver.Wk + 0.30 * wk_np[None, :, :]).astype(np.float32)
            if hasattr(solver, "Wv") and solver.Wv.ndim == 3:
                solver.Wv = (0.70 * solver.Wv + 0.30 * wv_np[None, :, :]).astype(np.float32)
            if hasattr(solver, "Wh") and solver.Wh.ndim == 3:
                if solver.Wh.shape[1:] == (self.hid_dim, self.hid_dim):
                    solver.Wh = (0.75 * solver.Wh + 0.25 * wh_np[None, :, :]).astype(np.float32)
                elif solver.Wh.shape[1] > self.hid_dim and solver.Wh.shape[2] == self.hid_dim:
                    solver.Wh[:, -self.hid_dim:, :] = (
                        0.75 * solver.Wh[:, -self.hid_dim:, :] + 0.25 * wh_np[None, :, :]
                    ).astype(np.float32)
            if hasattr(solver, "Wx") and solver.Wx.ndim == 3 and w_ret_np.shape == (self.hid_dim, 16):
                if solver.Wx.shape[1:] == (self.hid_dim, 16):
                    solver.Wx = (0.78 * solver.Wx + 0.22 * w_ret_np[None, :, :]).astype(np.float32)
                elif solver.Wx.shape[1] > self.hid_dim and solver.Wx.shape[2] == 16:
                    solver.Wx[:, -self.hid_dim:, :] = (
                        0.78 * solver.Wx[:, -self.hid_dim:, :] + 0.22 * w_ret_np[None, :, :]
                    ).astype(np.float32)
            if hasattr(solver, "W_macro_out") and solver.W_macro_out.ndim == 3:
                solver.W_macro_out = (0.70 * solver.W_macro_out + 0.30 * w_mac_o[None, :, :]).astype(np.float32)
            if hasattr(solver, "W_macro_in") and solver.W_macro_in.ndim == 3:
                solver.W_macro_in = (0.70 * solver.W_macro_in + 0.30 * w_mac_i[None, :, :]).astype(np.float32)
            if hasattr(solver, "W_mono_out") and solver.W_mono_out.ndim == 3:
                solver.W_mono_out = (0.70 * solver.W_mono_out + 0.30 * w_mon_o[None, :, :]).astype(np.float32)
            if hasattr(solver, "W_mono_in") and solver.W_mono_in.ndim == 3:
                solver.W_mono_in = (0.70 * solver.W_mono_in + 0.30 * w_mon_i[None, :, :]).astype(np.float32)
            if hasattr(solver, "Wpred") and solver.Wpred.ndim == 3 and solver.Wpred.shape[1:] == w_pred_np.shape:
                solver.Wpred = (0.80 * solver.Wpred + 0.20 * w_pred_np[None, :, :]).astype(np.float32)
            if hasattr(solver, "Wo") and solver.Wo.ndim == 3 and solver.Wo.shape[1:] == w_act2d_np.shape:
                solver.Wo = (0.85 * solver.Wo + 0.15 * w_act2d_np[None, :, :]).astype(np.float32)

            solver.W_symbol_lexicon = neo.W_symbol_lexicon.detach().cpu().numpy().astype(np.float32).copy()
            solver.W_body_assembly = neo.W_body_assembly.detach().cpu().numpy().astype(np.float32).copy()
            solver.W_parietal_gain = neo.W_parietal_gain.detach().cpu().numpy().astype(np.float32).copy()
            solver.W_procedural_attractor = neo.W_procedural_attractor.detach().cpu().numpy().astype(np.float32).copy()
            solver.W_procedural_chain = neo.W_procedural_chain.detach().cpu().numpy().astype(np.float32).copy()
            solver.W_syntax_transition = neo.W_syntax_transition.detach().cpu().numpy().astype(np.float32).copy()
            solver.W_mirror_inverse = neo.W_mirror_inverse.detach().cpu().numpy().astype(np.float32).copy()
            solver.W_fovea_v1_spikes = neo.W_fovea_v1_spikes.detach().cpu().numpy().astype(np.float32).copy()
            solver.W_fovea_stroke_assoc = neo.W_fovea_stroke_assoc.detach().cpu().numpy().astype(np.float32).copy()
            solver.myelination_stage_gates = neo.myelination_stage_gates.detach().cpu().numpy().astype(np.float32).copy()

    def sync_from_stage2_solver(self, solver):
        with torch.no_grad():
            neo = self.neocortex_logic
            for solver_attr, neo_attr in [
                ("Wq", "Wq"),
                ("Wk", "Wk"),
                ("Wv", "Wv"),
                ("Wh", "Wh"),
                ("W_macro_out", "W_macro_out"),
                ("W_macro_in", "W_macro_in"),
                ("W_mono_out", "W_mono_out"),
                ("W_mono_in", "W_mono_in"),
                ("Wpred", "W_pred"),
                ("W_symbol_lexicon", "W_symbol_lexicon"),
                ("W_body_assembly", "W_body_assembly"),
                ("W_parietal_gain", "W_parietal_gain"),
                ("W_procedural_attractor", "W_procedural_attractor"),
                ("W_procedural_chain", "W_procedural_chain"),
                ("W_syntax_transition", "W_syntax_transition"),
                ("W_mirror_inverse", "W_mirror_inverse"),
                ("W_fovea_v1_spikes", "W_fovea_v1_spikes"),
                ("W_fovea_stroke_assoc", "W_fovea_stroke_assoc"),
                ("myelination_stage_gates", "myelination_stage_gates"),
            ]:
                if hasattr(solver, solver_attr):
                    arr = getattr(solver, solver_attr)
                    if arr is not None:
                        np_arr = np.asarray(arr, dtype=np.float32)
                        if np_arr.ndim == 3:
                            np_arr = np.mean(np_arr, axis=0)
                        if solver_attr == "Wh" and np_arr.shape[0] > self.hid_dim:
                            np_arr = np_arr[-self.hid_dim:, :]
                        param = getattr(neo, neo_attr)
                        if np_arr.shape == tuple(param.shape):
                            param.copy_(torch.from_numpy(np_arr))

            if hasattr(solver, "Wo") and solver.Wo is not None:
                wo_np = np.asarray(solver.Wo, dtype=np.float32)
                if wo_np.ndim == 3:
                    wo_np = np.mean(wo_np, axis=0)
                if wo_np.shape == tuple(self.motor_reflex.W_act_2d.weight.shape):
                    self.motor_reflex.W_act_2d.weight.copy_(torch.from_numpy(wo_np))

    def save_all_brains(self, extra_meta=None):
        p = self.paths
        torch.save(self.sensory_cortex.state_dict(), p["sensory_pth"])
        self.sensory_cortex.export_onnx(p["sensory_onnx"])

        torch.save(self.neocortex_logic.state_dict(), p["neocortex_pth"])
        self.neocortex_logic.export_gguf(p["neocortex_gguf"], extra_meta=extra_meta)

        torch.save(self.motor_reflex.state_dict(), p["motor_pth"])
        self.motor_reflex.export_onnx(p["motor_onnx"])

        parent_dir = os.path.dirname(os.path.abspath(self.artifact_dir))
        brains_dir = os.path.join(parent_dir, "brains")
        if os.path.isdir(brains_dir) and os.path.abspath(brains_dir) != os.path.abspath(self.artifact_dir):
            for fpath in p.values():
                if os.path.isfile(fpath):
                    dst_path = os.path.join(brains_dir, os.path.basename(fpath))
                    with open(fpath, "rb") as f_src:
                        raw_bytes = f_src.read()
                    with open(dst_path, "wb") as f_dst:
                        f_dst.write(raw_bytes)
        return p

    def load_all_brains(self):
        p = self.paths
        loaded = {
            "sensory_pth": False,
            "sensory_onnx": False,
            "neocortex_pth": False,
            "neocortex_gguf": False,
            "motor_pth": False,
            "motor_onnx": False,
        }
        if os.path.exists(p["sensory_pth"]):
            try:
                sd = torch.load(p["sensory_pth"], map_location="cpu", weights_only=True)
                self.sensory_cortex.load_state_dict(sd, strict=False)
                loaded["sensory_pth"] = True
            except Exception:
                pass
        if os.path.exists(p["sensory_onnx"]):
            loaded["sensory_onnx"] = self.sensory_cortex.load_onnx_session(p["sensory_onnx"])

        if os.path.exists(p["neocortex_pth"]):
            try:
                sd = torch.load(p["neocortex_pth"], map_location="cpu", weights_only=True)
                self.neocortex_logic.load_state_dict(sd, strict=False)
                loaded["neocortex_pth"] = True
            except Exception:
                pass
        if os.path.exists(p["neocortex_gguf"]):
            try:
                self.neocortex_logic.load_gguf(p["neocortex_gguf"])
                loaded["neocortex_gguf"] = True
            except Exception:
                pass

        if os.path.exists(p["motor_pth"]):
            try:
                sd = torch.load(p["motor_pth"], map_location="cpu", weights_only=True)
                self.motor_reflex.load_state_dict(sd, strict=False)
                loaded["motor_pth"] = True
            except Exception:
                pass
        if os.path.exists(p["motor_onnx"]):
            loaded["motor_onnx"] = self.motor_reflex.load_onnx_session(p["motor_onnx"])

        return loaded
