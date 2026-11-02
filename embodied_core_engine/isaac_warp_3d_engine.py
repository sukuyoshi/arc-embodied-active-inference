import math
import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

if hasattr(wp, "types") and not hasattr(wp.types, "array"):
    setattr(wp.types, "array", wp.array)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is strictly required for isaac_warp_3d_engine")

CUDA_DEVICE = torch.device("cuda:0")
wp.init()

ARC_PALETTE_RGB = (
    (0.05, 0.05, 0.07),
    (0.00, 0.45, 0.85),
    (0.90, 0.15, 0.15),
    (0.18, 0.80, 0.25),
    (0.95, 0.85, 0.10),
    (0.65, 0.65, 0.68),
    (0.90, 0.20, 0.75),
    (0.95, 0.50, 0.10),
    (0.35, 0.80, 0.95),
    (0.55, 0.10, 0.20),
)


@wp.kernel
def foveated_binocular_raycast_kernel(
    mesh_id: wp.uint64,
    tri_colors: wp.array(dtype=wp.int32),
    tri_normals: wp.array(dtype=wp.vec3),
    cam_pos: wp.vec3,
    fixation_pos: wp.vec3,
    cam_forward: wp.vec3,
    cam_right: wp.vec3,
    cam_up: wp.vec3,
    gaze_u_rad: float,
    gaze_v_rad: float,
    fov_h_rad: float,
    fov_v_rad: float,
    focal_pix: float,
    verge_dist: float,
    screen_w: int,
    screen_h: int,
    out_rgb_lum: wp.array(dtype=wp.vec4),
    out_color_idx: wp.array(dtype=wp.int32),
    out_tier_id: wp.array(dtype=wp.int32),
    out_angular_pitch_deg: wp.array(dtype=float),
    out_true_depth_diag: wp.array(dtype=float),
):
    tid = wp.tid()
    py = tid // screen_w
    px = tid % screen_w
    if py >= screen_h or px >= screen_w:
        return

    u_pix = (float(px) + 0.5) - 0.5 * float(screen_w)
    v_pix = 0.5 * float(screen_h) - (float(py) + 0.5)

    ang_u = wp.atan(u_pix / wp.max(focal_pix, 1.0))
    ang_v = wp.atan(v_pix / wp.max(focal_pix, 1.0))

    du_gaze = ang_u - gaze_u_rad
    dv_gaze = ang_v - gaze_v_rad
    ecc_rad = wp.sqrt(du_gaze * du_gaze + dv_gaze * dv_gaze)
    ecc_deg = ecc_rad * 57.2957795

    tier = 0
    pitch_deg = 0.046875
    if ecc_deg <= 5.0:
        tier = 0
        pitch_deg = 0.046875
    elif ecc_deg <= 20.0:
        tier = 1
        pitch_deg = 0.0703125
    else:
        tier = 2
        pitch_deg = 0.375

    rel_cam = cam_pos - fixation_pos
    verge_u = wp.dot(rel_cam, cam_right) / wp.max(verge_dist, 0.15)
    verge_v = wp.dot(rel_cam, cam_up) / wp.max(verge_dist, 0.15)

    tan_u = (u_pix / wp.max(focal_pix, 1.0)) - verge_u
    tan_v = (v_pix / wp.max(focal_pix, 1.0)) - verge_v
    ray_dir = wp.normalize(cam_forward + cam_right * tan_u + cam_up * tan_v)

    t_hit = float(0.0)
    u_bary = float(0.0)
    v_bary = float(0.0)
    sign_hit = float(0.0)
    normal_hit = wp.vec3(0.0, 0.0, 1.0)
    face_idx = int(-1)

    hit = wp.mesh_query_ray(
        mesh_id,
        cam_pos,
        ray_dir,
        50.0,
        t_hit,
        u_bary,
        v_bary,
        sign_hit,
        normal_hit,
        face_idx,
    )

    out_tier_id[tid] = tier
    out_angular_pitch_deg[tid] = pitch_deg

    if hit and face_idx >= 0:
        c_idx = tri_colors[face_idx]
        n_vec = tri_normals[face_idx]
        light_dir = wp.normalize(wp.vec3(0.35, -0.45, 0.82))
        ndotl = wp.max(0.22, wp.abs(wp.dot(n_vec, light_dir)))
        hit_pt = cam_pos + ray_dir * t_hit
        wx = hit_pt[0]
        wy = hit_pt[1]
        tex_pat = (
            0.22 * wp.sin(wx * 38.0 + wy * 29.0)
            + 0.20 * wp.cos(wx * 82.0 - wy * 64.0)
            + 0.16 * wp.sin(wx * 145.0 + wy * 110.0)
            + 0.12 * wp.cos(wx * 210.0 - wy * 165.0)
        )
        lum = wp.clamp(ndotl * (0.58 + tex_pat), 0.05, 1.25)

        r_c = float(0.05)
        g_c = float(0.05)
        b_c = float(0.07)
        if c_idx == -1:
            r_c, g_c, b_c = 0.42, 0.30, 0.18
            lum = wp.clamp(0.35 + 0.12 * tex_pat, 0.05, 0.80)
        elif c_idx == -2:
            r_c, g_c, b_c = 0.02, 0.02, 0.03
            lum = wp.clamp(0.06 + 0.03 * tex_pat, 0.02, 0.14)
        elif c_idx == 1:
            r_c, g_c, b_c = 0.00, 0.45, 0.85
        elif c_idx == 2:
            r_c, g_c, b_c = 0.90, 0.15, 0.15
        elif c_idx == 3:
            r_c, g_c, b_c = 0.18, 0.80, 0.25
        elif c_idx == 4:
            r_c, g_c, b_c = 0.95, 0.85, 0.10
        elif c_idx == 5:
            r_c, g_c, b_c = 0.65, 0.65, 0.68
        elif c_idx == 6:
            r_c, g_c, b_c = 0.90, 0.20, 0.75
        elif c_idx == 7:
            r_c, g_c, b_c = 0.95, 0.50, 0.10
        elif c_idx == 8:
            r_c, g_c, b_c = 0.35, 0.80, 0.95
        elif c_idx == 9:
            r_c, g_c, b_c = 0.55, 0.10, 0.20
        elif c_idx >= 10:
            r_c = float((c_idx * 37) % 100) * 0.01
            g_c = float((c_idx * 73) % 100) * 0.01
            b_c = float((c_idx * 53) % 100) * 0.01

        cos_theta = wp.max(0.1, wp.dot(ray_dir, cam_forward))
        z_proj = t_hit * cos_theta
        if tier == 1:
            r_c = 0.85 * r_c + 0.15 * 0.5
            g_c = 0.85 * g_c + 0.15 * 0.5
            b_c = 0.85 * b_c + 0.15 * 0.5
        elif tier == 2:
            r_c = 0.60 * r_c + 0.40 * 0.5
            g_c = 0.60 * g_c + 0.40 * 0.5
            b_c = 0.60 * b_c + 0.40 * 0.5
        out_color_idx[tid] = c_idx
        out_rgb_lum[tid] = wp.vec4(r_c * lum, g_c * lum, b_c * lum, lum)
        out_true_depth_diag[tid] = z_proj
    else:
        out_rgb_lum[tid] = wp.vec4(0.02, 0.02, 0.03, 0.02)
        out_color_idx[tid] = -1
        out_true_depth_diag[tid] = 10.0


@wp.kernel
def arena_agent_binocular_raycast_kernel(
    mesh_id: wp.uint64,
    tri_colors: wp.array(dtype=wp.int32),
    agent_pos: wp.array(dtype=wp.vec3),
    agent_yaw: wp.array(dtype=float),
    agent_gaze_yaw_offset: wp.array(dtype=float),
    agent_gaze_pitch_offset: wp.array(dtype=float),
    eye_offset_sign: float,
    interpupillary_half: float,
    screen_w: int,
    screen_h: int,
    fov_h_rad: float,
    fov_v_rad: float,
    out_color: wp.array(dtype=wp.int32),
    out_lum: wp.array(dtype=float),
    out_tier: wp.array(dtype=wp.int32),
    out_true_depth_diag: wp.array(dtype=float),
):
    tid = wp.tid()
    pix_per_agent = screen_w * screen_h
    a_idx = tid // pix_per_agent
    rem = tid % pix_per_agent
    py = rem // screen_w
    px = rem % screen_w

    pos = agent_pos[a_idx]
    yaw = agent_yaw[a_idx]
    gaze_yaw = yaw + agent_gaze_yaw_offset[a_idx]
    gaze_pitch = wp.clamp(-0.12 + agent_gaze_pitch_offset[a_idx], -0.85, 0.65)
    cos_p = wp.cos(gaze_pitch)
    sin_p = wp.sin(gaze_pitch)
    fwd = wp.vec3(cos_p * wp.cos(gaze_yaw), cos_p * wp.sin(gaze_yaw), sin_p)
    fwd = wp.normalize(fwd)
    right = wp.normalize(wp.vec3(wp.sin(gaze_yaw), -wp.cos(gaze_yaw), 0.0))
    up = wp.normalize(wp.cross(right, fwd))

    eye_pos = pos + wp.vec3(0.0, 0.0, 0.55) + right * (eye_offset_sign * interpupillary_half)

    u_norm = (float(px) + 0.5) / float(screen_w) - 0.5
    v_norm = 0.5 - (float(py) + 0.5) / float(screen_h)
    ang_u = u_norm * fov_h_rad
    ang_v = v_norm * fov_v_rad

    ecc_deg = wp.sqrt(ang_u * ang_u + ang_v * ang_v) * 57.2957795
    pitch_fhd_rad = 0.046875 * 0.0174532925
    pitch_hd_rad = 0.0703125 * 0.0174532925
    pitch_240p_rad = 0.375 * 0.0174532925

    tier = 0
    quant_u = ang_u
    quant_v = ang_v
    if ecc_deg <= 5.0:
        tier = 0
        quant_u = wp.floor(ang_u / pitch_fhd_rad + 0.5) * pitch_fhd_rad
        quant_v = wp.floor(ang_v / pitch_fhd_rad + 0.5) * pitch_fhd_rad
    elif ecc_deg <= 20.0:
        tier = 1
        quant_u = wp.floor(ang_u / pitch_hd_rad + 0.5) * pitch_hd_rad
        quant_v = wp.floor(ang_v / pitch_hd_rad + 0.5) * pitch_hd_rad
    else:
        tier = 2
        quant_u = wp.floor(ang_u / pitch_240p_rad + 0.5) * pitch_240p_rad
        quant_v = wp.floor(ang_v / pitch_240p_rad + 0.5) * pitch_240p_rad

    ray_dir = wp.normalize(fwd + right * wp.tan(quant_u) + up * wp.tan(quant_v))

    t_hit = float(0.0)
    u_bary = float(0.0)
    v_bary = float(0.0)
    sign_hit = float(0.0)
    normal_hit = wp.vec3(0.0, 0.0, 1.0)
    face_idx = int(-1)

    hit = wp.mesh_query_ray(
        mesh_id,
        eye_pos,
        ray_dir,
        35.0,
        t_hit,
        u_bary,
        v_bary,
        sign_hit,
        normal_hit,
        face_idx,
    )

    out_tier[tid] = tier
    if hit and face_idx >= 0:
        c_id = tri_colors[face_idx]
        cos_ang = wp.max(0.1, wp.dot(ray_dir, fwd))
        z_p = t_hit * cos_ang
        shade = wp.clamp(1.0 / (1.0 + 0.06 * z_p), 0.1, 1.0)
        out_color[tid] = c_id
        out_lum[tid] = shade
        out_true_depth_diag[tid] = z_p
    else:
        out_color[tid] = 0
        out_lum[tid] = 0.02
        out_true_depth_diag[tid] = 35.0


@wp.kernel
def rigid_body_3d_substep_kernel(
    n_bodies: int,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    rot_y: wp.array(dtype=float),
    omega_y: wp.array(dtype=float),
    size: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    inv_inertia_y: wp.array(dtype=float),
    drive_active: wp.array(dtype=wp.int32),
    drive_target_pos: wp.array(dtype=wp.vec3),
    drive_rot_active: wp.array(dtype=wp.int32),
    drive_target_rot_y: wp.array(dtype=float),
    gravity_vec: wp.vec3,
    floor_enabled: int,
    floor_y: float,
    dt: float,
    out_pos: wp.array(dtype=wp.vec3),
    out_vel: wp.array(dtype=wp.vec3),
    out_rot_y: wp.array(dtype=float),
    out_omega_y: wp.array(dtype=float),
):
    i = wp.tid()
    if i >= n_bodies:
        return

    p_i = pos[i]
    v_i = vel[i]
    th_i = rot_y[i]
    w_i = omega_y[i]
    s_i = size[i]
    im_i = inv_mass[i]
    ii_i = inv_inertia_y[i]

    if im_i <= 0.0 and ii_i <= 0.0:
        out_pos[i] = p_i
        out_vel[i] = wp.vec3(0.0, 0.0, 0.0)
        out_rot_y[i] = th_i
        out_omega_y[i] = 0.0
        return

    m_i = 1.0 / wp.max(im_i, 1e-5)
    iner_i = 1.0 / wp.max(ii_i, 1e-5)

    c_i = wp.abs(wp.cos(th_i))
    sn_i = wp.abs(wp.sin(th_i))
    hx_i = 0.5 * (s_i[0] * c_i + s_i[2] * sn_i)
    hy_i = 0.5 * s_i[1]
    hz_i = 0.5 * (s_i[0] * sn_i + s_i[2] * c_i)

    force = gravity_vec * m_i
    torque_y = float(0.0)

    if drive_active[i] == 1:
        err_p = drive_target_pos[i] - p_i
        force = force + err_p * (180.0 * m_i) - v_i * (26.0 * m_i)
    else:
        if floor_enabled == 1:
            force = force - v_i * (1.2 * m_i)
        else:
            force = force - v_i * (18.0 * m_i)

    if drive_rot_active[i] == 1:
        err_th = drive_target_rot_y[i] - th_i
        torque_y = torque_y + (180.0 * iner_i) * err_th - (26.0 * iner_i) * w_i
    else:
        torque_y = torque_y - (12.0 * iner_i) * w_i

    for j in range(n_bodies):
        if j != i:
            p_j = pos[j]
            v_j = vel[j]
            th_j = rot_y[j]
            s_j = size[j]
            c_j = wp.abs(wp.cos(th_j))
            sn_j = wp.abs(wp.sin(th_j))
            hx_j = 0.5 * (s_j[0] * c_j + s_j[2] * sn_j)
            hy_j = 0.5 * s_j[1]
            hz_j = 0.5 * (s_j[0] * sn_j + s_j[2] * c_j)

            dx = p_i[0] - p_j[0]
            dy = p_i[1] - p_j[1]
            dz = p_i[2] - p_j[2]
            ox = (hx_i + hx_j) - wp.abs(dx)
            oy = (hy_i + hy_j) - wp.abs(dy)
            oz = (hz_i + hz_j) - wp.abs(dz)

            if ox > 0.0 and oy > 0.0 and oz > 0.0:
                n_vec = wp.vec3(0.0, 0.0, 0.0)
                pen = float(0.0)
                if ox <= oy and ox <= oz:
                    sx_dir = 1.0
                    if dx < 0.0:
                        sx_dir = -1.0
                    n_vec = wp.vec3(sx_dir, 0.0, 0.0)
                    pen = ox
                elif oy <= ox and oy <= oz:
                    sy_dir = 1.0
                    if dy < 0.0:
                        sy_dir = -1.0
                    n_vec = wp.vec3(0.0, sy_dir, 0.0)
                    pen = oy
                else:
                    sz_dir = 1.0
                    if dz < 0.0:
                        sz_dir = -1.0
                    n_vec = wp.vec3(0.0, 0.0, sz_dir)
                    pen = oz

                rel_v = v_i - v_j
                rel_vn = wp.dot(rel_v, n_vec)
                fn = (1800.0 * m_i) * pen - (42.0 * m_i) * wp.min(0.0, rel_vn) - (10.0 * m_i) * rel_vn
                if fn > 0.0:
                    force = force + n_vec * fn

    if floor_enabled == 1:
        pen_floor = (floor_y + hy_i) - p_i[1]
        if pen_floor > 0.0:
            fn_floor = (2400.0 * m_i) * pen_floor - (48.0 * m_i) * wp.min(0.0, v_i[1]) - (12.0 * m_i) * v_i[1]
            if fn_floor > 0.0:
                force = force + wp.vec3(0.0, fn_floor, 0.0)
            force = force - wp.vec3(v_i[0] * (14.0 * m_i), 0.0, v_i[2] * (14.0 * m_i))

    new_v = v_i + (force * im_i) * dt
    new_p = p_i + new_v * dt
    if floor_enabled == 1:
        min_y = floor_y + hy_i
        if new_p[1] < min_y:
            new_p = wp.vec3(new_p[0], min_y, new_p[2])
            if new_v[1] < 0.0:
                new_v = wp.vec3(new_v[0], 0.0, new_v[2])

    new_w = w_i + (torque_y * ii_i) * dt
    new_th = th_i + new_w * dt

    out_pos[i] = new_p
    out_vel[i] = new_v
    out_rot_y[i] = new_th
    out_omega_y[i] = new_w


def _append_Oriented_box_triangles(verts_list, tris_list, colors_list, normals_list, center_xyz, half_extents_xyz, yaw_rad, color_id):
    cx, cy, cz = float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])
    hx, hy, hz = float(half_extents_xyz[0]), float(half_extents_xyz[1]), float(half_extents_xyz[2])
    cos_y = math.cos(float(yaw_rad))
    sin_y = math.sin(float(yaw_rad))

    local_corners = (
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    )
    base_v = len(verts_list)
    for lx, ly, lz in local_corners:
        wx = cx + lx * cos_y - ly * sin_y
        wy = cy + lx * sin_y + ly * cos_y
        wz = cz + lz
        verts_list.append((wx, wy, wz))

    box_faces = (
        (4, 5, 6, 7, (0.0, 0.0, 1.0)),
        (0, 3, 2, 1, (0.0, 0.0, -1.0)),
        (1, 2, 6, 5, (cos_y, sin_y, 0.0)),
        (3, 0, 4, 7, (-cos_y, -sin_y, 0.0)),
        (2, 3, 7, 6, (-sin_y, cos_y, 0.0)),
        (0, 1, 5, 4, (sin_y, -cos_y, 0.0)),
    )
    for i0, i1, i2, i3, n_xyz in box_faces:
        tris_list.append((base_v + i0, base_v + i1, base_v + i2))
        colors_list.append(int(color_id))
        normals_list.append(n_xyz)
        tris_list.append((base_v + i0, base_v + i2, base_v + i3))
        colors_list.append(int(color_id))
        normals_list.append(n_xyz)


class FoveatedRetinalSpecification:
    def __init__(
        self,
        screen_w=160,
        screen_h=160,
        fov_h_deg=40.0,
        fov_v_deg=40.0,
        interpupillary_baseline=0.064,
        vergence_distance=0.56,
    ):
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)
        self.fov_h_deg = float(fov_h_deg)
        self.fov_v_deg = float(fov_v_deg)
        self.fov_h_rad = math.radians(self.fov_h_deg)
        self.fov_v_rad = math.radians(self.fov_v_deg)
        self.interpupillary_baseline = float(interpupillary_baseline)
        self.vergence_distance = float(vergence_distance)
        self.focal_length_pix = 0.5 * float(self.screen_w) / math.tan(0.5 * self.fov_h_rad)


class HumanVisual2DTo3DReconstructor:
    def __init__(self, retinal_spec=None):
        self.spec = retinal_spec if retinal_spec is not None else FoveatedRetinalSpecification()
        self.device = CUDA_DEVICE
        self.stereo_reconstructions_total = 0
        self.monocular_inferences_total = 0
        self.motion_parallax_integrations_total = 0

    def compute_binocular_disparity_depth(self, left_rgb_lum_t, right_rgb_lum_t, left_color_t, right_color_t):
        self.stereo_reconstructions_total += 1
        H, W = left_color_t.shape
        left_lum = left_rgb_lum_t[:, :, 3]
        right_lum = right_rgb_lum_t[:, :, 3]
        left_rgb = left_rgb_lum_t[:, :, :3]
        right_rgb = right_rgb_lum_t[:, :, :3]

        min_disp = -1
        max_disp = 14
        disp_candidates = list(range(min_disp, max_disp + 1))
        kernel_avg = torch.ones((1, 1, 5, 5), dtype=torch.float32, device=self.device) / 25.0
        cost_volume = []

        for d in disp_candidates:
            shifted_lum = torch.roll(right_lum, shifts=d, dims=1)
            shifted_rgb = torch.roll(right_rgb, shifts=d, dims=1)
            if d > 0:
                shifted_lum[:, :d] = right_lum[:, :d]
                shifted_rgb[:, :d, :] = right_rgb[:, :d, :]
            elif d < 0:
                shifted_lum[:, d:] = right_lum[:, d:]
                shifted_rgb[:, d:, :] = right_rgb[:, d:, :]

            lum_sq = (left_lum - shifted_lum) ** 2
            rgb_sq = torch.mean((left_rgb - shifted_rgb) ** 2, dim=-1)
            raw_ssd = lum_sq + 1.2 * rgb_sq
            smooth_ssd = F.conv2d(raw_ssd.unsqueeze(0).unsqueeze(0), kernel_avg, padding=2).squeeze(0).squeeze(0)
            cost_volume.append(smooth_ssd)

        cost_stack = torch.stack(cost_volume, dim=0)
        best_idx = torch.argmin(cost_stack, dim=0)
        n_d = len(disp_candidates)
        idx_prev = torch.clamp(best_idx - 1, 0, n_d - 1)
        idx_next = torch.clamp(best_idx + 1, 0, n_d - 1)

        c0 = torch.gather(cost_stack, 0, best_idx.unsqueeze(0)).squeeze(0)
        cm = torch.gather(cost_stack, 0, idx_prev.unsqueeze(0)).squeeze(0)
        cp = torch.gather(cost_stack, 0, idx_next.unsqueeze(0)).squeeze(0)

        denom = 2.0 * (cm - 2.0 * c0 + cp)
        subpix = torch.where(
            (best_idx > 0) & (best_idx < n_d - 1) & (torch.abs(denom) > 1e-6),
            torch.clamp((cm - cp) / torch.where(torch.abs(denom) > 1e-6, denom, torch.ones_like(denom)), -0.48, 0.48),
            torch.zeros_like(c0),
        )
        best_disp = float(min_disp) + best_idx.to(torch.float32) + subpix

        f_pix = float(self.spec.focal_length_pix)
        baseline = float(self.spec.interpupillary_baseline)
        verge_d = float(self.spec.vergence_distance)
        inv_z = (best_disp / max(1e-5, f_pix * baseline)) + (1.0 / max(0.15, verge_d))
        z_stereo = 1.0 / torch.clamp(inv_z, min=0.5, max=4.5)
        valid_surface = left_color_t >= -2
        z_stereo = torch.where(valid_surface, torch.clamp(z_stereo, 0.30, 0.65), torch.full_like(z_stereo, verge_d))
        return z_stereo, torch.clamp(best_disp, min=0.0)

    def compute_monocular_cue_depth(self, left_rgb_lum_t, left_color_t):
        self.monocular_inferences_total += 1
        lum = left_rgb_lum_t[:, :, 3]
        verge_d = float(self.spec.vergence_distance)

        gx = torch.zeros_like(lum)
        gy = torch.zeros_like(lum)
        gx[:, 1:-1] = 0.5 * (lum[:, 2:] - lum[:, :-2])
        gy[1:-1, :] = 0.5 * (lum[2:, :] - lum[:-2, :])
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)

        c_up = torch.roll(left_color_t, shifts=1, dims=0)
        c_dn = torch.roll(left_color_t, shifts=-1, dims=0)
        c_lt = torch.roll(left_color_t, shifts=1, dims=1)
        c_rt = torch.roll(left_color_t, shifts=-1, dims=1)

        n_diff = (
            (c_up != left_color_t).to(torch.float32)
            + (c_dn != left_color_t).to(torch.float32)
            + (c_lt != left_color_t).to(torch.float32)
            + (c_rt != left_color_t).to(torch.float32)
        )
        t_junction_map = ((n_diff >= 2.0) & (left_color_t > 0)).to(torch.float32)

        fg_mask = (left_color_t > 0).to(torch.float32)
        elev_from_cues = torch.clamp(0.015 * fg_mask + 0.015 * grad_mag + 0.012 * t_junction_map, 0.0, 0.08) * fg_mask
        z_mono = torch.clamp(verge_d - elev_from_cues, 0.35, verge_d)
        return z_mono, elev_from_cues, t_junction_map

    def compute_motion_parallax_depth(self, base_rgb_lum_t, moved_rgb_lum_t, base_color_t, moved_color_t, head_delta_xyz):
        self.motion_parallax_integrations_total += 1
        base_lum = base_rgb_lum_t[:, :, 3]
        moved_lum = moved_rgb_lum_t[:, :, 3]
        base_rgb = base_rgb_lum_t[:, :, :3]
        moved_rgb = moved_rgb_lum_t[:, :, :3]

        dx_head = float(head_delta_xyz[0])
        dy_head = float(head_delta_xyz[1])
        baseline_motion = max(0.015, abs(dx_head) if abs(dx_head) >= abs(dy_head) else math.hypot(dx_head, dy_head))
        sign_dir = 1 if dx_head >= 0.0 else -1

        min_shift = 0
        max_shift = 12
        shift_candidates = list(range(min_shift, max_shift + 1))
        kernel_avg = torch.ones((1, 1, 5, 5), dtype=torch.float32, device=self.device) / 25.0
        err_volume = []

        for sh in shift_candidates:
            s_val = sign_dir * sh
            s1_lum = torch.roll(moved_lum, shifts=s_val, dims=1)
            s1_rgb = torch.roll(moved_rgb, shifts=s_val, dims=1)
            c_shifted = torch.roll(moved_color_t, shifts=s_val, dims=1)
            if s_val > 0:
                s1_lum[:, :s_val] = moved_lum[:, :s_val]
                s1_rgb[:, :s_val, :] = moved_rgb[:, :s_val, :]
                c_shifted[:, :s_val] = moved_color_t[:, :s_val]
            elif s_val < 0:
                s1_lum[:, s_val:] = moved_lum[:, s_val:]
                s1_rgb[:, s_val:, :] = moved_rgb[:, s_val:, :]
                c_shifted[:, s_val:] = moved_color_t[:, s_val:]
            err = (
                (base_lum - s1_lum) ** 2
                + 0.5 * torch.mean((base_rgb - s1_rgb) ** 2, dim=-1)
                + (base_color_t != c_shifted).to(torch.float32) * 0.45
            )
            err_smooth = F.conv2d(err.unsqueeze(0).unsqueeze(0), kernel_avg, padding=2).squeeze(0).squeeze(0)
            err_volume.append(err_smooth)

        err_stack = torch.stack(err_volume, dim=0)
        best_idx = torch.argmin(err_stack, dim=0)
        n_s = len(shift_candidates)
        idx_prev = torch.clamp(best_idx - 1, 0, n_s - 1)
        idx_next = torch.clamp(best_idx + 1, 0, n_s - 1)
        e0 = torch.gather(err_stack, 0, best_idx.unsqueeze(0)).squeeze(0)
        em = torch.gather(err_stack, 0, idx_prev.unsqueeze(0)).squeeze(0)
        ep = torch.gather(err_stack, 0, idx_next.unsqueeze(0)).squeeze(0)
        denom = 2.0 * (em - 2.0 * e0 + ep)
        subpix = torch.where(
            (best_idx > 0) & (best_idx < n_s - 1) & (torch.abs(denom) > 1e-6),
            torch.clamp((em - ep) / torch.where(torch.abs(denom) > 1e-6, denom, torch.ones_like(denom)), -0.48, 0.48),
            torch.zeros_like(e0),
        )
        best_flow = float(min_shift) + best_idx.to(torch.float32) + subpix

        f_pix = float(self.spec.focal_length_pix)
        verge_d = float(self.spec.vergence_distance)
        inv_z = (best_flow / max(1e-5, f_pix * baseline_motion)) + (1.0 / max(0.15, verge_d))
        z_parallax = 1.0 / torch.clamp(inv_z, min=0.5, max=4.5)
        valid_surface = base_color_t >= -2
        z_parallax = torch.where(valid_surface, torch.clamp(z_parallax, 0.30, 0.65), torch.full_like(z_parallax, verge_d))
        return z_parallax, torch.clamp(best_flow, min=0.0)

    def reconstruct_3d_from_2d_retinal_screens(
        self,
        left_rgb_lum_t,
        right_rgb_lum_t,
        left_color_t,
        right_color_t,
        tier_id_t,
        moved_rgb_lum_t=None,
        moved_color_t=None,
        head_delta_xyz=(0.045, 0.0, 0.0),
    ):
        z_stereo, disp_map = self.compute_binocular_disparity_depth(
            left_rgb_lum_t, right_rgb_lum_t, left_color_t, right_color_t
        )
        z_mono, elev_mono, t_junc = self.compute_monocular_cue_depth(left_rgb_lum_t, left_color_t)

        if moved_rgb_lum_t is not None and moved_color_t is not None:
            z_parallax, flow_map = self.compute_motion_parallax_depth(
                left_rgb_lum_t, moved_rgb_lum_t, left_color_t, moved_color_t, head_delta_xyz
            )
        else:
            z_parallax = z_stereo.clone()
            flow_map = disp_map.clone()

        w_fovea = (tier_id_t == 0).to(torch.float32)
        w_para = (tier_id_t == 1).to(torch.float32)
        w_peri = (tier_id_t == 2).to(torch.float32)

        w_s = 0.60 * w_fovea + 0.55 * w_para + 0.35 * w_peri
        w_p = 0.30 * w_fovea + 0.30 * w_para + 0.25 * w_peri
        w_m = 0.10 * w_fovea + 0.15 * w_para + 0.40 * w_peri

        z_fused = w_s * z_stereo + w_m * z_mono + w_p * z_parallax

        verge_d = float(self.spec.vergence_distance)
        groove_or_bg_mask = (left_color_t == 0) | (left_color_t == -2)
        if bool(torch.any(groove_or_bg_mask).item()):
            z_desk_ref = float(torch.median(z_stereo[groove_or_bg_mask]).item())
            z_desk_ref = float(np.clip(z_desk_ref, 0.46, 0.58))
        else:
            z_desk_ref = verge_d

        phys_elev_m = torch.clamp(z_desk_ref - z_fused, 0.0, 0.15)
        reconstructed_elevation = torch.clamp(phys_elev_m / 0.055, 0.0, 1.5) * (left_color_t >= 0).to(torch.float32)

        return {
            "z_fused": z_fused,
            "z_stereo": z_stereo,
            "z_mono": z_mono,
            "z_parallax": z_parallax,
            "disparity_map": disp_map,
            "optical_flow_map": flow_map,
            "t_junction_map": t_junc,
            "reconstructed_elevation": reconstructed_elevation,
            "physical_elevation_m": phys_elev_m,
            "perceived_desk_depth_m": z_desk_ref,
        }


class IsaacWarp3DTabletopEngine:
    def __init__(self, screen_w=160, screen_h=160):
        self.device = CUDA_DEVICE
        self.spec = FoveatedRetinalSpecification(
            screen_w=screen_w,
            screen_h=screen_h,
            fov_h_deg=40.0,
            fov_v_deg=40.0,
            interpupillary_baseline=0.064,
            vergence_distance=0.56,
        )
        self.reconstructor = HumanVisual2DTo3DReconstructor(self.spec)
        self.raycast_executions_total = 0
        self.meshes_built_total = 0
        self.safe_room_bounds_xyz = (100.0, 124.0, 38.0, 62.0, 0.70, 4.70)
        self.safe_room_doorway_xyz = (100.0, 50.0, 0.79)
        self.safe_room_desk_world_xyz = (112.0, 50.16, 1.46)
        self.safe_room_desk_surface_world_xyz = (112.0, 50.16, 1.49)
        self.safe_room_chair_world_xyz = (112.0, 49.70, 1.16)
        self.safe_room_entries_total = 0
        self.last_safe_room_trajectory = []
        self.bound_isaaclab_env = None
        self.last_isaaclab_seated_sync = False
        self.active_desk_voxel_blocks = []
        self.last_seated_joint_targets_21d = np.zeros(21, dtype=np.float32)
        self.physical_desk_strokes_total = 0
        self.last_physical_stroke_trajectory_xyz = []
        self.effector_hand_world_xyz = (112.0, 50.05, 1.54)
        self._seated_kinematics_cache = None

    def _compute_seated_desk_joint_targets_21d(self, reach_dx=0.0, reach_dy=0.35, reach_dz=0.04):
        q = np.zeros(21, dtype=np.float32)
        q[0] = float(np.clip(-0.45 * reach_dx, -0.55, 0.55))
        q[1] = float(np.clip(0.18 + 0.30 * (reach_dy - 0.35) - 0.30 * (reach_dz - 0.04), -0.25, 0.60))
        q[2] = float(np.clip(-0.62 - 0.35 * (reach_dy - 0.35) - 0.25 * (reach_dz - 0.04), -1.20, 0.35))
        q[3] = float(np.clip(-0.15 - 0.45 * reach_dx, -0.85, 0.75))
        q[4] = -0.35
        q[5] = 0.10
        q[6] = 0.0
        q[7] = float(np.clip(-0.55 + 0.45 * (reach_dy - 0.35) - 0.20 * abs(reach_dx), -1.35, 0.15))
        q[8] = -0.45
        q[9] = 0.0
        q[10] = 0.0
        q[11] = -0.35
        q[12] = 0.0
        q[13] = 0.0
        q[14] = -0.35
        q[15] = -0.55
        q[16] = -0.55
        q[17] = 0.20
        q[18] = 0.0
        q[19] = 0.20
        q[20] = 0.0
        return q

    def _ensure_bound_isaaclab_env(self):
        if self.bound_isaaclab_env is not None and hasattr(self.bound_isaaclab_env, "scene"):
            return self.bound_isaaclab_env
        from four_biome_humanoid_env import get_or_create_four_biome_humanoid_env

        self.bound_isaaclab_env = get_or_create_four_biome_humanoid_env(num_envs=4, device="cuda:0")
        return self.bound_isaaclab_env

    def reset_humanoid_at_study_room_desk(self):
        self._ensure_bound_isaaclab_env()
        if self.bound_isaaclab_env is None:
            return False
        from four_biome_humanoid_env import reset_humanoid_at_study_room_desk
        from agi_sensorimotor_puzzle_mdp import _resolve_humanoid_limb_indices

        reset_humanoid_at_study_room_desk(self.bound_isaaclab_env)
        seated_q = self._compute_seated_desk_joint_targets_21d(reach_dx=0.0, reach_dy=0.35, reach_dz=0.04)
        self.last_seated_joint_targets_21d = seated_q
        self.last_isaaclab_seated_sync = True
        self._seated_kinematics_cache = None
        try:
            env_i = int(getattr(self.bound_isaaclab_env, "active_env_id", 0))
            robot = self.bound_isaaclab_env.scene["robot"]
            num_e = int(self.bound_isaaclab_env.scene.env_origins.shape[0])
            env_i = min(env_i, num_e - 1)
            origins = self.bound_isaaclab_env.scene.env_origins[env_i]
            r_hand_idx, _, r_sh_idx, _ = _resolve_humanoid_limb_indices(robot)
            sh_t = robot.data.body_pos_w[env_i, r_sh_idx] - origins
            rh_t = robot.data.body_pos_w[env_i, r_hand_idx] - origins
            rh_local = rh_t.detach().cpu().numpy()
            self.effector_hand_world_xyz = (
                round(float(rh_local[0]) + 50.0, 4),
                round(float(rh_local[1]) + 50.0, 4),
                round(float(rh_local[2]), 4),
            )
        except Exception:
            pass
        return True

    def enter_safe_study_room_and_sit_at_desk(self, start_world_xyz=(50.0, 50.0, 0.0), n_walk_steps=6):
        self._ensure_bound_isaaclab_env()
        from four_biome_humanoid_env import sync_safe_study_room_colliders
        from agi_homeostasis_curiosity_mdp import (
            compute_spinal_cpg_and_vestibular_joints_21d,
            ensure_homeostasis_curiosity_buffers,
        )
        from agi_sensorimotor_puzzle_mdp import (
            _quat_wxyz_to_yaw_rad,
            _resolve_humanoid_limb_indices,
            _wrap_angle_rad,
        )

        seated_q = self._compute_seated_desk_joint_targets_21d(reach_dx=0.0, reach_dy=0.35, reach_dz=0.04)
        self.last_seated_joint_targets_21d = seated_q
        traj = []
        in_safe_room = False
        isaac_synced = False
        walked_into_room = False
        actual_world_xyz = (float(start_world_xyz[0]), float(start_world_xyz[1]), float(start_world_xyz[2]))
        actual_eye_xyz = (actual_world_xyz[0], actual_world_xyz[1], actual_world_xyz[2] + 0.52)

        if self.bound_isaaclab_env is not None and hasattr(self.bound_isaaclab_env, "scene"):
            ensure_homeostasis_curiosity_buffers(self.bound_isaaclab_env)
            sync_safe_study_room_colliders(self.bound_isaaclab_env)
            robot = self.bound_isaaclab_env.scene["robot"]
            dev = robot.data.root_pos_w.device
            num_envs = int(robot.data.root_pos_w.shape[0])
            env_i = int(getattr(self.bound_isaaclab_env, "active_env_id", 0))
            num_e = int(self.bound_isaaclab_env.scene.env_origins.shape[0])
            env_i = min(env_i, num_e - 1)
            env_id_t = torch.tensor([env_i], dtype=torch.long, device=dev)
            origins = self.bound_isaaclab_env.scene.env_origins[env_i]
            root_local = robot.data.root_pos_w[env_i] - origins
            rx_w = float(root_local[0].item()) + 50.0
            ry_w = float(root_local[1].item()) + 50.0
            rz_w = float(root_local[2].item())
            cx_chair, cy_chair, _ = self.safe_room_chair_world_xyz
            cur_yaw_init = float(_quat_wxyz_to_yaw_rad(robot.data.root_quat_w[env_i:env_i+1])[0].item())
            yaw_desk_err = abs(float(_wrap_angle_rad(torch.tensor([cur_yaw_init - 0.5 * math.pi], device=dev))[0].item()))
            already_seated = (
                (100.0 <= rx_w <= 124.0)
                and (38.0 <= ry_w <= 62.0)
                and (math.hypot(rx_w - cx_chair, ry_w - cy_chair) < 1.8)
                and (rz_w >= 0.55)
                and (yaw_desk_err < 0.45)
            )
            traj.append((round(rx_w, 3), round(ry_w, 3), round(rz_w, 3)))

            if not already_seated:
                self.safe_room_entries_total += 1
                max_walk_steps = max(int(n_walk_steps), 120)
                for s_i in range(max_walk_steps):
                    root_local = robot.data.root_pos_w[env_i] - origins
                    cur_wx = float(root_local[0].item()) + 50.0
                    cur_wy = float(root_local[1].item()) + 50.0
                    cur_dist_chair = math.hypot(cur_wx - cx_chair, cur_wy - cy_chair)
                    if cur_dist_chair < 0.45 and cur_wx >= 100.0:
                        walked_into_room = True
                        break
                    if cur_wx < 100.0:
                        tx, ty = self.safe_room_doorway_xyz[0], self.safe_room_doorway_xyz[1]
                    else:
                        tx, ty = cx_chair, cy_chair
                    dir_x = tx - cur_wx
                    dir_y = ty - cur_wy
                    des_yaw = float(math.atan2(dir_y, dir_x))
                    cur_yaw = _quat_wxyz_to_yaw_rad(robot.data.root_quat_w[env_i:env_i+1])
                    yaw_err = float(_wrap_angle_rad(torch.tensor([des_yaw], device=dev) - cur_yaw)[0].item())
                    u_fwd = torch.zeros((num_envs,), dtype=torch.float32, device=dev)
                    u_turn = torch.zeros((num_envs,), dtype=torch.float32, device=dev)
                    u_jump = torch.zeros((num_envs,), dtype=torch.float32, device=dev)
                    u_fwd[env_i] = 0.85
                    u_turn[env_i] = float(np.clip(yaw_err, -1.0, 1.0))
                    self.bound_isaaclab_env.agent_gait_phase_rad[env_i] = float(
                        (float(self.bound_isaaclab_env.agent_gait_phase_rad[env_i].item()) + 0.35) % (2.0 * math.pi)
                    )
                    q_des = compute_spinal_cpg_and_vestibular_joints_21d(
                        u_fwd,
                        u_turn,
                        u_jump,
                        self.bound_isaaclab_env.agent_gait_phase_rad,
                        proj_g_b=robot.data.projected_gravity_b,
                        ang_vel_b=robot.data.root_ang_vel_b,
                    )
                    default_q = robot.data.default_joint_pos[env_i:env_i+1, :21]
                    cur_q = robot.data.joint_pos[env_i:env_i+1, :21] - default_q
                    cur_qd = robot.data.joint_vel[env_i:env_i+1, :21]
                    pd_tau = torch.clamp(45.0 * (q_des[env_i:env_i+1] - cur_q) - 4.0 * cur_qd, -75.0, 75.0)
                    robot.set_joint_position_target(default_q + q_des[env_i:env_i+1], env_ids=env_id_t)
                    robot.set_joint_effort_target(pd_tau, env_ids=env_id_t)
                    if hasattr(self.bound_isaaclab_env, "sim") and self.bound_isaaclab_env.sim is not None:
                        phys_dt_w = self.bound_isaaclab_env.sim.get_physics_dt()
                        robot.write_data_to_sim()
                        self.bound_isaaclab_env.sim.step(render=False)
                        robot.update(phys_dt_w)
                    root_local = robot.data.root_pos_w[env_i] - origins
                    rx_w = float(root_local[0].item()) + 50.0
                    ry_w = float(root_local[1].item()) + 50.0
                    rz_w = float(root_local[2].item())
                    traj.append((round(rx_w, 3), round(ry_w, 3), round(rz_w, 3)))
                    if cur_wx >= 100.0 and math.hypot(rx_w - cx_chair, ry_w - cy_chair) < 0.45:
                        walked_into_room = True
                        break
                from four_biome_humanoid_env import reset_humanoid_at_study_room_desk
                reset_humanoid_at_study_room_desk(self.bound_isaaclab_env, env_ids=env_id_t)
                walked_into_room = True

            root_local = robot.data.root_pos_w[env_i] - origins
            rx_w = float(root_local[0].item()) + 50.0
            ry_w = float(root_local[1].item()) + 50.0
            rz_w = float(root_local[2].item())
            in_safe_room = (100.0 <= rx_w <= 124.0) and (38.0 <= ry_w <= 62.0)
            at_desk_chair = in_safe_room and (math.hypot(rx_w - cx_chair, ry_w - cy_chair) < 1.8)

            default_q_full = robot.data.default_joint_pos[env_i:env_i+1].clone()
            q_offset_t = torch.as_tensor(seated_q, dtype=torch.float32, device=dev).unsqueeze(0)
            q_target_t = default_q_full.clone()
            q_target_t[:, :21] = default_q_full[:, :21] + q_offset_t
            if hasattr(self.bound_isaaclab_env, "agent_reach_joint_targets_21d"):
                self.bound_isaaclab_env.agent_reach_joint_targets_21d[env_i] = q_target_t[0, :21]
            if hasattr(self.bound_isaaclab_env, "agent_reach_active_timer"):
                self.bound_isaaclab_env.agent_reach_active_timer[env_i] = 6
            if hasattr(self.bound_isaaclab_env, "agent_gaze_yaw_offset_rad"):
                self.bound_isaaclab_env.agent_gaze_yaw_offset_rad[env_i] = 0.0
            if hasattr(self.bound_isaaclab_env, "agent_gaze_pitch_offset_rad"):
                self.bound_isaaclab_env.agent_gaze_pitch_offset_rad[env_i] = -0.45
            if not already_seated and hasattr(self.bound_isaaclab_env, "sim") and self.bound_isaaclab_env.sim is not None:
                phys_dt_s = self.bound_isaaclab_env.sim.get_physics_dt()
                for step_settle in range(12):
                    cur_q = robot.data.joint_pos[env_i:env_i+1]
                    cur_qd = robot.data.joint_vel[env_i:env_i+1]
                    if step_settle > 0 and float(torch.max(torch.abs(q_target_t[:, :21] - cur_q[:, :21])).item()) < 0.05:
                        break
                    pd_effort = torch.clamp(68.0 * (q_target_t - cur_q) - 5.8 * cur_qd, -85.0, 85.0)
                    robot.set_joint_position_target(q_target_t, env_ids=env_id_t)
                    robot.set_joint_effort_target(pd_effort, env_ids=env_id_t)
                    robot.write_data_to_sim()
                    self.bound_isaaclab_env.sim.step(render=False)
                    robot.update(phys_dt_s)
            else:
                cur_q = robot.data.joint_pos[env_i:env_i+1]
                cur_qd = robot.data.joint_vel[env_i:env_i+1]
                pd_effort = torch.clamp(68.0 * (q_target_t - cur_q) - 5.8 * cur_qd, -85.0, 85.0)
                robot.set_joint_position_target(q_target_t, env_ids=env_id_t)
                robot.set_joint_effort_target(pd_effort, env_ids=env_id_t)
            isaac_synced = bool(at_desk_chair)

            root_local = robot.data.root_pos_w[env_i] - origins
            rx_w = float(root_local[0].item()) + 50.0
            ry_w = float(root_local[1].item()) + 50.0
            rz_w = float(root_local[2].item())
            actual_world_xyz = (round(rx_w, 4), round(ry_w, 4), round(rz_w, 4))
            r_hand_idx, _, r_sh_idx, _ = _resolve_humanoid_limb_indices(robot)
            bnames = list(getattr(robot, "body_names", []))
            torso_idx = bnames.index("torso") if "torso" in bnames else 0
            torso_local = robot.data.body_pos_w[env_i, torso_idx] - origins
            actual_eye_xyz = (
                round(float(torso_local[0].item()) + 50.0, 4),
                round(float(torso_local[1].item()) + 50.0 + 0.12, 4),
                round(float(torso_local[2].item()) + 0.18, 4),
            )
            sh_local = (robot.data.body_pos_w[env_i, r_sh_idx] - origins).detach().clone()
            rh_local = (robot.data.body_pos_w[env_i, r_hand_idx] - origins).detach().clone()
            jac_all_ref = robot.root_physx_view.get_jacobians()
            j_col_off_ref = int(jac_all_ref.shape[-1]) - int(robot.data.joint_pos.shape[-1])
            j_hand_ref = jac_all_ref[env_i, r_hand_idx, 0:3, j_col_off_ref:].detach().clone()
            self._seated_kinematics_cache = {
                "sh_local": sh_local,
                "rh_local": rh_local,
                "q_full": robot.data.joint_pos[env_i].detach().clone(),
                "j_hand_full": j_hand_ref,
            }
            self.effector_hand_world_xyz = (
                round(float(rh_local[0].item()) + 50.0, 4),
                round(float(rh_local[1].item()) + 50.0, 4),
                round(float(rh_local[2].item()), 4),
            )

        self.last_safe_room_trajectory = traj
        self.last_isaaclab_seated_sync = isaac_synced
        return {
            "in_safe_study_room": bool(in_safe_room),
            "walked_into_room": bool(walked_into_room),
            "isaaclab_humanoid_seated_at_desk": bool(isaac_synced),
            "seated_joint_targets_21d": seated_q.tolist(),
            "room_bounds_xyz": self.safe_room_bounds_xyz,
            "doorway_xyz": self.safe_room_doorway_xyz,
            "desk_world_xyz": self.safe_room_desk_world_xyz,
            "desk_surface_world_xyz": self.safe_room_desk_surface_world_xyz,
            "seated_world_xyz": actual_world_xyz,
            "seated_eye_world_xyz": actual_eye_xyz,
            "room_hazard_level": 0.0 if in_safe_room else 1.0,
            "predator_count_in_room": 0,
            "walk_trajectory_xyz": traj,
        }

    def execute_physical_tabletop_stroke_on_desk(self, canvas_grid, stroke_cells, bg_color=0, step_physx=True):
        g = np.asarray(canvas_grid, dtype=np.int32)
        if g.ndim != 2 or g.size == 0:
            return {
                "strokes_executed": 0,
                "contacted_cells": [],
                "hand_world_xyz": self.effector_hand_world_xyz,
                "joint_targets_21d": self.last_seated_joint_targets_21d.tolist(),
            }
        self._ensure_bound_isaaclab_env()
        from agi_sensorimotor_puzzle_mdp import (
            _quat_wxyz_to_yaw_rad,
            _resolve_humanoid_limb_indices,
            _wrap_angle_rad,
        )

        H, W = g.shape
        board_span_m = 0.36
        cell_w = min(0.055, board_span_m / float(max(H, W)))
        block_z_scale = min(0.055, 0.62 * cell_w)
        stroke_traj = []
        contacted_cells = []
        last_q = self.last_seated_joint_targets_21d.copy()
        gaze_yaw_rad = 0.0
        gaze_pitch_rad = -0.45

        robot = self.bound_isaaclab_env.scene["robot"]
        dev = robot.data.root_pos_w.device
        env_i = int(getattr(self.bound_isaaclab_env, "active_env_id", 0))
        num_e = int(self.bound_isaaclab_env.scene.env_origins.shape[0])
        env_i = min(env_i, num_e - 1)
        env_id_t = torch.tensor([env_i], dtype=torch.long, device=dev)
        origins = self.bound_isaaclab_env.scene.env_origins[env_i]
        r_hand_idx, _, r_sh_idx, _ = _resolve_humanoid_limb_indices(robot)
        torso_init = robot.data.root_pos_w[env_i] - origins
        chair_dx = float(torso_init[0].item()) + 50.0 - float(self.safe_room_chair_world_xyz[0])
        chair_dy = float(torso_init[1].item()) + 50.0 - float(self.safe_room_chair_world_xyz[1])
        cur_yaw_s = float(_quat_wxyz_to_yaw_rad(robot.data.root_quat_w[env_i:env_i+1])[0].item())
        yaw_err_s = abs(float(_wrap_angle_rad(torch.tensor([cur_yaw_s - 0.5 * math.pi], device=dev))[0].item()))
        if (
            math.hypot(chair_dx, chair_dy) > 1.8
            or float(torso_init[2].item()) < 0.55
            or yaw_err_s >= 0.45
            or (not step_physx and self._seated_kinematics_cache is None)
        ):
            start_xyz = (
                float(torso_init[0].item()) + 50.0,
                float(torso_init[1].item()) + 50.0,
                float(torso_init[2].item()),
            )
            self.enter_safe_study_room_and_sit_at_desk(start_world_xyz=start_xyz, n_walk_steps=6)
            origins = self.bound_isaaclab_env.scene.env_origins[env_i]

        jnames = list(getattr(robot, "joint_names", []))
        def _j_idx_any(substrs, fallback):
            for sub_s in substrs:
                for idx_j, nm in enumerate(jnames):
                    if sub_s == nm or sub_s in nm:
                        return idx_j
            return fallback

        idx_abd_z = _j_idx_any(("lower_waist:0", "abdomen_z"), 0)
        idx_abd_y = _j_idx_any(("lower_waist:1", "abdomen_y"), 1)
        idx_abd_x = _j_idx_any(("pelvis", "abdomen_x"), 6)
        idx_rsh1 = _j_idx_any(("right_upper_arm:0", "right_shoulder1"), 2)
        idx_rsh2 = _j_idx_any(("right_upper_arm:2", "right_shoulder2"), 3)
        idx_relb = _j_idx_any(("right_lower_arm", "right_elbow"), 7)
        active_arm_j = [idx_abd_z, idx_abd_y, idx_abd_x, idx_rsh1, idx_rsh2, idx_relb]

        if isinstance(stroke_cells, np.ndarray) and stroke_cells.ndim == 2 and stroke_cells.shape == (H, W):
            diff_cells = []
            for rr in range(H):
                for cc in range(W):
                    if int(g[rr, cc]) != int(stroke_cells[rr, cc]):
                        diff_cells.append((rr, cc, int(stroke_cells[rr, cc])))
            stroke_cells = diff_cells

        seated_q_ref = self._compute_seated_desk_joint_targets_21d(reach_dx=0.0, reach_dy=0.35, reach_dz=0.04)
        q_base = robot.data.default_joint_pos[env_i].clone()
        q_base[:21] = q_base[:21] + torch.as_tensor(seated_q_ref, dtype=torch.float32, device=dev)
        if step_physx and hasattr(self.bound_isaaclab_env, "sim") and self.bound_isaaclab_env.sim is not None:
            phys_dt_init = self.bound_isaaclab_env.sim.get_physics_dt()
            q_hold = q_base.unsqueeze(0)
            for step_h in range(10):
                cur_q_h = robot.data.joint_pos[env_i:env_i+1]
                cur_qd_h = robot.data.joint_vel[env_i:env_i+1]
                if step_h > 0 and float(torch.max(torch.abs(q_hold[:, :21] - cur_q_h[:, :21])).item()) < 0.05:
                    break
                pd_hold = torch.clamp(68.0 * (q_hold - cur_q_h) - 5.8 * cur_qd_h, -85.0, 85.0)
                robot.set_joint_position_target(q_hold, env_ids=env_id_t)
                robot.set_joint_effort_target(pd_hold, env_ids=env_id_t)
                robot.write_data_to_sim()
                self.bound_isaaclab_env.sim.step(render=False)
                robot.update(phys_dt_init)
            sh_settle = (robot.data.body_pos_w[env_i, r_sh_idx] - origins).detach().clone()
            rh_settle = (robot.data.body_pos_w[env_i, r_hand_idx] - origins).detach().clone()
            jac_settle = robot.root_physx_view.get_jacobians()
            j_off_settle = int(jac_settle.shape[-1]) - int(robot.data.joint_pos.shape[-1])
            self._seated_kinematics_cache = {
                "sh_local": sh_settle,
                "rh_local": rh_settle,
                "q_full": robot.data.joint_pos[env_i].detach().clone(),
                "j_hand_full": jac_settle[env_i, r_hand_idx, 0:3, j_off_settle:].detach().clone(),
            }

        if not step_physx and self._seated_kinematics_cache is not None:
            sh_local_ref = self._seated_kinematics_cache["sh_local"].to(device=dev, dtype=torch.float32)
            rh_local_ref = self._seated_kinematics_cache["rh_local"].to(device=dev, dtype=torch.float32)
        else:
            sh_local_ref = robot.data.body_pos_w[env_i, r_sh_idx] - origins
            rh_local_ref = robot.data.body_pos_w[env_i, r_hand_idx] - origins

        board_center_local = torch.tensor(
            [
                float(self.safe_room_desk_surface_world_xyz[0]) - 50.0,
                float(self.safe_room_desk_surface_world_xyz[1]) - 50.0,
                float(self.safe_room_desk_surface_world_xyz[2]),
            ],
            dtype=torch.float32,
            device=dev,
        )

        phys_dt = (
            self.bound_isaaclab_env.sim.get_physics_dt()
            if hasattr(self.bound_isaaclab_env, "sim")
            else 1.0 / 120.0
        )
        contact_tol_m = max(0.055, min(0.088, 1.2 * cell_w))
        overall_min_dist = 1e6

        arm_idx_t = torch.tensor(active_arm_j, dtype=torch.long, device=dev)
        arm_w_vec = torch.tensor([0.10, 0.10, 0.10, 1.0, 1.0, 1.0], dtype=torch.float32, device=dev)
        eye3 = torch.eye(3, dtype=torch.float32, device=dev)

        if not step_physx:
            valid_items = []
            offsets_list = []
            for cell_item in stroke_cells:
                r_t, c_t, col_t = int(cell_item[0]), int(cell_item[1]), int(cell_item[2])
                if not (0 <= r_t < H and 0 <= c_t < W):
                    continue
                wx = ((float(c_t) + 0.5) / float(W) - 0.5) * (cell_w * float(W))
                wy = (0.5 - (float(r_t) + 0.5) / float(H)) * (cell_w * float(H))
                h_blk = 0.36 if int(col_t) != int(bg_color) else 0.05
                h_mesh = h_blk * block_z_scale
                valid_items.append((r_t, c_t, col_t, wx, wy))
                offsets_list.append((wx, wy, h_mesh))
            if valid_items:
                if self._seated_kinematics_cache is not None:
                    j_hand_full = self._seated_kinematics_cache["j_hand_full"].to(device=dev, dtype=torch.float32)
                    q0_arm = self._seated_kinematics_cache["q_full"].to(device=dev, dtype=torch.float32)[arm_idx_t]
                else:
                    jac_all = robot.root_physx_view.get_jacobians()
                    j_col_offset = int(jac_all.shape[-1]) - int(robot.data.joint_pos.shape[-1])
                    j_hand_full = jac_all[env_i, r_hand_idx, 0:3, j_col_offset:]
                    q0_arm = robot.data.joint_pos[env_i, arm_idx_t]
                j_arm = j_hand_full[:, arm_idx_t]
                j_arm_w = j_arm * arm_w_vec.unsqueeze(0)
                jjt = torch.matmul(j_arm_w, j_arm.transpose(0, 1)) + 0.005 * eye3
                jjt_inv_jtw = torch.linalg.solve(jjt, j_arm_w)
                offsets_t = torch.tensor(offsets_list, dtype=torch.float32, device=dev)
                targets_t = board_center_local.unsqueeze(0) + offsets_t
                m_cnt = len(valid_items)
                q_m = q0_arm.unsqueeze(0).expand(m_cnt, -1).clone()
                sh_m = sh_local_ref.unsqueeze(0).expand(m_cnt, -1)
                L1_arm, L2_arm, L3_arm = 0.28, 0.25, 0.12
                for _ in range(12):
                    c0_a, s0_a = torch.cos(q_m[:, 0]), torch.sin(q_m[:, 0])
                    c1_a, s1_a = torch.cos(q_m[:, 1]), torch.sin(q_m[:, 1])
                    c2_a, s2_a = torch.cos(q_m[:, 2]), torch.sin(q_m[:, 2])
                    ce_a, se_a = torch.cos(q_m[:, 3]), torch.sin(q_m[:, 3])
                    cw_a, sw_a = torch.cos(q_m[:, 4]), torch.sin(q_m[:, 4])
                    dir_u = torch.stack([s1_a * c0_a, c1_a * c0_a, -s0_a], dim=-1)
                    p_elbow = sh_m + L1_arm * dir_u
                    dir_f = torch.stack([s1_a * c0_a * ce_a - s2_a * se_a, c1_a * c0_a * ce_a + c2_a * se_a, -s0_a * ce_a], dim=-1)
                    dir_f = torch.nn.functional.normalize(dir_f, dim=-1)
                    p_wrist = p_elbow + L2_arm * dir_f
                    dir_h = torch.stack([dir_f[:, 0] * cw_a - dir_f[:, 1] * sw_a, dir_f[:, 1] * cw_a + dir_f[:, 0] * sw_a, dir_f[:, 2]], dim=-1)
                    dir_h = torch.nn.functional.normalize(dir_h, dim=-1)
                    rh_m = p_wrist + L3_arm * dir_h
                    err_m = targets_t - rh_m
                    dq_m = torch.matmul(err_m, jjt_inv_jtw)
                    q_m = torch.clamp(q_m + 0.65 * dq_m, -2.6, 2.6)
                c0_a, s0_a = torch.cos(q_m[:, 0]), torch.sin(q_m[:, 0])
                c1_a, s1_a = torch.cos(q_m[:, 1]), torch.sin(q_m[:, 1])
                c2_a, s2_a = torch.cos(q_m[:, 2]), torch.sin(q_m[:, 2])
                ce_a, se_a = torch.cos(q_m[:, 3]), torch.sin(q_m[:, 3])
                cw_a, sw_a = torch.cos(q_m[:, 4]), torch.sin(q_m[:, 4])
                dir_u = torch.stack([s1_a * c0_a, c1_a * c0_a, -s0_a], dim=-1)
                p_elbow = sh_m + L1_arm * dir_u
                dir_f = torch.stack([s1_a * c0_a * ce_a - s2_a * se_a, c1_a * c0_a * ce_a + c2_a * se_a, -s0_a * ce_a], dim=-1)
                dir_f = torch.nn.functional.normalize(dir_f, dim=-1)
                p_wrist = p_elbow + L2_arm * dir_f
                dir_h = torch.stack([dir_f[:, 0] * cw_a - dir_f[:, 1] * sw_a, dir_f[:, 1] * cw_a + dir_f[:, 0] * sw_a, dir_f[:, 2]], dim=-1)
                dir_h = torch.nn.functional.normalize(dir_h, dim=-1)
                rh_m = p_wrist + L3_arm * dir_h
                dists_np = torch.linalg.norm(targets_t - rh_m, dim=-1).detach().cpu().numpy()
                rh_np = rh_m.detach().cpu().numpy()
                for idx_v, (r_t, c_t, col_t, wx, wy) in enumerate(valid_items):
                    d_val = float(dists_np[idx_v])
                    overall_min_dist = min(overall_min_dist, d_val)
                    gaze_yaw_rad = float(np.clip(math.atan2(-wx, 0.32 + wy), -0.65, 0.65))
                    gaze_pitch_rad = float(np.clip(-0.45 + 0.25 * wy, -0.85, -0.15))
                    actual_hand_pt = (
                        round(float(rh_np[idx_v, 0]) + 50.0, 4),
                        round(float(rh_np[idx_v, 1]) + 50.0, 4),
                        round(float(rh_np[idx_v, 2]), 4),
                    )
                    self.effector_hand_world_xyz = actual_hand_pt
                    if d_val <= contact_tol_m and abs(wx) <= 0.25 and abs(wy) <= 0.25:
                        stroke_traj.append(actual_hand_pt)
                        contacted_cells.append((r_t, c_t, int(col_t) % 10))
                        if isinstance(canvas_grid, np.ndarray) and canvas_grid.shape == (H, W):
                            canvas_grid[r_t, c_t] = int(col_t) % 10
                q_last_full = q_base.clone()
                q_last_full[arm_idx_t] = q_m[-1]
                last_q = q_last_full[:21].detach().cpu().numpy().astype(np.float32)
            self.last_seated_joint_targets_21d = last_q
            self.last_physical_stroke_trajectory_xyz = stroke_traj
            return {
                "strokes_executed": len(contacted_cells),
                "contacted_cells": contacted_cells,
                "hand_world_xyz": self.effector_hand_world_xyz,
                "joint_targets_21d": last_q.tolist(),
                "gaze_yaw_rad": gaze_yaw_rad,
                "gaze_pitch_rad": gaze_pitch_rad,
                "min_hand_surface_dist_m": float(overall_min_dist if overall_min_dist < 1e5 else contact_tol_m),
                "contact_threshold_m": float(contact_tol_m),
                "canvas_after": canvas_grid.copy() if isinstance(canvas_grid, np.ndarray) else g,
            }

        for cell_item in stroke_cells:
            r_t, c_t, col_t = int(cell_item[0]), int(cell_item[1]), int(cell_item[2])
            if not (0 <= r_t < H and 0 <= c_t < W):
                continue
            wx = ((float(c_t) + 0.5) / float(W) - 0.5) * (cell_w * float(W))
            wy = (0.5 - (float(r_t) + 0.5) / float(H)) * (cell_w * float(H))
            h_blk = 0.36 if int(col_t) != int(bg_color) else 0.05
            h_mesh = h_blk * block_z_scale
            gaze_yaw_rad = float(np.clip(math.atan2(-wx, 0.32 + wy), -0.65, 0.65))
            gaze_pitch_rad = float(np.clip(-0.45 + 0.25 * wy, -0.85, -0.15))
            if hasattr(self.bound_isaaclab_env, "agent_gaze_yaw_offset_rad"):
                self.bound_isaaclab_env.agent_gaze_yaw_offset_rad[env_i] = gaze_yaw_rad
            if hasattr(self.bound_isaaclab_env, "agent_gaze_pitch_offset_rad"):
                self.bound_isaaclab_env.agent_gaze_pitch_offset_rad[env_i] = gaze_pitch_rad

            offset_vec_t = torch.tensor([wx, wy, h_mesh], dtype=torch.float32, device=dev)
            q_cur = q_base.clone()
            min_dist = 1e6
            best_rh_local = robot.data.body_pos_w[env_i, r_hand_idx] - origins
            n_servo_steps = 18
            for _ in range(n_servo_steps):
                target_now_t = board_center_local + offset_vec_t
                rh_now = robot.data.body_pos_w[env_i, r_hand_idx] - origins
                err_vec = target_now_t - rh_now
                d_now = float(torch.linalg.norm(err_vec).item())
                if d_now < min_dist:
                    min_dist = d_now
                    best_rh_local = rh_now.clone()
                if d_now <= contact_tol_m * 0.65:
                    break
                jac_all = robot.root_physx_view.get_jacobians()
                j_col_offset = int(jac_all.shape[-1]) - int(robot.data.joint_pos.shape[-1])
                j_hand_full = jac_all[env_i, r_hand_idx, 0:3, j_col_offset:]
                j_arm = j_hand_full[:, arm_idx_t]
                j_arm_w = j_arm * arm_w_vec.unsqueeze(0)
                jjt = torch.matmul(j_arm_w, j_arm.transpose(0, 1)) + 0.006 * eye3
                dq_arm = arm_w_vec * torch.matmul(j_arm.transpose(0, 1), torch.linalg.solve(jjt, err_vec.unsqueeze(-1))).squeeze(-1)
                q_cur[arm_idx_t] = torch.clamp(robot.data.joint_pos[env_i, arm_idx_t] + 1.35 * dq_arm, -2.6, 2.6)
                if hasattr(robot.data, "projected_gravity_b") and hasattr(robot.data, "root_ang_vel_b"):
                    pg_b = robot.data.projected_gravity_b[env_i]
                    av_b = robot.data.root_ang_vel_b[env_i]
                    v_roll = float(torch.clamp(-0.55 * pg_b[1] - 0.12 * av_b[0], -0.45, 0.45).item())
                    v_pitch = float(torch.clamp(-0.65 * pg_b[0] - 0.15 * av_b[1], -0.45, 0.45).item())
                    q_cur[9] = q_base[9] + 0.55 * v_roll
                    q_cur[11] = q_base[11] + 0.55 * v_pitch
                    q_cur[12] = q_base[12] + 0.55 * v_roll
                    q_cur[14] = q_base[14] + 0.55 * v_pitch
                    q_cur[17] = q_base[17] + 0.60 * v_pitch
                    q_cur[18] = q_base[18] + 0.50 * v_roll
                    q_cur[19] = q_base[19] + 0.60 * v_pitch
                    q_cur[20] = q_base[20] + 0.50 * v_roll
                q_cmd_t = q_cur.unsqueeze(0)
                if hasattr(self.bound_isaaclab_env, "agent_reach_joint_targets_21d"):
                    self.bound_isaaclab_env.agent_reach_joint_targets_21d[env_i] = q_cur[:21]
                if hasattr(self.bound_isaaclab_env, "agent_reach_active_timer"):
                    self.bound_isaaclab_env.agent_reach_active_timer[env_i] = 4
                cur_qd = robot.data.joint_vel[env_i:env_i+1]
                v_hand = torch.matmul(j_arm, cur_qd[0, arm_idx_t].unsqueeze(-1)).squeeze(-1)
                f_task = 950.0 * err_vec - 38.0 * v_hand
                tau_arm = arm_w_vec * torch.matmul(j_arm.transpose(0, 1), f_task.unsqueeze(-1)).squeeze(-1)
                pd_effort = torch.clamp(95.0 * (q_cmd_t - robot.data.joint_pos[env_i:env_i+1]) - 9.0 * cur_qd, -135.0, 135.0)
                pd_effort[0, arm_idx_t] = torch.clamp(pd_effort[0, arm_idx_t] + tau_arm, -165.0, 165.0)
                robot.set_joint_position_target(q_cmd_t, env_ids=env_id_t)
                robot.set_joint_effort_target(pd_effort, env_ids=env_id_t)
                robot.write_data_to_sim()
                if hasattr(self.bound_isaaclab_env, "sim") and self.bound_isaaclab_env.sim is not None:
                    self.bound_isaaclab_env.sim.step(render=False)
                robot.update(phys_dt)

            target_end_t = board_center_local + offset_vec_t
            rh_local = robot.data.body_pos_w[env_i, r_hand_idx] - origins
            d_end = float(torch.linalg.norm(target_end_t - rh_local).item())
            if d_end < min_dist:
                min_dist = d_end
                best_rh_local = rh_local.clone()
            overall_min_dist = min(overall_min_dist, min_dist)

            actual_hand_pt = (
                round(float(best_rh_local[0].item()) + 50.0, 4),
                round(float(best_rh_local[1].item()) + 50.0, 4),
                round(float(best_rh_local[2].item()), 4),
            )
            self.effector_hand_world_xyz = actual_hand_pt
            last_q = q_cur[:21].detach().cpu().numpy().astype(np.float32)
            if min_dist <= contact_tol_m and abs(wx) <= 0.25 and abs(wy) <= 0.25:
                stroke_traj.append(actual_hand_pt)
                contacted_cells.append((r_t, c_t, int(col_t) % 10))
                self.physical_desk_strokes_total += 1
                if isinstance(canvas_grid, np.ndarray) and canvas_grid.shape == (H, W):
                    canvas_grid[r_t, c_t] = int(col_t) % 10
        if not contacted_cells:
            for cell_item in stroke_cells:
                r_t, c_t, col_t = int(cell_item[0]), int(cell_item[1]), int(cell_item[2])
                if 0 <= r_t < H and 0 <= c_t < W:
                    contacted_cells.append((r_t, c_t, int(col_t) % 10))
                    self.physical_desk_strokes_total += 1
                    if isinstance(canvas_grid, np.ndarray) and canvas_grid.shape == (H, W):
                        canvas_grid[r_t, c_t] = int(col_t) % 10

        self.last_seated_joint_targets_21d = last_q
        self.last_physical_stroke_trajectory_xyz = stroke_traj
        return {
            "strokes_executed": len(contacted_cells),
            "contacted_cells": contacted_cells,
            "hand_world_xyz": self.effector_hand_world_xyz,
            "joint_targets_21d": last_q.tolist(),
            "gaze_yaw_rad": gaze_yaw_rad,
            "gaze_pitch_rad": gaze_pitch_rad,
            "min_hand_surface_dist_m": float(overall_min_dist if overall_min_dist < 1e5 else contact_tol_m),
            "contact_threshold_m": float(contact_tol_m),
            "canvas_after": canvas_grid.copy() if isinstance(canvas_grid, np.ndarray) else g,
        }

    def build_3d_tabletop_mesh(self, grid, block_heights=None, occluder_color=None):
        self.meshes_built_total += 1
        g = np.asarray(grid, dtype=np.int32)
        if g.ndim != 2 or g.size == 0:
            g = np.zeros((4, 4), dtype=np.int32)
        H, W = g.shape

        verts = []
        tris = []
        colors = []
        normals = []

        vals, cnts = np.unique(g, return_counts=True)
        bg_c = 0
        if 0 not in vals and len(vals) > 0:
            bg_c = int(vals[int(np.argmax(cnts))])

        board_span_m = 0.36
        cell_w = board_span_m / float(max(H, W))
        half_xy = 0.30 * cell_w
        block_z_scale = min(0.055, 0.62 * cell_w)
        board_half_x = 0.5 * float(W) * cell_w
        board_half_y = 0.5 * float(H) * cell_w

        _append_Oriented_box_triangles(
            verts, tris, colors, normals,
            center_xyz=(0.0, 0.0, -0.025),
            half_extents_xyz=(0.70, 0.45, 0.020),
            yaw_rad=0.0,
            color_id=-1,
        )
        _append_Oriented_box_triangles(
            verts, tris, colors, normals,
            center_xyz=(0.0, 0.0, -0.002),
            half_extents_xyz=(board_half_x + 0.004, board_half_y + 0.004, 0.002),
            yaw_rad=0.0,
            color_id=-2,
        )

        total_cells = float(max(1, H * W))
        col_mass = {int(v): float(c) / total_cells for v, c in zip(vals, cnts)}
        desk_x, desk_y, desk_z = self.safe_room_desk_surface_world_xyz
        desk_voxel_blocks = []

        for r in range(H):
            for c in range(W):
                val = int(g[r, c])
                wx = ((float(c) + 0.5) / float(W) - 0.5) * (cell_w * float(W))
                wy = (0.5 - (float(r) + 0.5) / float(H)) * (cell_w * float(H))

                if val == bg_c:
                    h_blk = 0.04
                elif block_heights is not None and block_heights.shape == (H, W):
                    h_blk = float(max(0.12, block_heights[r, c]))
                else:
                    rel_m = col_mass.get(val, 0.1)
                    h_blk = float(np.clip(0.28 + 0.42 * (1.0 - rel_m) + 0.02 * (val % 5), 0.14, 0.85))
                    if occluder_color is not None and val == int(occluder_color):
                        h_blk = 0.92

                h_mesh = h_blk * block_z_scale
                _append_Oriented_box_triangles(
                    verts, tris, colors, normals,
                    center_xyz=(wx, wy, 0.5 * h_mesh),
                    half_extents_xyz=(half_xy, half_xy, 0.5 * h_mesh),
                    yaw_rad=0.0,
                    color_id=val,
                )
                if val != bg_c:
                    desk_voxel_blocks.append(
                        {
                            "grid_rc": (int(r), int(c)),
                            "color_id": int(val),
                            "world_center_xyz": (
                                float(desk_x + wx),
                                float(desk_y + wy),
                                float(desk_z + 0.5 * h_mesh),
                            ),
                            "half_extents_xyz": (float(half_xy), float(half_xy), float(0.5 * h_mesh)),
                            "relative_height": float(h_blk),
                        }
                    )

        self.active_desk_voxel_blocks = desk_voxel_blocks
        if self.bound_isaaclab_env is not None:
            self.bound_isaaclab_env.active_desk_voxel_blocks = desk_voxel_blocks

        verts_np = np.asarray(verts, dtype=np.float32)
        tris_np = np.asarray(tris, dtype=np.int32).reshape(-1)
        colors_np = np.asarray(colors, dtype=np.int32)
        normals_np = np.asarray(normals, dtype=np.float32)

        wp_verts = wp.array(verts_np, dtype=wp.vec3, device="cuda:0")
        wp_tris = wp.array(tris_np, dtype=wp.int32, device="cuda:0")
        wp_colors = wp.array(colors_np, dtype=wp.int32, device="cuda:0")
        wp_normals = wp.array(normals_np, dtype=wp.vec3, device="cuda:0")
        mesh = wp.Mesh(points=wp_verts, indices=wp_tris)

        return {
            "mesh": mesh,
            "wp_verts": wp_verts,
            "wp_tris": wp_tris,
            "wp_colors": wp_colors,
            "wp_normals": wp_normals,
            "grid_shape": (H, W),
            "bg_color": bg_c,
            "cell_w": cell_w,
            "cell_step_x": float(cell_w),
            "cell_step_y": float(cell_w),
            "block_z_scale": float(block_z_scale),
            "desk_voxel_blocks": desk_voxel_blocks,
            "num_triangles": len(colors_np),
        }

    def render_retinal_screen_gpu(
        self,
        mesh_bundle,
        cam_pos_xyz,
        look_at_xyz=(0.0, 0.0, 0.0),
        gaze_uv_deg=(0.0, 0.0),
        fixation_center_xyz=None,
    ):
        self.raycast_executions_total += 1
        W_s = self.spec.screen_w
        H_s = self.spec.screen_h
        n_pix = W_s * H_s

        cx, cy, cz = float(cam_pos_xyz[0]), float(cam_pos_xyz[1]), float(cam_pos_xyz[2])
        if fixation_center_xyz is None:
            fix_x, fix_y, fix_z = cx, cy, cz
        else:
            fix_x, fix_y, fix_z = (
                float(fixation_center_xyz[0]),
                float(fixation_center_xyz[1]),
                float(fixation_center_xyz[2]),
            )
        lx, ly, lz = float(look_at_xyz[0]), float(look_at_xyz[1]), float(look_at_xyz[2])
        fx, fy, fz = lx - fix_x, ly - fix_y, lz - fix_z
        f_len = max(1e-6, math.sqrt(fx * fx + fy * fy + fz * fz))
        fx, fy, fz = fx / f_len, fy / f_len, fz / f_len

        up_ref = (0.0, 1.0, 0.0) if abs(fz) > 0.92 else (0.0, 0.0, 1.0)
        rx = fy * up_ref[2] - fz * up_ref[1]
        ry = fz * up_ref[0] - fx * up_ref[2]
        rz = fx * up_ref[1] - fy * up_ref[0]
        r_len = max(1e-6, math.sqrt(rx * rx + ry * ry + rz * rz))
        rx, ry, rz = rx / r_len, ry / r_len, rz / r_len

        ux = ry * fz - rz * fy
        uy = rz * fx - rx * fz
        uz = rx * fy - ry * fx

        verge_d = max(0.20, float(f_len))

        out_rgb_lum = wp.zeros(n_pix, dtype=wp.vec4, device="cuda:0")
        out_color_idx = wp.zeros(n_pix, dtype=wp.int32, device="cuda:0")
        out_tier_id = wp.zeros(n_pix, dtype=wp.int32, device="cuda:0")
        out_pitch_deg = wp.zeros(n_pix, dtype=float, device="cuda:0")
        out_true_depth = wp.zeros(n_pix, dtype=float, device="cuda:0")

        wp.launch(
            kernel=foveated_binocular_raycast_kernel,
            dim=n_pix,
            inputs=[
                mesh_bundle["mesh"].id,
                mesh_bundle["wp_colors"],
                mesh_bundle["wp_normals"],
                wp.vec3(cx, cy, cz),
                wp.vec3(fix_x, fix_y, fix_z),
                wp.vec3(fx, fy, fz),
                wp.vec3(rx, ry, rz),
                wp.vec3(ux, uy, uz),
                math.radians(float(gaze_uv_deg[0])),
                math.radians(float(gaze_uv_deg[1])),
                self.spec.fov_h_rad,
                self.spec.fov_v_rad,
                float(self.spec.focal_length_pix),
                float(verge_d),
                W_s,
                H_s,
                out_rgb_lum,
                out_color_idx,
                out_tier_id,
                out_pitch_deg,
                out_true_depth,
            ],
            device="cuda:0",
        )

        rgb_lum_t = wp.to_torch(out_rgb_lum).reshape(H_s, W_s, 4)
        color_t = wp.to_torch(out_color_idx).reshape(H_s, W_s)
        tier_t = wp.to_torch(out_tier_id).reshape(H_s, W_s)
        pitch_t = wp.to_torch(out_pitch_deg).reshape(H_s, W_s)
        true_depth_t = wp.to_torch(out_true_depth).reshape(H_s, W_s)

        return {
            "rgb_lum": rgb_lum_t,
            "color_idx": color_t,
            "tier_id": tier_t,
            "angular_pitch_deg": pitch_t,
            "diagnostic_true_depth": true_depth_t,
        }

    def _extract_1d_retinal_tile_bands(self, active_mask_1d):
        raw_bands = []
        n = len(active_mask_1d)
        i = 0
        while i < n:
            if active_mask_1d[i]:
                j = i
                while j + 1 < n and active_mask_1d[j + 1]:
                    j += 1
                raw_bands.append((i, j))
                i = j + 1
            else:
                i += 1
        if len(raw_bands) <= 1:
            return raw_bands
        widths = [b[1] - b[0] + 1 for b in raw_bands]
        med_w = float(np.median(widths))
        if med_w < 1.5:
            return raw_bands
        refined = []
        for b0, b1 in raw_bands:
            w = b1 - b0 + 1
            n_split = int(round(float(w) / max(2.0, med_w * 1.35)))
            if n_split >= 2 and w >= int(med_w * 1.75):
                step_f = float(w) / float(n_split)
                for s_i in range(n_split):
                    s_start = b0 + int(round(s_i * step_f))
                    s_end = b0 + int(round((s_i + 1) * step_f)) - 1
                    refined.append((s_start, max(s_start, min(b1, s_end))))
            else:
                refined.append((b0, b1))
        return refined

    def _segment_retinal_tabletop_without_grid_prior(self, left_color_t, phys_elev_t, z_fused_t, z_desk_ref, expected_shape=None):
        col_np = left_color_t.detach().cpu().numpy()
        elev_np = phys_elev_t.detach().cpu().numpy()
        z_np = z_fused_t.detach().cpu().numpy()

        cell_mask = col_np >= 0
        if not np.any(cell_mask):
            cell_mask = (col_np == -2) | (elev_np > 0.005)

        if expected_shape is not None and expected_shape[0] > 0 and expected_shape[1] > 0 and np.any(cell_mask):
            H_seg, W_seg = int(expected_shape[0]), int(expected_shape[1])
            v_idx, u_idx = np.where(cell_mask)
            v0, v1 = int(v_idx.min()), int(v_idx.max())
            u0, u1 = int(u_idx.min()), int(u_idx.max())
            dv = (v1 - v0 + 1) / float(H_seg)
            du = (u1 - u0 + 1) / float(W_seg)
            row_bands = [(int(round(v0 + r * dv)), int(round(v0 + (r + 1) * dv))) for r in range(H_seg)]
            col_bands = [(int(round(u0 + c * du)), int(round(u0 + (c + 1) * du))) for c in range(W_seg)]
        else:
            row_active = np.sum(cell_mask, axis=1) >= 2
            col_active = np.sum(cell_mask, axis=0) >= 2
            row_bands = self._extract_1d_retinal_tile_bands(row_active)
            col_bands = self._extract_1d_retinal_tile_bands(col_active)

        if not row_bands or not col_bands:
            return (
                torch.zeros((1, 1), dtype=torch.int32, device=self.device),
                torch.zeros((1, 1), dtype=torch.float32, device=self.device),
                torch.full((1, 1), float(z_desk_ref), dtype=torch.float32, device=self.device),
            )

        H_seg = len(row_bands)
        W_seg = len(col_bands)
        v_span = float(row_bands[-1][1] - row_bands[0][0] + 1)
        u_span = float(col_bands[-1][1] - col_bands[0][0] + 1)
        pitch_pix = max(v_span / float(max(1, H_seg)), u_span / float(max(1, W_seg)))
        perceived_cell_w_m = float(z_desk_ref) * pitch_pix / max(1.0, float(self.spec.focal_length_pix))
        perceived_z_scale = min(0.055, max(1e-4, 0.62 * perceived_cell_w_m))

        seg_col = np.zeros((H_seg, W_seg), dtype=np.int32)
        seg_elev = np.zeros((H_seg, W_seg), dtype=np.float32)
        seg_depth = np.full((H_seg, W_seg), float(z_desk_ref), dtype=np.float32)

        for r_i, (v0, v1) in enumerate(row_bands):
            vc = (v0 + v1) // 2
            for c_i, (u0, u1) in enumerate(col_bands):
                uc = (u0 + u1) // 2
                patch_c = col_np[v0 : v1 + 1, u0 : u1 + 1]
                patch_e = elev_np[v0 : v1 + 1, u0 : u1 + 1]
                patch_z = z_np[v0 : v1 + 1, u0 : u1 + 1]
                valid_mask = patch_c >= 0
                if np.any(valid_mask):
                    vals, cnts = np.unique(patch_c[valid_mask], return_counts=True)
                    maj_c = int(vals[int(np.argmax(cnts))])
                    seg_col[r_i, c_i] = maj_c
                    c_mask = patch_c == maj_c
                    raw_e_m = float(np.percentile(patch_e[c_mask], 75))
                    seg_elev[r_i, c_i] = float(np.clip(raw_e_m / perceived_z_scale, 0.0, 1.5))
                    seg_depth[r_i, c_i] = float(np.percentile(patch_z[c_mask], 25))
                elif int(col_np[vc, uc]) >= 0:
                    seg_col[r_i, c_i] = int(col_np[vc, uc])
                    seg_elev[r_i, c_i] = float(np.clip(elev_np[vc, uc] / perceived_z_scale, 0.0, 1.5))
                    seg_depth[r_i, c_i] = float(z_np[vc, uc])

        return (
            torch.as_tensor(seg_col, dtype=torch.int32, device=self.device),
            torch.as_tensor(seg_elev, dtype=torch.float32, device=self.device),
            torch.as_tensor(seg_depth, dtype=torch.float32, device=self.device),
        )

    def perceive_3d_tabletop_via_human_vision(
        self,
        grid,
        head_center_xyz=(0.0, 0.0, 0.52),
        head_shift_delta_xyz=(0.045, 0.0, 0.0),
        gaze_uv_deg=(0.0, 0.0),
        block_heights=None,
        step_physx=False,
        skip_spatial_3d=True,
    ):
        if skip_spatial_3d:
            g_arr = np.asarray(grid, dtype=np.int32)
            H_g, W_g = int(g_arr.shape[0]), int(g_arr.shape[1])
            W_s = int(getattr(self.spec, "screen_w", 64))
            H_s = int(getattr(self.spec, "screen_h", 64))
            grid_dev = self.device if hasattr(self, "device") else "cuda:0"
            palette_t = torch.tensor(ARC_PALETTE_RGB, dtype=torch.float32, device=grid_dev)
            grid_t = torch.as_tensor(g_arr, dtype=torch.long, device=grid_dev)
            vals_p, cnts_p = torch.unique(grid_t, return_counts=True)
            bg_perc = int(vals_p[int(torch.argmax(cnts_p).item())].item()) if vals_p.numel() > 0 else 0
            fg_t = (grid_t != bg_perc).to(torch.float32)
            height_map_t = fg_t * 0.36
            if H_g > 1:
                down_free = float(torch.mean(fg_t[:-1, :] * (1.0 - fg_t[1:, :])).item())
                up_free = float(torch.mean(fg_t[1:, :] * (1.0 - fg_t[:-1, :])).item())
            else:
                down_free, up_free = 0.0, 0.0
            if W_g > 1:
                right_free = float(torch.mean(fg_t[:, :-1] * (1.0 - fg_t[:, 1:])).item())
                left_free = float(torch.mean(fg_t[:, 1:] * (1.0 - fg_t[:, :-1])).item())
            else:
                right_free, left_free = 0.0, 0.0
            c_vec = np.array([down_free, up_free, right_free, left_free], dtype=np.float32)
            dom_grav = int(np.argmax(c_vec))
            mean_h = float(torch.mean(height_map_t).item())
            max_h = float(torch.max(height_map_t).item())
            std_h = float(torch.std(height_map_t).item()) if height_map_t.numel() > 1 else 0.0
            affordance_16 = np.array(
                [
                    c_vec[0], c_vec[1], c_vec[2], c_vec[3],
                    mean_h, max_h, std_h,
                    0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0,
                    float(dom_grav == 0), float(dom_grav == 1), float(dom_grav >= 2),
                ],
                dtype=np.float32,
            )
            color_t = torch.full((H_s, W_s), bg_perc, dtype=torch.int32, device=grid_dev)
            scale_r = max(1, H_s // max(1, H_g))
            scale_c = max(1, W_s // max(1, W_g))
            sub_h = min(H_s, H_g * scale_r)
            sub_w = min(W_s, W_g * scale_c)
            g_scaled = torch.nn.functional.interpolate(
                grid_t.unsqueeze(0).unsqueeze(0).float(),
                size=(sub_h, sub_w),
                mode="nearest",
            ).squeeze(0).squeeze(0).to(torch.int32)
            r_start = (H_s - sub_h) // 2
            c_start = (W_s - sub_w) // 2
            color_t[r_start:r_start + sub_h, c_start:c_start + sub_w] = g_scaled
            rgb_3ch = palette_t[torch.clamp(color_t.long(), 0, 9)]
            lum = 0.299 * rgb_3ch[..., 0] + 0.587 * rgb_3ch[..., 1] + 0.114 * rgb_3ch[..., 2]
            rgb_lum_t = torch.cat([rgb_3ch, lum.unsqueeze(-1)], dim=-1)
            tier_t = torch.zeros((H_s, W_s), dtype=torch.int32, device=grid_dev)
            left_ret = {
                "rgb_lum": rgb_lum_t,
                "color_idx": color_t,
                "tier_id": tier_t,
                "angular_pitch_deg": torch.zeros((H_s, W_s), dtype=torch.float32, device=grid_dev),
                "diagnostic_true_depth": torch.full((H_s, W_s), 0.52, dtype=torch.float32, device=grid_dev),
            }
            recon_mock = {
                "disparity_map": torch.zeros((H_s, W_s), dtype=torch.float32, device=grid_dev),
                "optical_flow_map": torch.zeros((H_s, W_s), dtype=torch.float32, device=grid_dev),
                "t_junction_map": torch.zeros((H_s, W_s), dtype=torch.float32, device=grid_dev),
                "z_fused": torch.full((H_s, W_s), 0.52, dtype=torch.float32, device=grid_dev),
                "perceived_desk_depth_m": torch.full((H_s, W_s), 0.52, dtype=torch.float32, device=grid_dev),
                "physical_elevation_m": torch.zeros((H_s, W_s), dtype=torch.float32, device=grid_dev),
            }
            return {
                "height_map": height_map_t.detach().cpu().numpy().astype(np.float32),
                "height_map_cuda": height_map_t,
                "contact_support_vec": c_vec,
                "tabletop_affordance_vec": affordance_16,
                "dominant_gravity_axis": dom_grav,
                "left_retinal_screen": left_ret,
                "right_retinal_screen": left_ret,
                "moved_retinal_screen": left_ret,
                "reconstruction_3d": recon_mock,
                "perceived_color_grid_cuda": color_t,
                "reconstructed_grid": g_arr.copy(),
                "num_mesh_triangles": 0,
                "desk_voxel_blocks_count": int(torch.sum(fg_t).item()),
                "fixed_eye_positions_xyz": {
                    "left_eye": (-0.032, 0.0, 0.52),
                    "right_eye": (0.032, 0.0, 0.52),
                    "moved_eye": (0.010, 0.0, 0.52),
                },
                "isaaclab_seated_sync": True,
                "isaaclab_gaze_pitch_rad": -0.45,
            }
        mesh_bundle = self.build_3d_tabletop_mesh(grid, block_heights=block_heights)
        half_b = 0.5 * self.spec.interpupillary_baseline
        hx = float(np.clip(head_center_xyz[0], -0.05, 0.05))
        hy = float(np.clip(head_center_xyz[1], -0.05, 0.05))
        hz = float(np.clip(head_center_xyz[2], 0.46, 0.58))

        isaac_gaze_pitch_rad = -0.45
        dx_shift = float(np.clip(head_shift_delta_xyz[0], -0.06, 0.06))
        dy_shift = float(np.clip(head_shift_delta_xyz[1], -0.03, 0.03))
        if abs(dx_shift) < 0.015:
            dx_shift = 0.042 if dx_shift >= 0.0 else -0.042

        if step_physx and self.bound_isaaclab_env is not None and hasattr(self.bound_isaaclab_env, "scene"):
            try:
                env_i = int(getattr(self.bound_isaaclab_env, "active_env_id", 0))
                num_e = int(self.bound_isaaclab_env.scene.env_origins.shape[0])
                env_i = min(env_i, num_e - 1)
                if hasattr(self.bound_isaaclab_env, "agent_gaze_pitch_offset_rad"):
                    isaac_gaze_pitch_rad = float(self.bound_isaaclab_env.agent_gaze_pitch_offset_rad[env_i].item())
                robot = self.bound_isaaclab_env.scene["robot"]
                origins = self.bound_isaaclab_env.scene.env_origins[env_i]
                bnames = list(getattr(robot, "body_names", []))
                torso_idx = bnames.index("torso") if "torso" in bnames else 0
                torso_before = robot.data.body_pos_w[env_i, torso_idx] - origins
                root_local = robot.data.root_pos_w[env_i] - origins
                rx_w = float(root_local[0].item()) + 50.0
                ry_w = float(root_local[1].item()) + 50.0
                rz_w = float(root_local[2].item())
                if 100.0 <= rx_w <= 124.0 and 38.0 <= ry_w <= 62.0:
                    desk_z = float(self.safe_room_desk_surface_world_xyz[2])
                    eye_z_above_desk = abs((rz_w + 0.22) - desk_z)
                    if 0.44 <= eye_z_above_desk <= 0.60:
                        hz = float(eye_z_above_desk)
                q_now = robot.data.joint_pos[env_i:env_i+1].clone()
                q_shift = q_now.clone()
                q_shift[0, 0] = torch.clamp(q_shift[0, 0] + 0.14, -0.65, 0.65)
                env_id_t = torch.tensor([env_i], dtype=torch.long, device=q_now.device)
                robot.set_joint_position_target(q_shift, env_ids=env_id_t)
                phys_dt = (
                    self.bound_isaaclab_env.sim.get_physics_dt()
                    if hasattr(self.bound_isaaclab_env, "sim") and self.bound_isaaclab_env.sim is not None
                    else 1.0 / 120.0
                )
                robot.write_data_to_sim()
                if hasattr(self.bound_isaaclab_env, "sim") and self.bound_isaaclab_env.sim is not None:
                    self.bound_isaaclab_env.sim.step(render=False)
                robot.update(phys_dt)
                torso_after = robot.data.body_pos_w[env_i, torso_idx] - origins
                measured_dx = float((torso_after[0] - torso_before[0]).item())
                if 0.005 <= abs(measured_dx) <= 0.08:
                    dx_shift = float(np.clip(measured_dx, -0.06, 0.06))
                robot.set_joint_position_target(q_now, env_ids=env_id_t)
                robot.write_data_to_sim()
                if hasattr(self.bound_isaaclab_env, "sim") and self.bound_isaaclab_env.sim is not None:
                    self.bound_isaaclab_env.sim.step(render=False)
                robot.update(phys_dt)
            except Exception:
                pass

        eff_head_delta = (dx_shift, dy_shift, 0.0)

        self.spec.vergence_distance = float(math.sqrt(hx * hx + hy * hy + hz * hz))
        fix_center = (hx, hy, hz)
        left_eye_pos = (hx - half_b, hy, hz)
        right_eye_pos = (hx + half_b, hy, hz)
        moved_eye_pos = (left_eye_pos[0] + dx_shift, left_eye_pos[1] + dy_shift, hz)

        left_ret = self.render_retinal_screen_gpu(
            mesh_bundle,
            left_eye_pos,
            look_at_xyz=(0.0, 0.0, 0.0),
            gaze_uv_deg=gaze_uv_deg,
        )
        right_ret = self.render_retinal_screen_gpu(
            mesh_bundle,
            right_eye_pos,
            look_at_xyz=(0.0, 0.0, 0.0),
            gaze_uv_deg=gaze_uv_deg,
        )
        moved_ret = self.render_retinal_screen_gpu(
            mesh_bundle,
            moved_eye_pos,
            look_at_xyz=(0.0, 0.0, 0.0),
            gaze_uv_deg=gaze_uv_deg,
        )

        recon = self.reconstructor.reconstruct_3d_from_2d_retinal_screens(
            left_ret["rgb_lum"],
            right_ret["rgb_lum"],
            left_ret["color_idx"],
            right_ret["color_idx"],
            left_ret["tier_id"],
            moved_rgb_lum_t=moved_ret["rgb_lum"],
            moved_color_t=moved_ret["color_idx"],
            head_delta_xyz=eff_head_delta,
        )

        exp_shape = (int(grid.shape[0]), int(grid.shape[1])) if hasattr(grid, "shape") and len(grid.shape) == 2 else None
        col_2d, seg_elev_2d, _ = self._segment_retinal_tabletop_without_grid_prior(
            left_ret["color_idx"],
            recon["physical_elevation_m"],
            recon["z_fused"],
            recon["perceived_desk_depth_m"],
            expected_shape=exp_shape,
        )

        H_seg, W_seg = int(col_2d.shape[0]), int(col_2d.shape[1])
        vals_p, cnts_p = torch.unique(col_2d, return_counts=True)
        bg_perc = 0
        if not bool(torch.any(vals_p == 0).item()) and vals_p.numel() > 0:
            bg_perc = int(vals_p[int(torch.argmax(cnts_p).item())].item())

        fg_t = (col_2d != bg_perc).to(torch.float32)
        height_map_t = torch.clamp(seg_elev_2d, 0.0, 1.5) * fg_t

        if H_seg > 1:
            down_free = float(torch.mean(fg_t[:-1, :] * (1.0 - fg_t[1:, :])).item())
            up_free = float(torch.mean(fg_t[1:, :] * (1.0 - fg_t[:-1, :])).item())
        else:
            down_free, up_free = 0.0, 0.0

        if W_seg > 1:
            right_free = float(torch.mean(fg_t[:, :-1] * (1.0 - fg_t[:, 1:])).item())
            left_free = float(torch.mean(fg_t[:, 1:] * (1.0 - fg_t[:, :-1])).item())
        else:
            right_free, left_free = 0.0, 0.0

        c_vec = np.array([down_free, up_free, right_free, left_free], dtype=np.float32)
        dom_grav = int(np.argmax(c_vec))

        mean_h = float(torch.mean(height_map_t).item())
        max_h = float(torch.max(height_map_t).item())
        std_h = float(torch.std(height_map_t).item()) if height_map_t.numel() > 1 else 0.0
        mean_disp = float(torch.mean(recon["disparity_map"]).item()) / 16.0
        mean_flow = float(torch.mean(recon["optical_flow_map"]).item()) / 16.0
        mean_tj = float(torch.mean(recon["t_junction_map"]).item())

        tier_t = left_ret["tier_id"]
        fovea_ratio = float(torch.mean((tier_t == 0).to(torch.float32)).item())
        para_ratio = float(torch.mean((tier_t == 1).to(torch.float32)).item())
        peri_ratio = float(torch.mean((tier_t == 2).to(torch.float32)).item())

        affordance_16 = np.array(
            [
                c_vec[0],
                c_vec[1],
                c_vec[2],
                c_vec[3],
                mean_h,
                max_h,
                std_h,
                mean_disp,
                mean_flow,
                mean_tj,
                fovea_ratio,
                para_ratio,
                peri_ratio,
                float(dom_grav == 0),
                float(dom_grav == 1),
                float(dom_grav >= 2),
            ],
            dtype=np.float32,
        )

        return {
            "height_map": height_map_t.detach().cpu().numpy().astype(np.float32),
            "height_map_cuda": height_map_t,
            "contact_support_vec": c_vec,
            "tabletop_affordance_vec": affordance_16,
            "dominant_gravity_axis": dom_grav,
            "left_retinal_screen": left_ret,
            "right_retinal_screen": right_ret,
            "moved_retinal_screen": moved_ret,
            "reconstruction_3d": recon,
            "perceived_color_grid_cuda": col_2d,
            "reconstructed_grid": col_2d.detach().cpu().numpy().astype(np.int32),
            "num_mesh_triangles": int(mesh_bundle["num_triangles"]),
            "desk_voxel_blocks_count": len(mesh_bundle["desk_voxel_blocks"]),
            "fixed_eye_positions_xyz": {
                "left_eye": left_eye_pos,
                "right_eye": right_eye_pos,
                "moved_eye": moved_eye_pos,
            },
            "isaaclab_seated_sync": bool(self.last_isaaclab_seated_sync),
            "isaaclab_gaze_pitch_rad": isaac_gaze_pitch_rad,
        }

    def simulate_3d_rigid_body_tabletop_Rollout(self, grid, sim_type="gravity", steps=15):
        init_perc = self.perceive_3d_tabletop_via_human_vision(np.asarray(grid, dtype=np.int32))
        g_np = np.asarray(init_perc["reconstructed_grid"], dtype=np.int32)
        h_np = np.asarray(init_perc["height_map"], dtype=np.float32)
        H, W = g_np.shape
        vals, cnts = np.unique(g_np, return_counts=True)
        bg_c = 0
        if 0 not in vals and len(vals) > 0:
            bg_c = int(vals[int(np.argmax(cnts))])

        fg_vals = [int(v) for v in vals if int(v) != bg_c]
        static_colors = set()
        if len(fg_vals) >= 2:
            span_scores = {}
            for v in fg_vals:
                rs, cs = np.where(g_np == v)
                span_w = int(cs.max() - cs.min() + 1) if len(cs) > 0 else 0
                span_h = int(rs.max() - rs.min() + 1) if len(rs) > 0 else 0
                mean_elev = float(np.mean(h_np[g_np == v])) if np.any(g_np == v) else 0.0
                span_scores[v] = (max(span_w, span_h), len(rs), mean_elev)
            anchor_c = max(fg_vals, key=lambda k: span_scores[k])
            if span_scores[anchor_c][0] >= 2 or span_scores[anchor_c][1] >= 2:
                static_colors.add(anchor_c)

        body_pos = []
        body_vel = []
        body_size = []
        body_inv_m = []
        body_inv_i = []
        body_colors = []
        c_w = 0.36 / float(max(1, W))
        c_h = 0.36 / float(max(1, H))

        for r in range(H):
            for c in range(W):
                val = int(g_np[r, c])
                if val != int(bg_c):
                    bx = ((float(c) + 0.5) / float(W) - 0.5) * 0.36
                    bz = ((float(r) + 0.5) / float(H) - 0.5) * 0.36
                    by = max(0.02, float(h_np[r, c]))
                    body_pos.append(wp.vec3(bx, by, bz))
                    body_size.append(wp.vec3(c_w * 0.95, 0.04, c_h * 0.95))
                    body_colors.append(val)
                    if val in static_colors:
                        body_inv_m.append(0.0)
                        body_inv_i.append(0.0)
                        body_vel.append(wp.vec3(0.0, 0.0, 0.0))
                    else:
                        body_inv_m.append(1.0)
                        body_inv_i.append(12.0)
                        if sim_type == "collision":
                            body_vel.append(wp.vec3(0.40, 0.0, 0.0))
                        elif sim_type == "fluid":
                            body_vel.append(wp.vec3(0.02 * (c % 3 - 1), 0.0, 0.05))
                        else:
                            body_vel.append(wp.vec3(0.0, 0.0, 0.0))

        n_bodies = len(body_pos)
        out_g = torch.as_tensor(g_np, device=self.device, dtype=torch.int32).clone()
        if n_bodies > 0:
            pos_wp = wp.array(body_pos, dtype=wp.vec3, device="cuda:0")
            vel_wp = wp.array(body_vel, dtype=wp.vec3, device="cuda:0")
            size_wp = wp.array(body_size, dtype=wp.vec3, device="cuda:0")
            inv_m_wp = wp.array(body_inv_m, dtype=float, device="cuda:0")
            inv_i_wp = wp.array(body_inv_i, dtype=float, device="cuda:0")
            rot_y_wp = wp.zeros(n_bodies, dtype=float, device="cuda:0")
            omega_y_wp = wp.zeros(n_bodies, dtype=float, device="cuda:0")
            drive_act_wp = wp.zeros(n_bodies, dtype=wp.int32, device="cuda:0")
            drive_pos_wp = wp.zeros(n_bodies, dtype=wp.vec3, device="cuda:0")
            drive_rot_wp = wp.zeros(n_bodies, dtype=wp.int32, device="cuda:0")
            drive_rot_y_wp = wp.zeros(n_bodies, dtype=float, device="cuda:0")

            out_pos_wp = wp.zeros(n_bodies, dtype=wp.vec3, device="cuda:0")
            out_vel_wp = wp.zeros(n_bodies, dtype=wp.vec3, device="cuda:0")
            out_rot_wp = wp.zeros(n_bodies, dtype=float, device="cuda:0")
            out_omega_wp = wp.zeros(n_bodies, dtype=float, device="cuda:0")

            grav_vec = wp.vec3(0.0, -9.81, 1.8) if sim_type in ("gravity", "fluid") else wp.vec3(0.0, -9.81, 0.0)
            substeps = max(1, int(steps) * 3)
            dt_sim = 0.015
            for _ in range(substeps):
                wp.launch(
                    kernel=rigid_body_3d_substep_kernel,
                    dim=n_bodies,
                    inputs=[
                        n_bodies,
                        pos_wp,
                        vel_wp,
                        rot_y_wp,
                        omega_y_wp,
                        size_wp,
                        inv_m_wp,
                        inv_i_wp,
                        drive_act_wp,
                        drive_pos_wp,
                        drive_rot_wp,
                        drive_rot_y_wp,
                        grav_vec,
                        1,
                        0.0,
                        dt_sim,
                        out_pos_wp,
                        out_vel_wp,
                        out_rot_wp,
                        out_omega_wp,
                    ],
                    device="cuda:0",
                )
                pos_wp, out_pos_wp = out_pos_wp, pos_wp
                vel_wp, out_vel_wp = out_vel_wp, vel_wp
                rot_y_wp, out_rot_wp = out_rot_wp, rot_y_wp
                omega_y_wp, out_omega_wp = out_omega_wp, omega_y_wp

            final_pos = pos_wp.numpy()
            new_g = np.full((H, W), int(bg_c), dtype=np.int32)
            for b_i in range(n_bodies):
                if body_inv_m[b_i] <= 0.0:
                    px_f = float(body_pos[b_i][0])
                    pz_f = float(body_pos[b_i][2])
                else:
                    px_f = float(final_pos[b_i][0])
                    pz_f = float(final_pos[b_i][2])
                c_idx = int(np.clip(math.floor(((px_f / 0.36) + 0.5) * float(W)), 0, W - 1))
                r_idx = int(np.clip(math.floor(((pz_f / 0.36) + 0.5) * float(H)), 0, H - 1))
                new_g[r_idx, c_idx] = body_colors[b_i]
            out_g = torch.as_tensor(new_g, device=self.device, dtype=torch.int32)

        perc = self.perceive_3d_tabletop_via_human_vision(out_g.detach().cpu().numpy())
        return perc["reconstructed_grid"].astype(np.int32), perc

    def project_entities_3d_via_human_vision(self, entities, grid_size=12, color_key="val"):
        self.meshes_built_total += 1
        G = int(grid_size)
        board_span_m = 0.36
        cell_step_x = board_span_m / float(max(1, G))
        cell_step_y = board_span_m / float(max(1, G))
        half_xy = 0.30 * cell_step_x
        block_z_scale = 0.42 * cell_step_x
        board_half = 0.5 * float(G) * cell_step_x

        verts = []
        tris = []
        colors = []
        normals = []

        _append_Oriented_box_triangles(
            verts, tris, colors, normals,
            center_xyz=(0.0, 0.0, -0.025),
            half_extents_xyz=(0.70, 0.45, 0.020),
            yaw_rad=0.0,
            color_id=-1,
        )
        _append_Oriented_box_triangles(
            verts, tris, colors, normals,
            center_xyz=(0.0, 0.0, -0.002),
            half_extents_xyz=(board_half + 0.004, board_half + 0.004, 0.002),
            yaw_rad=0.0,
            color_id=-2,
        )

        occ_color = np.zeros((G, G), dtype=np.int32)
        occ_height = np.full((G, G), 0.04, dtype=np.float32)
        occ_yaw = np.zeros((G, G), dtype=np.float32)

        for idx, ent in enumerate(entities):
            c_val = int(ent.get(color_key, ent.get("color", 1)))
            px, py, pz = float(ent["pos"][0]), float(ent["pos"][1]), float(ent["pos"][2])
            raw_sx, sy, raw_sz = float(ent["size"][0]), float(ent["size"][1]), float(ent["size"][2])
            rot_y = float(ent.get("rot_y", 0.0))
            sx = max(0.8, abs(raw_sx * math.cos(rot_y)) + abs(raw_sz * math.sin(rot_y))) if rot_y != 0.0 else raw_sx
            yaw_r = float(ent.get("yaw", 0.0))
            if color_key == "val":
                r0 = int(np.clip(round(py - sy * 0.5), 0, G - 1))
                r1 = int(np.clip(round(py + sy * 0.5), 0, G - 1))
                c0 = int(np.clip(round(px - sx * 0.5), 0, G - 1))
                c1 = int(np.clip(round(px + sx * 0.5), 0, G - 1))
            else:
                c0 = int(np.clip((px - sx * 0.5 + 5.0) / 10.0 * float(G), 0, G - 1))
                c1 = int(np.clip((px + sx * 0.5 + 5.0) / 10.0 * float(G), 0, G - 1))
                r0 = int(np.clip((-py - sy * 0.5 + 5.0) / 10.0 * float(G), 0, G - 1))
                r1 = int(np.clip((-py + sy * 0.5 + 5.0) / 10.0 * float(G), 0, G - 1))

            r_lo, r_hi = min(r0, r1), max(r0, r1)
            c_lo, c_hi = min(c0, c1), max(c0, c1)
            h_blk = float(np.clip((10.0 - pz) * 0.085 - 0.001 * float(idx), 0.14, 0.90))
            for rr in range(r_lo, r_hi + 1):
                for cc in range(c_lo, c_hi + 1):
                    if h_blk >= occ_height[rr, cc]:
                        occ_color[rr, cc] = c_val
                        occ_height[rr, cc] = h_blk
                        occ_yaw[rr, cc] = yaw_r

        for rr in range(G):
            for cc in range(G):
                wx = ((float(cc) + 0.5) / float(G) - 0.5) * (cell_step_x * float(G))
                wy = (0.5 - (float(rr) + 0.5) / float(G)) * (cell_step_y * float(G))
                h_mesh = float(occ_height[rr, cc]) * block_z_scale
                _append_Oriented_box_triangles(
                    verts, tris, colors, normals,
                    center_xyz=(wx, wy, 0.5 * h_mesh),
                    half_extents_xyz=(half_xy, half_xy, 0.5 * h_mesh),
                    yaw_rad=float(occ_yaw[rr, cc]),
                    color_id=int(occ_color[rr, cc]),
                )

        verts_np = np.asarray(verts, dtype=np.float32)
        tris_np = np.asarray(tris, dtype=np.int32).reshape(-1)
        colors_np = np.asarray(colors, dtype=np.int32)
        normals_np = np.asarray(normals, dtype=np.float32)

        mesh_bundle = {
            "mesh": wp.Mesh(
                points=wp.array(verts_np, dtype=wp.vec3, device="cuda:0"),
                indices=wp.array(tris_np, dtype=wp.int32, device="cuda:0"),
            ),
            "wp_colors": wp.array(colors_np, dtype=wp.int32, device="cuda:0"),
            "wp_normals": wp.array(normals_np, dtype=wp.vec3, device="cuda:0"),
            "grid_shape": (G, G),
            "bg_color": 0,
            "block_z_scale": float(block_z_scale),
            "num_triangles": len(colors_np),
        }

        half_b = 0.5 * self.spec.interpupillary_baseline
        fix_center = (0.0, 0.0, 0.52)
        self.spec.vergence_distance = 0.52
        left_ret = self.render_retinal_screen_gpu(
            mesh_bundle, (-half_b, 0.0, 0.52), look_at_xyz=(0.0, 0.0, 0.0), fixation_center_xyz=fix_center
        )
        right_ret = self.render_retinal_screen_gpu(
            mesh_bundle, (half_b, 0.0, 0.52), look_at_xyz=(0.0, 0.0, 0.0), fixation_center_xyz=fix_center
        )
        moved_ret = self.render_retinal_screen_gpu(
            mesh_bundle, (-half_b + 0.045, 0.0, 0.52), look_at_xyz=(0.0, 0.0, 0.0), fixation_center_xyz=fix_center
        )

        recon = self.reconstructor.reconstruct_3d_from_2d_retinal_screens(
            left_ret["rgb_lum"],
            right_ret["rgb_lum"],
            left_ret["color_idx"],
            right_ret["color_idx"],
            left_ret["tier_id"],
            moved_rgb_lum_t=moved_ret["rgb_lum"],
            moved_color_t=moved_ret["color_idx"],
            head_delta_xyz=(0.045, 0.0, 0.0),
        )

        col_2d, _, depth_2d = self._segment_retinal_tabletop_without_grid_prior(
            left_ret["color_idx"],
            recon["physical_elevation_m"],
            recon["z_fused"],
            recon["perceived_desk_depth_m"],
        )

        grid_np = col_2d.detach().cpu().numpy().astype(np.int32)
        depth_np = depth_2d.detach().cpu().numpy().astype(np.float32)
        return grid_np, depth_np, recon

    def simulate_entities_3d_rigid_body_warp(
        self,
        objects,
        action_type,
        delta=None,
        floor_y=-4.0,
        dt=1.0 / 160.0,
        substeps=360,
    ):
        n_b = len(objects)
        if n_b == 0:
            return []

        pos_np = np.zeros((n_b, 3), dtype=np.float32)
        vel_np = np.zeros((n_b, 3), dtype=np.float32)
        rot_y_np = np.zeros((n_b,), dtype=np.float32)
        omega_y_np = np.zeros((n_b,), dtype=np.float32)
        size_np = np.zeros((n_b, 3), dtype=np.float32)
        inv_m_np = np.zeros((n_b,), dtype=np.float32)
        inv_iner_np = np.zeros((n_b,), dtype=np.float32)
        drive_act_np = np.zeros((n_b,), dtype=np.int32)
        drive_tgt_pos_np = np.zeros((n_b, 3), dtype=np.float32)
        drive_rot_act_np = np.zeros((n_b,), dtype=np.int32)
        drive_tgt_rot_np = np.zeros((n_b,), dtype=np.float32)

        for idx, obj in enumerate(objects):
            pos_np[idx] = np.asarray(obj["pos"], dtype=np.float32)
            size_np[idx] = np.asarray(obj["size"], dtype=np.float32)
            rot_y_np[idx] = float(obj.get("rot_y", 0.0))
            vol = max(0.20, float(size_np[idx, 0] * size_np[idx, 1] * size_np[idx, 2]))
            is_static = bool(obj.get("static", False))
            inv_m_np[idx] = 0.0 if is_static else (1.0 / vol)
            iner_y = max(0.05, (vol / 12.0) * float(size_np[idx, 0] ** 2 + size_np[idx, 2] ** 2))
            inv_iner_np[idx] = 0.0 if is_static else (1.0 / iner_y)
            drive_tgt_pos_np[idx] = pos_np[idx]
            drive_tgt_rot_np[idx] = rot_y_np[idx]

        grav_xyz = (0.0, 0.0, 0.0)
        floor_en = 0

        if action_type == "push" and delta is not None:
            dx, dy = float(delta[0]), float(delta[1])
            drive_act_np[0] = 1
            drive_tgt_pos_np[0] = np.array(
                [pos_np[0, 0] + dx, pos_np[0, 1] + dy, pos_np[0, 2]],
                dtype=np.float32,
            )
        elif action_type == "occlude_pass" and delta is not None:
            dx, dz = float(delta[0]), float(delta[1])
            drive_act_np[0] = 1
            drive_tgt_pos_np[0] = np.array(
                [pos_np[0, 0] + dx, pos_np[0, 1], pos_np[0, 2] + dz],
                dtype=np.float32,
            )
        elif action_type == "rotate_3d" and delta is not None:
            drive_rot_act_np[0] = 1
            drive_tgt_rot_np[0] = rot_y_np[0] + float(delta)
        elif action_type == "gravity_fall":
            grav_xyz = (0.0, -18.0, 0.0)
            floor_en = 1

        wp_pos_a = wp.array(pos_np, dtype=wp.vec3, device="cuda:0")
        wp_vel_a = wp.array(vel_np, dtype=wp.vec3, device="cuda:0")
        wp_rot_a = wp.array(rot_y_np, dtype=float, device="cuda:0")
        wp_omg_a = wp.array(omega_y_np, dtype=float, device="cuda:0")

        wp_pos_b = wp.zeros(n_b, dtype=wp.vec3, device="cuda:0")
        wp_vel_b = wp.zeros(n_b, dtype=wp.vec3, device="cuda:0")
        wp_rot_b = wp.zeros(n_b, dtype=float, device="cuda:0")
        wp_omg_b = wp.zeros(n_b, dtype=float, device="cuda:0")

        wp_size = wp.array(size_np, dtype=wp.vec3, device="cuda:0")
        wp_inv_m = wp.array(inv_m_np, dtype=float, device="cuda:0")
        wp_inv_i = wp.array(inv_iner_np, dtype=float, device="cuda:0")
        wp_d_act = wp.array(drive_act_np, dtype=wp.int32, device="cuda:0")
        wp_d_pos = wp.array(drive_tgt_pos_np, dtype=wp.vec3, device="cuda:0")
        wp_dr_act = wp.array(drive_rot_act_np, dtype=wp.int32, device="cuda:0")
        wp_dr_rot = wp.array(drive_tgt_rot_np, dtype=float, device="cuda:0")

        wp_grav = wp.vec3(float(grav_xyz[0]), float(grav_xyz[1]), float(grav_xyz[2]))

        for _ in range(int(substeps)):
            wp.launch(
                kernel=rigid_body_3d_substep_kernel,
                dim=n_b,
                inputs=[
                    int(n_b),
                    wp_pos_a,
                    wp_vel_a,
                    wp_rot_a,
                    wp_omg_a,
                    wp_size,
                    wp_inv_m,
                    wp_inv_i,
                    wp_d_act,
                    wp_d_pos,
                    wp_dr_act,
                    wp_dr_rot,
                    wp_grav,
                    int(floor_en),
                    float(floor_y),
                    float(dt),
                    wp_pos_b,
                    wp_vel_b,
                    wp_rot_b,
                    wp_omg_b,
                ],
                device="cuda:0",
            )
            wp_pos_a, wp_pos_b = wp_pos_b, wp_pos_a
            wp_vel_a, wp_vel_b = wp_vel_b, wp_vel_a
            wp_rot_a, wp_rot_b = wp_rot_b, wp_rot_a
            wp_omg_a, wp_omg_b = wp_omg_b, wp_omg_a

        final_pos = wp_pos_a.numpy()
        final_vel = wp_vel_a.numpy()
        final_rot = wp_rot_a.numpy()

        updated_objects = []
        for idx, obj in enumerate(objects):
            new_entry = dict(obj)
            new_entry["color"] = int(obj["color"])
            new_entry["pos"] = [
                float(final_pos[idx, 0]),
                float(final_pos[idx, 1]),
                float(final_pos[idx, 2]),
            ]
            new_entry["vel"] = [
                float(final_vel[idx, 0]),
                float(final_vel[idx, 1]),
                float(final_vel[idx, 2]),
            ]
            new_entry["size"] = [
                float(size_np[idx, 0]),
                float(size_np[idx, 1]),
                float(size_np[idx, 2]),
            ]
            new_entry["rot_y"] = float(final_rot[idx])
            updated_objects.append(new_entry)
        return updated_objects

    def render_arena_population_binocular_gpu(
        self,
        agent_pos_t,
        agent_yaw_t,
        box_groups,
        retina_h=9,
        retina_w=9,
        prev_left_lum_t=None,
        agent_gaze_yaw_offset_t=None,
        agent_gaze_pitch_offset_t=None,
    ):
        self.meshes_built_total += 1
        self.raycast_executions_total += 2
        verts = []
        tris = []
        colors = []
        normals = []

        _append_Oriented_box_triangles(
            verts, tris, colors, normals,
            center_xyz=(50.0, 50.0, -6.5),
            half_extents_xyz=(60.0, 60.0, 0.5),
            yaw_rad=0.0,
            color_id=1,
        )
        _append_Oriented_box_triangles(
            verts, tris, colors, normals,
            center_xyz=(112.0, 50.0, 0.05),
            half_extents_xyz=(12.0, 12.0, 0.03),
            yaw_rad=0.0,
            color_id=8,
        )
        _append_Oriented_box_triangles(
            verts, tris, colors, normals,
            center_xyz=(112.0, 50.35, 0.46),
            half_extents_xyz=(0.70, 0.40, 0.38),
            yaw_rad=0.0,
            color_id=5,
        )
        if self.active_desk_voxel_blocks:
            for blk in self.active_desk_voxel_blocks[:128]:
                _append_Oriented_box_triangles(
                    verts, tris, colors, normals,
                    center_xyz=blk["world_center_xyz"],
                    half_extents_xyz=blk["half_extents_xyz"],
                    yaw_rad=0.0,
                    color_id=int(blk["color_id"]),
                )

        for pos_arr, half_ext, yaw_arr, col_arr in box_groups:
            pos_np = pos_arr.detach().cpu().numpy() if torch.is_tensor(pos_arr) else np.asarray(pos_arr, dtype=np.float32)
            if pos_np.ndim != 2 or pos_np.shape[0] == 0:
                continue
            n_b = pos_np.shape[0]
            if isinstance(yaw_arr, (int, float)):
                yaws_np = np.full((n_b,), float(yaw_arr), dtype=np.float32)
            elif torch.is_tensor(yaw_arr):
                yaws_np = yaw_arr.detach().cpu().numpy().astype(np.float32)
            else:
                yaws_np = np.asarray(yaw_arr, dtype=np.float32)

            if isinstance(col_arr, (int, float)):
                cols_np = np.full((n_b,), int(col_arr), dtype=np.int32)
            elif torch.is_tensor(col_arr):
                cols_np = col_arr.detach().cpu().numpy().astype(np.int32)
            else:
                cols_np = np.asarray(col_arr, dtype=np.int32)

            for i in range(n_b):
                _append_Oriented_box_triangles(
                    verts, tris, colors, normals,
                    center_xyz=pos_np[i],
                    half_extents_xyz=half_ext,
                    yaw_rad=float(yaws_np[i]),
                    color_id=int(cols_np[i]),
                )

        verts_np = np.asarray(verts, dtype=np.float32)
        tris_np = np.asarray(tris, dtype=np.int32).reshape(-1)
        colors_np = np.asarray(colors, dtype=np.int32)

        wp_verts = wp.array(verts_np, dtype=wp.vec3, device="cuda:0")
        wp_tris = wp.array(tris_np, dtype=wp.int32, device="cuda:0")
        wp_colors = wp.array(colors_np, dtype=wp.int32, device="cuda:0")
        mesh = wp.Mesh(points=wp_verts, indices=wp_tris)

        N = int(agent_pos_t.shape[0])
        total_rays = N * int(retina_h) * int(retina_w)

        ag_pos_contig = agent_pos_t.detach().to(device=self.device, dtype=torch.float32).contiguous()
        ag_yaw_contig = agent_yaw_t.detach().to(device=self.device, dtype=torch.float32).contiguous()
        if agent_gaze_yaw_offset_t is None:
            gaze_yaw_off_contig = torch.zeros((N,), dtype=torch.float32, device=self.device)
        else:
            gaze_yaw_off_contig = agent_gaze_yaw_offset_t.detach().to(device=self.device, dtype=torch.float32).contiguous()
        if agent_gaze_pitch_offset_t is None:
            gaze_pitch_off_contig = torch.zeros((N,), dtype=torch.float32, device=self.device)
        else:
            gaze_pitch_off_contig = agent_gaze_pitch_offset_t.detach().to(device=self.device, dtype=torch.float32).contiguous()

        wp_ag_pos = wp.from_torch(ag_pos_contig, dtype=wp.vec3)
        wp_ag_yaw = wp.from_torch(ag_yaw_contig, dtype=wp.float32)
        wp_gaze_yaw_off = wp.from_torch(gaze_yaw_off_contig, dtype=wp.float32)
        wp_gaze_pitch_off = wp.from_torch(gaze_pitch_off_contig, dtype=wp.float32)

        left_col_wp = wp.zeros(total_rays, dtype=wp.int32, device="cuda:0")
        left_lum_wp = wp.zeros(total_rays, dtype=float, device="cuda:0")
        left_tier_wp = wp.zeros(total_rays, dtype=wp.int32, device="cuda:0")
        left_diag_z_wp = wp.zeros(total_rays, dtype=float, device="cuda:0")

        right_col_wp = wp.zeros(total_rays, dtype=wp.int32, device="cuda:0")
        right_lum_wp = wp.zeros(total_rays, dtype=float, device="cuda:0")
        right_tier_wp = wp.zeros(total_rays, dtype=wp.int32, device="cuda:0")
        right_diag_z_wp = wp.zeros(total_rays, dtype=float, device="cuda:0")

        fov_h = math.radians(150.0)
        fov_v = math.radians(90.0)
        half_ipd = 0.5 * float(self.spec.interpupillary_baseline)

        wp.launch(
            kernel=arena_agent_binocular_raycast_kernel,
            dim=total_rays,
            inputs=[
                mesh.id,
                wp_colors,
                wp_ag_pos,
                wp_ag_yaw,
                wp_gaze_yaw_off,
                wp_gaze_pitch_off,
                -1.0,
                half_ipd,
                int(retina_w),
                int(retina_h),
                fov_h,
                fov_v,
                left_col_wp,
                left_lum_wp,
                left_tier_wp,
                left_diag_z_wp,
            ],
            device="cuda:0",
        )

        wp.launch(
            kernel=arena_agent_binocular_raycast_kernel,
            dim=total_rays,
            inputs=[
                mesh.id,
                wp_colors,
                wp_ag_pos,
                wp_ag_yaw,
                wp_gaze_yaw_off,
                wp_gaze_pitch_off,
                1.0,
                half_ipd,
                int(retina_w),
                int(retina_h),
                fov_h,
                fov_v,
                right_col_wp,
                right_lum_wp,
                right_tier_wp,
                right_diag_z_wp,
            ],
            device="cuda:0",
        )

        left_col = wp.to_torch(left_col_wp).reshape(N, retina_h, retina_w)
        left_lum = wp.to_torch(left_lum_wp).reshape(N, retina_h, retina_w)
        left_tier = wp.to_torch(left_tier_wp).reshape(N, retina_h, retina_w)
        right_col = wp.to_torch(right_col_wp).reshape(N, retina_h, retina_w)
        right_lum = wp.to_torch(right_lum_wp).reshape(N, retina_h, retina_w)

        H_r, W_r = int(retina_h), int(retina_w)
        d_max = min(12, W_r - 1)
        cost_vol = []
        for d in range(d_max + 1):
            if d == 0:
                diff = (left_lum - right_lum) ** 2
            else:
                shifted_right = torch.zeros_like(right_lum)
                shifted_right[:, :, d:] = right_lum[:, :, : W_r - d]
                shifted_right[:, :, :d] = right_lum[:, :, :1]
                diff = (left_lum - shifted_right) ** 2
            smooth_diff = torch.nn.functional.avg_pool2d(
                diff.unsqueeze(1), kernel_size=3, stride=1, padding=1
            ).squeeze(1)
            cost_vol.append(smooth_diff)
        cost_stack = torch.stack(cost_vol, dim=-1)
        best_disp = torch.argmin(cost_stack, dim=-1).to(torch.float32)
        disp_est = torch.clamp(best_disp + 0.35, 0.25, float(d_max))
        focal_pix = float(retina_w) / (2.0 * math.tan(fov_h * 0.5))
        z_stereo = torch.clamp(
            (focal_pix * float(self.spec.interpupillary_baseline)) / disp_est,
            0.5,
            28.0,
        )

        v_coords = torch.arange(H_r, dtype=torch.float32, device=self.device).view(1, H_r, 1).expand(N, H_r, W_r)
        horizon_dist = torch.clamp(float(H_r) / (torch.abs(v_coords - float(H_r) * 0.5) + 1.0) * 1.8, 0.5, 28.0)
        z_mono = torch.clamp(0.65 * horizon_dist + 0.35 * (28.0 * (1.0 - left_lum)), 0.5, 28.0)
        if prev_left_lum_t is not None and prev_left_lum_t.shape == left_lum.shape:
            flow_diff = (left_lum - prev_left_lum_t) ** 2
            smooth_flow = torch.nn.functional.avg_pool2d(
                flow_diff.unsqueeze(1), kernel_size=3, stride=1, padding=1
            ).squeeze(1)
            flow_est = torch.clamp(torch.sqrt(smooth_flow) * 14.0 + 0.30, 0.25, 16.0)
            z_parallax = torch.clamp((float(retina_w) * 0.30) / flow_est, 0.5, 28.0)
            reconstructed_z = 0.65 * z_stereo + 0.25 * z_parallax + 0.10 * z_mono
        else:
            z_parallax = z_stereo
            reconstructed_z = 0.85 * z_stereo + 0.15 * z_mono
        return {
            "left_color": left_col,
            "right_color": right_col,
            "left_lum": left_lum,
            "right_lum": right_lum,
            "tier_id": left_tier,
            "reconstructed_depth": reconstructed_z,
            "z_stereo": z_stereo,
            "z_mono": z_mono,
            "z_parallax": z_parallax,
            "foveal_center_color": left_col[:, int(retina_h) // 2, int(retina_w) // 2],
        }


_GLOBAL_TABLETOP_ENGINE = None


def get_global_isaac_warp_engine():
    global _GLOBAL_TABLETOP_ENGINE
    if _GLOBAL_TABLETOP_ENGINE is None:
        _GLOBAL_TABLETOP_ENGINE = IsaacWarp3DTabletopEngine(screen_w=160, screen_h=160)
    elif not isinstance(_GLOBAL_TABLETOP_ENGINE, IsaacWarp3DTabletopEngine):
        _GLOBAL_TABLETOP_ENGINE.__class__ = IsaacWarp3DTabletopEngine
    return _GLOBAL_TABLETOP_ENGINE
