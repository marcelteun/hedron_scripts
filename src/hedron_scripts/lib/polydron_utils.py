# polydron_utils.py
# Version: 2.8.0
# Addons required: numpy, pygame, moderngl

import json
import math
import os
import tkinter as tk
from tkinter import filedialog

import moderngl
import numpy as np
import pygame

_DIR_STATE_PATH = os.path.join(
    os.path.expanduser("~"), ".polydron_paths.json")
_dir_state_cache = None


def _load_dir_state():
    global _dir_state_cache
    if _dir_state_cache is not None:
        return _dir_state_cache

    state = {}
    try:
        with open(_DIR_STATE_PATH, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            state = loaded
    except (OSError, ValueError):
        state = {}

    _dir_state_cache = state
    return _dir_state_cache


def _save_dir_state(state):
    global _dir_state_cache
    _dir_state_cache = state
    try:
        with open(_DIR_STATE_PATH, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
    except OSError:
        pass


def get_last_export_dir():
    state = _load_dir_state()
    folder = state.get("last_export_dir", "")
    if folder and os.path.isdir(folder):
        return folder
    return os.getcwd()


def set_last_export_dir(folder):
    if folder and os.path.isdir(folder):
        state = dict(_load_dir_state())
        state["last_export_dir"] = os.path.abspath(folder)
        _save_dir_state(state)


def get_last_open_dir():
    state = _load_dir_state()
    folder = state.get("last_open_dir", "")
    if folder and os.path.isdir(folder):
        return folder
    return get_last_export_dir()


def set_last_open_dir(folder):
    if folder and os.path.isdir(folder):
        state = dict(_load_dir_state())
        state["last_open_dir"] = os.path.abspath(folder)
        _save_dir_state(state)


def get_next_output_path(prefix="output", ext=".off", directory=None):
    base_prefix = os.path.join(directory, prefix) if directory else prefix
    idx = 1
    while True:
        candidate = f"{base_prefix}({idx}){ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def choose_open_off_path(title="Open OFF file"):
    tk_root = tk.Tk()
    try:
        tk_root.withdraw()
        tk_root.attributes("-topmost", True)
        off_path = filedialog.askopenfilename(
            title=title,
            initialdir=get_last_open_dir(),
            filetypes=[("OFF files", "*.off"), ("All files", "*.*")])
    finally:
        tk_root.destroy()

    if off_path:
        set_last_open_dir(os.path.dirname(off_path))
    return off_path


def choose_save_off_path(title="Export OFF file", default_stem="output"):
    export_dir = get_last_export_dir()
    suggested = get_next_output_path(
        prefix=default_stem, ext=".off", directory=export_dir)
    tk_root = tk.Tk()
    try:
        tk_root.withdraw()
        tk_root.attributes("-topmost", True)
        save_path = filedialog.asksaveasfilename(
            title=title,
            initialdir=export_dir,
            initialfile=os.path.basename(suggested),
            defaultextension=".off",
            filetypes=[("OFF files", "*.off"), ("All files", "*.*")],
            confirmoverwrite=False)
    finally:
        tk_root.destroy()

    if save_path:
        set_last_export_dir(os.path.dirname(save_path))
    return save_path


# -----------------------------------------------------------------------------
# Camera & Projection
# -----------------------------------------------------------------------------


def make_perspective(fovy_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
    return np.array([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [
            0.0,
            0.0,
            (far + near) / (near - far),
            (2.0 * far * near) / (near - far)],
        [0.0, 0.0, -1.0, 0.0]], dtype=np.float32)


def make_look_at(eye, target, up):
    f = (target - eye) / np.linalg.norm(target - eye)
    s = np.cross(f, up) / np.linalg.norm(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def project_to_screen(points_3d, mvp, width, height):
    pts_homo = np.hstack([
        points_3d, np.ones((len(points_3d), 1), dtype=np.float32)])
    clip = pts_homo @ mvp.T
    w = clip[:, 3:4]
    valid = w[:, 0] > 0.05
    ndc = clip[:, :3] / np.maximum(w, 1e-6)
    screen_x = (ndc[:, 0] + 1.0) * 0.5 * width
    screen_y = (1.0 - ndc[:, 1]) * 0.5 * height
    return np.column_stack([screen_x, screen_y, ndc[:, 2]]), valid


def point_to_segment_dist_2d(p, a, b):
    ab = b - a
    l2 = np.dot(ab, ab)
    if l2 == 0.0:
        return np.linalg.norm(p - a)
    t = np.clip(np.dot(p - a, ab) / l2, 0.0, 1.0)
    return np.linalg.norm(p - (a + t * ab))


# -----------------------------------------------------------------------------
# 3D Rivet Node Generator & Mesh Helpers
# -----------------------------------------------------------------------------

OCT_VERTS = np.array([
    [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)

OCT_TRIS = [
    (0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
    (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5)]

OCT_NORMALS = []
for _i0, _i1, _i2 in OCT_TRIS:
    _n = np.cross(
        OCT_VERTS[_i1] - OCT_VERTS[_i0], OCT_VERTS[_i2] - OCT_VERTS[_i0])
    _n /= np.linalg.norm(_n) + 1e-8
    OCT_NORMALS.append(_n)


def append_vertex_marker(buf, centre, radius, colour):
    cx, cy, cz = float(centre[0]), float(centre[1]), float(centre[2])
    r0, g0, b0 = colour[0], colour[1], colour[2]
    for (i0, i1, i2), norm in zip(OCT_TRIS, OCT_NORMALS):
        nx, ny, nz = float(norm[0]), float(norm[1]), float(norm[2])
        for idx in (i0, i1, i2):
            v = OCT_VERTS[idx]
            buf.extend([
                cx + float(v[0]) * radius,
                cy + float(v[1]) * radius,
                cz + float(v[2]) * radius,
                nx, ny, nz, r0, g0, b0])


def append_face_triangles(buf, face, norm, col):
    pts = face["world_pts"]
    nx, ny, nz = float(norm[0]), float(norm[1]), float(norm[2])
    c0, c1, c2 = col[0], col[1], col[2]
    p = face["gonality"]
    q = face.get("density", 1)

    if q > 1:
        centre = np.mean(pts, axis=0)
        hull = [None] * p
        for m in range(p):
            hull[(m * q) % p] = pts[m]
        ratio = (
            math.cos(q * math.pi / p)
            / (math.cos((q - 1) * math.pi / p) * math.cos(math.pi / p)))
        valleys = [
            centre + ratio * (0.5 * (hull[j] + hull[(j + 1) % p]) - centre)
            for j in range(p)]
        for j in range(p):
            u_cur = hull[j]
            u_nxt = hull[(j + 1) % p]
            w_cur = valleys[j]
            for tri_pts in ((centre, u_cur, w_cur), (centre, w_cur, u_nxt)):
                for pt in tri_pts:
                    buf.extend([
                        float(pt[0]), float(pt[1]), float(pt[2]),
                        nx, ny, nz, c0, c1, c2])
        return

    p0 = pts[0]
    for i in range(1, len(pts) - 1):
        p1, p2 = pts[i], pts[i + 1]
        for pt in (p0, p1, p2):
            buf.extend([
                float(pt[0]), float(pt[1]), float(pt[2]),
                nx, ny, nz, c0, c1, c2])


def upload_vbo(vbo, data_list, stride):
    if not data_list:
        return 0
    arr = np.asarray(data_list, dtype=np.float32)
    if arr.nbytes > vbo.size:
        vbo.orphan(arr.nbytes * 2)
    vbo.write(arr.tobytes())
    return len(arr) // stride


# -----------------------------------------------------------------------------
# Shaders & OpenGL Setup Helpers
# -----------------------------------------------------------------------------

SURFACE_VS = """
#version 330
uniform mat4 u_mvp;
uniform mat3 u_rot;
in vec3 in_pos;
in vec3 in_norm;
in vec3 in_col;
out vec3 v_norm;
out vec3 v_col;
void main() {
    v_norm = u_rot * in_norm;
    v_col = in_col;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

SURFACE_FS = """
#version 330
in vec3 v_norm;
in vec3 v_col;
out vec4 fragColor;
void main() {
    vec3 n = normalize(v_norm);
    if (!gl_FrontFacing) n = -n;
    vec3 light = normalize(vec3(0.5, 0.8, 1.0));
    float diff = max(dot(n, light), 0.0);
    fragColor = vec4((0.4 + 0.6 * diff) * v_col, 1.0);
}
"""

LINE_VS = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
in vec3 in_col;
out vec3 v_col;
void main() {
    v_col = in_col;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

LINE_FS = """
#version 330
in vec3 v_col;
out vec4 fragColor;
void main() {
    fragColor = vec4(v_col, 1.0);
}
"""

HELP_VS = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

HELP_FS = """
#version 330
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    fragColor = texture(u_tex, v_uv);
}
"""


def create_gl_programs(ctx):
    surf_prog = ctx.program(vertex_shader=SURFACE_VS, fragment_shader=SURFACE_FS)
    line_prog = ctx.program(vertex_shader=LINE_VS, fragment_shader=LINE_FS)
    help_prog = ctx.program(vertex_shader=HELP_VS, fragment_shader=HELP_FS)
    return surf_prog, line_prog, help_prog


def create_help_overlay(ctx, help_prog, help_lines, width, height):
    help_font = (
        pygame.font.SysFont("consolas", 18)
        or pygame.font.Font(None, 22))
    line_h = help_font.get_linesize() + 2
    pad_x, pad_y = 20, 16
    max_w = max(help_font.size(line)[0] for line in help_lines)
    tex_w = max_w + 2 * pad_x
    tex_h = len(help_lines) * line_h + 2 * pad_y
    if tex_w % 2:
        tex_w += 1
    if tex_h % 2:
        tex_h += 1

    bg_rgb = (18, 22, 30)
    help_surf = pygame.Surface((tex_w, tex_h), pygame.SRCALPHA)
    help_surf.fill((*bg_rgb, 248))
    for idx, line in enumerate(help_lines):
        fg = (255, 225, 110) if idx == 0 else (242, 246, 252)
        txt_surf = help_font.render(line, True, fg, bg_rgb)
        help_surf.blit(txt_surf, (pad_x, pad_y + idx * line_h))
    pygame.draw.rect(
        help_surf, (130, 145, 175, 255), (0, 0, tex_w, tex_h), width=2)

    help_tex = ctx.texture(
        (tex_w, tex_h), 4, pygame.image.tostring(help_surf, "RGBA", True))
    help_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    help_tex.repeat_x = False
    help_tex.repeat_y = False

    hx1 = (width - tex_w) // 2
    hy1 = (height - tex_h) // 2
    hx2 = hx1 + tex_w
    hy2 = hy1 + tex_h
    nx1 = (hx1 / width) * 2.0 - 1.0
    nx2 = (hx2 / width) * 2.0 - 1.0
    ny1 = 1.0 - (hy1 / height) * 2.0
    ny2 = 1.0 - (hy2 / height) * 2.0
    help_verts = np.array([
        nx1, ny1, 0.0, 1.0,
        nx1, ny2, 0.0, 0.0,
        nx2, ny2, 1.0, 0.0,
        nx1, ny1, 0.0, 1.0,
        nx2, ny2, 1.0, 0.0,
        nx2, ny1, 1.0, 1.0], dtype=np.float32)
    help_vbo = ctx.buffer(help_verts.tobytes())
    help_vao = ctx.vertex_array(
        help_prog, [(help_vbo, "2f 2f", "in_pos", "in_uv")])
    return help_tex, help_vao


def create_divider_vao(ctx, line_prog):
    divider_verts = np.array([
        0.0, -1.0, -0.99, 0.55, 0.58, 0.64,
        0.0, 1.0, -0.99, 0.55, 0.58, 0.64], dtype=np.float32)
    divider_vbo = ctx.buffer(divider_verts.tobytes())
    return ctx.vertex_array(
        line_prog, [(divider_vbo, "3f 3f", "in_pos", "in_col")])


def build_ui_buttons_data(
        pane_w, height, undo_rect, delete_rect,
        hover_undo, hover_delete, can_delete):
    ui_surf_data = []
    ui_line_data = []

    def append_ui_btn(rect, bg_col, border_col):
        x1, y1, x2, y2 = rect
        nx1 = (x1 / pane_w) * 2.0 - 1.0
        nx2 = (x2 / pane_w) * 2.0 - 1.0
        ny1 = 1.0 - (y1 / height) * 2.0
        ny2 = 1.0 - (y2 / height) * 2.0
        z = -0.99
        norm = [0.0, 0.0, 1.0]

        quad_pts = [
            (nx1, ny1), (nx1, ny2), (nx2, ny2),
            (nx1, ny1), (nx2, ny2), (nx2, ny1)]
        for px, py in quad_pts:
            ui_surf_data.extend([px, py, z, *norm, *bg_col])

        border_segs = [
            ((nx1, ny1), (nx2, ny1)),
            ((nx2, ny1), (nx2, ny2)),
            ((nx2, ny2), (nx1, ny2)),
            ((nx1, ny2), (nx1, ny1))]
        for (ax, ay), (bx, by) in border_segs:
            ui_line_data.extend([
                ax, ay, z, *border_col, bx, by, z, *border_col])

    bg_u = [0.35, 0.38, 0.45] if hover_undo else [0.22, 0.24, 0.28]
    bd_u = [0.65, 0.70, 0.85] if hover_undo else [0.40, 0.44, 0.50]
    append_ui_btn(undo_rect, bg_u, bd_u)

    if can_delete:
        bg_d = [0.65, 0.22, 0.25] if hover_delete else [0.45, 0.18, 0.20]
        bd_d = [0.95, 0.40, 0.45] if hover_delete else [0.70, 0.28, 0.32]
    else:
        bg_d = [0.18, 0.16, 0.18]
        bd_d = [0.30, 0.26, 0.28]
    append_ui_btn(delete_rect, bg_d, bd_d)

    return ui_surf_data, ui_line_data