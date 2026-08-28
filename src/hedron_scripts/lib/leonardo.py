# Version: 7.0
# Addons required: glfw, moderngl, numpy, imgui-bundle, pyrr, sys, PyOpenGL

import ctypes
import os
import re
import sys
import time
from collections import deque

import glfw
import moderngl
import numpy as np
from imgui_bundle import imgui
from pyrr import Matrix44

# =========================================================
# CONFIGURATION
# =========================================================
DEBUG_MODE = False

def dprint(*args):
    if DEBUG_MODE:
        print("DEBUG:", *args)
        sys.stdout.flush()

# =========================================================
# HELP TEXT
# =========================================================
HELP_TEXT = """This model builds a Leonardo style version of an input polyhedron.

It can read OFF files (and certain VRML files).  Just drag and drop onto the screen.

Two main parameters W and D set the width and depth of the frames.  They normally work off teh sliedrs but a CTRL left mouse click will aloow direct keyboard entry.

The 'no overlap' toggle stops the lateral side walls from overlapping by limiting their depth.

Default colour is to take colours from the input file.  The alternative colour mode is that those in the planes of the input file (the 'top' edges of the frames) are red, the 'side' edges yellow and the 'bottom' edges blue.

Clicking on the Source Polyhedron Button makes a slider visible where the source polyhedron can be viewed as a transparent layer.

The Export button writes an OFF file, with '-leonardo' appended to the input directory.  The source polyhedron is not included.

WARNING.  If W and D are equal then trying to perform Leonardos on a cube results in degeneracy.  In general all repeat Leonardos can be problematic, use them at your own risk.  Crossed faces and hemihedra are also at your risk.

Jim McNeill/Google AI Studio 2026"""

# =========================================================
# GEOMETRY UTILS
# =========================================================

def calculate_face_normal_and_area(coords):
    if len(coords) < 3:
        return np.array([0.0, 0.0, 0.0]), 0.0
    nx = ny = nz = 0.0
    for i in range(len(coords)):
        v_curr = coords[i]
        v_next = coords[(i + 1) % len(coords)]
        nx += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
        ny += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
        nz += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])
    length = np.sqrt(nx**2 + ny**2 + nz**2)
    if length == 0:
        return np.array([0.0, 0.0, 0.0]), 0.0
    return np.array([nx/length, ny/length, nz/length]), 0.5 * length

def is_crossed_or_concave(coords, face_normal):
    if len(coords) < 4:
        return False  
    for i in range(len(coords)):
        v_prev = coords[i - 1]
        v_curr = coords[i]
        v_next = coords[(i + 1) % len(coords)]
        e1 = (v_curr[0] - v_prev[0], v_curr[1] - v_prev[1], v_curr[2] - v_prev[2])
        e2 = (v_next[0] - v_curr[0], v_next[1] - v_curr[1], v_next[2] - v_curr[2])
        local_nx = e1[1] * e2[2] - e1[2] * e2[1]
        local_ny = e1[2] * e2[0] - e1[0] * e2[2]
        local_nz = e1[0] * e2[1] - e1[1] * e2[0]
        dot = local_nx * face_normal[0] + local_ny * face_normal[1] + local_nz * face_normal[2]
        if dot < -1e-12:
            return True
    return False

def merge_vertices(verts, faces, tolerance=1e-10):
    dprint("Starting merge_vertices...")
    unique_verts = []
    v_map = {}
    new_faces = []
    inv_tol = 1.0 / tolerance
    for i, face in enumerate(faces):
        new_face = []
        for v_idx in face:
            v = verts[v_idx]
            key = tuple((v * inv_tol).astype(np.int64))
            if key not in v_map:
                v_map[key] = len(unique_verts)
                unique_verts.append(v)
            new_face.append(v_map[key])
        clean_face = []
        for idx in new_face:
            if not clean_face or idx != clean_face[-1]:
                clean_face.append(idx)
        if len(clean_face) > 2 and clean_face[0] == clean_face[-1]:
            clean_face.pop()
        if len(clean_face) >= 3:
            new_faces.append(clean_face)
        if i % 1000 == 0 and i > 0:
            dprint(f"  merge_vertices: processed {i} faces...")
    dprint(f"merge_vertices complete. Verts: {len(unique_verts)}, Faces: {len(new_faces)}")
    return np.array(unique_verts, dtype=np.float64), new_faces

def ensure_consistent_normals(vertices, faces_with_data, label="Source"):
    if not faces_with_data: 
        dprint(f"ensure_consistent_normals ({label}): No faces.")
        return faces_with_data
    num_faces = len(faces_with_data)
    dprint(f"Starting ensure_consistent_normals ({label}) for {num_faces} faces...")
    visited, face_adj = [False] * num_faces, {}
    dprint(f"  ({label}) Building adjacency map...")
    for i, (face, _) in enumerate(faces_with_data):
        f_len = len(face)
        for j in range(f_len):
            v1, v2 = face[j], face[(j + 1) % f_len]
            if v1 == v2:
                continue 
            face_adj.setdefault(tuple(sorted((v1, v2))), []).append(i)
    processed_faces = [[list(f), col] for f, col in faces_with_data]
    total_visited = 0
    for start_idx in range(num_faces):
        if visited[start_idx]:
            continue
        queue = deque([start_idx])
        visited[start_idx] = True
        total_visited += 1
        while queue:
            curr_idx = queue.popleft()
            curr_face, _ = processed_faces[curr_idx]
            f_len = len(curr_face)
            for j in range(f_len):
                v1, v2 = curr_face[j], curr_face[(j + 1) % f_len]
                if v1 == v2:
                    continue
                for neighbor_idx in face_adj.get(tuple(sorted((v1, v2))), []):
                    if not visited[neighbor_idx]:
                        neighbor, _ = processed_faces[neighbor_idx]
                        n_len, match = len(neighbor), False
                        for k in range(n_len):
                            nv1, nv2 = neighbor[k], neighbor[(k + 1) % n_len]
                            if nv1 == v1 and nv2 == v2:
                                processed_faces[neighbor_idx][0] = neighbor[::-1]
                                match = True
                                break
                            elif nv1 == v2 and nv2 == v1:
                                match = True
                                break
                        if match:
                            visited[neighbor_idx] = True
                            total_visited += 1
                            queue.append(neighbor_idx)
                            if total_visited % 1000 == 0:
                                dprint(f"  ({label}) Traversed {total_visited}/{num_faces}...")
    dprint(f"ensure_consistent_normals ({label}) complete.")
    return [(tuple(f), col) for f, col in processed_faces]

def get_signed_volume(verts, faces, label="Source"):
    dprint(f"Calculating signed volume ({label})...")
    vol = 0
    for i, f in enumerate(faces):
        v0 = verts[f[0]]
        for j in range(1, len(f) - 1):
            vol += np.dot(v0, np.cross(verts[f[j]], verts[f[j+1]]))
        if i % 5000 == 0 and i > 0:
            dprint(f"  volume ({label}): {i} faces...")
    res = vol / 6.0
    dprint(f"Signed volume ({label}) complete: {res}")
    return res

def get_truncated_icosahedron():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico_v = np.array([[-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0], [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi], [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]], dtype=np.float64)
    ico_f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    edge_to_pts, final_verts, v_map, inv_tol = {}, [], {}, 1.0 / 1e-10
    def add_v(v):
        key = tuple((v * inv_tol).astype(np.int64))
        if key not in v_map:
            v_map[key] = len(final_verts)
            final_verts.append(v)
        return v_map[key]
    for f in ico_f:
        for i in range(3):
            u, v = f[i], f[(i+1)%3]
            pair = tuple(sorted((u, v)))
            if pair not in edge_to_pts:
                edge_to_pts[pair] = {u: add_v(ico_v[u]+(ico_v[v]-ico_v[u])/3.0), v: add_v(ico_v[u]+2.0*(ico_v[v]-ico_v[u])/3.0)}
    faces, cols = [], []
    for f in ico_f:
        hex_f = []
        for i in range(3):
            u, v = f[i], f[(i+1)%3]
            pair = tuple(sorted((u, v)))
            hex_f.extend([edge_to_pts[pair][u], edge_to_pts[pair][v]])
        faces.append(hex_f)
        cols.append([1.0, 1.0, 0.1])
    for i in range(12):
        neighbors = []
        for f in ico_f:
            if i in f:
                neighbors.append(f[(f.index(i)+1)%3])
                break
        while len(neighbors) < 5:
            curr = neighbors[-1]
            for f in ico_f:
                if i in f and curr in f:
                    cand = f[3 - f.index(i) - f.index(curr)]
                    if cand not in neighbors:
                        neighbors.append(cand)
                        break
        faces.append([edge_to_pts[tuple(sorted((i, n)))][i] for n in neighbors])
        cols.append([1.0, 0.1, 0.1])
    return np.array(final_verts, dtype=np.float64), faces, cols

def extract_vrml_array(text, tag):
    pattern = re.compile(r'\b' + re.escape(tag) + r'\b.*?\[(.*?)\]', re.DOTALL)
    match = pattern.search(text)
    if not match:
        return []
    return re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', match.group(1))

def load_poly_data(path):
    if not path or not os.path.exists(path):
        return get_truncated_icosahedron()
    ext = os.path.splitext(path)[1].lower()
    try:
        verts, faces, cols = None, [], []
        if ext in ['.wrl', '.vrml']:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = "\n".join([line.split('#')[0] for line in f])
            all_pts, all_colors = extract_vrml_array(content, "point"), extract_vrml_array(content, "color")
            ifs_matches = list(re.finditer(r'IndexedFaceSet\s*\{', content))
            models = []
            for match in ifs_matches:
                start = match.start()
                name_match = re.search(r'DEF\s+(\w+)', content[max(0, start-150):start])
                name = name_match.group(1) if name_match else "Unknown"
                bc, end = 0, start + content[start:].find('{')
                for i in range(end, len(content)):
                    if content[i] == '{':
                        bc += 1
                    elif content[i] == '}':
                        bc -= 1
                        if bc == 0:
                            block = content[start:i+1]
                            pts, idx = extract_vrml_array(block, "point") or all_pts, extract_vrml_array(block, "coordIndex")
                            if pts and idx and len(pts) % 3 == 0:
                                models.append({'name': name, 'pts': pts, 'idx': idx, 'block': block})
                            break
            if not models:
                return None
            target = next((m for m in models if m['name'] == "SOLID"), models[0])
            verts = np.array(target['pts'], dtype=np.float64).reshape(-1, 3)
            curr = []
            for i in target['idx']:
                val = int(float(i))
                if val == -1:
                    if curr:
                        faces.append(curr)
                        curr = []
                else:
                    curr.append(val)
            pal = np.array([[0.7, 0.7, 0.7]])
            ctk = extract_vrml_array(target['block'], "color") or all_colors
            if ctk and len(ctk) >= 3:
                pal = np.array(ctk, dtype=np.float64).reshape(-1, 3)
            cit = extract_vrml_array(target['block'], "colorIndex")
            if cit:
                cidxs = [int(float(t)) for t in cit if int(float(t)) != -1]
                cols = [pal[cidxs[i]].tolist() if i < len(cidxs) else pal[0].tolist() for i in range(len(faces))]
            else:
                cols = [pal[0].tolist()] * len(faces)
        else:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_lines = f.readlines()
            
            lines = []
            for l in raw_lines:
                stripped = l.strip()
                if stripped.startswith('\ufeff'):
                    stripped = stripped.replace('\ufeff', '')
                if stripped and not stripped.startswith('#'):
                    lines.append(stripped)
            
            if not lines:
                return None
                
            if lines[0].startswith('OFF'):
                if len(lines[0]) > 3:
                    header = lines[0][3:].split()
                    start = 1
                else:
                    header = lines[1].split()
                    start = 2
            else:
                header = lines[0].split()
                start = 1
                
            nv, nf, _ = map(int, header)
            verts = np.array([list(map(float, l.split()[:3])) for l in lines[start:start+nv]], dtype=np.float64)
            for i in range(nf):
                p = lines[start+nv+i].split()
                n = int(p[0])
                faces.append(list(map(int, p[1:1+n])))
                c = [0.7, 0.7, 0.7]
                if len(p) > 1+n:
                    raw = [float(x) for x in p[1+n:1+n+3]]
                    c = [x/255.0 if x > 1.0 else x for x in raw]
                cols.append(c)
        return verts, faces, cols
    except (OSError, ValueError, IndexError, KeyError):
        return None
        
# =========================================================
# LEONARDO SOLVER
# =========================================================

def get_miter_vectors(vertices, faces, depth):
    dprint("Calculating miter vectors...")
    face_normals, v_to_f = [], [[] for _ in range(len(vertices))]
    for fi, f in enumerate(faces):
        v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
        n = np.cross(v1-v0, v2-v0)
        n /= (np.linalg.norm(n) + 1e-15)
        face_normals.append(n)
        for vi in f:
            v_to_f[vi].append(fi)
    miters = np.zeros_like(vertices)
    for vi in range(len(vertices)):
        adj = list(set(v_to_f[vi]))
        if len(adj) < 2:
            continue
        A, b = np.array([face_normals[fi] for fi in adj]), np.full((len(adj),), -float(depth))
        m, _, _, _ = np.linalg.lstsq(A, b, rcond=-1)
        miters[vi] = m
    return miters

def build_leonardo_mesh(vertices, faces, cols, width, depth, alt_mode):
    dprint(f"Starting build_leonardo_mesh (W={width}, D={depth})...")
    raw_quads, global_miters = [], get_miter_vectors(vertices, faces, depth)
    for fi, face in enumerate(faces):
        vi, flen = list(face), len(face)
        fn = np.cross(vertices[vi[1]]-vertices[vi[0]], vertices[vi[2]]-vertices[vi[0]])
        fn /= (np.linalg.norm(fn) + 1e-15)
        ortho_ext = -fn * float(depth)
        t_surf, t_bot = [], []
        for i in range(flen):
            v, vp, vn = vertices[vi[i]], vertices[vi[i-1]], vertices[vi[(i+1)%flen]]
            e1, e2 = (vp-v)/(np.linalg.norm(vp-v)+1e-15), (vn-v)/(np.linalg.norm(vn-v)+1e-15)
            bisect = e1 + e2
            bisect /= (np.linalg.norm(bisect) + 1e-15)
            ts = v + bisect * (float(width) / (np.sqrt(max(0.0, 1.0 - np.dot(bisect, e1)**2)) + 1e-15))
            t_surf.append(ts)
            t_bot.append(ts + ortho_ext)
        for i in range(flen):
            v0, v1, ts0, ts1, tb0, tb1 = vertices[vi[i]], vertices[vi[(i+1)%flen]], t_surf[i], t_surf[(i+1)%flen], t_bot[i], t_bot[(i+1)%flen]
            vb0, vb1 = v0 + global_miters[vi[i]], v1 + global_miters[vi[(i+1)%flen]]
            c_t, c_s, c_b = ([0.9, 0.1, 0.1], [0.9, 0.9, 0.1], [0.1, 0.1, 0.9]) if alt_mode else ([cols[fi]]*3)
            raw_quads.extend([
                (v0, v1, ts1, ts0, "F1", c_t),
                (v0, v1, vb1, vb0, "F2", c_s),
                (ts1, ts0, tb0, tb1, "F2", c_s),
                (v0, ts0, tb0, vb0, "F4", cols[fi]),
                (v1, ts1, tb1, vb1, "F4", cols[fi]),
                (vb0, vb1, tb1, tb0, "F3", c_b)])
    uv, vm, inv_tol = [], {}, 1.0 / 1e-10
    def get_idx(v):
        key = tuple((v * inv_tol).astype(np.int64))
        if key not in vm:
            vm[key] = len(uv)
            uv.append(v)
        return vm[key]
    dprint(f"Processing {len(raw_quads)} raw quads...")
    temp_faces, fc = [], {}
    for i, (*pts, ft, col) in enumerate(raw_quads):
        idxs = tuple(get_idx(p) for p in pts)
        if len(set(idxs)) < 3:
            continue
        s_idxs = tuple(sorted(idxs))
        fc[s_idxs] = fc.get(s_idxs, 0) + 1
        temp_faces.append((idxs, ft, col, s_idxs))
        if i % 10000 == 0 and i > 0:
            dprint(f"  raw_quads: {i}...")
    boundary_faces = [(idxs, col) for idxs, ft, col, s_idxs in temp_faces if not ((ft in ["F4", "F2"]) and fc[s_idxs] > 1)]
    final_v_np = np.array(uv, 'f8')
    manifold_faces = ensure_consistent_normals(final_v_np, boundary_faces, label="Leonardo")
    if get_signed_volume(final_v_np, [f[0] for f in manifold_faces], label="Leonardo") < 0:
        dprint("Flipping whole model (negative volume)...")
        manifold_faces = [(f[::-1], col) for f, col in manifold_faces]
    ov, oc, on = [], [], []
    dprint("Generating output vertex buffers...")
    for i, (idxs, col) in enumerate(manifold_faces):
        pts = [final_v_np[idx] for idx in idxs]
        n = np.cross(pts[1]-pts[0], pts[2]-pts[0])
        n /= (np.linalg.norm(n) + 1e-15)
        for idx in [0, 1, 2, 0, 2, 3]:
            ov.append(pts[idx])
            oc.append(col)
            on.append(n)
        if i % 5000 == 0 and i > 0:
            dprint(f"  output buffer: {i} faces...")
    return final_v_np, manifold_faces, np.array(ov, 'f4'), np.array(oc, 'f4'), np.array(on, 'f4')

# =========================================================
# VIEWER
# =========================================================

class Viewer:

    def __init__(self):
        self.model_orientation = Matrix44.identity(dtype='float32')
        if os.name == 'nt':
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("McNeill.Leonardo.Solver.v7.0")
        glfw.init()
        title = "Leonardo Solver v7.0"
        if DEBUG_MODE:
            title += " [DEBUG]"
        self.win = glfw.create_window(1100, 800, title, None, None)
        glfw.make_context_current(self.win)
        if os.name == 'nt':
            try:
                hwnd = glfw.get_win32_window(self.win)
                source_dir = getattr(self, 'source_dir', '.')
                base_filename = getattr(self, 'base_filename', 'leonardo')
                ico_path = os.path.join(source_dir, f"{base_filename}.ico")
                if os.path.exists(ico_path):
                    hicon = ctypes.windll.user32.LoadImageW(0, ico_path, 1, 0, 0, 0x00000010)
                    if hicon:
                        ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 0, hicon)
                        ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 1, hicon)
            except (OSError, AttributeError) as e:
                dprint(f"Icon Load Failed: {e}")
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.prog = self.ctx.program(
            vertex_shader="""#version 330
            in vec3 in_pos, in_col, in_norm; uniform mat4 model, view, proj; uniform mat3 normal_matrix;
            out vec3 v_col, v_norm;
            void main() { v_col = in_col; v_norm = normal_matrix * in_norm; gl_Position = proj * view * model * vec4(in_pos, 1.0); }""",
            fragment_shader="""#version 330
            in vec3 v_col, v_norm; out vec4 f_color; uniform bool is_line; uniform float u_alpha;
            void main() {
                if (is_line) f_color = vec4(0,0,0,u_alpha);
                else { vec3 N = normalize(v_norm); float d = max(dot(N, normalize(vec3(0.5, 0.8, 0.5))), 0.0); f_color = vec4(v_col * (0.35 + 0.65 * d), u_alpha); }
            }"""
        )
        imgui.create_context()
        imgui.backends.glfw_init_for_opengl(ctypes.cast(self.win, ctypes.c_void_p).value, True)
        imgui.backends.opengl3_init("#version 330")
        self.base_v, self.faces, self.cols, self.vbo, self.vao, self.line_vao = None, None, None, None, None, None
        self.src_vao, self.show_source, self.src_alpha = None, False, 0.3
        self.frame_w, self.frame_d, self.alt_mode = 0.05, 0.03, False
        self.no_overlap, self.f2min = False, 0.0
        self.yaw, self.pitch, self.zoom = 0.0, 0.0, 3.5
        self.drag, self.last_mouse, self.dirty = False, (0, 0), False
        self.status_msg, self.status_time = "", 0
        self.pending_load, self.warn_crossed, self.warn_hemi, self.warn_zero, self.warn_repeat, self.cli_error = None, False, False, False, False, None
        self.trigger_warning = False
        glfw.set_drop_callback(self.win, lambda w, p: self.load(p[0]))
        glfw.set_scroll_callback(self.win, lambda w, x, y: self.on_scroll(y))
        
        initial_file = None
        if len(sys.argv) > 1:
            arg = sys.argv[1]
            if arg and arg != "%1":
                initial_file = arg
        self.load(initial_file)

    def on_scroll(self, y):
        if not imgui.get_io().want_capture_mouse:
            self.zoom = max(0.5, min(10.0, self.zoom - y * 0.2))

    def load(self, path):
        res = load_poly_data(path)
        if res is None:
            if path:
                self.cli_error = f"Failed to parse model:\n{path}\n\nReverting to default."
            v, f, c = get_truncated_icosahedron()
            path = None
        else:
            v, f, c = res
        v -= np.mean(v, axis=0)
        v /= np.max(np.linalg.norm(v, axis=1))
        is_c, is_h, is_z, is_r, TOL = False, False, False, False, 1e-7
        for face in f:
            if len(set(face)) != len(face):
                is_r = True
            fpts = v[np.array(face, dtype=int)]
            norm, area = calculate_face_normal_and_area(fpts)
            if area < TOL:
                is_z = True
            if not is_c and is_crossed_or_concave(fpts, norm):
                is_c = True
            if not is_h and np.all(np.abs(np.mean(fpts, axis=0)) < TOL):
                is_h = True
        if any([is_c, is_h, is_z, is_r]):
            print(f"Warnings found! Crossed:{is_c} Hemi:{is_h} Zero:{is_z} RepeatV:{is_r}")
            sys.stdout.flush()
        if is_c or is_h or is_z or is_r:
            self.pending_load, self.trigger_warning = (v, f, c, path), True
            self.warn_crossed, self.warn_hemi, self.warn_zero, self.warn_repeat = is_c, is_h, is_z, is_r
            return
        self.apply_load(v, f, c, path)

    def apply_load(self, v, f, c, path):
        v, f = merge_vertices(v, f)
        f = [fi[0] for fi in ensure_consistent_normals(v, [[face, None] for face in f if len(face) >= 3], label="ApplyLoad")]
        self.base_v, self.faces, self.cols, self.dirty = v, f, c, True
        self.source_dir = os.path.dirname(path) if path else "."
        self.base_filename = os.path.splitext(os.path.basename(path))[0] if path else "truncated_icosahedron"
        sv, sc, sn = [], [], []
        for fi, face in enumerate(f):
            v0 = v[face[0]]
            n = np.cross(v[face[1]]-v0, v[face[2]]-v0)
            n /= (np.linalg.norm(n)+1e-15)
            for i in range(1, len(face)-1):
                for idx in [0, i, i+1]:
                    sv.append(v[face[idx]])
                    sc.append(c[fi])
                    sn.append(n)
        if self.src_vao:
            self.src_vao.release()
        if sv:
            self.src_vao = self.ctx.vertex_array(self.prog, [(self.ctx.buffer(np.hstack([np.array(sv, 'f4'), np.array(sc, 'f4'), np.array(sn, 'f4')]).astype('f4').tobytes()), "3f 3f 3f", "in_pos", "in_col", "in_norm")])
        self.calc_f2min()

    def calc_f2min(self):
        edge_to_f, face_norms = {}, []
        for fi, face in enumerate(self.faces):
            for i in range(len(face)):
                edge_to_f.setdefault(tuple(sorted((face[i], face[(i+1)%len(face)]))), []).append(fi)
            v0, v1, v2 = self.base_v[face[0]], self.base_v[face[1]], self.base_v[face[2]]
            face_norms.append(np.cross(v1-v0, v2-v0) / (np.linalg.norm(np.cross(v1-v0, v2-v0)) + 1e-15))
        def get_ts(fi, vi):
            face = self.faces[fi]
            v_idx = face.index(vi)
            v = self.base_v[face[v_idx]]
            vp, vn = self.base_v[face[v_idx-1]], self.base_v[face[(v_idx+1)%len(face)]]
            e1, e2 = (vp-v)/(np.linalg.norm(vp-v)+1e-15), (vn-v)/(np.linalg.norm(vn-v)+1e-15)
            bisect = (e1 + e2)
            bisect /= (np.linalg.norm(bisect) + 1e-15)
            return v + bisect * (float(self.frame_w) / (np.sqrt(max(0.0, 1.0 - np.dot(bisect, e1)**2)) + 1e-15))
        ds = []
        for edge, f_idxs in edge_to_f.items():
            if len(f_idxs) == 2:
                fi1, fi2 = f_idxs
                n1, n2 = face_norms[fi1], face_norms[fi2]
                div = np.sum((n1 - n2)**2)
                if div > 1e-12:
                    for v_idx in edge:
                        ds.append(np.dot(get_ts(fi1, v_idx) - get_ts(fi2, v_idx), n1 - n2) / div)
        self.f2min = min(ds) if ds else 0.0

    def update_mesh(self):
        if self.base_v is None:
            return
        self.ex_v, self.ex_f, v, c, n = build_leonardo_mesh(self.base_v, self.faces, self.cols, self.frame_w, self.frame_d, self.alt_mode)
        if self.vbo:
            self.vbo.release()
        self.vbo = self.ctx.buffer(np.hstack([v, c, n]).astype('f4').tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "3f 3f 3f", "in_pos", "in_col", "in_norm")])
        lns = []
        for idxs, _ in self.ex_f:
            for i in range(len(idxs)):
                lns.extend(self.ex_v[idxs[i]].astype('f4'))
                lns.extend(self.ex_v[idxs[(i+1)%len(idxs)]].astype('f4'))
        if self.line_vao:
            self.line_vao.release()
        self.line_vao = self.ctx.vertex_array(self.prog, [(self.ctx.buffer(np.array(lns, 'f4').tobytes()), "3f", "in_pos")], skip_errors=True)
        self.dirty = False

    def render(self):
        w, h = glfw.get_framebuffer_size(self.win)
        if w <= 0 or h <= 0:
            return
        if self.dirty:
            self.update_mesh()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.1, 0.1, 0.1)
        
        # --- FIXED VIEW-SPACE TRANSFORMATION PIPELINE ---
        if not imgui.get_io().want_capture_mouse:
            if glfw.get_mouse_button(self.win, 0) or glfw.get_mouse_button(self.win, 1):
                cx, cy = glfw.get_cursor_pos(self.win)
                if self.drag:
                    dx, dy = cx - self.last_mouse[0], cy - self.last_mouse[1]
                    if imgui.get_io().key_ctrl: 
                        self.zoom = max(0.5, min(10.0, self.zoom + dy * 0.01))
                    elif glfw.get_mouse_button(self.win, 0):
                        delta_yaw = Matrix44.from_y_rotation(-dx * 0.01)
                        delta_pitch = Matrix44.from_x_rotation(-dy * 0.01)
                        self.model_orientation = self.model_orientation @ delta_yaw @ delta_pitch
                        
                self.last_mouse, self.drag = (cx, cy), True
            else:
                self.drag = False

        model = self.model_orientation
        view = Matrix44.look_at((0, 0, self.zoom), (0, 0, 0), (0, 1, 0))
        proj = Matrix44.perspective_projection(45, w/h, 0.1, 100)
        
        for name, val in [("model", model), ("view", view), ("proj", proj)]: 
            self.prog[name].write(val.astype('f4'))
            
        self.prog["normal_matrix"].write(np.ascontiguousarray(np.linalg.inv(np.array(model))[:3, :3].T.astype('f4')))
        
        if self.vao:
            self.prog["u_alpha"].value, self.prog["is_line"].value = 1.0, False
            self.vao.render()
        if self.show_source and self.src_vao:
            self.ctx.polygon_offset = 1.0, 1.0
            self.prog["u_alpha"].value, self.prog["is_line"].value = self.src_alpha, False
            self.src_vao.render()
            self.ctx.polygon_offset = 0.0, 0.0
        if self.line_vao:
            self.prog["u_alpha"].value, self.prog["is_line"].value = 1.0, True
            self.line_vao.render(moderngl.LINES)

        imgui.backends.opengl3_new_frame()
        imgui.backends.glfw_new_frame()
        imgui.new_frame()

        if self.trigger_warning: 
            imgui.set_next_window_size((500, 500))  
            imgui.open_popup("Geometry Warning") 
            self.trigger_warning = False
        if imgui.begin_popup_modal("Geometry Warning", True)[0]:
            msg = "WARNING: Quality issues found:\n"
            if self.warn_crossed:
                msg += " - Crossed (self-intersecting/non-convex) faces.\n"
            if self.warn_hemi:
                msg += " - Hemihedral faces (origin-centered).\n"
            if self.warn_zero:
                msg += " - Zero-area (degenerate) faces.\n"
            if self.warn_repeat:
                msg += " - Faces with repeated vertex indices.\n"
            msg += "\nOrientation and consistency checks may hang or fail. Proceed?"
            imgui.text_wrapped(msg)
            imgui.separator()
            if imgui.button("Proceed", (120, 0)):
                self.apply_load(*self.pending_load)
                self.pending_load = None
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel", (120, 0)):
                self.pending_load = None
                imgui.close_current_popup()
            imgui.end_popup()
            
        if self.cli_error:
            imgui.open_popup("File Error")
        if imgui.begin_popup_modal("File Error", True)[0]:
            imgui.text_wrapped(self.cli_error)
            if imgui.button("OK", (-1, 0)):
                self.cli_error = None
                imgui.close_current_popup()
            imgui.end_popup()
        imgui.begin("Control Panel", flags=imgui.WindowFlags_.always_auto_resize.value)
        u1, self.frame_w = imgui.slider_float("Width (W)", self.frame_w, 0.001, 0.3)
        if u1:
            self.frame_w = round(self.frame_w, 3)
            self.calc_f2min()
            self.dirty = True
        max_d = 0.3
        if self.no_overlap:
            max_d = max(0.001, min(0.3, self.f2min))
            self.frame_d = min(self.frame_d, max_d)
        u2, self.frame_d = imgui.slider_float("Depth (D)", self.frame_d, 0.001, max_d)
        if u2:
            if abs(self.frame_d - self.f2min) >= 0.001:
                self.frame_d = round(self.frame_d, 3)
            self.dirty = True
        u3, self.alt_mode = imgui.checkbox("Alt Colour Mode", self.alt_mode)
        u4, self.no_overlap = imgui.checkbox("No Overlap", self.no_overlap)
        imgui.same_line()
        imgui.text(f"Max: {self.f2min:.4f}")
        imgui.separator()
        _, self.show_source = imgui.checkbox("Show Source", self.show_source)
        if self.show_source:
            _, self.src_alpha = imgui.slider_float("Alpha", self.src_alpha, 0.0, 1.0)
        if u3 or u4:
            self.dirty = True
        if imgui.button("Export OFF", (-1, 0)): 
            try:
                p = os.path.join(self.source_dir, f"{self.base_filename}_leonardo.off")
                with open(p, "w") as f:
                    f.write(f"OFF\n{len(self.ex_v)} {len(self.ex_f)} 0\n")
                    f.writelines(f"{v[0]:.16f} {v[1]:.16f} {v[2]:.16f}\n" for v in self.ex_v)
                    f.writelines(f"{len(idxs)} {' '.join(map(str, idxs))} {int(col[0]*255)} {int(col[1]*255)} {int(col[2]*255)}\n" for idxs, col in self.ex_f)
                self.status_msg, self.status_time = f"Saved: {os.path.basename(p)}", time.time()
            except OSError as e:
                self.status_msg, self.status_time = f"Error: {e}", time.time()
        if imgui.button("HELP", (-1, 0)):
            imgui.open_popup("HelpWindow")
        imgui.set_next_window_size((500, 0))
        if imgui.begin_popup_modal("HelpWindow", True)[0]:
            imgui.text_wrapped(HELP_TEXT)
            imgui.separator()
            if imgui.button("OK", (-1, 0)):
                imgui.close_current_popup()
            imgui.end_popup()
        imgui.end()
        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())
        glfw.swap_buffers(self.win)
        
    def run(self):
        while not glfw.window_should_close(self.win):
            glfw.poll_events()
            self.render()
        imgui.backends.opengl3_shutdown()
        imgui.backends.glfw_shutdown()
        imgui.destroy_context()
        glfw.terminate()

if __name__ == "__main__":
    Viewer().run()