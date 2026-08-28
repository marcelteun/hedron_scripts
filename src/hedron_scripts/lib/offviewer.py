# Version 5.56
# Addons: pygame, moderngl, numpy
# offviewer.py
"""
HELP TEXT
When you open a grid window the sort mode will be greyed out while it calculates
the metrics for the off files. Once it has done this and you select a sort order,
duplicate values of the sort parameter are highlighted in pink. The delete
duplicates button will keep the oldest file with any one value. Float parameters
have a small tolerance built in.

- Auto rotate is fun but wait until the sort button blacks up or you'll slow it down.
- + (or =) and - will zoom the cells on the grid screen.
- Any polyhedron can be rotated by left dragging with the mouse or by using the arrow keys.
- Double click on an image to get to the enlarged screen.
  - The left and right arrows are previous/next file.
  - 'f' on the enlarged screen will cycle through the face types one by one.
  - 'v' cycles through the vertex types.
  - 'e' does some edge options (also on the grid view).
  - 'c' cycles through the parts of a compound.
  - 'x' to return to the original view.
- The views also have some right click functions.
  - delete, rename, view source, configurable external programs via 'external_tools.json'.
- The offcheck.py file is also a standalone checker. If you change the metrics
  that this is calculating they should flow through into offviewer automatically.
- Files offcheck.py and facetings_math.py are required to be in the same folder as this file.
Change log (debugging not included):
140826 - The path and 'search' at the top of the grid screen now work.  Search accepts * and ?
       - Symmetry is enabled in the offcheck link.  See variable below to switch it off if required.
       - 'f' mode on the enlarged view shows coplanar faces to the focus face as translucent.  Use ALPHA to control.
       - 'f' now cycles through face types rather than all faces
       - 'e' now also works as an edge on/off switch in grid view
150826 - added '-' to search box to mean the opposite of the rest of the input eg '-*a*' anything that does not contain an 'a'.
       - 'v' now cycles through vertex types rather than all vertices
190826 - 'c' added to enlarged mode to cycle through parts of a compound
       - Refresh added to Grid view.  Subfolder sorting and scrolling added.
270826 - external_tools.json added so right click is configurable and not machine specific
       - Caching added to speed up display
       - Polling of focus folder added to detect new files
"""

import csv
import fnmatch
import hashlib
import importlib.util
import json
import math
import os
import pickle
import re
import shutil
import sys
import threading
import tkinter as tk
from collections import defaultdict
from tkinter import messagebox, simpledialog

import moderngl
import numpy as np
import pygame
from pygame.locals import (
    KEYDOWN,
    K_c,
    K_DOWN,
    K_e,
    K_EQUALS,
    K_ESCAPE,
    K_f,
    K_KP_MINUS,
    K_KP_PLUS,
    K_LEFT,
    K_MINUS,
    K_PLUS,
    K_RIGHT,
    K_UP,
    K_v,
    K_x,
    MOUSEBUTTONDOWN,
    MOUSEBUTTONUP,
    MOUSEMOTION,
    QUIT,
    VIDEORESIZE,
)

# Configuration
AUTO_ROTATION_SPEED = -0.0002
SORT_TOLERANCE = 0.0001
RUN_SYMMETRY_CALCULATION = (
    False  # Set to True to enable symmetry calculation in offcheck
)
ALPHA = 0.2  # Transparency value for coplanar faces in Enlarged view
CACHE_THRESHOLD = 10
FOLDER_POLL_INTERVAL_MS = 1000  # Poll interval to detect new/deleted files

# Thread-safe signals for background analysis
bg_csv_ready = [False]
bg_csv_path = [None]
bg_run_id = [0]

# Package setup and dynamic imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    from hedron_scripts.lib import offcheck
except ModuleNotFoundError:
    _spec = importlib.util.spec_from_file_location("offcheck", "offcheck.py")
    if _spec is None:
        print("Cannot import offcheck, checker unavailable")
        offcheck = None
    else:
        offcheck = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(offcheck)


def format_sort_value(val):
    if val is None or val == "":
        return ""
    try:
        f = float(val)
        if f.is_integer():
            return str(int(f))
        return f"{f:.5g}"
    except (ValueError, TypeError):
        return str(val)


def parse_color(parts, default_color=None):
    if default_color is None:
        default_color = [178, 204, 230, 255]
    if len(parts) == 0:
        return default_color
    try:
        color = []
        for x in parts[:4]:
            val = float(x)
            color.append(val)
        if len(color) < 4:
            color.append(255.0)
        if any(c > 1.0 for c in color[:3]):
            color = [c / 255.0 for c in color]
        return color
    except Exception:
        return default_color


def seg_intersect(A, B, C, D):
    denom = (B[0] - A[0]) * (D[1] - C[1]) - (B[1] - A[1]) * (D[0] - C[0])
    if abs(denom) < 1e-9:
        return None
    t = ((C[0] - A[0]) * (D[1] - C[1]) - (C[1] - A[1]) * (D[0] - C[0])) / denom
    u = ((C[0] - A[0]) * (B[1] - A[1]) - (C[1] - A[1]) * (B[0] - A[0])) / denom
    if 1e-9 < t < 1.0 - 1e-9 and 1e-9 < u < 1.0 - 1e-9:
        return t, u
    return None


def triangulate_simple_polygon(indices, verts_2d):
    n = len(indices)
    if n < 3:
        return []
    if n == 3:
        return [indices]

    poly = list(indices)
    triangles = []

    area = 0.0
    for i in range(len(poly)):
        p1 = verts_2d[poly[i]]
        p2 = verts_2d[poly[(i + 1) % len(poly)]]
        area += p1[0] * p2[1] - p2[0] * p1[1]

    ccw = area >= 0

    def is_convex(p_prev, p_curr, p_next):
        cross = (p_curr[0] - p_prev[0]) * (p_next[1] - p_curr[1]) - (
            p_curr[1] - p_prev[1]
        ) * (p_next[0] - p_curr[0])
        return cross > 1e-9 if ccw else cross < -1e-9

    def is_point_in_tri(p, a, b, c):
        cp1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        cp2 = (c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0])
        cp3 = (a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0])
        if ccw:
            return cp1 >= -1e-9 and cp2 >= -1e-9 and cp3 >= -1e-9
        return cp1 <= 1e-9 and cp2 <= 1e-9 and cp3 <= 1e-9

    limit = len(poly) * len(poly) + 100
    while len(poly) > 3 and limit > 0:
        limit -= 1
        ear_found = False
        m = len(poly)
        for i in range(m):
            prev_idx = poly[(i - 1) % m]
            curr_idx = poly[i]
            next_idx = poly[(i + 1) % m]

            p_prev = verts_2d[prev_idx]
            p_curr = verts_2d[curr_idx]
            p_next = verts_2d[next_idx]

            if is_convex(p_prev, p_curr, p_next):
                any_inside = False
                for other_idx in poly:
                    if other_idx in (prev_idx, curr_idx, next_idx):
                        continue
                    p_other = verts_2d[other_idx]
                    if (
                        (p_other[0] == p_prev[0] and p_other[1] == p_prev[1])
                        or (p_other[0] == p_curr[0] and p_other[1] == p_curr[1])
                        or (p_other[0] == p_next[0] and p_other[1] == p_next[1])
                    ):
                        continue
                    if is_point_in_tri(p_other, p_prev, p_curr, p_next):
                        any_inside = True
                        break
                if not any_inside:
                    triangles.append([prev_idx, curr_idx, next_idx])
                    poly.pop(i)
                    ear_found = True
                    break
        if not ear_found:
            triangles.append([poly[0], poly[1], poly[2]])
            poly.pop(1)

    if len(poly) == 3:
        triangles.append(list(poly))
    return triangles


def load_external_tools():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "external_tools.json")

    loaded_tools = []
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cleaned_lines = []
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("//") or stripped.startswith("#"):
                        continue
                    cleaned_lines.append(line)
                cleaned_content = "".join(cleaned_lines)
                loaded_tools = json.loads(cleaned_content)
        except Exception:
            loaded_tools = []

    if not loaded_tools:
        loaded_tools = [
            {"prompt": "View Source", "program": "notepad.exe"},
            {"prompt": "Rename", "action": "rename"},
            {"prompt": "Delete", "action": "delete"},
        ]

    valid_tools = []
    for tool in loaded_tools:
        prog = tool.get("program", "")
        action = tool.get("action", "")
        if (
            action
            or os.path.exists(prog)
            or (
                prog
                and (
                    shutil.which(prog)
                    or prog.lower() in ("py", "python", "pythonw", "python3")
                )
            )
        ):
            valid_tools.append(tool)

    return valid_tools


def watch_edit_tool(proc, filepath, state, ctx, prog_3d, viewers):
    proc.wait()

    prog_dir = os.path.dirname(os.path.abspath(__file__))
    folder_hash = hashlib.md5(
        os.path.abspath(state["current_folder"]).encode("utf-8")
    ).hexdigest()
    cache_dir = os.path.join(prog_dir, ".cache", folder_hash)
    metrics_cache_path = os.path.join(cache_dir, "metrics.json")
    geo_cache_path = os.path.join(cache_dir, os.path.basename(filepath) + ".bin")

    if os.path.exists(geo_cache_path):
        try:
            os.remove(geo_cache_path)
        except Exception:
            pass

    if os.path.exists(metrics_cache_path):
        try:
            with open(metrics_cache_path, "r", encoding="utf-8") as cf:
                cache_data = json.load(cf)
            fname = os.path.basename(filepath)
            if "files" in cache_data and fname in cache_data["files"]:
                del cache_data["files"][fname]
                with open(metrics_cache_path, "w", encoding="utf-8") as cf:
                    json.dump(cache_data, cf, indent=4)
        except Exception as e:
            print(f"Failed to clear cache: {e}")

    state["pending_reload"] = filepath
    state["needs_redraw"] = 2


def load_off(filepath):
    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]

        if not lines:
            print(f"Error: {filepath} is empty or contains only comments.")
            return None, None, None, None, None, None

        header = lines[0]
        is_coff = header.startswith("COFF")

        start_idx = 1
        if header in ["OFF", "COFF"]:
            parts = lines[1].split()
            start_idx = 2
        elif header.startswith("OFF") or header.startswith("COFF"):
            parts = header[4:].split() if is_coff else header[3:].split()
        else:
            print(f"Error: {filepath} has invalid OFF header line: '{header}'")
            return None, None, None, None, None, None

        num_verts = int(parts[0])
        num_faces = int(parts[1])

        vertex_lines = lines[start_idx : start_idx + num_verts]
        try:
            vertices_data = np.fromstring("\n".join(vertex_lines), sep=" ")
            num_cols = len(vertices_data) // num_verts
            vertices_data = vertices_data.reshape((num_verts, num_cols))
            vertices = vertices_data[:, :3].tolist()
            if is_coff and num_cols >= 6:
                vert_colors = [
                    parse_color(vertices_data[i, 3:7].tolist())
                    for i in range(num_verts)
                ]
            else:
                vert_colors = [[0.7, 0.8, 0.9, 1.0]] * num_verts
        except Exception:
            vertices = []
            vert_colors = []
            for i in range(num_verts):
                line_parts = lines[start_idx + i].split()
                vertices.append([float(x) for x in line_parts[:3]])
                if is_coff and len(line_parts) >= 6:
                    vert_colors.append(parse_color(line_parts[3:7]))
                else:
                    vert_colors.append([0.7, 0.8, 0.9, 1.0])

        raw_faces_data = []
        max_val = -1
        min_val = 99999999

        for i in range(num_faces):
            line_parts = lines[start_idx + num_verts + i].split()
            if not line_parts:
                continue
            n_v = int(line_parts[0])
            face_verts = [int(x) for x in line_parts[1 : 1 + n_v]]
            if face_verts:
                max_val = max(max_val, max(face_verts))
                min_val = min(min_val, min(face_verts))
            raw_faces_data.append((n_v, face_verts, line_parts[1 + n_v :]))

        if max_val >= num_verts and min_val >= 1:
            raw_faces_data = [
                (n_v, [v - 1 for v in face_verts], color_parts)
                for n_v, face_verts, color_parts in raw_faces_data
            ]

        orig_faces = [item[1] for item in raw_faces_data]

        faces = []
        face_colors = []
        triangle_to_face_map = []
        global_inter_map = {}

        for i, (n_v, face_verts, color_parts) in enumerate(raw_faces_data):
            f_color = (
                parse_color(color_parts, default_color=None) if color_parts else None
            )

            if n_v < 3:
                continue

            pts_3d = [np.array(vertices[idx], dtype="f4") for idx in face_verts]

            N = np.zeros(3, dtype="f4")
            for j in range(1, n_v - 1):
                for k in range(j + 1, n_v):
                    cross_prod = np.cross(pts_3d[j] - pts_3d[0], pts_3d[k] - pts_3d[0])
                    if np.linalg.norm(cross_prod) > 1e-6:
                        N = cross_prod
                        break
                if np.linalg.norm(N) > 1e-6:
                    break

            norm_N = np.linalg.norm(N)
            if norm_N < 1e-9:
                for j in range(1, n_v - 1):
                    faces.append([face_verts[0], face_verts[j], face_verts[j + 1]])
                    face_colors.append(f_color)
                    triangle_to_face_map.append(i)
                continue
            N = N / norm_N

            U = pts_3d[1] - pts_3d[0]
            U = U - np.dot(U, N) * N
            norm_U = np.linalg.norm(U)
            if norm_U < 1e-9:
                for idx_u in range(2, n_v):
                    U = pts_3d[idx_u] - pts_3d[0]
                    U = U - np.dot(U, N) * N
                    norm_U = np.linalg.norm(U)
                    if norm_U >= 1e-9:
                        break

            if norm_U < 1e-9:
                if abs(N[0]) > 0.9:
                    U = np.array([0.0, 1.0, 0.0], dtype="f4")
                else:
                    U = np.array([1.0, 0.0, 0.0], dtype="f4")
                U = U - np.dot(U, N) * N
                U = U / np.linalg.norm(U)
            else:
                U = U / norm_U

            V = np.cross(N, U)

            pts_2d = []
            for p in pts_3d:
                pts_2d.append(
                    [
                        float(np.dot(p - pts_3d[0], U)),
                        float(np.dot(p - pts_3d[0], V)),
                    ]
                )

            has_intersection = False
            edge_intersections = {k: [] for k in range(n_v)}

            local_vertices = []
            for idx in range(n_v):
                local_vertices.append(
                    {
                        "2d": np.array(pts_2d[idx]),
                        "3d": pts_3d[idx].tolist(),
                        "color": vert_colors[face_verts[idx]],
                    }
                )

            for k in range(n_v):
                for m in range(k + 2, n_v):
                    if k == 0 and m == n_v - 1:
                        continue
                    A, B = pts_2d[k], pts_2d[(k + 1) % n_v]
                    C, D = pts_2d[m], pts_2d[(m + 1) % n_v]
                    res = seg_intersect(A, B, C, D)
                    if res is not None:
                        t, u = res
                        has_intersection = True

                        p3d_orig = (1 - t) * pts_3d[k] + t * pts_3d[(k + 1) % n_v]
                        p2d_orig = (1 - t) * np.array(A) + t * np.array(B)
                        color_A = np.array(vert_colors[face_verts[k]])
                        color_B = np.array(vert_colors[face_verts[(k + 1) % n_v]])
                        p_color = ((1 - t) * color_A + t * color_B).tolist()

                        v_idx = -1
                        for idx, v in enumerate(local_vertices):
                            if np.linalg.norm(v["2d"] - p2d_orig) < 1e-5:
                                v_idx = idx
                                break
                        if v_idx == -1:
                            local_vertices.append(
                                {
                                    "2d": p2d_orig,
                                    "3d": p3d_orig.tolist(),
                                    "color": p_color,
                                }
                            )
                            v_idx = len(local_vertices) - 1

                        edge_intersections[k].append((t, v_idx))
                        edge_intersections[m].append((u, v_idx))

            if not has_intersection:
                local_indices = list(range(n_v))
                local_tris = triangulate_simple_polygon(local_indices, pts_2d)
                for tri in local_tris:
                    faces.append(
                        [
                            face_verts[tri[0]],
                            face_verts[tri[1]],
                            face_verts[tri[2]],
                        ]
                    )
                    face_colors.append(f_color)
                    triangle_to_face_map.append(i)
                continue

            segments = []
            for k in range(n_v):
                v_start = k
                v_end = (k + 1) % n_v
                inters = sorted(edge_intersections[k], key=lambda x: x[0])

                curr_idx = v_start
                for _, inter_idx in inters:
                    segments.append((curr_idx, inter_idx))
                    curr_idx = inter_idx
                segments.append((curr_idx, v_end))

            cleaned_segments = []
            seen_segments = set()
            for u, v in segments:
                if u == v:
                    continue
                seg_key = tuple(sorted((u, v)))
                if seg_key not in seen_segments:
                    seen_segments.add(seg_key)
                    cleaned_segments.append((u, v))

            adj = {idx: [] for idx in range(len(local_vertices))}
            half_edges = []
            half_edge_lookup = {}

            for u, v in cleaned_segments:
                half_edges.append({"from": u, "to": v, "visited": False})
                half_edge_lookup[(u, v)] = len(half_edges) - 1
                adj[u].append(v)

                half_edges.append({"from": v, "to": u, "visited": False})
                half_edge_lookup[(v, u)] = len(half_edges) - 1
                adj[v].append(u)

            sorted_adj = {}
            for u in range(len(local_vertices)):

                def get_angle(v, u_idx=u):
                    diff = local_vertices[v]["2d"] - local_vertices[u_idx]["2d"]
                    return math.atan2(diff[1], diff[0])

                sorted_adj[u] = sorted(adj[u], key=get_angle)

            loops = []
            for he_idx, he in enumerate(half_edges):
                if he["visited"]:
                    continue

                cycle = []
                curr_he_idx = he_idx

                while True:
                    curr_he = half_edges[curr_he_idx]
                    if curr_he["visited"]:
                        break
                    curr_he["visited"] = True
                    u = curr_he["from"]
                    v = curr_he["to"]
                    cycle.append(v)

                    v_neighbors = sorted_adj[v]
                    u_pos = v_neighbors.index(u)
                    next_neighbor = v_neighbors[(u_pos - 1) % len(v_neighbors)]
                    curr_he_idx = half_edge_lookup[(v, next_neighbor)]

                if len(cycle) >= 3:
                    loops.append(cycle)

            interior_loops = []
            for cycle in loops:
                area = 0.0
                n_c = len(cycle)
                for idx_c in range(n_c):
                    p1 = local_vertices[cycle[idx_c]]["2d"]
                    p2 = local_vertices[cycle[(idx_c + 1) % n_c]]["2d"]
                    area += p1[0] * p2[1] - p2[0] * p1[1]
                area = 0.5 * area
                if area > 1e-7:
                    interior_loops.append(cycle)

            coord_to_global = {}
            for idx in face_verts:
                p = vertices[idx]
                key = (round(p[0], 6), round(p[1], 6), round(p[2], 6))
                coord_to_global[key] = idx

            def get_global_index(p3d, p_col):
                key = (round(p3d[0], 6), round(p3d[1], 6), round(p3d[2], 6))
                if key in coord_to_global:
                    return coord_to_global[key]
                if key in global_inter_map:
                    return global_inter_map[key]
                g_idx = len(vertices)
                vertices.append(p3d)
                vert_colors.append(p_col)
                global_inter_map[key] = g_idx
                return g_idx

            for loop in interior_loops:
                loop_2d = [local_vertices[v_idx]["2d"] for v_idx in loop]
                local_indices = list(range(len(loop_2d)))
                local_tris = triangulate_simple_polygon(local_indices, loop_2d)
                for tri in local_tris:
                    g_tri = [
                        get_global_index(
                            local_vertices[loop[tri[0]]]["3d"],
                            local_vertices[loop[tri[0]]]["color"],
                        ),
                        get_global_index(
                            local_vertices[loop[tri[1]]]["3d"],
                            local_vertices[loop[tri[1]]]["color"],
                        ),
                        get_global_index(
                            local_vertices[loop[tri[2]]]["3d"],
                            local_vertices[loop[tri[2]]]["color"],
                        ),
                    ]
                    faces.append(g_tri)
                    face_colors.append(f_color)
                    triangle_to_face_map.append(i)

        if len(faces) == 0:
            print(f"Warning: {filepath} has 0 faces after triangulation.")

        verts_out = np.array(vertices, dtype="f4")
        faces_out = np.array(faces, dtype="i4")

        return (
            verts_out,
            faces_out,
            vert_colors,
            face_colors,
            triangle_to_face_map,
            orig_faces,
        )
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None, None, None, None, None


def create_perspective_matrix(fovy, aspect, near, far):
    scale = math.tan(math.radians(fovy) / 2)
    try:
        ty = 1 / scale
    except ZeroDivisionError:
        ty = 1.0
    tx = ty / aspect
    tz = -(far + near) / (far - near)
    tw = -2 * far * near / (far - near)
    return np.array(
        [[tx, 0, 0, 0], [0, ty, 0, 0], [0, 0, tz, -1], [0, 0, tw, 0]], dtype="f4"
    )


def create_lookat_matrix(eye, target, up):
    zaxis = eye - target
    zaxis_norm = np.linalg.norm(zaxis)
    zaxis = zaxis / (zaxis_norm if zaxis_norm != 0 else 1.0)
    xaxis = np.cross(up, zaxis)
    xaxis_norm = np.linalg.norm(xaxis)
    xaxis = xaxis / (xaxis_norm if xaxis_norm != 0 else 1.0)
    yaxis = np.cross(zaxis, xaxis)

    return np.array(
        [
            [xaxis[0], yaxis[0], zaxis[0], 0],
            [xaxis[1], yaxis[1], zaxis[1], 0],
            [xaxis[2], yaxis[2], zaxis[2], 0],
            [-np.dot(xaxis, eye), -np.dot(yaxis, eye), -np.dot(zaxis, eye), 1],
        ],
        dtype="f4",
    )


def create_sphere_mesh(radius=0.02, rings=8, sectors=8):
    verts = []
    faces = []
    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(sectors):
            theta = 2 * math.pi * s / sectors
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.cos(phi)
            z = radius * math.sin(phi) * math.sin(theta)
            verts.append([x, y, z])
    for r in range(rings):
        for s in range(sectors):
            r0 = r * sectors + s
            r1 = r * sectors + (s + 1) % sectors
            r2 = (r + 1) * sectors + s
            r3 = (r + 1) * sectors + (s + 1) % sectors
            faces.append([r0, r2, r1])
            faces.append([r1, r2, r3])
    return np.array(verts, dtype="f4"), np.array(faces, dtype="i4")


class ModelViewer:
    def __init__(self, filepath, ctx, program_3d, use_cache=False, cache_dir=None):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.ctx = ctx
        self.program = program_3d
        self.metrics = {}
        self.rot_mat = np.eye(3, dtype="f4")
        self.valid = False

        cache_file = (
            os.path.join(cache_dir, self.filename + ".bin")
            if use_cache and cache_dir
            else None
        )
        loaded = self._try_load_cache(filepath, cache_file) if cache_file else False

        if not loaded:
            if not self._build_from_off(filepath, cache_file):
                self.valid = False
                return

        self._finalize_topology()
        self.valid = True

    def _try_load_cache(self, filepath, cache_file):
        if not os.path.exists(cache_file):
            return False
        try:
            current_mtime = round(os.path.getmtime(filepath), 4)
            current_size = os.path.getsize(filepath)
            with open(cache_file, "rb") as f:
                pkg = pickle.load(f)
            if (
                abs(pkg.get("mtime", 0.0) - current_mtime) >= 1e-3
                or pkg.get("size") != current_size
            ):
                return False

            data = pkg["data"]
            self.verts = data["verts"]
            self.vbo_data = data["vbo_data"]
            self.orig_faces = data.get("orig_faces", [])
            self.num_original_faces = data["num_original_faces"]
            self.face_ranges = data["face_ranges"]
            self.face_planes = data["face_planes"]
            self.face_types = data["face_types"]
            self.face_type_keys = data["face_type_keys"]
            self.representative_faces = data["representative_faces"]
            self.components = data["components"]

            if self.orig_faces:
                self.face_boundary_edges = {
                    i: [(f[j], f[(j + 1) % len(f)]) for j in range(len(f))]
                    for i, f in enumerate(self.orig_faces)
                }
            else:
                self.face_boundary_edges = data.get("face_boundary_edges", {})
            self.all_edges = list(
                {
                    tuple(sorted(e))
                    for edges in self.face_boundary_edges.values()
                    for e in edges
                }
            )

            self._create_gpu_resources(data["component_data"])
            return True
        except Exception:
            return False

    def _build_from_off(self, filepath, cache_file=None):
        verts, faces, vert_colors, face_colors, tri_to_face, orig_faces = load_off(
            filepath
        )
        if verts is None or faces is None or len(faces) == 0 or len(verts) == 0:
            return False

        self.orig_faces = orig_faces if orig_faces else []

        centroid = np.mean(verts, axis=0)
        verts = verts - centroid
        max_span = np.max(np.linalg.norm(verts, axis=1))
        self.verts = verts / max_span if max_span > 0 else verts

        tri_verts = self.verts[faces]
        v0, v1, v2 = tri_verts[:, 0, :], tri_verts[:, 1, :], tri_verts[:, 2, :]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normals = normals / norms

        num_triangles = len(faces)
        vbo_array = np.zeros((num_triangles, 3, 10), dtype="f4")
        vbo_array[:, :, 0:3] = self.verts[faces]
        vbo_array[:, :, 3:7] = np.array(vert_colors, dtype="f4")[faces]
        vbo_array[:, :, 7:10] = normals[:, np.newaxis, :]
        for i, f_col in enumerate(face_colors):
            if f_col is not None:
                vbo_array[i, :, 3:7] = f_col
        self.vbo_data = vbo_array.ravel()

        self.face_ranges = {}
        curr_v = 0
        for i in range(len(faces)):
            f_idx = tri_to_face[i]
            if f_idx not in self.face_ranges:
                self.face_ranges[f_idx] = [curr_v, 0]
            self.face_ranges[f_idx][1] += 3
            curr_v += 3
        self.num_original_faces = len(self.face_ranges)

        self.face_planes = {}
        for f_idx, (start_v, _) in self.face_ranges.items():
            t_idx = start_v // 3
            self.face_planes[f_idx] = (
                normals[t_idx],
                np.dot(normals[t_idx], v0[t_idx]),
            )

        face_triangles = defaultdict(list)
        for i in range(len(faces)):
            face_triangles[tri_to_face[i]].append(faces[i])

        self.face_boundary_edges = {}
        if self.orig_faces:
            for f_idx, f_verts in enumerate(self.orig_faces):
                n_v = len(f_verts)
                self.face_boundary_edges[f_idx] = [
                    (f_verts[j], f_verts[(j + 1) % n_v]) for j in range(n_v)
                ]
        else:
            for f_idx, tris in face_triangles.items():
                ec = defaultdict(int)
                for tri in tris:
                    for u, v in [
                        (tri[0], tri[1]),
                        (tri[1], tri[2]),
                        (tri[2], tri[0]),
                    ]:
                        ec[tuple(sorted((u, v)))] += 1
                self.face_boundary_edges[f_idx] = [e for e, c in ec.items() if c == 1]
        self.all_edges = list(
            {
                tuple(sorted(e))
                for edges in self.face_boundary_edges.values()
                for e in edges
            }
        )

        self.face_types = defaultdict(list)
        for f_idx in range(self.num_original_faces):
            edge_lens = tuple(
                round(float(np.linalg.norm(self.verts[u] - self.verts[v])), 4)
                for u, v in self.face_boundary_edges[f_idx]
            )
            sorted_lens = tuple(sorted(edge_lens))
            area = sum(
                0.5
                * np.linalg.norm(
                    np.cross(
                        self.verts[t[1]] - self.verts[t[0]],
                        self.verts[t[2]] - self.verts[t[0]],
                    )
                )
                for t in face_triangles[f_idx]
            )
            gonality = (
                len(self.orig_faces[f_idx])
                if (self.orig_faces and f_idx < len(self.orig_faces))
                else len(self.face_boundary_edges[f_idx])
            )
            self.face_types[(gonality, sorted_lens, round(float(area), 4))].append(
                f_idx
            )

        self.face_type_keys = list(self.face_types.keys())
        self.representative_faces = [self.face_types[k][0] for k in self.face_type_keys]

        self._extract_components(
            faces,
            tri_to_face,
            face_colors,
            vert_colors,
            normals,
            num_triangles,
        )

        if cache_file:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                pkg = {
                    "mtime": round(os.path.getmtime(filepath), 4),
                    "size": os.path.getsize(filepath),
                    "data": {
                        "verts": self.verts,
                        "vbo_data": self.vbo_data,
                        "orig_faces": self.orig_faces,
                        "num_original_faces": self.num_original_faces,
                        "face_ranges": self.face_ranges,
                        "face_planes": self.face_planes,
                        "face_boundary_edges": self.face_boundary_edges,
                        "face_types": self.face_types,
                        "face_type_keys": self.face_type_keys,
                        "representative_faces": self.representative_faces,
                        "components": self.components,
                        "component_data": self._cached_comp_data,
                    },
                }
                with open(cache_file, "wb") as f:
                    pickle.dump(pkg, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f"Warning: Failed to save geometry cache: {e}")

        self._create_gpu_resources(self._cached_comp_data)
        return True

    def _extract_components(
        self,
        faces,
        tri_to_face,
        face_colors,
        vert_colors,
        normals,
        num_triangles,
    ):
        orig_components = []
        if self.orig_faces:
            e_to_f = defaultdict(list)
            for f_idx, f_verts in enumerate(self.orig_faces):
                n = len(f_verts)
                for i in range(n):
                    e_to_f[tuple(sorted((f_verts[i], f_verts[(i + 1) % n])))].append(
                        f_idx
                    )
            adj = defaultdict(list)
            for f_indices in e_to_f.values():
                for idx1 in range(len(f_indices)):
                    for idx2 in range(idx1 + 1, len(f_indices)):
                        adj[f_indices[idx1]].append(f_indices[idx2])
                        adj[f_indices[idx2]].append(f_indices[idx1])
            visited = [False] * len(self.orig_faces)
            for i in range(len(self.orig_faces)):
                if not visited[i]:
                    comp, q = [], [i]
                    visited[i] = True
                    while q:
                        curr = q.pop(0)
                        comp.append(curr)
                        for nbr in adj[curr]:
                            if not visited[nbr]:
                                visited[nbr] = True
                                q.append(nbr)
                    orig_components.append(comp)

        if orig_components:
            f_to_c = {
                f_idx: c_idx
                for c_idx, comp in enumerate(orig_components)
                for f_idx in comp
            }
            comp_tris = defaultdict(list)
            for t_idx in range(num_triangles):
                comp_tris[f_to_c.get(tri_to_face[t_idx], 0)].append(t_idx)
            self.components = [comp_tris[c] for c in sorted(comp_tris.keys())]
        else:
            e_to_t = defaultdict(list)
            for i, tri in enumerate(faces):
                for u, v in [
                    (tri[0], tri[1]),
                    (tri[1], tri[2]),
                    (tri[2], tri[0]),
                ]:
                    e_to_t[tuple(sorted((u, v)))].append(i)
            adj = defaultdict(list)
            for t_indices in e_to_t.values():
                for idx1 in range(len(t_indices)):
                    for idx2 in range(idx1 + 1, len(t_indices)):
                        adj[t_indices[idx1]].append(t_indices[idx2])
                        adj[t_indices[idx2]].append(t_indices[idx1])
            visited = [False] * num_triangles
            self.components = []
            for i in range(num_triangles):
                if not visited[i]:
                    comp, q = [], [i]
                    visited[i] = True
                    while q:
                        curr = q.pop(0)
                        comp.append(curr)
                        for nbr in adj[curr]:
                            if not visited[nbr]:
                                visited[nbr] = True
                                q.append(nbr)
                    self.components.append(comp)

        self._cached_comp_data = []
        for comp in self.components:
            n_t = len(comp)
            vbo_comp = np.zeros((n_t, 3, 10), dtype="f4")
            vbo_comp[:, :, 0:3] = self.verts[faces[comp]]
            vbo_comp[:, :, 3:7] = np.array(vert_colors, dtype="f4")[faces[comp]]
            vbo_comp[:, :, 7:10] = normals[comp, np.newaxis, :]
            for out_i, t_idx in enumerate(comp):
                if face_colors[t_idx] is not None:
                    vbo_comp[out_i, :, 3:7] = face_colors[t_idx]
            comp_edges = {
                e
                for t_idx in comp
                for e in self.face_boundary_edges.get(tri_to_face[t_idx], [])
            }
            comp_verts = {v for t_idx in comp for v in faces[t_idx]}
            self._cached_comp_data.append(
                {
                    "vbo_bytes": vbo_comp.ravel().tobytes(),
                    "edges": list(comp_edges),
                    "num_v": len(comp_verts),
                    "num_f": len(set(tri_to_face[t_idx] for t_idx in comp)),
                }
            )

    def _create_gpu_resources(self, comp_data):
        self.vbo = self.ctx.buffer(self.vbo_data.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, "3f 4f 3f", "in_position", "in_color", "in_normal")],
        )
        self.component_resources = []
        for c in comp_data:
            vbo_comp = self.ctx.buffer(c["vbo_bytes"])
            vao_comp = self.ctx.vertex_array(
                self.program,
                [(vbo_comp, "3f 4f 3f", "in_position", "in_color", "in_normal")],
            )
            self.component_resources.append(
                {
                    "vbo": vbo_comp,
                    "vao": vao_comp,
                    "edges": c["edges"],
                    "num_v": c["num_v"],
                    "num_f": c["num_f"],
                }
            )
        self.sphere_local_verts, self.sphere_faces = create_sphere_mesh(
            radius=0.015, rings=8, sectors=8
        )

    def _finalize_topology(self):
        if self.orig_faces:
            self.num_original_vertices = (
                max((max(f) for f in self.orig_faces), default=-1) + 1
            )
        else:
            self.num_original_vertices = len(self.verts)

        self.vertex_to_original_faces = {
            v_idx: set() for v_idx in range(self.num_original_vertices)
        }
        if self.orig_faces:
            for orig_face_idx, f_verts in enumerate(self.orig_faces):
                for v_idx in f_verts:
                    if v_idx < self.num_original_vertices:
                        self.vertex_to_original_faces[v_idx].add(orig_face_idx)

        face_to_type_idx = {}
        for type_idx, shape_key in enumerate(self.face_type_keys):
            for f_idx in self.face_types[shape_key]:
                face_to_type_idx[f_idx] = type_idx

        self.vertex_types = defaultdict(list)
        for v_idx in range(self.num_original_vertices):
            v_face_types = sorted(
                [
                    face_to_type_idx[f]
                    for f in self.vertex_to_original_faces[v_idx]
                    if f in face_to_type_idx
                ]
            )
            self.vertex_types[(len(v_face_types), tuple(v_face_types))].append(v_idx)

        self.vertex_type_keys = list(self.vertex_types.keys())
        self.representative_vertices = [
            self.vertex_types[k][0] for k in self.vertex_type_keys
        ]

    def rotate_camera(self, dx, dy):
        if dx != 0:
            c, s = math.cos(dx), math.sin(dx)
            ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype="f4")
            self.rot_mat = self.rot_mat @ ry

        if dy != 0:
            c, s = math.cos(dy), math.sin(dy)
            rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype="f4")
            self.rot_mat = self.rot_mat @ rx

    def render_sphere_at(self, pos):
        translated_data = []
        black_color = [0.0, 0.0, 0.0, 1.0]
        for tri in self.sphere_faces:
            for idx in tri:
                v = self.sphere_local_verts[idx]
                p = v + pos
                n = (
                    v / np.linalg.norm(v)
                    if np.linalg.norm(v) > 0
                    else np.array([0.0, 1.0, 0.0], dtype="f4")
                )
                translated_data.extend(
                    [p[0], p[1], p[2]] + black_color + [n[0], n[1], n[2]]
                )

        temp_vbo = self.ctx.buffer(np.array(translated_data, dtype="f4").tobytes())
        temp_vao = self.ctx.vertex_array(
            self.program,
            [(temp_vbo, "3f 4f 3f", "in_position", "in_color", "in_normal")],
        )
        temp_vao.render(moderngl.TRIANGLES)
        temp_vao.release()
        temp_vbo.release()

    def render_edges(self, edges_to_draw, verts):
        if not edges_to_draw:
            return
        edge_vbo_data = []
        black_color = [0.0, 0.0, 0.0, 1.0]
        dummy_normal = [0.0, 0.0, 0.0]
        for u, v in edges_to_draw:
            pos_u = verts[u]
            pos_v = verts[v]
            edge_vbo_data.extend(
                [pos_u[0], pos_u[1], pos_u[2]] + black_color + dummy_normal
            )
            edge_vbo_data.extend(
                [pos_v[0], pos_v[1], pos_v[2]] + black_color + dummy_normal
            )

        edge_vbo_bytes = np.array(edge_vbo_data, dtype="f4").tobytes()
        temp_vbo = self.ctx.buffer(edge_vbo_bytes)
        temp_vao = self.ctx.vertex_array(
            self.program,
            [(temp_vbo, "3f 4f 3f", "in_position", "in_color", "in_normal")],
        )
        temp_vao.render(moderngl.LINES)
        temp_vao.release()
        temp_vbo.release()

    def render_3d(
        self,
        viewport,
        aspect,
        active_face_index=-1,
        target_x=0.0,
        active_vertex_index=-1,
        edge_mode=0,
        active_part_idx=-1,
    ):
        if not self.valid:
            return
        self.ctx.viewport = viewport

        eye = self.rot_mat @ np.array([0.0, 0.0, 3.0], dtype="f4")
        up = self.rot_mat @ np.array([0.0, 1.0, 0.0], dtype="f4")

        if target_x != 0.0:
            zaxis = eye - np.array([0.0, 0.0, 0.0], dtype="f4")
            zaxis = zaxis / np.linalg.norm(zaxis)
            right = np.cross(up, zaxis)
            right_norm = np.linalg.norm(right)
            right = right / (right_norm if right_norm != 0 else 1.0)
            target = target_x * right
        else:
            target = np.array([0.0, 0.0, 0.0], dtype="f4")

        proj = create_perspective_matrix(45.0, aspect, 0.1, 10.0)
        view = create_lookat_matrix(eye, target, up)

        self.program["u_proj"].write(proj.tobytes())
        self.program["u_view"].write(view.tobytes())

        if "u_alpha_override" in self.program:
            self.program["u_alpha_override"].value = -1.0

        if active_part_idx != -1 and 0 <= active_part_idx < len(
            self.component_resources
        ):
            res = self.component_resources[active_part_idx]
            res["vao"].render(moderngl.TRIANGLES)
            if edge_mode > 0:
                self.render_edges(res["edges"], self.verts)
        else:
            if active_face_index != -1 and active_face_index in self.face_ranges:
                active_plane = self.face_planes[active_face_index]
                coplanar_indices = []
                for f_idx, plane in self.face_planes.items():
                    if f_idx != active_face_index:
                        n1, d1 = active_plane
                        n2, d2 = plane
                        if abs(abs(np.dot(n1, n2)) - 1.0) < 1e-4:
                            if abs(d1 - d2 * np.sign(np.dot(n1, n2))) < 1e-4:
                                coplanar_indices.append(f_idx)

                if coplanar_indices:
                    self.program["u_alpha_override"].value = ALPHA
                    self.ctx.depth_write = False
                    for f_idx in coplanar_indices:
                        start_v, count_v = self.face_ranges[f_idx]
                        self.vao.render(
                            moderngl.TRIANGLES,
                            vertices=count_v,
                            first=start_v,
                        )
                    self.ctx.depth_write = True

                self.program["u_alpha_override"].value = -1.0
                start_v, count_v = self.face_ranges[active_face_index]
                self.vao.render(moderngl.TRIANGLES, vertices=count_v, first=start_v)
            elif (
                active_vertex_index != -1
                and active_vertex_index in self.vertex_to_original_faces
            ):
                for orig_face_idx in self.vertex_to_original_faces[active_vertex_index]:
                    if orig_face_idx in self.face_ranges:
                        start_v, count_v = self.face_ranges[orig_face_idx]
                        self.vao.render(
                            moderngl.TRIANGLES,
                            vertices=count_v,
                            first=start_v,
                        )
            else:
                self.vao.render(moderngl.TRIANGLES)

            if active_vertex_index != -1:
                pos = self.verts[active_vertex_index]
                self.render_sphere_at(pos)

            if edge_mode > 0:
                edges_to_draw = []
                if active_vertex_index != -1:
                    if edge_mode == 1:
                        edges_to_draw = [
                            e
                            for e in self.all_edges
                            if e[0] == active_vertex_index
                            or e[1] == active_vertex_index
                        ]
                    elif edge_mode == 2:
                        edges_to_draw = self.all_edges
                elif active_face_index != -1:
                    if edge_mode == 1:
                        edges_to_draw = self.face_boundary_edges.get(
                            active_face_index, []
                        )
                    elif edge_mode == 2:
                        edges_to_draw = self.all_edges
                else:
                    edges_to_draw = self.all_edges

                self.render_edges(edges_to_draw, self.verts)


def run_bg_analysis(folder, viewers_list):
    bg_run_id[0] += 1
    my_run_id = bg_run_id[0]

    def worker():
        if not viewers_list:
            return

        if offcheck is None or not hasattr(offcheck, "verify_off_logic"):
            print(
                "Cannot run background metric analysis: offcheck module is unavailable."
            )
            return

        verify_off_logic = offcheck.verify_off_logic
        results = []
        all_keys = set()

        prog_dir = os.path.dirname(os.path.abspath(__file__))

        offcheck_mtime = 0.0
        candidate_paths = [
            getattr(offcheck, "__file__", None),
            os.path.join(prog_dir, "offcheck.py"),
            os.path.join(prog_dir, "offviewer", "lib", "offcheck.py"),
        ]
        for cp in candidate_paths:
            if cp:
                abs_cp = (
                    os.path.abspath(cp)
                    if os.path.isabs(cp)
                    else os.path.abspath(os.path.join(prog_dir, cp))
                )
                if os.path.exists(abs_cp):
                    try:
                        offcheck_mtime = round(os.path.getmtime(abs_cp), 4)
                        break
                    except Exception:
                        pass

        folder_hash = hashlib.md5(os.path.abspath(folder).encode("utf-8")).hexdigest()
        cache_dir = os.path.join(prog_dir, ".cache", folder_hash)
        cache_path = os.path.join(cache_dir, "metrics.json")

        cache_data = {}
        cache_dirty = False

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as cf:
                    loaded = json.load(cf)
                    cached_mtime = loaded.get("offcheck_mtime", 0.0)
                    if abs(cached_mtime - offcheck_mtime) < 1e-3:
                        cache_data = loaded.get("files", {})
            except Exception as e:
                print(f"Failed to read metrics cache: {e}")

        for v in viewers_list:
            if bg_run_id[0] != my_run_id:
                return
            fpath = os.path.abspath(v.filepath)
            fname = os.path.basename(fpath)

            try:
                current_mtime = round(os.path.getmtime(fpath), 4)
                current_size = os.path.getsize(fpath)
            except Exception:
                current_mtime = 0.0
                current_size = 0

            cached_entry = cache_data.get(fname)
            use_cache = False
            if cached_entry:
                cached_file_mtime = cached_entry.get("mtime", 0.0)
                if (
                    abs(cached_file_mtime - current_mtime) < 1e-3
                    and cached_entry.get("size") == current_size
                ):
                    use_cache = True

            if use_cache:
                stats = cached_entry.get("metrics")
            else:
                try:
                    res, stats = verify_off_logic(
                        fpath,
                        return_stats=True,
                        run_symmetry=RUN_SYMMETRY_CALCULATION,
                    )
                    if stats is None:
                        stats = {"Filename": fname, "Detail": str(res)}
                except Exception as e:
                    print(f"Error calculating metrics for {fname}: {e}")
                    stats = {"Filename": fname, "Detail": f"Error: {e}"}

                cache_data[fname] = {
                    "mtime": current_mtime,
                    "size": current_size,
                    "metrics": stats,
                }
                cache_dirty = True

            if stats is not None:
                results.append(stats)
                all_keys.update(stats.keys())

        if bg_run_id[0] != my_run_id:
            return

        if cache_dirty:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as cf:
                    json.dump(
                        {"offcheck_mtime": offcheck_mtime, "files": cache_data},
                        cf,
                        indent=4,
                    )
            except Exception as e:
                print(f"Failed to write metrics cache: {e}")

        headers_list = list(all_keys)
        if "Filename" in headers_list:
            headers_list.remove("Filename")
        if "Detail" in headers_list:
            headers_list.remove("Detail")

        def extract_num(key_str):
            m = re.search(r"\d+", key_str)
            return int(m.group()) if m else 999999

        dynamic_headers = [
            h
            for h in headers_list
            if h.startswith("Faces (") or h.startswith("Valence (")
        ]
        base_headers = [h for h in headers_list if h not in dynamic_headers]

        base_headers.sort()
        dynamic_headers.sort(key=extract_num)

        headers = ["Filename"] + base_headers + dynamic_headers + ["Detail"]

        csv_path = os.path.join(folder, ".offviewer_temp.csv")
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                for row in results:
                    if bg_run_id[0] != my_run_id:
                        return
                    row_data = {h: row.get(h, "") for h in headers}
                    writer.writerow(row_data)

            if bg_run_id[0] != my_run_id:
                try:
                    os.remove(csv_path)
                except Exception:
                    pass
                return

            bg_csv_path[0] = csv_path
            bg_csv_ready[0] = True
        except Exception as e:
            print(f"Failed to write background CSV: {e}")

    t = threading.Thread(target=worker, daemon=True)
    t.start()


def run_single_analysis(filepath, state):
    state["active_report"] = "Generating offcheck report..."
    state["needs_redraw"] = 2

    def worker():
        try:
            verify_off_logic = offcheck.verify_off_logic
            report = verify_off_logic(
                os.path.abspath(filepath), run_symmetry=RUN_SYMMETRY_CALCULATION
            )

            if "Valence 0" in report:
                lines = report.split("\n")
                cleaned_lines = []
                has_valence_zero = False
                for line in lines:
                    if re.search(r"Valence\s+0\s*:", line):
                        has_valence_zero = True
                        continue
                    cleaned_lines.append(line)
                if has_valence_zero:
                    report = "\n".join(cleaned_lines)
                    if "CONCLUSION:" in report:
                        report = report.replace(
                            "CONCLUSION:",
                            "CONCLUSION: WARNING: No 0 vertex in file\n",
                        )
                    else:
                        report += "\nCONCLUSION: WARNING: No 0 vertex in file"

            state["active_report"] = report
            state["needs_redraw"] = 2
        except Exception as e:
            state["active_report"] = f"Error during offcheck execution:\n{e}"
            state["needs_redraw"] = 2

    t = threading.Thread(target=worker, daemon=True)
    t.start()


def load_viewers_from_folder(folder, ctx, prog_3d):
    viewers_list = []
    try:
        off_files = [f for f in os.listdir(folder) if f.lower().endswith(".off")]
        use_cache = len(off_files) > CACHE_THRESHOLD
        cache_dir = None

        if use_cache:
            prog_dir = os.path.dirname(os.path.abspath(__file__))
            folder_hash = hashlib.md5(
                os.path.abspath(folder).encode("utf-8")
            ).hexdigest()
            cache_dir = os.path.join(prog_dir, ".cache", folder_hash)

            if os.path.exists(cache_dir):
                active_cache_names = {f"{f}.bin" for f in off_files}
                active_cache_names.add("metrics.json")
                for cached_file in os.listdir(cache_dir):
                    if cached_file not in active_cache_names:
                        try:
                            os.remove(os.path.join(cache_dir, cached_file))
                        except Exception:
                            pass

        for f in sorted(off_files):
            v = ModelViewer(
                os.path.join(folder, f),
                ctx,
                prog_3d,
                use_cache=use_cache,
                cache_dir=cache_dir,
            )
            if v.valid:
                viewers_list.append(v)
    except Exception as e:
        print(f"Error scanning folder {folder}: {e}")
    return viewers_list


# Version: 5.57
def poll_folder_changes(state, viewers, ctx, prog_3d):
    folder = state.get("current_folder")
    if not folder or not os.path.isdir(folder):
        return

    try:
        disk_files = {f for f in os.listdir(folder) if f.lower().endswith(".off")}
    except Exception:
        return

    current_viewer_map = {v.filename: v for v in state.get("all_viewers", [])}
    current_files = set(current_viewer_map.keys())

    added = disk_files - current_files
    removed = current_files - disk_files
    existing = disk_files & current_files

    prog_dir = os.path.dirname(os.path.abspath(__file__))
    folder_hash = hashlib.md5(os.path.abspath(folder).encode("utf-8")).hexdigest()
    cache_dir = os.path.join(prog_dir, ".cache", folder_hash)
    use_cache = len(disk_files) > CACHE_THRESHOLD

    changed = False

    # Check existing files for modification
    for fname in existing:
        fpath = os.path.join(folder, fname)
        try:
            current_mtime = round(os.path.getmtime(fpath), 4)
            current_size = os.path.getsize(fpath)
        except Exception:
            continue

        v = current_viewer_map[fname]
        if not hasattr(v, "_poll_mtime"):
            v._poll_mtime = current_mtime
            v._poll_size = current_size
            continue

        if abs(v._poll_mtime - current_mtime) >= 1e-3 or v._poll_size != current_size:
            # Invalidate disk caches for the modified file
            geo_cache_path = os.path.join(cache_dir, fname + ".bin")
            if os.path.exists(geo_cache_path):
                try:
                    os.remove(geo_cache_path)
                except Exception:
                    pass

            metrics_cache_path = os.path.join(cache_dir, "metrics.json")
            if os.path.exists(metrics_cache_path):
                try:
                    with open(metrics_cache_path, "r", encoding="utf-8") as cf:
                        cdata = json.load(cf)
                    if "files" in cdata and fname in cdata["files"]:
                        del cdata["files"][fname]
                        with open(metrics_cache_path, "w", encoding="utf-8") as cf:
                            json.dump(cdata, cf, indent=4)
                except Exception:
                    pass

            new_v = ModelViewer(
                fpath, ctx, prog_3d, use_cache=use_cache, cache_dir=cache_dir
            )
            if new_v.valid:
                new_v._poll_mtime = current_mtime
                new_v._poll_size = current_size
                for idx, old_v in enumerate(state["all_viewers"]):
                    if old_v.filename == fname:
                        state["all_viewers"][idx] = new_v
                        break
                changed = True

                # If the modified file is currently enlarged, rerun single report
                if (
                    state.get("fullscreen_index", -1) != -1
                    and 0 <= state["fullscreen_index"] < len(viewers)
                    and viewers[state["fullscreen_index"]].filename == fname
                ):
                    run_single_analysis(fpath, state)

    if removed:
        state["all_viewers"] = [
            v for v in state["all_viewers"] if v.filename not in removed
        ]
        changed = True

    if added:
        for fname in sorted(added):
            fpath = os.path.join(folder, fname)
            try:
                current_mtime = round(os.path.getmtime(fpath), 4)
                current_size = os.path.getsize(fpath)
            except Exception:
                current_mtime = 0.0
                current_size = 0

            v = ModelViewer(
                fpath, ctx, prog_3d, use_cache=use_cache, cache_dir=cache_dir
            )
            if v.valid:
                v._poll_mtime = current_mtime
                v._poll_size = current_size
                state["all_viewers"].append(v)
                changed = True

    if changed:
        apply_search_and_sort(state, viewers)
        run_bg_analysis(folder, state["all_viewers"])
        state["needs_redraw"] = 2


def get_subfolders(folder):
    subfolders_list = []
    try:
        parent = os.path.dirname(folder)
        if parent and parent != folder:
            subfolders_list.append("..")
        dirs = []
        for d in os.listdir(folder):
            if os.path.isdir(os.path.join(folder, d)) and not d.startswith("."):
                dirs.append(d)
        dirs.sort(key=lambda s: s.lower())
        subfolders_list.extend(dirs)
    except Exception as e:
        print(f"Error listing subfolders: {e}")
    return subfolders_list


def apply_search_and_sort(state, viewers):
    all_v = state.get("all_viewers", [])
    query = state.get("search_query", "").strip()
    if query:
        negate = False
        pattern = query
        if query.startswith("-") and len(query) > 1:
            negate = True
            pattern = query[1:]

        filtered = []
        for v in all_v:
            name_lower = v.filename.lower()
            base_lower = os.path.splitext(v.filename)[0].lower()
            pattern_lower = pattern.lower()

            is_match = fnmatch.fnmatch(name_lower, pattern_lower) or fnmatch.fnmatch(
                base_lower, pattern_lower
            )

            if negate:
                if not is_match:
                    filtered.append(v)
            else:
                if is_match:
                    filtered.append(v)
        viewers[:] = filtered
    else:
        viewers[:] = list(all_v)

    selected_metric = state.get("selected_sort_metric", "Default")
    sort_viewers_by_metric(selected_metric, viewers)


def sort_viewers_by_metric(metric, viewers):
    if metric == "Default":
        viewers.sort(key=lambda v: v.filename.lower())
    elif metric.upper() in ("V-E-F", "VEF"):

        def get_sort_key(v):
            try:
                v_num = int(
                    float(v.metrics.get("Vertices", v.metrics.get("vertices", 0)))
                )
                e_num = int(float(v.metrics.get("Edges", v.metrics.get("edges", 0))))
                f_num = int(float(v.metrics.get("Faces", v.metrics.get("faces", 0))))
                if v_num > 0 or e_num > 0 or f_num > 0:
                    return (0, v_num, e_num, f_num)
            except (ValueError, TypeError):
                pass

            val = v.metrics.get(metric, "")
            if val == "" or val is None:
                return (2, (0, 0, 0))
            if val in ("N/A", "Error", "nan", "NaN"):
                return (1, val.lower())

            nums = [int(n) for n in re.findall(r"\d+", str(val))]
            if len(nums) >= 3:
                return (0, nums[0], nums[1], nums[2])
            elif len(nums) > 0:
                nums.extend([0] * (3 - len(nums)))
                return (0, nums[0], nums[1], nums[2])

            return (1, str(val).lower())

        viewers.sort(key=get_sort_key)
    else:

        def get_sort_key(v):
            val = v.metrics.get(metric, "")
            if val == "" or val is None:
                return (2, "")
            if val in ("N/A", "Error", "nan", "NaN"):
                return (1, val.lower())
            try:
                return (0, float(val))
            except ValueError:
                return (1, str(val).lower())

        viewers.sort(key=get_sort_key)


def init_gl_resources(ctx):
    vs_3d = """
    #version 330
    in vec3 in_position;
    in vec4 in_color;
    in vec3 in_normal;
    out vec4 v_color;
    out vec3 v_normal;
    uniform mat4 u_proj;
    uniform mat4 u_view;
    void main() {
        gl_Position = u_proj * u_view * vec4(in_position, 1.0);
        v_color = in_color;
        v_normal = mat3(u_view) * in_normal;
    }
    """

    fs_3d = """
    #version 330
    in vec4 v_color;
    in vec3 v_normal;
    out vec4 fragColor;
    uniform float u_alpha_override;
    void main() {
        vec3 normal = normalize(v_normal);
        if (!gl_FrontFacing) {
            normal = -normal;
        }
        vec3 light_dir1 = normalize(vec3(0.5, 0.5, 1.0));
        vec3 light_dir2 = normalize(vec3(-0.5, -0.3, 0.8));
        vec3 light_dir3 = normalize(vec3(0.0, 1.0, -0.5));

        float diff1 = max(dot(normal, light_dir1), 0.0) * 0.50;
        float diff2 = max(dot(normal, light_dir2), 0.0) * 0.25;
        float diff3 = max(dot(normal, light_dir3), 0.0) * 0.15;

        float ambient = 0.35;
        vec3 final_light = v_color.rgb * (diff1 + diff2 + diff3 + ambient);
        float alpha = (u_alpha_override >= 0.0) ? u_alpha_override : v_color.a;
        fragColor = vec4(final_light, alpha);
    }
    """

    vs_2d = """
    #version 330
    in vec2 in_position;
    in vec2 in_texcoord;
    out vec2 v_texcoord;
    void main() {
        gl_Position = vec4(in_position, 0.0, 1.0);
        v_texcoord = in_texcoord;
    }
    """

    fs_2d = """
    #version 330
    in vec2 v_texcoord;
    out vec4 fragColor;
    uniform sampler2D u_texture;
    void main() {
        fragColor = texture(u_texture, v_texcoord);
    }
    """

    prog_3d = ctx.program(vertex_shader=vs_3d, fragment_shader=fs_3d)
    prog_2d = ctx.program(vertex_shader=vs_2d, fragment_shader=fs_2d)

    quad_data = np.array(
        [
            -1.0,
            -1.0,
            0.0,
            0.0,
            1.0,
            -1.0,
            1.0,
            0.0,
            -1.0,
            1.0,
            0.0,
            1.0,
            -1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        dtype="f4",
    )
    quad_vbo = ctx.buffer(quad_data.tobytes())
    quad_vao = ctx.vertex_array(
        prog_2d, [(quad_vbo, "2f 2f", "in_position", "in_texcoord")]
    )

    return prog_3d, prog_2d, quad_vao


def show_bespoke_text_window(filepath):
    def run():
        root = tk.Tk()
        root.title(f"Source: {os.path.basename(filepath)}")
        root.geometry("650x550")

        scrollbar_y = tk.Scrollbar(root, orient=tk.VERTICAL)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x = tk.Scrollbar(root, orient=tk.HORIZONTAL)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        text_widget = tk.Text(
            root,
            wrap=tk.NONE,
            yscrollcommand=scrollbar_y.set,
            xscrollcommand=scrollbar_x.set,
            font=("Courier New", 10),
        )
        text_widget.pack(fill=tk.BOTH, expand=True)

        scrollbar_y.config(command=text_widget.yview)
        scrollbar_x.config(command=text_widget.xview)

        try:
            with open(filepath, "r", errors="ignore") as f:
                content = f.read()
            text_widget.insert(tk.END, content)
        except Exception as e:
            text_widget.insert(tk.END, f"Error opening file:\n{e}")

        text_widget.config(state=tk.DISABLED)
        root.mainloop()

    t = threading.Thread(target=run, daemon=True)
    t.start()


def _perform_full_refresh(state, viewers, subfolders, ctx, prog_3d):
    prog_dir = os.path.dirname(os.path.abspath(__file__))
    folder_hash = hashlib.md5(
        os.path.abspath(state["current_folder"]).encode("utf-8")
    ).hexdigest()
    cache_dir = os.path.join(prog_dir, ".cache", folder_hash)
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
        except Exception as e:
            print(f"Error clearing cache: {e}")

    current_file = None
    if state["fullscreen_index"] != -1 and 0 <= state["fullscreen_index"] < len(
        viewers
    ):
        current_file = viewers[state["fullscreen_index"]].filename

    state["all_viewers"] = load_viewers_from_folder(
        state["current_folder"], ctx, prog_3d
    )
    apply_search_and_sort(state, viewers)
    subfolders[:] = get_subfolders(state["current_folder"])

    if current_file:
        new_idx = next(
            (i for i, v in enumerate(viewers) if v.filename == current_file), -1
        )
        state["fullscreen_index"] = new_idx
        if new_idx != -1:
            run_single_analysis(viewers[new_idx].filepath, state)
        else:
            state["active_report"] = None

    state["sort_dropdown_enabled"] = False
    state["selected_sort_metric"] = "Default"
    state["sort_headers"] = []
    state["dropdown_open"] = False
    state["dropdown_scroll"] = 0
    bg_csv_ready[0] = False
    bg_csv_path[0] = None
    run_bg_analysis(state["current_folder"], state["all_viewers"])
    state["needs_redraw"] = 2


def _handle_drop_file(dropped_path, state, viewers, subfolders, ctx, prog_3d):
    if os.path.isdir(dropped_path):
        state["current_folder"] = os.path.abspath(dropped_path)
        state["fullscreen_index"] = -1
    else:
        state["current_folder"] = os.path.abspath(os.path.dirname(dropped_path))
        state["target_file"] = os.path.basename(dropped_path)

    state["all_viewers"] = load_viewers_from_folder(
        state["current_folder"], ctx, prog_3d
    )
    state["search_query"] = ""
    apply_search_and_sort(state, viewers)
    subfolders[:] = get_subfolders(state["current_folder"])
    state["scroll_y"] = 0.0
    state["subfolder_scroll_y"] = 0.0
    state["hovered_index"] = -1
    state["active_face_index"] = -1
    state["active_rep_idx"] = -1
    state["active_vertex_index"] = -1
    state["active_vertex_rep_idx"] = -1
    state["active_part_idx"] = -1
    state["active_report"] = None

    if not os.path.isdir(dropped_path):
        for idx, v in enumerate(viewers):
            if v.filename == state["target_file"]:
                state["fullscreen_index"] = idx
                run_single_analysis(v.filepath, state)
                break

    state["sort_dropdown_enabled"] = False
    state["selected_sort_metric"] = "Default"
    state["sort_headers"] = []
    state["dropdown_open"] = False
    state["dropdown_scroll"] = 0
    bg_csv_ready[0] = False
    bg_csv_path[0] = None
    state["context_menu_open"] = False
    run_bg_analysis(state["current_folder"], state["all_viewers"])
    state["needs_redraw"] = 2


def _handle_keydown(event, state, viewers, subfolders, ctx, prog_3d):
    if event.key in (pygame.K_r, pygame.K_F5):
        _perform_full_refresh(state, viewers, subfolders, ctx, prog_3d)
    elif state["fullscreen_index"] != -1:
        v = viewers[state["fullscreen_index"]]
        if event.key in (K_ESCAPE, K_x):
            state["fullscreen_index"] = -1
            state["active_face_index"] = -1
            state["active_rep_idx"] = -1
            state["active_vertex_index"] = -1
            state["active_vertex_rep_idx"] = -1
            state["active_part_idx"] = -1
            state["active_report"] = None
            state["needs_redraw"] = 2
        elif event.key in (K_LEFT, K_RIGHT):
            step = -1 if event.key == K_LEFT else 1
            new_idx = (state["fullscreen_index"] + step) % len(viewers)
            state["fullscreen_index"] = new_idx
            state["active_face_index"] = -1
            state["active_rep_idx"] = -1
            state["active_vertex_index"] = -1
            state["active_vertex_rep_idx"] = -1
            state["active_part_idx"] = -1
            nv = viewers[new_idx]
            state["target_file"] = nv.filename
            run_single_analysis(nv.filepath, state)
            state["needs_redraw"] = 2
        elif event.key == K_f:
            num_rep = len(v.representative_faces)
            if num_rep > 0:
                state["active_rep_idx"] = (state["active_rep_idx"] + 2) % (
                    num_rep + 1
                ) - 1
                state["active_face_index"] = (
                    v.representative_faces[state["active_rep_idx"]]
                    if state["active_rep_idx"] != -1
                    else -1
                )
                state["active_vertex_index"] = -1
                state["active_vertex_rep_idx"] = -1
                state["active_part_idx"] = -1
                state["needs_redraw"] = 2
        elif event.key == K_v:
            num_rep_v = len(v.representative_vertices)
            if num_rep_v > 0:
                state["active_vertex_rep_idx"] = (
                    state["active_vertex_rep_idx"] + 2
                ) % (num_rep_v + 1) - 1
                state["active_vertex_index"] = (
                    v.representative_vertices[state["active_vertex_rep_idx"]]
                    if state["active_vertex_rep_idx"] != -1
                    else -1
                )
                state["active_face_index"] = -1
                state["active_rep_idx"] = -1
                state["active_part_idx"] = -1
                state["needs_redraw"] = 2
        elif event.key == K_e:
            state["edge_mode"] = (state["edge_mode"] + 1) % 3
            state["needs_redraw"] = 2
        elif event.key == K_c:
            num_parts = len(v.component_resources)
            if num_parts > 0:
                state["active_part_idx"] = (state["active_part_idx"] + 2) % (
                    num_parts + 1
                ) - 1
                state["active_face_index"] = -1
                state["active_rep_idx"] = -1
                state["active_vertex_index"] = -1
                state["active_vertex_rep_idx"] = -1
                state["needs_redraw"] = 2
    else:
        if event.key in (K_PLUS, K_EQUALS, K_KP_PLUS):
            state["item_w"] = min(400, state["item_w"] + 10)
            state["needs_redraw"] = 2
        elif event.key in (K_MINUS, K_KP_MINUS):
            state["item_w"] = max(80, state["item_w"] - 10)
            state["needs_redraw"] = 2
        elif event.key == K_e:
            state["grid_edges_on"] = not state["grid_edges_on"]
            state["needs_redraw"] = 2


def _handle_context_menu_action(relative_y, target_v, state, viewers, ctx, prog_3d):
    import subprocess

    root_temp = tk.Tk()
    root_temp.option_add("*Entry.width", 60)
    root_temp.withdraw()

    num_tools = len(state.get("external_tools", []))
    if relative_y < 20:
        old_name = target_v.filename
        new_name = simpledialog.askstring(
            "Rename File", "Enter new name:", initialvalue=old_name
        )
        if new_name:
            if not new_name.lower().endswith(".off"):
                new_name += ".off"
            new_path = os.path.join(os.path.dirname(target_v.filepath), new_name)
            try:
                os.rename(target_v.filepath, new_path)
                target_v.filepath = new_path
                target_v.filename = new_name
                run_bg_analysis(state["current_folder"], state["all_viewers"])
            except Exception as e:
                messagebox.showerror("Error", f"Could not rename file: {e}")
    elif relative_y < 40:
        confirm = messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to permanently delete {target_v.filename}?",
        )
        if confirm:
            try:
                os.remove(target_v.filepath)
                viewers.pop(state["context_menu_index"])
                if target_v in state["all_viewers"]:
                    state["all_viewers"].remove(target_v)
                run_bg_analysis(state["current_folder"], state["all_viewers"])
            except Exception as e:
                messagebox.showerror("Error", f"Could not delete file: {e}")
    elif relative_y < 60:
        show_bespoke_text_window(target_v.filepath)
    elif relative_y < 60 + num_tools * 20:
        tool_idx = int((relative_y - 60) // 20)
        tool = state["external_tools"][tool_idx]
        filepath = target_v.filepath
        try:
            cmd = [tool["program"]] + tool.get("args", []) + [os.path.abspath(filepath)]
            proc = subprocess.Popen(cmd)
            if tool["prompt"].strip().lower().startswith("edit"):
                t = threading.Thread(
                    target=watch_edit_tool,
                    args=(proc, filepath, state, ctx, prog_3d, viewers),
                    daemon=True,
                )
                t.start()
        except Exception as e:
            messagebox.showerror("Error", f"Could not launch {tool['prompt']}: {e}")

    root_temp.update()
    root_temp.destroy()
    state["context_menu_open"] = False


def _handle_mouse_scroll(event_button, mx, my, win_h, sidebar_w, top_bar_h, state):
    if event_button == 4:
        if state["dropdown_open"] and pygame.Rect(
            15, 102, 170, min(350, win_h - 120)
        ).collidepoint(mx, my):
            state["dropdown_scroll"] = max(0, state["dropdown_scroll"] - 20)
        elif mx <= sidebar_w and my > 160:
            state["subfolder_scroll_y"] = max(0.0, state["subfolder_scroll_y"] - 25)
        elif state["fullscreen_index"] == -1 and mx > sidebar_w and my > top_bar_h:
            state["scroll_y"] = max(
                0.0, min(state["scroll_y"] - 50, state["max_scroll"])
            )
    elif event_button == 5:
        if state["dropdown_open"] and pygame.Rect(
            15, 102, 170, min(350, win_h - 120)
        ).collidepoint(mx, my):
            all_opts_len = len(["Default"] + state["sort_headers"])
            max_dp_scroll = max(0, all_opts_len * 20 - min(350, win_h - 120))
            state["dropdown_scroll"] = min(max_dp_scroll, state["dropdown_scroll"] + 20)
        elif mx <= sidebar_w and my > 160:
            state["subfolder_scroll_y"] = min(
                state.get("max_subfolder_scroll", 0),
                state["subfolder_scroll_y"] + 25,
            )
        elif state["fullscreen_index"] == -1 and mx > sidebar_w and my > top_bar_h:
            state["scroll_y"] = max(
                0.0, min(state["scroll_y"] + 50, state["max_scroll"])
            )


def _delete_duplicates(state, viewers):
    selected_metric = state.get("selected_sort_metric", "Default")
    if selected_metric == "Default" or len(viewers) <= 1:
        return

    groups = []
    current_group = [viewers[0]]
    for v in viewers[1:]:
        prev_v = current_group[-1]
        val1 = v.metrics.get(selected_metric, "")
        val2 = prev_v.metrics.get(selected_metric, "")
        is_equal = False
        if val1 is not None and val2 is not None and val1 != "" and val2 != "":
            if val1 == val2:
                is_equal = True
            else:
                try:
                    f1, f2 = float(val1), float(val2)
                    if abs(f1 - f2) < SORT_TOLERANCE * abs(f2):
                        is_equal = True
                except (ValueError, TypeError):
                    pass
        if is_equal:
            current_group.append(v)
        else:
            groups.append(current_group)
            current_group = [v]
    groups.append(current_group)

    to_delete = []
    for group in groups:
        if len(group) > 1:
            mtimes = []
            for v in group:
                try:
                    mtimes.append((os.path.getmtime(v.filepath), v))
                except Exception:
                    mtimes.append((float("inf"), v))
            mtimes.sort(key=lambda x: x[0])
            for m, v in mtimes[1:]:
                to_delete.append(v)

    if to_delete:
        deleted_count = 0
        for v in to_delete:
            try:
                if os.path.exists(v.filepath):
                    os.remove(v.filepath)
                if v in viewers:
                    viewers.remove(v)
                if v in state["all_viewers"]:
                    state["all_viewers"].remove(v)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting duplicate {v.filename}: {e}")

        run_bg_analysis(state["current_folder"], state["all_viewers"])
        state["needs_redraw"] = 2

        root_temp = tk.Tk()
        root_temp.withdraw()
        messagebox.showinfo(
            "Delete Duplicates",
            f"Deleted {deleted_count} duplicate files. Kept the oldest from each set.",
        )
        root_temp.destroy()


def handle_events(events, state, viewers, subfolders, ctx, prog_3d):
    mx, my = pygame.mouse.get_pos()
    double_click_threshold = 300
    sidebar_w = 200
    top_bar_h = 75
    win_w, win_h = pygame.display.get_window_size()
    grid_w = win_w - sidebar_w

    for event in events:
        if event.type == QUIT:
            state["running"] = False
        elif event.type == VIDEORESIZE:
            pygame.display.set_mode(
                event.size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
            )
        elif event.type == pygame.DROPFILE:
            _handle_drop_file(event.file, state, viewers, subfolders, ctx, prog_3d)
        elif event.type == KEYDOWN:
            _handle_keydown(event, state, viewers, subfolders, ctx, prog_3d)
        elif event.type == MOUSEBUTTONDOWN:
            if event.button in (4, 5):
                _handle_mouse_scroll(
                    event.button, mx, my, win_h, sidebar_w, top_bar_h, state
                )
            elif event.button == 3:
                target_idx = (
                    state["fullscreen_index"]
                    if state["fullscreen_index"] != -1
                    else state["hovered_index"]
                )
                if target_idx != -1:
                    state["context_menu_open"] = True
                    state["context_menu_index"] = target_idx
                    state["context_menu_pos"] = (mx, my)
            elif event.button == 1:
                event_handled = False
                if state["context_menu_open"]:
                    cx, cy = state["context_menu_pos"]
                    num_tools = len(state.get("external_tools", []))
                    menu_h = 60 + num_tools * 20
                    menu_w = state.get("context_menu_width", 120)
                    if pygame.Rect(cx, cy, menu_w, menu_h).collidepoint(mx, my):
                        _handle_context_menu_action(
                            my - cy,
                            viewers[state["context_menu_index"]],
                            state,
                            viewers,
                            ctx,
                            prog_3d,
                        )
                    else:
                        state["context_menu_open"] = False
                    event_handled = True

                elif state["fullscreen_index"] != -1 and len(viewers) > 0:
                    prev_btn = pygame.Rect(win_w - 210, win_h - 45, 40, 30)
                    next_btn = pygame.Rect(win_w - 160, win_h - 45, 40, 30)
                    grid_btn = pygame.Rect(win_w - 110, win_h - 45, 90, 30)

                    if prev_btn.collidepoint(mx, my) or next_btn.collidepoint(mx, my):
                        step = -1 if prev_btn.collidepoint(mx, my) else 1
                        new_idx = (state["fullscreen_index"] + step) % len(viewers)
                        state["fullscreen_index"] = new_idx
                        state["active_face_index"] = -1
                        state["active_rep_idx"] = -1
                        state["active_vertex_index"] = -1
                        state["active_vertex_rep_idx"] = -1
                        state["active_part_idx"] = -1
                        v = viewers[new_idx]
                        state["target_file"] = v.filename
                        run_single_analysis(v.filepath, state)
                        state["needs_redraw"] = 2
                        event_handled = True
                    elif grid_btn.collidepoint(mx, my):
                        state["fullscreen_index"] = -1
                        state["active_face_index"] = -1
                        state["active_rep_idx"] = -1
                        state["active_vertex_index"] = -1
                        state["active_vertex_rep_idx"] = -1
                        state["active_part_idx"] = -1
                        state["active_report"] = None
                        state["needs_redraw"] = 2
                        event_handled = True

                if (
                    not event_handled
                    and state["dropdown_open"]
                    and state["sort_dropdown_enabled"]
                ):
                    dp_h = min(350, win_h - 120)
                    if pygame.Rect(15, 102, 170, dp_h).collidepoint(mx, my):
                        all_opts = ["Default"] + state["sort_headers"]
                        clicked_idx = (my - 102 + state["dropdown_scroll"]) // 20
                        if 0 <= clicked_idx < len(all_opts):
                            state["selected_sort_metric"] = all_opts[clicked_idx]
                            sort_viewers_by_metric(
                                state["selected_sort_metric"], viewers
                            )
                            state["dropdown_open"] = False
                        event_handled = True
                    else:
                        state["dropdown_open"] = False

                if not event_handled and pygame.Rect(15, 32, 170, 18).collidepoint(
                    mx, my
                ):
                    state["auto_rotate"] = not state["auto_rotate"]
                    event_handled = True

                if (
                    not event_handled
                    and state["sort_dropdown_enabled"]
                    and pygame.Rect(15, 75, 170, 22).collidepoint(mx, my)
                ):
                    state["dropdown_open"] = not state["dropdown_open"]
                    event_handled = True

                if not event_handled and pygame.Rect(15, 103, 170, 16).collidepoint(
                    mx, my
                ):
                    _delete_duplicates(state, viewers)
                    event_handled = True

                if not event_handled and pygame.Rect(15, 125, 170, 24).collidepoint(
                    mx, my
                ):
                    _perform_full_refresh(state, viewers, subfolders, ctx, prog_3d)
                    event_handled = True

                if not event_handled and state["fullscreen_index"] == -1:
                    if pygame.Rect(win_w - 115, 30, 100, 30).collidepoint(mx, my):
                        root_temp = tk.Tk()
                        root_temp.option_add("*Entry.width", 60)
                        root_temp.withdraw()
                        query = simpledialog.askstring(
                            "Search Files",
                            "Enter search pattern (? and * wildcards allowed):",
                            initialvalue=state.get("search_query", ""),
                        )
                        if query is not None:
                            state["search_query"] = query
                            apply_search_and_sort(state, viewers)
                            state["scroll_y"] = 0.0
                            state["hovered_index"] = -1
                            state["fullscreen_index"] = -1
                        root_temp.update()
                        root_temp.destroy()
                        event_handled = True
                    elif pygame.Rect(sidebar_w + 15, 30, grid_w - 150, 30).collidepoint(
                        mx, my
                    ):
                        root_temp = tk.Tk()
                        root_temp.option_add("*Entry.width", 80)
                        root_temp.withdraw()
                        new_folder = simpledialog.askstring(
                            "Edit Folder Path",
                            "Enter new folder path:",
                            initialvalue=state["current_folder"],
                        )
                        if new_folder:
                            new_folder_abs = os.path.abspath(new_folder)
                            if os.path.isdir(new_folder_abs):
                                state["current_folder"] = new_folder_abs
                                state["all_viewers"] = load_viewers_from_folder(
                                    state["current_folder"], ctx, prog_3d
                                )
                                state["search_query"] = ""
                                apply_search_and_sort(state, viewers)
                                subfolders[:] = get_subfolders(state["current_folder"])
                                state["scroll_y"] = 0.0
                                state["subfolder_scroll_y"] = 0.0
                                state["hovered_index"] = -1
                                state["fullscreen_index"] = -1
                                state["active_face_index"] = -1
                                state["active_rep_idx"] = -1
                                state["active_vertex_index"] = -1
                                state["active_vertex_rep_idx"] = -1
                                state["active_part_idx"] = -1
                                state["active_report"] = None
                                state["sort_dropdown_enabled"] = False
                                state["selected_sort_metric"] = "Default"
                                state["sort_headers"] = []
                                state["dropdown_open"] = False
                                state["dropdown_scroll"] = 0
                                bg_csv_ready[0] = False
                                bg_csv_path[0] = None
                                state["context_menu_open"] = False
                                run_bg_analysis(
                                    state["current_folder"],
                                    state["all_viewers"],
                                )
                            else:
                                messagebox.showerror(
                                    "Error",
                                    f"Directory does not exist:\n{new_folder_abs}",
                                )
                        root_temp.update()
                        root_temp.destroy()
                        event_handled = True

                if not event_handled:
                    grid_h = win_h - top_bar_h
                    if (
                        (mx >= win_w - 12)
                        and state["fullscreen_index"] == -1
                        and state["max_scroll"] > 0
                    ):
                        bar_h = max(
                            30,
                            int((grid_h / (state["max_scroll"] + grid_h)) * grid_h),
                        )
                        bar_y = top_bar_h + int(
                            (state["scroll_y"] / state["max_scroll"]) * (grid_h - bar_h)
                        )
                        state["scrollbar_dragging"] = True
                        state["scrollbar_click_offset"] = (
                            (my - bar_y)
                            if (bar_y <= my <= bar_y + bar_h)
                            else (bar_h // 2)
                        )
                        track_h = grid_h - bar_h
                        if track_h > 0 and not (bar_y <= my <= bar_y + bar_h):
                            ratio = max(
                                0.0,
                                min(
                                    1.0,
                                    (my - top_bar_h - state["scrollbar_click_offset"])
                                    / track_h,
                                ),
                            )
                            state["scroll_y"] = ratio * state["max_scroll"]
                    elif mx <= sidebar_w and my > 160:
                        sub_rect = pygame.Rect(10, 185, sidebar_w - 20, win_h - 195)
                        if sub_rect.collidepoint(mx, my):
                            clicked_idx = int(
                                (my - (185 - state.get("subfolder_scroll_y", 0))) // 25
                            )
                            if 0 <= clicked_idx < len(subfolders):
                                target = subfolders[clicked_idx]
                                state["current_folder"] = (
                                    os.path.abspath(
                                        os.path.dirname(state["current_folder"])
                                    )
                                    if target == ".."
                                    else os.path.abspath(
                                        os.path.join(state["current_folder"], target)
                                    )
                                )
                                state["all_viewers"] = load_viewers_from_folder(
                                    state["current_folder"], ctx, prog_3d
                                )
                                state["search_query"] = ""
                                apply_search_and_sort(state, viewers)
                                subfolders[:] = get_subfolders(state["current_folder"])
                                state["scroll_y"] = 0.0
                                state["subfolder_scroll_y"] = 0.0
                                state["hovered_index"] = -1
                                state["fullscreen_index"] = -1
                                state["active_face_index"] = -1
                                state["active_rep_idx"] = -1
                                state["active_vertex_index"] = -1
                                state["active_vertex_rep_idx"] = -1
                                state["active_part_idx"] = -1
                                state["active_report"] = None
                                state["sort_dropdown_enabled"] = False
                                state["selected_sort_metric"] = "Default"
                                state["sort_headers"] = []
                                state["dropdown_open"] = False
                                state["dropdown_scroll"] = 0
                                bg_csv_ready[0] = False
                                bg_csv_path[0] = None
                                state["context_menu_open"] = False
                                run_bg_analysis(
                                    state["current_folder"],
                                    state["all_viewers"],
                                )
                    else:
                        now = pygame.time.get_ticks()
                        if now - state["last_click_time"] < double_click_threshold:
                            if state["fullscreen_index"] != -1:
                                state["fullscreen_index"] = -1
                                state["active_face_index"] = -1
                                state["active_rep_idx"] = -1
                                state["active_vertex_index"] = -1
                                state["active_vertex_rep_idx"] = -1
                                state["active_part_idx"] = -1
                                state["active_report"] = None
                            elif state["hovered_index"] != -1:
                                state["fullscreen_index"] = state["hovered_index"]
                                run_single_analysis(
                                    viewers[state["hovered_index"]].filepath,
                                    state,
                                )
                        else:
                            drag_idx = (
                                state["fullscreen_index"]
                                if state["fullscreen_index"] != -1
                                else state["hovered_index"]
                            )
                            if mx > sidebar_w and my > top_bar_h and drag_idx != -1:
                                state["dragging"] = True
                                state["dragged_index"] = drag_idx
                                pygame.mouse.get_rel()
                        state["last_click_time"] = now

        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:
                state["dragging"] = False
                state["dragged_index"] = -1
                state["scrollbar_dragging"] = False
        elif event.type == MOUSEMOTION:
            grid_h = win_h - top_bar_h
            if state["scrollbar_dragging"] and state["max_scroll"] > 0:
                bar_h = max(
                    30,
                    int((grid_h / (state["max_scroll"] + grid_h)) * grid_h),
                )
                track_h = grid_h - bar_h
                if track_h > 0:
                    ratio = max(
                        0.0,
                        min(
                            1.0,
                            (my - top_bar_h - state["scrollbar_click_offset"])
                            / track_h,
                        ),
                    )
                    state["scroll_y"] = ratio * state["max_scroll"]
            elif (
                state["dragging"]
                and state["dragged_index"] != -1
                and 0 <= state["dragged_index"] < len(viewers)
            ):
                dx_pixels, dy_pixels = event.rel
                viewers[state["dragged_index"]].rotate_camera(
                    -dx_pixels * 0.005, -dy_pixels * 0.005
                )


def draw_ui_surface(
    win_w, win_h, state, viewers, subfolders, font, font_bold, mx, my, off_files_count
):
    sidebar_w = 200
    top_bar_h = 75
    padding = 25
    text_area_h = 45
    item_w = state.get("item_w", 180)
    item_h = item_w + text_area_h
    grid_w = win_w - sidebar_w
    grid_h = win_h - top_bar_h
    cols = max(1, (grid_w - padding) // (item_w + padding))

    ui_surf = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
    ui_surf.fill((0, 0, 0, 0))

    pygame.draw.rect(ui_surf, (255, 255, 255), (sidebar_w, top_bar_h, grid_w, grid_h))

    if state["fullscreen_index"] != -1:
        if 0 <= state["fullscreen_index"] < len(viewers):
            v = viewers[state["fullscreen_index"]]

            pygame.draw.rect(
                ui_surf,
                (255, 255, 255),
                (
                    sidebar_w + 10,
                    top_bar_h + 10,
                    grid_w - 20,
                    win_h - top_bar_h - 20,
                ),
            )
            pygame.draw.rect(
                ui_surf,
                (150, 150, 150),
                (
                    sidebar_w + 10,
                    top_bar_h + 10,
                    grid_w - 20,
                    win_h - top_bar_h - 20,
                ),
                1,
            )

            view_h = win_h - top_bar_h - 60
            report_active = bool(state.get("active_report"))
            viewport_w_hole = grid_w - 440 - 30 if report_active else grid_w - 22

            ui_surf.fill(
                (0, 0, 0, 0),
                (sidebar_w + 11, top_bar_h + 15, viewport_w_hole, view_h),
            )

            report_text = state.get("active_report")
            if report_text:
                report_w = 440
                report_x = win_w - report_w - 20
                report_y = top_bar_h + 20
                report_h = win_h - top_bar_h - 80

                pygame.draw.rect(
                    ui_surf,
                    (248, 249, 250),
                    (report_x, report_y, report_w, report_h),
                )
                pygame.draw.rect(
                    ui_surf,
                    (220, 224, 230),
                    (report_x, report_y, report_w, report_h),
                    1,
                )

                font_mono = pygame.font.SysFont("consolas", 12)
                lines = report_text.split("\n")
                ty = report_y + 10
                for line in lines:
                    if ty + 16 > report_y + report_h - 10:
                        break
                    txt_surface = font_mono.render(line, True, (40, 40, 40))
                    ui_surf.blit(txt_surface, (report_x + 15, ty))
                    ty += 16

            selected_metric = state.get("selected_sort_metric", "Default")
            label_x = sidebar_w + 30
            if selected_metric != "Default":
                lbl = font_bold.render(v.filename, True, (40, 40, 40))
                ui_surf.blit(lbl, (label_x, win_h - 45))
                m_val = format_sort_value(v.metrics.get(selected_metric, ""))
                m_lbl = font.render(
                    f"{selected_metric}: {m_val}", True, (100, 100, 100)
                )
                ui_surf.blit(m_lbl, (label_x, win_h - 25))
            else:
                lbl = font_bold.render(v.filename, True, (40, 40, 40))
                ui_surf.blit(lbl, (label_x, win_h - 38))

            active_face = state.get("active_face_index", -1)
            if active_face != -1:
                shape_name = "Custom Face"
                for shape_key, faces_list in v.face_types.items():
                    if active_face in faces_list:
                        gonality = shape_key[0]
                        shape_name = f"{gonality}-sided Face Shape"
                        break
                face_lbl = font_bold.render(
                    f"Displaying Representative {shape_name} (Face {active_face + 1})",
                    True,
                    (255, 50, 50),
                )
                ui_surf.blit(face_lbl, (label_x, top_bar_h + 30))

            active_vertex = state.get("active_vertex_index", -1)
            if active_vertex != -1:
                valence = len(v.vertex_to_original_faces[active_vertex])
                vertex_lbl = font_bold.render(
                    f"Displaying Representative Valence-{valence} Vertex (Vertex {active_vertex + 1})",
                    True,
                    (255, 50, 50),
                )
                ui_surf.blit(vertex_lbl, (label_x, top_bar_h + 30))

            active_part_idx = state.get("active_part_idx", -1)
            if active_part_idx != -1 and 0 <= active_part_idx < len(
                v.component_resources
            ):
                res = v.component_resources[active_part_idx]
                part_lbl = font_bold.render(
                    f"Displaying Compound Part {active_part_idx + 1}/{len(v.component_resources)} (V={res['num_v']}, F={res['num_f']})",
                    True,
                    (255, 50, 50),
                )
                ui_surf.blit(part_lbl, (label_x, top_bar_h + 30))

            prev_btn_rect = pygame.Rect(win_w - 210, win_h - 45, 40, 30)
            next_btn_rect = pygame.Rect(win_w - 160, win_h - 45, 40, 30)
            grid_btn_rect = pygame.Rect(win_w - 110, win_h - 45, 90, 30)

            if prev_btn_rect.collidepoint(mx, my):
                pygame.draw.rect(ui_surf, (220, 230, 245), prev_btn_rect)
                pygame.draw.rect(ui_surf, (100, 150, 220), prev_btn_rect, 1)
            else:
                pygame.draw.rect(ui_surf, (240, 240, 240), prev_btn_rect)
                pygame.draw.rect(ui_surf, (180, 180, 180), prev_btn_rect, 1)
            prev_lbl = font_bold.render("<", True, (50, 50, 50))
            ui_surf.blit(prev_lbl, prev_lbl.get_rect(center=prev_btn_rect.center))

            if next_btn_rect.collidepoint(mx, my):
                pygame.draw.rect(ui_surf, (220, 230, 245), next_btn_rect)
                pygame.draw.rect(ui_surf, (100, 150, 220), next_btn_rect, 1)
            else:
                pygame.draw.rect(ui_surf, (240, 240, 240), next_btn_rect)
                pygame.draw.rect(ui_surf, (180, 180, 180), next_btn_rect, 1)
            next_lbl = font_bold.render(">", True, (50, 50, 50))
            ui_surf.blit(next_lbl, next_lbl.get_rect(center=next_btn_rect.center))

            if grid_btn_rect.collidepoint(mx, my):
                pygame.draw.rect(ui_surf, (220, 230, 245), grid_btn_rect)
                pygame.draw.rect(ui_surf, (100, 150, 220), grid_btn_rect, 1)
            else:
                pygame.draw.rect(ui_surf, (240, 240, 240), grid_btn_rect)
                pygame.draw.rect(ui_surf, (180, 180, 180), grid_btn_rect, 1)
            grid_lbl = font_bold.render("Grid View", True, (50, 50, 50))
            ui_surf.blit(grid_lbl, grid_lbl.get_rect(center=grid_btn_rect.center))
    else:
        prev_val = None
        selected_metric = state.get("selected_sort_metric", "Default")

        for i, v in enumerate(viewers):
            row = i // cols
            col = i % cols

            x = sidebar_w + padding + col * (item_w + padding)
            y = top_bar_h + padding + row * (item_h + padding) - int(state["scroll_y"])

            current_val = (
                v.metrics.get(selected_metric, None)
                if selected_metric != "Default"
                else None
            )

            if y + item_h < top_bar_h or y > win_h:
                prev_val = current_val
                continue

            is_hov = i == state["hovered_index"]
            bg_col = (233, 244, 253, 255) if is_hov else (255, 255, 255, 255)
            border_col = (153, 201, 239) if is_hov else (229, 229, 229)

            pygame.draw.rect(ui_surf, bg_col, (x, y, item_w, item_h))
            pygame.draw.rect(ui_surf, border_col, (x, y, item_w, item_h), 1)

            role_y = max(top_bar_h, y + 1)
            role_h = min(win_h, y + item_w - 1) - role_y
            if role_h > 0:
                ui_surf.fill((0, 0, 0, 0), (x + 1, role_y, item_w - 2, role_h))

            pygame.draw.rect(ui_surf, border_col, (x, y, item_w, item_w), 1)

            is_equal = False
            if (
                selected_metric != "Default"
                and current_val is not None
                and prev_val is not None
                and current_val != ""
                and prev_val != ""
            ):
                if current_val == prev_val:
                    is_equal = True
                else:
                    try:
                        f1 = float(current_val)
                        f2 = float(prev_val)
                        if abs(f1 - f2) < SORT_TOLERANCE * abs(f2):
                            is_equal = True
                    except (ValueError, TypeError):
                        pass

            if selected_metric != "Default" and is_equal:
                shading_rect = pygame.Rect(
                    x + 5, y + item_w + 5, item_w - 10, text_area_h - 10
                )
                pygame.draw.rect(
                    ui_surf, (255, 215, 225), shading_rect, border_radius=4
                )

            if selected_metric != "Default":
                lbl = font.render(v.filename, True, (40, 40, 40))
                lbl_rect = lbl.get_rect(center=(x + item_w // 2, y + item_w + 15))
                if top_bar_h < lbl_rect.centery < win_h:
                    ui_surf.blit(lbl, lbl_rect)
                m_val = format_sort_value(v.metrics.get(selected_metric, ""))
                m_lbl = font.render(m_val, True, (100, 100, 100))
                m_rect = m_lbl.get_rect(center=(x + item_w // 2, y + item_w + 32))
                if top_bar_h < m_rect.centery < win_h:
                    ui_surf.blit(m_lbl, m_rect)
            else:
                lbl = font.render(v.filename, True, (40, 40, 40))
                lbl_rect = lbl.get_rect(
                    center=(x + item_w // 2, y + item_w + text_area_h // 2)
                )
                if top_bar_h < lbl_rect.centery < win_h:
                    ui_surf.blit(lbl, lbl_rect)

            prev_val = current_val

    pygame.draw.rect(ui_surf, (252, 252, 252), (0, 0, sidebar_w, win_h))
    pygame.draw.rect(ui_surf, (243, 243, 243), (sidebar_w, 0, grid_w, top_bar_h))
    pygame.draw.line(ui_surf, (220, 220, 220), (sidebar_w, 0), (sidebar_w, win_h), 1)
    pygame.draw.line(
        ui_surf, (220, 220, 220), (sidebar_w, top_bar_h), (win_w, top_bar_h), 1
    )

    suffix = "file" if off_files_count == 1 else "files"
    txt_ctrl = font_bold.render(f"{off_files_count} OFF {suffix}", True, (0, 0, 0))
    ui_surf.blit(txt_ctrl, (15, 15))

    pygame.draw.rect(ui_surf, (230, 230, 230), (15, 35, 12, 12))
    pygame.draw.rect(ui_surf, (150, 150, 150), (15, 35, 12, 12), 1)
    if state["auto_rotate"]:
        pygame.draw.rect(ui_surf, (0, 102, 204), (18, 38, 6, 6))
    txt_rotate = font.render("Auto-rotate Models", True, (50, 50, 50))
    ui_surf.blit(txt_rotate, (35, 33))

    txt_sort_lbl = font.render("Sort models by:", True, (120, 120, 120))
    ui_surf.blit(txt_sort_lbl, (15, 58))

    bg_drop = (255, 255, 255) if state["sort_dropdown_enabled"] else (240, 240, 240)
    border_drop = (180, 180, 180) if state["sort_dropdown_enabled"] else (220, 220, 220)
    text_drop_col = (50, 50, 50) if state["sort_dropdown_enabled"] else (160, 160, 160)

    pygame.draw.rect(ui_surf, bg_drop, (15, 75, 170, 22))
    pygame.draw.rect(ui_surf, border_drop, (15, 75, 170, 22), 1)

    curr_val_txt = font.render(state["selected_sort_metric"], True, text_drop_col)
    ui_surf.blit(curr_val_txt, (22, 78))

    arrow_col = (100, 100, 100) if state["sort_dropdown_enabled"] else (180, 180, 180)
    pygame.draw.polygon(ui_surf, arrow_col, [(170, 83), (176, 83), (173, 88)])

    pygame.draw.rect(ui_surf, (230, 230, 230), (15, 105, 12, 12))
    pygame.draw.rect(ui_surf, (150, 150, 150), (15, 105, 12, 12), 1)
    txt_del_dup = font.render("Delete Duplicates", True, (50, 50, 50))
    ui_surf.blit(txt_del_dup, (35, 103))

    refresh_rect = pygame.Rect(15, 125, 170, 24)
    if refresh_rect.collidepoint(mx, my):
        pygame.draw.rect(ui_surf, (220, 230, 245), refresh_rect, border_radius=4)
        pygame.draw.rect(ui_surf, (100, 150, 220), refresh_rect, 1, border_radius=4)
    else:
        pygame.draw.rect(ui_surf, (240, 240, 240), refresh_rect, border_radius=4)
        pygame.draw.rect(ui_surf, (180, 180, 180), refresh_rect, 1, border_radius=4)
    refresh_txt = font_bold.render("Refresh Files", True, (50, 50, 50))
    ui_surf.blit(refresh_txt, refresh_txt.get_rect(center=refresh_rect.center))

    panel_title = font_bold.render("Subfolders Panel", True, (0, 0, 0))
    ui_surf.blit(panel_title, (15, 160))

    sub_clip_rect = pygame.Rect(10, 185, sidebar_w - 20, win_h - 195)
    clip_prev = ui_surf.get_clip()
    ui_surf.set_clip(sub_clip_rect)

    sy = 185 - int(state.get("subfolder_scroll_y", 0))
    for i, item in enumerate(subfolders):
        is_parent = item == ".."
        display_name = "[Parent Folder] .." if is_parent else f"📁 {item}"
        color = (0, 102, 204) if is_parent else (50, 50, 50)

        if (
            mx <= sidebar_w
            and sy <= my <= sy + 22
            and sub_clip_rect.collidepoint(mx, my)
        ):
            if (
                not state["dropdown_open"]
                or my < 102
                or my > 102 + min(350, win_h - 120)
            ):
                pygame.draw.rect(
                    ui_surf, (230, 240, 250), (10, sy - 2, sidebar_w - 20, 22)
                )
                pygame.draw.rect(
                    ui_surf,
                    (180, 200, 230),
                    (10, sy - 2, sidebar_w - 20, 22),
                    1,
                )

        txt = font.render(display_name, True, color)
        ui_surf.blit(txt, (20, sy))
        sy += 25

    ui_surf.set_clip(clip_prev)

    if state.get("max_subfolder_scroll", 0) > 0:
        sb_x = sidebar_w - 10
        sb_y = 185
        sb_w = 5
        sb_h = win_h - 195
        pygame.draw.rect(ui_surf, (245, 245, 245), (sb_x, sb_y, sb_w, sb_h))
        bar_h = max(
            15,
            int((sb_h / (state["max_subfolder_scroll"] + sb_h)) * sb_h),
        )
        bar_y = sb_y + int(
            (state["subfolder_scroll_y"] / state["max_subfolder_scroll"])
            * (sb_h - bar_h)
        )
        pygame.draw.rect(
            ui_surf, (200, 200, 200), (sb_x, bar_y, sb_w, bar_h), border_radius=3
        )

    pygame.draw.rect(ui_surf, (255, 255, 255), (sidebar_w + 15, 30, grid_w - 150, 30))
    pygame.draw.rect(
        ui_surf, (200, 200, 200), (sidebar_w + 15, 30, grid_w - 150, 30), 1
    )
    addr_txt = font.render(f"  {state['current_folder']}", True, (80, 80, 80))
    ui_surf.blit(addr_txt, (sidebar_w + 20, 36))

    search_rect = pygame.Rect(win_w - 115, 30, 100, 30)
    pygame.draw.rect(ui_surf, (255, 255, 255), search_rect)
    pygame.draw.rect(ui_surf, (200, 200, 200), search_rect, 1)

    display_search = state.get("search_query", "")
    if not display_search:
        search_txt = font.render("Search", True, (150, 150, 150))
    else:
        search_txt = font.render(display_search, True, (0, 102, 204))
    ui_surf.blit(search_txt, (win_w - 105, 36))

    if state["fullscreen_index"] == -1 and state["max_scroll"] > 0:
        bar_h = max(
            30,
            int((grid_h / (state["max_scroll"] + grid_h)) * grid_h),
        )
        bar_y = top_bar_h + int(
            (state["scroll_y"] / state["max_scroll"]) * (grid_h - bar_h)
        )
        pygame.draw.rect(ui_surf, (240, 240, 240), (win_w - 12, top_bar_h, 12, grid_h))
        pygame.draw.rect(
            ui_surf,
            (200, 200, 200),
            (win_w - 10, bar_y, 8, bar_h),
            border_radius=4,
        )

    if state["dropdown_open"] and state["sort_dropdown_enabled"]:
        all_options = ["Default"] + state["sort_headers"]
        dropdown_max_h = min(350, win_h - 120)
        opt_h = 20

        pygame.draw.rect(ui_surf, (255, 255, 255), (15, 102, 170, dropdown_max_h))
        pygame.draw.rect(ui_surf, (150, 150, 150), (15, 102, 170, dropdown_max_h), 1)

        clip_prev = ui_surf.get_clip()
        ui_surf.set_clip(pygame.Rect(16, 103, 168, dropdown_max_h - 2))

        for idx, opt in enumerate(all_options):
            oy = 102 + idx * opt_h - state["dropdown_scroll"]
            if oy + opt_h < 102 or oy > 102 + dropdown_max_h:
                continue

            is_opt_hov = pygame.Rect(15, oy, 170, opt_h).collidepoint(
                mx, my
            ) and pygame.Rect(15, 102, 170, dropdown_max_h).collidepoint(mx, my)
            if is_opt_hov:
                pygame.draw.rect(ui_surf, (230, 240, 250), (16, oy, 168, opt_h))

            opt_txt = font.render(opt, True, (40, 40, 40))
            ui_surf.blit(opt_txt, (22, oy + 3))

        ui_surf.set_clip(clip_prev)

    if state["context_menu_open"]:
        cx, cy = state["context_menu_pos"]
        num_tools = len(state.get("external_tools", []))
        menu_h = 60 + num_tools * 20

        prompts = ["Rename", "Delete", "View Source"] + [
            t["prompt"] for t in state.get("external_tools", [])
        ]
        menu_w = max(font.size(p)[0] for p in prompts) + 25
        state["context_menu_width"] = menu_w

        pygame.draw.rect(ui_surf, (255, 255, 255), (cx, cy, menu_w, menu_h))
        pygame.draw.rect(ui_surf, (150, 150, 150), (cx, cy, menu_w, menu_h), 1)

        hovered_row = (my - cy) // 20
        if 0 <= hovered_row < (3 + num_tools) and pygame.Rect(
            cx, cy, menu_w, menu_h
        ).collidepoint(mx, my):
            pygame.draw.rect(
                ui_surf,
                (230, 240, 250),
                (cx + 1, cy + hovered_row * 20 + 1, menu_w - 2, 18),
            )

        txt_rename = font.render("Rename", True, (40, 40, 40))
        txt_delete = font.render("Delete", True, (40, 40, 40))
        txt_view_source = font.render("View Source", True, (40, 40, 40))

        ui_surf.blit(txt_rename, (cx + 10, cy + 3))
        ui_surf.blit(txt_delete, (cx + 10, cy + 23))
        ui_surf.blit(txt_view_source, (cx + 10, cy + 43))

        for idx, tool in enumerate(state.get("external_tools", [])):
            txt_tool = font.render(tool["prompt"], True, (40, 40, 40))
            ui_surf.blit(txt_tool, (cx + 10, cy + 63 + idx * 20))

    return ui_surf


def main():
    pygame.init()
    pygame.font.init()

    win_w, win_h = 1300, 850

    try:
        screen_w, screen_h = 1920, 1080
        if sys.platform == "win32":
            import ctypes

            screen_w = ctypes.windll.user32.GetSystemMetrics(0)
            screen_h = ctypes.windll.user32.GetSystemMetrics(1)
        else:
            info = pygame.display.Info()
            screen_w = info.current_w
            screen_h = info.current_h

        x_pos = max(0, (screen_w - win_w) // 2)
        y_pos = max(0, (screen_h - win_h) // 2 - 30)
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x_pos},{y_pos}"
    except Exception as e:
        print(f"Error setting window position: {e}")

    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    pygame.display.set_mode(
        (win_w, win_h), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    )

    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.depth_func = "<="
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    try:
        import ctypes

        if sys.platform == "win32":
            ctypes.windll.opengl32.glEnable(32823)
        else:
            for name in [
                "libGL.so.1",
                "libGL.so",
                "/System/Library/Frameworks/OpenGL.framework/OpenGL",
            ]:
                try:
                    ctypes.CDLL(name).glEnable(32823)
                    break
                except Exception:
                    pass
    except Exception:
        pass

    ctx.polygon_offset = (1.0, 1.0)

    prog_3d, prog_2d, quad_vao = init_gl_resources(ctx)

    font = pygame.font.SysFont("segoeui", 12)
    font_bold = pygame.font.SysFont("segoeui", 13, bold=True)

    current_folder = os.path.abspath(".")
    target_file = None
    fullscreen_index = -1

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.isfile(arg):
            current_folder = os.path.dirname(os.path.abspath(arg))
            target_file = os.path.basename(arg)
        elif os.path.isdir(arg):
            current_folder = os.path.abspath(arg)

    viewers = load_viewers_from_folder(current_folder, ctx, prog_3d)
    subfolders = get_subfolders(current_folder)

    state = {
        "current_folder": current_folder,
        "target_file": target_file,
        "fullscreen_index": fullscreen_index,
        "active_face_index": -1,
        "active_rep_idx": -1,
        "active_vertex_index": -1,
        "active_vertex_rep_idx": -1,
        "active_part_idx": -1,
        "edge_mode": 0,
        "grid_edges_on": False,
        "active_report": None,
        "scroll_y": 0.0,
        "subfolder_scroll_y": 0.0,
        "max_scroll": 0,
        "max_subfolder_scroll": 0,
        "running": True,
        "hovered_index": -1,
        "last_click_time": 0,
        "dragging": False,
        "dragged_index": -1,
        "scrollbar_dragging": False,
        "scrollbar_click_offset": 0,
        "auto_rotate": False,
        "sort_dropdown_enabled": False,
        "sort_headers": [],
        "selected_sort_metric": "Default",
        "dropdown_open": False,
        "dropdown_scroll": 0,
        "context_menu_open": False,
        "context_menu_index": -1,
        "context_menu_pos": (0, 0),
        "needs_redraw": 2,
        "checker_proc": None,
        "item_w": 180,
        "all_viewers": list(viewers),
        "search_query": "",
        "external_tools": load_external_tools(),
    }

    if target_file:
        for idx, v in enumerate(viewers):
            if v.filename == target_file:
                fullscreen_index = idx
                state["fullscreen_index"] = idx
                run_single_analysis(v.filepath, state)
                break

    sidebar_w = 200
    top_bar_h = 75
    padding = 25
    text_area_h = 45

    clock = pygame.time.Clock()
    run_bg_analysis(state["current_folder"], state["all_viewers"])
    last_folder_poll = 0

    while state["running"]:
        dt = clock.tick(60)
        dt = min(100, dt)

        now_ticks = pygame.time.get_ticks()
        if now_ticks - last_folder_poll >= FOLDER_POLL_INTERVAL_MS:
            last_folder_poll = now_ticks
            poll_folder_changes(state, viewers, ctx, prog_3d)

        mx, my = pygame.mouse.get_pos()
        keys = pygame.key.get_pressed()
        keyboard_active = any(keys[k] for k in (K_LEFT, K_RIGHT, K_UP, K_DOWN))

        if (
            state["auto_rotate"]
            or keyboard_active
            or bg_csv_ready[0]
            or state["dragging"]
            or state["scrollbar_dragging"]
        ):
            state["needs_redraw"] = 2

        events = pygame.event.get()
        if len(events) > 0:
            state["needs_redraw"] = 2

        if state.get("pending_reload"):
            reload_path = state.pop("pending_reload")
            prog_dir = os.path.dirname(os.path.abspath(__file__))
            folder_hash = hashlib.md5(
                os.path.abspath(state["current_folder"]).encode("utf-8")
            ).hexdigest()
            off_files = [
                f
                for f in os.listdir(state["current_folder"])
                if f.lower().endswith(".off")
            ]
            use_cache = len(off_files) > CACHE_THRESHOLD
            cache_dir = (
                os.path.join(prog_dir, ".cache", folder_hash) if use_cache else None
            )

            new_v = ModelViewer(
                reload_path,
                ctx,
                prog_3d,
                use_cache=use_cache,
                cache_dir=cache_dir,
            )
            if new_v.valid:
                for idx, v in enumerate(state["all_viewers"]):
                    if os.path.abspath(v.filepath) == os.path.abspath(reload_path):
                        state["all_viewers"][idx] = new_v
                        break
                for i, f_v in enumerate(viewers):
                    if os.path.abspath(f_v.filepath) == os.path.abspath(reload_path):
                        viewers[i] = new_v
                        break
            run_bg_analysis(state["current_folder"], state["all_viewers"])

        handle_events(events, state, viewers, subfolders, ctx, prog_3d)

        win_w, win_h = pygame.display.get_window_size()
        fb_w, fb_h = win_w, win_h
        sx = 1.0
        sy = 1.0

        grid_w = win_w - sidebar_w
        grid_h = win_h - top_bar_h

        item_w = state.get("item_w", 180)
        item_h = item_w + text_area_h

        cols = max(1, (grid_w - padding) // (item_w + padding))
        total_rows = math.ceil(len(viewers) / cols)
        state["max_scroll"] = max(0, total_rows * (item_h + padding) + padding - grid_h)
        state["scroll_y"] = max(0.0, min(state["scroll_y"], state["max_scroll"]))

        sub_panel_h = win_h - 195
        state["max_subfolder_scroll"] = max(0, len(subfolders) * 25 - sub_panel_h)
        state["subfolder_scroll_y"] = max(
            0.0, min(state["subfolder_scroll_y"], state["max_subfolder_scroll"])
        )

        try:
            off_files_count = len(
                [
                    f
                    for f in os.listdir(state["current_folder"])
                    if f.lower().endswith(".off")
                ]
            )
        except Exception:
            off_files_count = 0

        if bg_csv_ready[0]:
            csv_path = bg_csv_path[0]
            bg_csv_ready[0] = False
            try:
                if csv_path and os.path.exists(csv_path):
                    with open(csv_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if reader.fieldnames:
                            state["sort_headers"] = [
                                h
                                for h in reader.fieldnames
                                if h not in ("Filename", "Detail")
                            ]
                            metrics_map = {row["Filename"]: row for row in rows}
                            for v in state["all_viewers"]:
                                v.metrics = metrics_map.get(v.filename, {})
                            apply_search_and_sort(state, viewers)
                            state["sort_dropdown_enabled"] = True
                            state["needs_redraw"] = 2
            except Exception as e:
                print("Error loading CSV results:", e)

        if state["needs_redraw"] > 0:
            if state["fullscreen_index"] != -1:
                pygame.display.set_caption("Enlarged View")
            else:
                pygame.display.set_caption("Grid View")

            active_viewer = None
            if state["fullscreen_index"] != -1:
                if 0 <= state["fullscreen_index"] < len(viewers):
                    active_viewer = viewers[state["fullscreen_index"]]
            elif state["hovered_index"] != -1:
                if 0 <= state["hovered_index"] < len(viewers):
                    active_viewer = viewers[state["hovered_index"]]

            if active_viewer and not state["dragging"]:
                rot_val = 0.003 * dt
                dx = 0.0
                dy = 0.0
                if keys[K_LEFT]:
                    dx = -rot_val
                if keys[K_RIGHT]:
                    dx = rot_val
                if keys[K_UP]:
                    dy = -rot_val
                if keys[K_DOWN]:
                    dy = rot_val
                active_viewer.rotate_camera(dx, dy)

            if state["auto_rotate"]:
                rot_val = AUTO_ROTATION_SPEED * dt
                for v in viewers:
                    v.rotate_camera(rot_val, 0.0)

            state["hovered_index"] = -1
            if state["fullscreen_index"] == -1:
                for i in range(len(viewers)):
                    row = i // cols
                    col = i % cols
                    x = sidebar_w + padding + col * (item_w + padding)
                    y = (
                        top_bar_h
                        + padding
                        + row * (item_h + padding)
                        - int(state["scroll_y"])
                    )
                    if x <= mx <= x + item_w and y <= my <= y + item_h:
                        if mx > sidebar_w and my > top_bar_h:
                            state["hovered_index"] = i
                            break

            ctx.clear(1.0, 1.0, 1.0, 1.0)

            if state["fullscreen_index"] != -1:
                if 0 <= state["fullscreen_index"] < len(viewers):
                    v = viewers[state["fullscreen_index"]]

                    viewport_x = sidebar_w + 10
                    v_y_gl = 45

                    report_active = bool(state.get("active_report"))
                    viewport_w = (
                        win_w - viewport_x - 440 - 30
                        if report_active
                        else win_w - viewport_x - 20
                    )
                    viewport_h = win_h - top_bar_h - 60

                    ctx.scissor = (
                        int(viewport_x * sx),
                        int(v_y_gl * sy),
                        int(viewport_w * sx),
                        int(viewport_h * sy),
                    )
                    v.render_3d(
                        (
                            int(viewport_x * sx),
                            int(v_y_gl * sy),
                            int(viewport_w * sx),
                            int(viewport_h * sy),
                        ),
                        viewport_w / max(1, viewport_h),
                        state["active_face_index"],
                        0.0,
                        state.get("active_vertex_index", -1),
                        state.get("edge_mode", 0),
                        state.get("active_part_idx", -1),
                    )
            else:
                for i, v in enumerate(viewers):
                    row = i // cols
                    col = i % cols

                    x = sidebar_w + padding + col * (item_w + padding)
                    y_from_top = (
                        top_bar_h
                        + padding
                        + row * (item_h + padding)
                        - int(state["scroll_y"])
                    )
                    y_from_bottom = win_h - (y_from_top + item_h)

                    if y_from_top + item_h < top_bar_h or y_from_top > win_h:
                        continue

                    grid_bottom_gl = 0
                    grid_top_gl = win_h - top_bar_h

                    v_y_bottom = y_from_bottom + text_area_h
                    v_y_top = y_from_bottom + item_h

                    clamped_bottom = max(grid_bottom_gl, v_y_bottom)
                    clamped_top = min(grid_top_gl, v_y_top)

                    clamped_h = clamped_top - clamped_bottom
                    if clamped_h > 0:
                        ctx.scissor = (
                            int((x + 1) * sx),
                            int(clamped_bottom * sy),
                            int((item_w - 2) * sx),
                            int(clamped_h * sy),
                        )
                        grid_edge_mode = 2 if state.get("grid_edges_on", False) else 0
                        v.render_3d(
                            (
                                int((x + 1) * sx),
                                int(v_y_bottom * sy),
                                int((item_w - 2) * sx),
                                int(item_w * sy),
                            ),
                            sx / sy,
                            edge_mode=grid_edge_mode,
                        )

            ctx.scissor = None
            ctx.viewport = (0, 0, int(fb_w), int(fb_h))

            ui_surf = draw_ui_surface(
                win_w,
                win_h,
                state,
                viewers,
                subfolders,
                font,
                font_bold,
                mx,
                my,
                off_files_count,
            )

            ui_data = pygame.image.tostring(ui_surf, "RGBA", True)
            ui_tex = ctx.texture((win_w, win_h), 4, ui_data)
            ui_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            ui_tex.use(0)

            quad_vao.render(moderngl.TRIANGLES)

            pygame.display.flip()
            ui_tex.release()

            state["needs_redraw"] = max(0, state["needs_redraw"] - 1)
        else:
            pygame.time.wait(15)

    if state.get("checker_proc"):
        try:
            state["checker_proc"].terminate()
        except Exception:
            pass

    if bg_csv_path[0] and os.path.exists(bg_csv_path[0]):
        try:
            os.remove(bg_csv_path[0])
        except Exception:
            pass

    pygame.quit()


if __name__ == "__main__":
    main()
