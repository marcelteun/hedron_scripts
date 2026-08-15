# Version 5.20
# Addons: pygame, moderngl, numpy
#
"""
HELP TEXT
When you open a grid window the sort mode will be greyed out while it calculates the metrics for the off files.  Once it has done this and you select a sort order, duplicate values of the sort parameter are highlighted in pink.  The delete duplicates button will keep the oldest file with any one value.   Float parameters have a small tolerance built in.

- Auto rotate is fun but wait until the sort button blacks up or you'll slow it down.
- + (or =) and - will zoom the cells on the grid screen-
- Any polyhedron can be rotated by left dragging with the mouse ot=r by using the arrow keys
- Double click on an image to get to the enlarged screen.
  - The left and right arrows are previous/next file.
  - 'f' on the enlarged screen will cycle through the face types one by one.
  - 'v' cycles through the vertices
  - 'e' does some edge options (also on the grid view)
  - 'x' to return to the original view
- The grid view also has some functions if you right click.
  - delete, rename, view source, open in Stella.
- The offcheck.py file is also a standalone checker.  If you change the metrics that this is calculating they should flow through into offviewer automatically.
- Files offcheck.py and facetings_math.py are required to be in the same folder as this file.
Change log:
140826 - The path and 'search' at the top of the grid screen now work.
       - Symmetry is enabled in the offcheck link.  See variable below to switch it off if required.
       - 'f' mode on the enlarged view shows coplanar faces to the focus face as translucent.  Use ALPHA to control.
       - 'f' now cycles through face types rather than all faces
       - 'e' now also works as an edge on/off switch in grid view

"""

import os
import sys
import math
import numpy as np
import pygame
from pygame.locals import *
import moderngl
import threading
import csv
import re
import tkinter as tk
import fnmatch

# from numba import njit
from tkinter import simpledialog, messagebox
from collections import defaultdict

# Configuration
AUTO_ROTATION_SPEED = -0.0002
SORT_TOLERANCE = 0.0001
RUN_SYMMETRY_CALCULATION = (
    False  # Set to True to enable symmetry calculation in offcheck
)
ALPHA = 0.2  # Transparency value for coplanar faces in Enlarged view

# Thread-safe signals for background analysis
bg_csv_ready = [False]
bg_csv_path = [None]
bg_run_id = [0]


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


def parse_color(parts, default_color=[178, 204, 230, 255]):
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
    except:
        return default_color


# @njit(cache=True)
def seg_intersect(A, B, C, D):
    denom = (B[0] - A[0]) * (D[1] - C[1]) - (B[1] - A[1]) * (D[0] - C[0])
    if abs(denom) < 1e-9:
        return None
    t = ((C[0] - A[0]) * (D[1] - C[1]) - (C[1] - A[1]) * (D[0] - C[0])) / denom
    u = ((C[0] - A[0]) * (B[1] - A[1]) - (C[1] - A[1]) * (B[0] - A[0])) / denom
    if 1e-9 < t < 1.0 - 1e-9 and 1e-9 < u < 1.0 - 1e-9:
        return t, u
    return None


# @njit(cache=True)
def get_crossing_count(point, poly_2d):
    x, y = point
    crossings = 0
    n = len(poly_2d)
    for i in range(n):
        p1 = poly_2d[i]
        p2 = poly_2d[(i + 1) % n]
        if (p1[1] <= y and p2[1] > y) or (p2[1] <= y and p1[1] > y):
            if x < p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]):
                crossings += 1
    return crossings


def triangulate_simple_polygon(indices, verts_2d):
    n = len(indices)
    if n < 3:
        return []
    if n == 3:
        return [indices]

    poly = list(indices)
    triangles = []

    def is_ccw(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]) > 0

    def is_inside_triangle(p, a, b, c):
        v0 = np.array(c) - np.array(a)
        v1 = np.array(b) - np.array(a)
        v2 = np.array(p) - np.array(a)
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)
        denom_val = dot00 * dot11 - dot01 * dot01
        invDenom = 1 / denom_val if denom_val != 0 else 1.0
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom
        return (u >= -1e-9) and (v >= -1e-9) and (u + v <= 1.0 + 1e-9)

    limit = 100
    while len(poly) > 3 and limit > 0:
        limit -= 1
        ear_found = False
        for i in range(len(poly)):
            prev_idx = poly[i - 1]
            curr_idx = poly[i]
            next_idx = poly[(i + 1) % len(poly)]

            p_prev = verts_2d[prev_idx]
            p_curr = verts_2d[curr_idx]
            p_next = verts_2d[next_idx]

            if is_ccw(p_prev, p_curr, p_next):
                any_inside = False
                for idx in poly:
                    if idx in (prev_idx, curr_idx, next_idx):
                        continue
                    if is_inside_triangle(verts_2d[idx], p_prev, p_curr, p_next):
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
            return None, None, None, None, None

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
            return None, None, None, None, None

        num_verts = int(parts[0])
        num_faces = int(parts[1])

        vertices = []
        vert_colors = []

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

        faces = []
        face_colors = []
        triangle_to_face_map = []
        global_inter_map = {}

        for i in range(num_faces):
            line_parts = lines[start_idx + num_verts + i].split()
            n_v = int(line_parts[0])
            face_verts = [int(x) for x in line_parts[1 : 1 + n_v]]

            color_parts = line_parts[1 + n_v :]
            f_color = (
                parse_color(color_parts, default_color=None) if color_parts else None
            )

            if n_v < 3:
                continue

            pts_3d = [np.array(vertices[idx], dtype="f4") for idx in face_verts]

            N = np.zeros(3, dtype="f4")
            for k in range(n_v):
                p_curr = pts_3d[k]
                p_next = pts_3d[(k + 1) % n_v]
                N[0] += (p_curr[1] - p_next[1]) * (p_curr[2] + p_next[2])
                N[1] += (p_curr[2] - p_next[2]) * (p_curr[0] + p_next[0])
                N[2] += (p_curr[0] - p_next[0]) * (p_curr[1] + p_next[1])

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
                U = pts_3d[2] - pts_3d[0]
                U = U - np.dot(U, N) * N
                norm_U = np.linalg.norm(U)

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
                    [float(np.dot(p - pts_3d[0], U)), float(np.dot(p - pts_3d[0], V))]
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
                        [face_verts[tri[0]], face_verts[tri[1]], face_verts[tri[2]]]
                    )
                    face_colors.append(f_color)
                    triangle_to_face_map.append(i)
                continue

            # Planar Graph Traversal Solver
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

                def get_angle(v):
                    diff = local_vertices[v]["2d"] - local_vertices[u]["2d"]
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
                    u_idx = v_neighbors.index(u)
                    next_neighbor = v_neighbors[(u_idx - 1) % len(v_neighbors)]
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

        return (
            np.array(vertices, dtype="f4"),
            np.array(faces, dtype="i4"),
            vert_colors,
            face_colors,
            triangle_to_face_map,
        )
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None, None, None


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
    def __init__(self, filepath, ctx, program_3d):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.ctx = ctx
        self.program = program_3d
        self.metrics = {}

        self.rot_mat = np.eye(3, dtype="f4")

        verts, faces, vert_colors, face_colors, tri_to_face = load_off(filepath)
        if verts is None or faces is None or len(faces) == 0 or len(verts) == 0:
            self.valid = False
            return
        self.valid = True

        centroid = np.mean(verts, axis=0)
        verts = verts - centroid
        max_span = np.max(np.linalg.norm(verts, axis=1))
        if max_span > 0:
            verts = verts / max_span

        tri_verts = verts[faces]
        v0 = tri_verts[:, 0, :]
        v1 = tri_verts[:, 1, :]
        v2 = tri_verts[:, 2, :]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normals = normals / norms

        num_triangles = len(faces)
        vbo_array = np.zeros((num_triangles, 3, 10), dtype="f4")
        vbo_array[:, :, 0:3] = verts[faces]
        vbo_array[:, :, 3:7] = default_colors = np.array(vert_colors, dtype="f4")[faces]
        vbo_array[:, :, 7:10] = normals[:, np.newaxis, :]

        face_colors_clean = [fc for fc in face_colors if fc is not None]
        if len(face_colors_clean) > 0:
            for i, f_col in enumerate(face_colors):
                if f_col is not None:
                    vbo_array[i, :, 3:7] = f_col

        self.vbo_data = vbo_array.ravel()
        self.vbo = ctx.buffer(self.vbo_data.tobytes())
        self.vao = ctx.vertex_array(
            self.program,
            [(self.vbo, "3f 4f 3f", "in_position", "in_color", "in_normal")],
        )

        # Track rendering vertex ranges for original polyhedron faces
        self.face_ranges = {}
        current_vertex = 0
        for i in range(len(faces)):
            orig_face_idx = tri_to_face[i]
            if orig_face_idx not in self.face_ranges:
                self.face_ranges[orig_face_idx] = [current_vertex, 0]
            self.face_ranges[orig_face_idx][1] += 3
            current_vertex += 3
        self.num_original_faces = len(self.face_ranges)

        # Precompute face planes for coplanar detection
        self.face_planes = {}
        for orig_face_idx, (start_v, count_v) in self.face_ranges.items():
            tri_idx = start_v // 3
            norm = normals[tri_idx]
            pt = v0[tri_idx]
            d = np.dot(norm, pt)
            self.face_planes[orig_face_idx] = (norm, d)

        # Track original faces adjacent to each vertex
        self.vertex_to_original_faces = {v_idx: set() for v_idx in range(len(verts))}
        for i in range(len(faces)):
            orig_face_idx = tri_to_face[i]
            for j in range(3):
                v_idx = faces[i][j]
                self.vertex_to_original_faces[v_idx].add(orig_face_idx)
        self.num_vertices = len(verts)

        # Track face triangulations to extract boundary edges
        face_triangles = defaultdict(list)
        for i in range(len(faces)):
            orig_face_idx = tri_to_face[i]
            face_triangles[orig_face_idx].append(faces[i])

        self.face_boundary_edges = {}
        for orig_face_idx, tris in face_triangles.items():
            edge_counts = defaultdict(int)
            for tri in tris:
                for u, v in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                    edge_key = tuple(sorted((u, v)))
                    edge_counts[edge_key] += 1
            self.face_boundary_edges[orig_face_idx] = [
                edge for edge, count in edge_counts.items() if count == 1
            ]

        all_edges_set = set()
        for edges in self.face_boundary_edges.values():
            all_edges_set.update(edges)
        self.all_edges = list(all_edges_set)

        # Precompute face types based on shape signature
        self.verts = verts
        self.face_types = defaultdict(list)
        for orig_face_idx in range(self.num_original_faces):
            # Calculate edge lengths from boundary edges
            edge_lengths = []
            for u, v in self.face_boundary_edges[orig_face_idx]:
                edge_lengths.append(np.linalg.norm(self.verts[u] - self.verts[v]))
            sorted_lens = tuple(round(float(l), 4) for l in sorted(edge_lengths))

            # Calculate area of the original face by summing tri areas
            area = 0.0
            for tri in face_triangles[orig_face_idx]:
                p0 = self.verts[tri[0]]
                p1 = self.verts[tri[1]]
                p2 = self.verts[tri[2]]
                area += 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
            rounded_area = round(float(area), 4)

            gonality = len(self.face_boundary_edges[orig_face_idx])
            shape_key = (gonality, sorted_lens, rounded_area)
            self.face_types[shape_key].append(orig_face_idx)

        self.face_type_keys = list(self.face_types.keys())

        # Select the first face of each face type as its representative
        self.representative_faces = [
            self.face_types[key][0] for key in self.face_type_keys
        ]

        # Store scaled verts and sphere mesh local data
        self.sphere_local_verts, self.sphere_faces = create_sphere_mesh(
            radius=0.015, rings=8, sectors=8
        )

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
    ):
        if not self.valid:
            return
        self.ctx.viewport = viewport

        eye = self.rot_mat @ np.array([0.0, 0.0, 3.0], dtype="f4")
        up = self.rot_mat @ np.array([0.0, 1.0, 0.0], dtype="f4")

        # Calculate horizontal camera pan offset relative to camera frame
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

        # Set alpha override back to default solid state
        if "u_alpha_override" in self.program:
            self.program["u_alpha_override"].value = -1.0

        if active_face_index != -1 and active_face_index in self.face_ranges:
            # Render coplanar faces as translucent
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
                    self.vao.render(moderngl.TRIANGLES, vertices=count_v, first=start_v)
                self.ctx.depth_write = True

            # Render focus face solid
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
                    self.vao.render(moderngl.TRIANGLES, vertices=count_v, first=start_v)
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
                        if e[0] == active_vertex_index or e[1] == active_vertex_index
                    ]
                elif edge_mode == 2:
                    edges_to_draw = self.all_edges
            elif active_face_index != -1:
                if edge_mode == 1:
                    edges_to_draw = self.face_boundary_edges.get(active_face_index, [])
                elif edge_mode == 2:
                    edges_to_draw = self.all_edges
            else:
                edges_to_draw = self.all_edges

            self.render_edges(edges_to_draw, self.verts)


def run_bg_analysis(folder, viewers_list):
    bg_run_id[0] += 1
    my_run_id = bg_run_id[0]

    def worker():
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("offcheck", "offcheck.py")
            if spec is None:
                return
            offcheck = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(offcheck)
            verify_off_logic = offcheck.verify_off_logic
        except Exception as e:
            print(f"Failed to load offcheck for background analysis: {e}")
            return

        results = []
        all_keys = set()

        for v in viewers_list:
            if bg_run_id[0] != my_run_id:
                return
            fpath = os.path.abspath(v.filepath)
            res, stats = verify_off_logic(
                fpath, return_stats=True, run_symmetry=RUN_SYMMETRY_CALCULATION
            )
            if stats is not None:
                results.append(stats)
                all_keys.update(stats.keys())
            else:
                basic_stats = {"Filename": os.path.basename(fpath), "Detail": res}
                results.append(basic_stats)
                all_keys.update(basic_stats.keys())

        if bg_run_id[0] != my_run_id:
            return

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
            import importlib.util

            spec = importlib.util.spec_from_file_location("offcheck", "offcheck.py")
            if spec is None:
                state["active_report"] = "Failed to load offcheck.py"
                state["needs_redraw"] = 2
                return
            offcheck = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(offcheck)
            verify_off_logic = offcheck.verify_off_logic

            report = verify_off_logic(
                os.path.abspath(filepath), run_symmetry=RUN_SYMMETRY_CALCULATION
            )
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
        for f in sorted(off_files):
            v = ModelViewer(os.path.join(folder, f), ctx, prog_3d)
            if v.valid:
                viewers_list.append(v)
    except Exception as e:
        print(f"Error scanning folder {folder}: {e}")
    return viewers_list


def get_subfolders(folder):
    subfolders_list = []
    try:
        parent = os.path.dirname(folder)
        if parent and parent != folder:
            subfolders_list.append("..")
        for d in sorted(os.listdir(folder)):
            if os.path.isdir(os.path.join(folder, d)) and not d.startswith("."):
                subfolders_list.append(d)
    except Exception as e:
        print(f"Error listing subfolders: {e}")
    return subfolders_list


def apply_search_and_sort(state, viewers):
    all_v = state.get("all_viewers", [])
    query = state.get("search_query", "").strip()
    if query:
        filtered = []
        for v in all_v:
            if fnmatch.fnmatch(v.filename.lower(), query.lower()):
                filtered.append(v)
            elif fnmatch.fnmatch(
                os.path.splitext(v.filename)[0].lower(), query.lower()
            ):
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


# Version 5.22


def handle_events(events, state, viewers, subfolders, ctx, prog_3d):
    import subprocess

    mx, my = pygame.mouse.get_pos()
    double_click_threshold = 300
    sidebar_w = 200
    top_bar_h = 75
    item_w = state.get("item_w", 180)
    text_area_h = 45
    item_h = item_w + text_area_h

    for event in events:
        if event.type == QUIT:
            state["running"] = False
        elif event.type == VIDEORESIZE:
            pygame.display.set_mode(
                event.size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
            )
        elif event.type == pygame.DROPFILE:
            dropped_path = event.file
            if os.path.isdir(dropped_path):
                state["current_folder"] = os.path.abspath(dropped_path)
            else:
                state["current_folder"] = os.path.abspath(os.path.dirname(dropped_path))
            state["all_viewers"] = load_viewers_from_folder(
                state["current_folder"], ctx, prog_3d
            )
            state["search_query"] = ""
            apply_search_and_sort(state, viewers)
            subfolders[:] = get_subfolders(state["current_folder"])
            state["scroll_y"] = 0.0
            state["hovered_index"] = -1
            state["fullscreen_index"] = -1
            state["active_face_index"] = -1
            state["active_rep_idx"] = -1
            state["active_vertex_index"] = -1
            state["active_report"] = None

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
        elif event.type == MOUSEBUTTONDOWN:
            event_handled = False
            win_w, win_h = pygame.display.get_window_size()
            grid_w = win_w - sidebar_w

            if event.button == 4:
                if state["dropdown_open"] and pygame.Rect(
                    15, 102, 170, min(350, win_h - 120)
                ).collidepoint(mx, my):
                    state["dropdown_scroll"] = max(0, state["dropdown_scroll"] - 20)
                    event_handled = True
                elif (
                    state["fullscreen_index"] == -1
                    and mx > sidebar_w
                    and my > top_bar_h
                ):
                    state["scroll_y"] = max(
                        0.0, min(state["scroll_y"] - 50, state["max_scroll"])
                    )
                    event_handled = True
            elif event.button == 5:
                if state["dropdown_open"] and pygame.Rect(
                    15, 102, 170, min(350, win_h - 120)
                ).collidepoint(mx, my):
                    all_opts_len = len(["Default"] + state["sort_headers"])
                    max_dp_scroll = max(0, all_opts_len * 20 - min(350, win_h - 120))
                    state["dropdown_scroll"] = min(
                        max_dp_scroll, state["dropdown_scroll"] + 20
                    )
                    event_handled = True
                elif (
                    state["fullscreen_index"] == -1
                    and mx > sidebar_w
                    and my > top_bar_h
                ):
                    state["scroll_y"] = max(
                        0.0, min(state["scroll_y"] + 50, state["max_scroll"])
                    )
                    event_handled = True
            elif event.button == 3:  # Right Click
                if state["fullscreen_index"] == -1 and state["hovered_index"] != -1:
                    state["context_menu_open"] = True
                    state["context_menu_index"] = state["hovered_index"]
                    state["context_menu_pos"] = (mx, my)
                    event_handled = True
            elif event.button == 1:  # Left Click
                # Clicked within active context menu
                if state["context_menu_open"]:
                    cx, cy = state["context_menu_pos"]
                    menu_rect = pygame.Rect(cx, cy, 120, 80)
                    if menu_rect.collidepoint(mx, my):
                        relative_y = my - cy

                        root_temp = tk.Tk()
                        root_temp.option_add("*Entry.width", 60)
                        root_temp.withdraw()

                        if relative_y < 20:  # Rename option
                            old_name = viewers[state["context_menu_index"]].filename
                            new_name = simpledialog.askstring(
                                "Rename File", "Enter new name:", initialvalue=old_name
                            )
                            if new_name:
                                if not new_name.lower().endswith(".off"):
                                    new_name += ".off"
                                old_path = viewers[state["context_menu_index"]].filepath
                                new_path = os.path.join(
                                    os.path.dirname(old_path), new_name
                                )
                                try:
                                    os.rename(old_path, new_path)
                                    viewers[
                                        state["context_menu_index"]
                                    ].filepath = new_path
                                    viewers[
                                        state["context_menu_index"]
                                    ].filename = new_name
                                    run_bg_analysis(
                                        state["current_folder"], state["all_viewers"]
                                    )
                                except Exception as e:
                                    messagebox.showerror(
                                        "Error", f"Could not rename file: {e}"
                                    )
                        elif relative_y < 40:  # Delete option
                            target_v = viewers[state["context_menu_index"]]
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
                                    run_bg_analysis(
                                        state["current_folder"], state["all_viewers"]
                                    )
                                except Exception as e:
                                    messagebox.showerror(
                                        "Error", f"Could not delete file: {e}"
                                    )
                        elif relative_y < 60:  # Open in Stella option
                            stella_path = os.path.join(
                                os.environ.get("ProgramFiles", "C:\\Program Files"),
                                "Stella4D",
                                "Stella4D.exe",
                            )
                            if not os.path.exists(stella_path):
                                stella_path = os.path.join(
                                    os.environ.get(
                                        "ProgramFiles(x86)", "C:\\Program Files (x86)"
                                    ),
                                    "Stella4D",
                                    "Stella4D.exe",
                                )
                            if os.path.exists(stella_path):
                                try:
                                    subprocess.Popen(
                                        [
                                            stella_path,
                                            os.path.abspath(
                                                viewers[
                                                    state["context_menu_index"]
                                                ].filepath
                                            ),
                                        ]
                                    )
                                except Exception as e:
                                    messagebox.showerror(
                                        "Error", f"Could not launch Stella4D: {e}"
                                    )
                            else:
                                messagebox.showerror(
                                    "Error",
                                    f"Stella4D not found at expected Program Files paths.",
                                )
                        else:  # View Source option
                            target_v = viewers[state["context_menu_index"]]
                            show_bespoke_text_window(target_v.filepath)

                        root_temp.update()
                        root_temp.destroy()
                        state["context_menu_open"] = False
                        event_handled = True
                    else:
                        state["context_menu_open"] = False

                # Check Left/Right enlarged view buttons
                elif state["fullscreen_index"] != -1 and len(viewers) > 0:
                    prev_btn_rect = pygame.Rect(win_w - 210, win_h - 45, 40, 30)
                    next_btn_rect = pygame.Rect(win_w - 160, win_h - 45, 40, 30)
                    grid_btn_rect = pygame.Rect(win_w - 110, win_h - 45, 90, 30)

                    if prev_btn_rect.collidepoint(mx, my):
                        new_idx = (state["fullscreen_index"] - 1) % len(viewers)
                        state["fullscreen_index"] = new_idx
                        state["active_face_index"] = -1
                        state["active_rep_idx"] = -1
                        state["active_vertex_index"] = -1
                        v = viewers[new_idx]
                        state["target_file"] = v.filename
                        run_single_analysis(v.filepath, state)
                        state["needs_redraw"] = 2
                        event_handled = True
                    elif next_btn_rect.collidepoint(mx, my):
                        new_idx = (state["fullscreen_index"] + 1) % len(viewers)
                        state["fullscreen_index"] = new_idx
                        state["active_face_index"] = -1
                        state["active_rep_idx"] = -1
                        state["active_vertex_index"] = -1
                        v = viewers[new_idx]
                        state["target_file"] = v.filename
                        run_single_analysis(v.filepath, state)
                        state["needs_redraw"] = 2
                        event_handled = True
                    elif grid_btn_rect.collidepoint(mx, my):
                        state["fullscreen_index"] = -1
                        state["active_face_index"] = -1
                        state["active_rep_idx"] = -1
                        state["active_vertex_index"] = -1
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
                        relative_y = my - 102 + state["dropdown_scroll"]
                        clicked_idx = relative_y // 20
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

                # Check 'Delete Duplicates' Checkbox Click
                if not event_handled and pygame.Rect(15, 103, 170, 16).collidepoint(
                    mx, my
                ):
                    selected_metric = state.get("selected_sort_metric", "Default")
                    if selected_metric != "Default" and len(viewers) > 1:
                        groups = []
                        current_group = [viewers[0]]

                        for v in viewers[1:]:
                            prev_v = current_group[-1]
                            val1 = v.metrics.get(selected_metric, "")
                            val2 = prev_v.metrics.get(selected_metric, "")

                            is_equal = False
                            if (
                                val1 is not None
                                and val2 is not None
                                and val1 != ""
                                and val2 != ""
                            ):
                                if val1 == val2:
                                    is_equal = True
                                else:
                                    try:
                                        f1 = float(val1)
                                        f2 = float(
                                            prev_v.metrics.get(selected_metric, "")
                                        )
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
                                oldest_v = mtimes[0][1]

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

                            run_bg_analysis(
                                state["current_folder"], state["all_viewers"]
                            )
                            state["needs_redraw"] = 2

                            root_temp = tk.Tk()
                            root_temp.withdraw()
                            messagebox.showinfo(
                                "Delete Duplicates",
                                f"Deleted {deleted_count} duplicate files. Kept the oldest from each set.",
                            )
                            root_temp.destroy()
                    event_handled = True

                # Check Interactive Search Button Click
                if not event_handled and state["fullscreen_index"] == -1:
                    search_btn_rect = pygame.Rect(win_w - 115, 30, 100, 30)
                    if search_btn_rect.collidepoint(mx, my):
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

                # Check Editable Folder Address Box Click
                if not event_handled and state["fullscreen_index"] == -1:
                    addr_box_rect = pygame.Rect(sidebar_w + 15, 30, grid_w - 150, 30)
                    if addr_box_rect.collidepoint(mx, my):
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
                                state["hovered_index"] = -1
                                state["fullscreen_index"] = -1
                                state["active_face_index"] = -1
                                state["active_rep_idx"] = -1
                                state["active_vertex_index"] = -1
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
                                    state["current_folder"], state["all_viewers"]
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
                    on_scrollbar_track = mx >= win_w - 12
                    grid_h = win_h - top_bar_h
                    if (
                        on_scrollbar_track
                        and state["fullscreen_index"] == -1
                        and state["max_scroll"] > 0
                    ):
                        bar_h = max(
                            30, int((grid_h / (state["max_scroll"] + grid_h)) * grid_h)
                        )
                        bar_y = top_bar_h + int(
                            (state["scroll_y"] / state["max_scroll"]) * (grid_h - bar_h)
                        )
                        if bar_y <= my <= bar_y + bar_h:
                            state["scrollbar_dragging"] = True
                            state["scrollbar_click_offset"] = my - bar_y
                        else:
                            state["scrollbar_dragging"] = True
                            state["scrollbar_click_offset"] = bar_h // 2
                            track_usable_h = grid_h - bar_h
                            if track_usable_h > 0:
                                click_y_in_track = (
                                    my - top_bar_h - state["scrollbar_click_offset"]
                                )
                                ratio = max(
                                    0.0, min(1.0, click_y_in_track / track_usable_h)
                                )
                                state["scroll_y"] = ratio * state["max_scroll"]
                    elif mx <= sidebar_w and my > 155:
                        folder_y = 160
                        clicked_idx = (my - folder_y) // 25
                        if 0 <= clicked_idx < len(subfolders):
                            target = subfolders[clicked_idx]
                            if target == "..":
                                state["current_folder"] = os.path.abspath(
                                    os.path.dirname(state["current_folder"])
                                )
                            else:
                                state["current_folder"] = os.path.abspath(
                                    os.path.join(state["current_folder"], target)
                                )
                            state["all_viewers"] = load_viewers_from_folder(
                                state["current_folder"], ctx, prog_3d
                            )
                            state["search_query"] = ""
                            apply_search_and_sort(state, viewers)
                            subfolders[:] = get_subfolders(state["current_folder"])
                            state["scroll_y"] = 0.0
                            state["hovered_index"] = -1
                            state["fullscreen_index"] = -1
                            state["active_face_index"] = -1
                            state["active_rep_idx"] = -1
                            state["active_vertex_index"] = -1
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
                                state["current_folder"], state["all_viewers"]
                            )
                    else:
                        current_time = pygame.time.get_ticks()
                        time_diff = current_time - state["last_click_time"]
                        state["last_click_time"] = current_time

                        if time_diff < double_click_threshold:
                            if state["fullscreen_index"] != -1:
                                state["fullscreen_index"] = -1
                                state["active_face_index"] = -1
                                state["active_rep_idx"] = -1
                                state["active_vertex_index"] = -1
                                state["active_report"] = None
                            elif state["hovered_index"] != -1:
                                active_viewer = viewers[state["hovered_index"]]
                                state["fullscreen_index"] = state["hovered_index"]
                                run_single_analysis(active_viewer.filepath, state)
                        else:
                            if state["fullscreen_index"] != -1:
                                if mx > sidebar_w and my > top_bar_h:
                                    state["dragging"] = True
                                    state["dragged_index"] = state["fullscreen_index"]
                                    pygame.mouse.get_rel()
                            else:
                                if (
                                    mx > sidebar_w
                                    and my > top_bar_h
                                    and state["hovered_index"] != -1
                                ):
                                    state["dragging"] = True
                                    state["dragged_index"] = state["hovered_index"]
                                    pygame.mouse.get_rel()
        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:
                state["dragging"] = False
                state["dragged_index"] = -1
                state["scrollbar_dragging"] = False
        elif event.type == MOUSEMOTION:
            win_w, win_h = pygame.display.get_window_size()
            grid_h = win_h - top_bar_h
            if state["scrollbar_dragging"] and state["max_scroll"] > 0:
                bar_h = max(30, int((grid_h / (state["max_scroll"] + grid_h)) * grid_h))
                track_usable_h = grid_h - bar_h
                if track_usable_h > 0:
                    click_y_in_track = my - top_bar_h - state["scrollbar_click_offset"]
                    ratio = max(0.0, min(1.0, click_y_in_track / track_usable_h))
                    state["scroll_y"] = ratio * state["max_scroll"]
            elif (
                state["dragging"]
                and state["dragged_index"] != -1
                and 0 <= state["dragged_index"] < len(viewers)
            ):
                dx_pixels, dy_pixels = event.rel
                rot_scale = 0.005
                viewers[state["dragged_index"]].rotate_camera(
                    -dx_pixels * rot_scale, -dy_pixels * rot_scale
                )
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                if state["fullscreen_index"] != -1:
                    state["fullscreen_index"] = -1
                    state["active_face_index"] = -1
                    state["active_rep_idx"] = -1
                    state["active_vertex_index"] = -1
                    state["active_report"] = None
                    state["needs_redraw"] = 2
            elif event.key == K_f:
                if state["fullscreen_index"] != -1 and 0 <= state[
                    "fullscreen_index"
                ] < len(viewers):
                    v = viewers[state["fullscreen_index"]]
                    num_reps = len(v.representative_faces)
                    if num_reps > 0:
                        state["active_vertex_index"] = -1
                        state["active_rep_idx"] = (
                            state.get("active_rep_idx", -1) + 2
                        ) % (num_reps + 1) - 1
                        if state["active_rep_idx"] != -1:
                            state["active_face_index"] = v.representative_faces[
                                state["active_rep_idx"]
                            ]
                        else:
                            state["active_face_index"] = -1
                        state["needs_redraw"] = 2
            elif event.key == K_v:
                if state["fullscreen_index"] != -1 and 0 <= state[
                    "fullscreen_index"
                ] < len(viewers):
                    v = viewers[state["fullscreen_index"]]
                    num_verts = v.num_vertices
                    if num_verts > 0:
                        state["active_face_index"] = -1
                        state["active_rep_idx"] = -1
                        state["active_vertex_index"] = (
                            state["active_vertex_index"] + 2
                        ) % (num_verts + 1) - 1
                        state["needs_redraw"] = 2
            elif event.key == K_e:
                if state["fullscreen_index"] != -1 and 0 <= state[
                    "fullscreen_index"
                ] < len(viewers):
                    if (
                        state["active_vertex_index"] != -1
                        or state["active_face_index"] != -1
                    ):
                        state["edge_mode"] = (state["edge_mode"] + 1) % 3
                    else:
                        state["edge_mode"] = 2 if state["edge_mode"] == 0 else 0
                    state["needs_redraw"] = 2
                else:
                    # Toggle edge display for Grid view independently
                    state["grid_edges_on"] = not state.get("grid_edges_on", False)
                    state["needs_redraw"] = 2
            elif event.key == K_x:
                if state["fullscreen_index"] != -1:
                    state["active_face_index"] = -1
                    state["active_rep_idx"] = -1
                    state["active_vertex_index"] = -1
                    state["needs_redraw"] = 2
            elif event.key in (K_PLUS, K_EQUALS, K_KP_PLUS):
                state["item_w"] = min(400, state.get("item_w", 180) + 20)
                state["needs_redraw"] = 2
            elif event.key in (K_MINUS, K_KP_MINUS):
                state["item_w"] = max(80, state.get("item_w", 180) - 20)
                state["needs_redraw"] = 2


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

            # Draw standard background panel and border
            pygame.draw.rect(
                ui_surf,
                (255, 255, 255),
                (sidebar_w + 10, top_bar_h + 10, grid_w - 20, win_h - top_bar_h - 20),
            )
            pygame.draw.rect(
                ui_surf,
                (150, 150, 150),
                (sidebar_w + 10, top_bar_h + 10, grid_w - 20, win_h - top_bar_h - 20),
                1,
            )

            # Clear transparent hole across the dynamic 3D display region so model can shine through
            view_h = win_h - top_bar_h - 60
            report_active = bool(state.get("active_report"))
            viewport_w_hole = grid_w - 440 - 30 if report_active else grid_w - 22

            ui_surf.fill(
                (0, 0, 0, 0), (sidebar_w + 11, top_bar_h + 15, viewport_w_hole, view_h)
            )

            # Draw solid report overlay box on the right of the cleared area if active
            report_text = state.get("active_report")
            if report_text:
                report_w = 440
                report_x = win_w - report_w - 20
                report_y = top_bar_h + 20
                report_h = win_h - top_bar_h - 80

                pygame.draw.rect(
                    ui_surf, (248, 249, 250), (report_x, report_y, report_w, report_h)
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
            active_rep_idx = state.get("active_rep_idx", -1)
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
                vertex_lbl = font_bold.render(
                    f"Displaying Faces Adjacent to Vertex {active_vertex + 1} / {v.num_vertices}",
                    True,
                    (255, 50, 50),
                )
                ui_surf.blit(vertex_lbl, (label_x, top_bar_h + 30))

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
            hole_h = min(win_h, y + item_w - 1) - role_y
            if hole_h > 0:
                ui_surf.fill((0, 0, 0, 0), (x + 1, role_y, item_w - 2, hole_h))

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

    panel_title = font_bold.render("Subfolders Panel", True, (0, 0, 0))
    ui_surf.blit(panel_title, (15, 130))

    sy = 155
    for i, item in enumerate(subfolders):
        if sy + 25 > win_h:
            break
        is_parent = item == ".."
        display_name = "[Parent Folder] .." if is_parent else f"📁 {item}"
        color = (0, 102, 204) if is_parent else (50, 50, 50)

        if (
            mx <= sidebar_w
            and sy <= my <= sy + 22
            and (
                not state["dropdown_open"]
                or my < 102
                or my > 102 + min(350, win_h - 120)
            )
        ):
            pygame.draw.rect(ui_surf, (230, 240, 250), (10, sy - 2, sidebar_w - 20, 22))
            pygame.draw.rect(
                ui_surf, (180, 200, 230), (10, sy - 2, sidebar_w - 20, 22), 1
            )

        txt = font.render(display_name, True, color)
        ui_surf.blit(txt, (20, sy))
        sy += 25

    # Address bar
    pygame.draw.rect(ui_surf, (255, 255, 255), (sidebar_w + 15, 30, grid_w - 150, 30))
    pygame.draw.rect(
        ui_surf, (200, 200, 200), (sidebar_w + 15, 30, grid_w - 150, 30), 1
    )
    addr_txt = font.render(f"  {state['current_folder']}", True, (80, 80, 80))
    ui_surf.blit(addr_txt, (sidebar_w + 20, 36))

    # Search button
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
        bar_h = max(30, int((grid_h / (state["max_scroll"] + grid_h)) * grid_h))
        bar_y = top_bar_h + int(
            (state["scroll_y"] / state["max_scroll"]) * (grid_h - bar_h)
        )
        pygame.draw.rect(ui_surf, (240, 240, 240), (win_w - 12, top_bar_h, 12, grid_h))
        pygame.draw.rect(
            ui_surf, (200, 200, 200), (win_w - 10, bar_y, 8, bar_h), border_radius=4
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
        pygame.draw.rect(ui_surf, (255, 255, 255), (cx, cy, 120, 80))
        pygame.draw.rect(ui_surf, (150, 150, 150), (cx, cy, 120, 80), 1)

        if pygame.Rect(cx, cy, 120, 20).collidepoint(mx, my):
            pygame.draw.rect(ui_surf, (230, 240, 250), (cx + 1, cy + 1, 118, 18))
        elif pygame.Rect(cx, cy + 20, 120, 20).collidepoint(mx, my):
            pygame.draw.rect(ui_surf, (230, 240, 250), (cx + 1, cy + 21, 118, 18))
        elif pygame.Rect(cx, cy + 40, 120, 20).collidepoint(mx, my):
            pygame.draw.rect(ui_surf, (230, 240, 250), (cx + 1, cy + 41, 118, 18))
        elif pygame.Rect(cx, cy + 60, 120, 20).collidepoint(mx, my):
            pygame.draw.rect(ui_surf, (230, 240, 250), (cx + 1, cy + 61, 118, 18))

        txt_rename = font.render("Rename", True, (40, 40, 40))
        txt_delete = font.render("Delete", True, (40, 40, 40))
        txt_stella = font.render("Open in Stella", True, (40, 40, 40))
        txt_view_source = font.render("View Source", True, (40, 40, 40))

        ui_surf.blit(txt_rename, (cx + 10, cy + 3))
        ui_surf.blit(txt_delete, (cx + 10, cy + 23))
        ui_surf.blit(txt_stella, (cx + 10, cy + 43))
        ui_surf.blit(txt_view_source, (cx + 10, cy + 63))

    return ui_surf


# Version 5.23


def main():
    import subprocess

    pygame.init()
    pygame.font.init()

    win_w, win_h = 1300, 850

    try:
        screen_w, screen_h = 1920, 1080  # Default fallback
        if sys.platform == "win32":
            import ctypes

            screen_w = ctypes.windll.user32.GetSystemMetrics(0)
            screen_h = ctypes.windll.user32.GetSystemMetrics(1)
        else:
            info = pygame.display.Info()
            screen_w = info.current_w
            screen_h = info.current_h

        # Center the window on the desktop
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
    ctx.depth_func = "<="  # Set depth function to Less-Than-or-Equal to eliminate coplanar Z-fighting
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    # Enable polygon offset to push filled faces slightly back in depth relative to wireframe edges
    try:
        import ctypes

        if sys.platform == "win32":
            ctypes.windll.opengl32.glEnable(32823)  # GL_POLYGON_OFFSET_FILL (0x8037)
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

    # Central application UI & interactive state
    state = {
        "current_folder": current_folder,
        "target_file": target_file,
        "fullscreen_index": fullscreen_index,
        "active_face_index": -1,
        "active_rep_idx": -1,
        "active_vertex_index": -1,
        "edge_mode": 0,
        "grid_edges_on": False,
        "active_report": None,
        "scroll_y": 0.0,
        "max_scroll": 0,
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

    while state["running"]:
        dt = clock.tick(60)
        dt = min(100, dt)

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

        handle_events(events, state, viewers, subfolders, ctx, prog_3d)

        # Dynamic layout calculations based on current window size
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

        # Process rendering only if marked as dirty
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

                    # Allocate comfortable spacing based on active report
                    report_active = bool(state.get("active_report"))
                    viewport_w = (
                        win_w - viewport_x - 440 - 30
                        if report_active
                        else win_w - viewport_x - 20
                    )
                    viewport_h = win_h - top_bar_h - 60

                    # Model remains centered locally inside its designated viewport (target_x = 0.0)
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
