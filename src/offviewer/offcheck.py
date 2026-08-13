# File: offcheck.py
# Version: 4.4
# Addons required: pip install windnd numpy

import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from collections import defaultdict
import math
import windnd 
import ctypes
import csv
import importlib.util
import re
import heapq
import numpy as np

def calculate_face_normal_and_area(coords):
    """Calculates the area and the Newell normal of a 3D polygon, handling self-intersections for absolute area."""
    if len(coords) < 3:
        return (0, 0, 0), 0.0
    nx = ny = nz = 0.0
    for i in range(len(coords)):
        v_curr = coords[i]
        v_next = coords[(i + 1) % len(coords)]
        nx += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
        ny += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
        nz += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])
    
    length = math.sqrt(nx**2 + ny**2 + nz**2)
    if length == 0:
        return (0, 0, 0), 0.0
    normal = (nx/length, ny/length, nz/length)

    # Helper to project 3D coordinates to 2D isometry
    def project_to_2d_isometry(pts, n_vec):
        n_arr = np.array(n_vec)
        if abs(n_arr[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(n_arr, ref)
        u /= np.linalg.norm(u)
        v = np.cross(n_arr, u)
        return [np.array([np.dot(p, u), np.dot(p, v)]) for p in pts]

    # Helper to compute absolute area of possibly self-intersecting 2D polygon
    def compute_absolute_area_2d(p2d):
        n_pts = len(p2d)
        for i in range(n_pts):
            for j in range(i + 2, n_pts):
                if i == 0 and j == n_pts - 1:
                    continue
                A = p2d[i]
                B = p2d[(i + 1) % n_pts]
                C = p2d[j]
                D = p2d[(j + 1) % n_pts]
                
                r = B - A
                s = D - C
                denom = r[0]*s[1] - r[1]*s[0]
                if abs(denom) > 1e-12:
                    num_t = (C[0]-A[0])*s[1] - (C[1]-A[1])*s[0]
                    num_u = (C[0]-A[0])*r[1] - (C[1]-A[1])*r[0]
                    t = num_t / denom
                    u = num_u / denom
                    if 1e-11 < t < 1.0 - 1e-11 and 1e-11 < u < 1.0 - 1e-11:
                        P = A + t * r
                        loop1 = p2d[i+1 : j+1] + [P]
                        loop2 = p2d[j+1:] + p2d[:i+1] + [P]
                        return compute_absolute_area_2d(loop1) + compute_absolute_area_2d(loop2)
        
        area = 0.0
        for i in range(n_pts):
            p1 = p2d[i]
            p2 = p2d[(i + 1) % n_pts]
            area += p1[0] * p2[1] - p2[0] * p1[1]
        return 0.5 * abs(area)

    p2d_proj = project_to_2d_isometry(coords, normal)
    abs_area = compute_absolute_area_2d(p2d_proj)
    return normal, abs_area

def get_convex_hull_normal(coords):
    """Computes a stable plane normal by finding the 2D convex hull of projected 3D coordinates."""
    if len(coords) < 3:
        return (0.0, 0.0, 1.0)
    
    pts = np.array(coords)
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    cov = np.dot(pts_centered.T, pts_centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    normal_pca = eigenvectors[:, 0]
    norm = np.linalg.norm(normal_pca)
    if norm > 1e-12:
        normal_pca /= norm
    else:
        normal_pca = np.array([0.0, 0.0, 1.0])

    if abs(normal_pca[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal_pca, ref)
    u_norm = np.linalg.norm(u)
    if u_norm > 1e-12:
        u /= u_norm
    v = np.cross(normal_pca, u)
    v_norm = np.linalg.norm(v)
    if v_norm > 1e-12:
        v /= v_norm

    pts_2d = []
    for i, p in enumerate(coords):
        px = np.dot(p - centroid, u)
        py = np.dot(p - centroid, v)
        pts_2d.append((px, py, i))

    sorted_pts = sorted(pts_2d, key=lambda p: (p[0], p[1]))
    if len(sorted_pts) < 3:
        return tuple(normal_pca)

    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 1e-11:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(sorted_pts):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 1e-11:
            upper.pop()
        upper.append(p)

    hull_pts_2d = lower[:-1] + upper[:-1]
    hull_coords = [coords[p[2]] for p in hull_pts_2d]

    hull_normal, _ = calculate_face_normal_and_area(hull_coords)
    return hull_normal

def is_concave_face(coords, face_normal):
    """Checks if a face is non-convex."""
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

def is_crossed_face(coords, normal):
    """Checks if a face is self-intersecting (crossed) by projecting to 2D."""
    if len(coords) < 4:
        return False
        
    nx, ny, nz = normal
    abs_x, abs_y, abs_z = abs(nx), abs(ny), abs(nz)
    if abs_z >= abs_x and abs_z >= abs_y:
        p2d = [(v[0], v[1]) for v in coords]
    elif abs_y >= abs_x:
        p2d = [(v[0], v[2]) for v in coords]
    else:
        p2d = [(v[1], v[2]) for v in coords]
        
    n = len(p2d)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            A = p2d[i]
            B = p2d[(i + 1) % n]
            C = p2d[j]
            D = p2d[(j + 1) % n]
            
            def cp(p1, p2, p3):
                return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            d1 = cp(C, D, A)
            d2 = cp(C, D, B)
            d3 = cp(A, B, C)
            d4 = cp(A, B, D)
            if (((d1 > 1e-12 and d2 < -1e-12) or (d1 < -1e-12 and d2 > 1e-12)) and
                ((d3 > 1e-12 and d4 < -1e-12) or (d3 < -1e-12 and d4 > 1e-12))):
                return True
    return False

def check_planarity(coords, normal, tolerance=1e-6):
    """Checks if all vertices of a face lie on the same plane."""
    if len(coords) <= 3:
        return False, 0.0 # Triangles are always planar
    
    v0 = coords[0]
    max_dist = 0.0
    for i in range(1, len(coords)):
        vec = (coords[i][0] - v0[0], coords[i][1] - v0[1], coords[i][2] - v0[2])
        dist = abs(vec[0]*normal[0] + vec[1]*normal[1] + vec[2]*normal[2])
        max_dist = max_dist if max_dist > dist else dist
        
    return max_dist > tolerance, max_dist

def find_closest_vertex_distance(unique_vertices, initial_estimate):
    """Calculates the absolute closest distance between any two distinct vertices."""
    if len(unique_vertices) < 2:
        return float('inf')
    
    d_min = initial_estimate if initial_estimate > 0 else float('inf')
    pts = sorted(unique_vertices, key=lambda p: p[0])
    n = len(pts)
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[j][0] - pts[i][0]
            if dx >= d_min:
                break
            dy = pts[j][1] - pts[i][1]
            dz = pts[j][2] - pts[i][2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if 0 < dist < d_min:
                d_min = dist
    return d_min

def verify_off_logic(filepath, return_stats=False, run_symmetry=True):
    """The core consistency checking logic."""
    one_vertex_faces_detected = False
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    except Exception as e:
        err = f"Error opening file: {e}"
        return (err, None) if return_stats else err

    if not lines or not lines[0].startswith('OFF'):
        err = "Error: Valid OFF header not found."
        return (err, None) if return_stats else err

    header_idx = 1 if lines[0] == 'OFF' else 0
    count_line = lines[header_idx]
    if header_idx == 0: count_line = count_line[3:].strip()
    
    try:
        counts = list(map(int, count_line.split()))
        num_vertices, num_faces = counts[0], counts[1]
    except:
        err = "Error: Could not parse vertex/face counts from header."
        return (err, None) if return_stats else err

    data_start_idx = header_idx + 1
    vertex_end_idx = data_start_idx + num_vertices

    if len(lines) < vertex_end_idx:
        err = f"Error: Header claims {num_vertices} vertices, but file ends prematurely."
        return (err, None) if return_stats else err

    # --- 1. Parse Vertices ---
    vertices = []
    vertex_positions = defaultdict(list)
    duplicate_coords_count = 0
    max_decimal_places = 0
    
    for i in range(data_start_idx, vertex_end_idx):
        try:
            raw_parts = lines[i].split()[:3]
            for part in raw_parts:
                if '.' in part:
                    dec_part = part.split('.')[1].split('e')[0].split('E')[0]
                    max_decimal_places = max(max_decimal_places, len(dec_part))
            
            coords = tuple(map(float, raw_parts))
            vertices.append(coords)
            vertex_positions[coords].append(i - data_start_idx)
        except ValueError:
            err = f"Error: Could not parse coordinates at vertex line {i}."
            return (err, None) if return_stats else err

    for pos in vertex_positions:
        if len(vertex_positions[pos]) > 1:
            duplicate_coords_count += 1

    # Calculate distance to origin for each vertex
    origin_distances = [math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for v in vertices]
    max_origin_dist = max(origin_distances) if origin_distances else 0.0
    min_origin_dist = min(origin_distances) if origin_distances else 0.0

    # --- 2. Parse Faces ---
    edge_dict = defaultdict(int)
    vertex_valence_map = [set() for _ in range(num_vertices)]
    face_type_counts = defaultdict(int) 
    face_lines = lines[vertex_end_idx : vertex_end_idx + num_faces]
    zero_area_faces = []
    hemihedral_faces = []
    faces_with_repeats = []
    crossed_faces = [] 
    concave_faces = []
    non_planar_faces = []
    min_face_area = float('inf')
    shortest_edge_len = float('inf')
    max_planar_err = 0.0
    TOLERANCE = 1e-12 

    processed_face_count = 0
    parsed_faces = []

    for i, line in enumerate(face_lines):
        raw_parts = line.split()
        if not raw_parts: continue
        try:
            n_v = int(raw_parts[0])
            v_indices = list(map(int, raw_parts[1:1+n_v]))
        except ValueError:
            err = f"Error: Face {i} has non-integer indices."
            return (err, None) if return_stats else err

        if len(v_indices) != n_v:
            err = f"Error: Face {i} claims {n_v} vertices but lists {len(v_indices)}."
            return (err, None) if return_stats else err

        parsed_faces.append(v_indices)
        face_type_counts[n_v] += 1
        processed_face_count += 1

        if len(set(v_indices)) != len(v_indices):
            faces_with_repeats.append(i)

        face_coords = []
        for j in range(n_v):
            v_idx = v_indices[j]
            v_next_idx = v_indices[(j + 1) % n_v]
            if v_idx < 0 or v_idx >= num_vertices:
                err = f"Error: Face {i} references invalid vertex index {v_idx}."
                return (err, None) if return_stats else err
            
            v1 = vertices[v_idx]
            face_coords.append(v1)
            v2 = vertices[v_next_idx]
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
            if dist > 0: 
                shortest_edge_len = min(shortest_edge_len, dist)

            edge = tuple(sorted((v_idx, v_next_idx)))
            edge_dict[edge] += 1
            vertex_valence_map[v_idx].add(edge)
            vertex_valence_map[v_next_idx].add(edge)

        # Check for Hemihedra (centered at origin)
        cx = sum(v[0] for v in face_coords) / n_v
        cy = sum(v[1] for v in face_coords) / n_v
        cz = sum(v[2] for v in face_coords) / n_v
        if abs(cx) < TOLERANCE and abs(cy) < TOLERANCE and abs(cz) < TOLERANCE:
            hemihedral_faces.append(i)

        normal, area = calculate_face_normal_and_area(face_coords)
        if processed_face_count == 1 or area < min_face_area:
            min_face_area = area
        
        if area < TOLERANCE: 
            zero_area_faces.append(i)
        else:
            if is_crossed_face(face_coords, normal):
                crossed_faces.append(i)
            if is_concave_face(face_coords, normal):
                concave_faces.append(i)
            
            # Using the stable normal of the face's convex hull for planarity checks
            hull_normal = get_convex_hull_normal(face_coords)
            is_np, err = check_planarity(face_coords, hull_normal)
            max_planar_err = max(max_planar_err, err)
            if is_np:
                non_planar_faces.append(i)

    # --- 3. Connected Components (Compound Parts) ---
    edge_to_faces = defaultdict(list)
    for f_idx, v_indices in enumerate(parsed_faces):
        n_v = len(v_indices)
        for j in range(n_v):
            v_idx = v_indices[j]
            v_next_idx = v_indices[(j + 1) % n_v]
            edge = tuple(sorted((v_idx, v_next_idx)))
            edge_to_faces[edge].append(f_idx)

    face_neighbors = defaultdict(list)
    for edge, f_indices in edge_to_faces.items():
        if len(f_indices) > 1:
            for f1 in f_indices:
                for f2 in f_indices:
                    if f1 != f2:
                        face_neighbors[f1].append(f2)

    actual_num_faces = len(parsed_faces)
    visited_faces = [False] * actual_num_faces
    compound_parts = 0
    for i in range(actual_num_faces):
        if not visited_faces[i]:
            compound_parts += 1
            queue = [i]
            visited_faces[i] = True
            head = 0
            while head < len(queue):
                curr = queue[head]
                head += 1
                for neighbor in face_neighbors[curr]:
                    if not visited_faces[neighbor]:
                        visited_faces[neighbor] = True
                        queue.append(neighbor)

    boundary = [e for e, c in edge_dict.items() if c == 1]
    non_manifold = [e for e, c in edge_dict.items() if c > 2]
    
    valence_distribution = defaultdict(int)
    for v_edges in vertex_valence_map:
        valence_distribution[len(v_edges)] += 1

    # Genus calculation: (2 - Euler Characteristic) / 2
    euler_chi = num_vertices - len(edge_dict) + actual_num_faces
    genus = (2 - euler_chi) / 2
    if genus.is_integer():
        genus = int(genus)

    # Gonality: Populated if all faces have the same gonality
    gonality = ""
    if len(face_type_counts) == 1:
        gonality = list(face_type_counts.keys())[0]

    # Valence: Populated only if all vertices of input polyhedron have the same valence
    shared_valence = ""
    if len(valence_distribution) == 1 and num_vertices > 0:
        shared_valence = list(valence_distribution.keys())[0]

    # --- 3a. Geometric Path Metrics & Weighted Wiener Index scaled to R = 1 ---
    # Find Centroid and Circumradius
    if num_vertices > 0:
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        zs = [v[2] for v in vertices]
        cx = sum(xs) / num_vertices
        cy = sum(ys) / num_vertices
        cz = sum(zs) / num_vertices
        
        max_r_sq = 0.0
        for v in vertices:
            r_sq = (v[0] - cx)**2 + (v[1] - cy)**2 + (v[2] - cz)**2
            if r_sq > max_r_sq:
                max_r_sq = r_sq
        circumradius = math.sqrt(max_r_sq) if max_r_sq > 0 else 1.0
    else:
        circumradius = 1.0

    scale_factor = 1.0 / circumradius

    # Scale the edge lengths
    edge_lengths = {}
    for (u, v), count in edge_dict.items():
        v1 = vertices[u]
        v2 = vertices[v]
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
        edge_lengths[(u, v)] = dist * scale_factor

    adj_weighted = [[] for _ in range(num_vertices)]
    for (u, v), length in edge_lengths.items():
        adj_weighted[u].append((v, length))
        adj_weighted[v].append((u, length))
        
    total_geom_dist = 0
    disconnected = False
    
    for start in range(num_vertices):
        dist_map = {start: 0.0}
        pq = [(0.0, start)]
        while pq:
            d, curr = heapq.heappop(pq)
            if d > dist_map[curr]:
                continue
            for nbr, weight in adj_weighted[curr]:
                new_d = d + weight
                if nbr not in dist_map or new_d < dist_map[nbr]:
                    dist_map[nbr] = new_d
                    heapq.heappush(pq, (new_d, nbr))
        
        if len(dist_map) < num_vertices:
            disconnected = True
        for d_val in dist_map.values():
            total_geom_dist += d_val

    if not disconnected and num_vertices > 1:
        wiener_index = total_geom_dist / 2
    else:
        wiener_index = float('nan')

    # --- 4. Symmetry Group and Order ---
    symmetry_order = ""
    symmetry_symbol = ""
    if run_symmetry:
        symmetry_order = "N/A"
        symmetry_symbol = "N/A"
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
            sym_path = os.path.join(script_dir, "offcheck-symmetry.py")
            if not os.path.exists(sym_path):
                sym_path = os.path.abspath("offcheck-symmetry.py")
                
            if os.path.exists(sym_path):
                spec = importlib.util.spec_from_file_location("offcheck_symmetry", sym_path)
                offcheck_symmetry = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(offcheck_symmetry)
                
                if one_vertex_faces_detected:
                    temp_filepath = filepath + ".symtemp.off"
                    with open(temp_filepath, 'w') as temp_f:
                        temp_f.write("OFF\n")
                        temp_f.write(f"{num_vertices} {len(parsed_faces)} 0\n")
                        for v in vertices:
                            temp_f.write(f"{v[0]} {v[1]} {v[2]}\n")
                        for face in parsed_faces:
                            temp_f.write(f"{len(face)} " + " ".join(map(str, face)) + "\n")
                    target_filepath = temp_filepath
                else:
                    target_filepath = filepath
                    
                try:
                    sym_report = offcheck_symmetry.analyze_symmetry(target_filepath)
                    if sym_report.startswith("Error"):
                        symmetry_order = "Error"
                        symmetry_symbol = sym_report
                    else:
                        for line in sym_report.splitlines():
                            if "TOTAL SYMMETRY GROUP ORDER:" in line:
                                symmetry_order = int(line.split()[-1])
                            elif "SCHOENFLIES SYMBOL:" in line:
                                symmetry_symbol = line.split()[-1]
                except Exception as inner_e:
                    import traceback
                    symmetry_order = "Error in analysis"
                    symmetry_symbol = str(inner_e)
                    print(traceback.format_exc(), file=sys.stderr)
                finally:
                    if one_vertex_faces_detected and os.path.exists(temp_filepath):
                        try:
                            os.remove(temp_filepath)
                        except Exception:
                            pass
            else:
                symmetry_order = "Missing helper file"
                symmetry_symbol = "N/A"
        except Exception as e:
            import traceback
            symmetry_order = "Load Error"
            symmetry_symbol = str(e)
            print(traceback.format_exc(), file=sys.stderr)

    def format_metric(val):
        if val == float('inf') or val == 0 or math.isnan(val): return "N/A"
        if val < 1e-4: return f"{val:.8e}"
        s = f"{val:.8f}".rstrip('0')
        return s + "0" if s.endswith('.') else s

    closest_vertex_dist = find_closest_vertex_distance(list(vertex_positions.keys()), shortest_edge_len)

    disp_shortest_edge = format_metric(shortest_edge_len)
    disp_closest_vertex = format_metric(closest_vertex_dist)
    disp_min_area = format_metric(min_face_area)
    disp_wiener = format_metric(wiener_index)
    disp_max_origin_dist = format_metric(max_origin_dist)
    disp_min_origin_dist = format_metric(min_origin_dist)

    face_comp_report = []
    for v_count in sorted(face_type_counts.keys()):
        label = {3: "Triangles", 4: "Quads", 5: "Pentagons"}.get(v_count, f"{v_count}-gons")
        face_comp_report.append(f"{label:<15} ({v_count} vts): {face_type_counts[v_count]}")

    valence_report = []
    for val in sorted(valence_distribution.keys()):
        valence_report.append(f"Valence {val:<6}: {valence_distribution[val]} vertices")

    report = [
        f"File: {filepath}",
        f"Vertices: {num_vertices} | Faces: {num_faces}",
        f"Edges: {len(edge_dict)} | Genus: {genus}",
    ]
    if run_symmetry:
        report.append(f"Symmetry Group:               {symmetry_symbol} (order {symmetry_order})")
    report.extend([
        f"V-E-F:                        {num_vertices}-{len(edge_dict)}-{actual_num_faces}",
        f"Compound Parts:               {compound_parts}",
        f"Coordinate Precision:         {max_decimal_places} decimal places",
        f"Wiener Index (Geom):          {disp_wiener}",
        "-" * 45,
        "FACE COMPOSITION:",
        *face_comp_report,
        "-" * 45,
        "VERTEX VALENCE DISTRIBUTION:",
        *valence_report,
        "-" * 45,
        f"Duplicate Coordinates:         {duplicate_coords_count} sets",
        f"Faces w/ Repeat Indices:       {len(faces_with_repeats)}",
        f"Crossed Faces:                 {len(crossed_faces)}",
        f"Concave Faces:                 {len(concave_faces)}",
        f"Non-Planar Faces:              {len(non_planar_faces)}",
        f"Max Planarity Error:           {max_planar_err:.8e}",
        f"Zero-Area Faces:               {len(zero_area_faces)}",
        f"Hemihedral Faces:              {len(hemihedral_faces)}",
        f"Smallest Face Area:            {disp_min_area}",
        f"Shortest Edge Length:          {disp_shortest_edge}",
        f"Closest Vertex Distance:       {disp_closest_vertex}",
        f"Max Vertex Dist (Origin):      {disp_max_origin_dist}",
        f"Min Vertex Dist (Origin):      {disp_min_origin_dist}",
        "-" * 45,
        f"Manifold Edges:                {len(edge_dict) - len(boundary) - len(non_manifold)}",
        f"Boundary Edges (1 face):       {len(boundary)}",
        f"Non-Manifold Edges (>2 faces): {len(non_manifold)}",
        "-" * 45
    ])
    
    issues = []
    if num_faces == 0: issues.append("Header contains zero faces")
    elif processed_face_count == 0: issues.append("No valid face definitions found")
    if duplicate_coords_count: issues.append("Duplicate vertex coordinates")
    if faces_with_repeats: issues.append("Faces with repeated indices")
    if non_planar_faces: issues.append("Non-planar faces (Quads+)")
    if zero_area_faces: issues.append("Zero-area (degenerate) faces")
    if hemihedral_faces: issues.append("Hemihedral faces")
    if boundary: issues.append("Open boundaries (holes)")
    if non_manifold: issues.append("Non-manifold geometry")
    
    if not issues and processed_face_count > 0:
        report.append("CONCLUSION: Perfectly closed manifold mesh.")
    else:
        report.append("ISSUES DETECTED:")
        if not issues and processed_face_count == 0:
            report.append(" - Malformed file: No faces processed.")
        else:
            for issue in issues: report.append(f" - {issue}")
        
    if return_stats:
        stats = {
            "Filename": os.path.basename(filepath),
            "Vertices": num_vertices,
            "Faces": num_faces,
            "Edges": len(edge_dict),
            "Genus": genus,
            "V-E-F": f"{num_vertices}-{len(edge_dict)}-{actual_num_faces}",
            "Concentration": gonality,  # Maintain name or leave empty
            "Gonality": gonality,
            "Valence": shared_valence,
            "Symmetry Group": f"{symmetry_symbol} (order {symmetry_order})" if (run_symmetry and symmetry_symbol) else "",
            "Compound Parts": compound_parts,
            "Coordinate Precision": max_decimal_places,
            "Wiener Index (Geom)": wiener_index,
            "Duplicate Coords": duplicate_coords_count,
            "Repeat Indices": len(faces_with_repeats),
            "Crossed Faces": len(crossed_faces),
            "Concave Faces": len(concave_faces),
            "Non-Planar Faces": len(non_planar_faces),
            "Max Planarity Error": max_planar_err,
            "Zero-Area Faces": len(zero_area_faces),
            "Hemihedral Faces": len(hemihedral_faces),
            "Smallest Face Area": min_face_area,
            "Shortest Edge Length": shortest_edge_len,
            "Closest Vertex Distance": closest_vertex_dist,
            "Max Vertex Dist (Origin)": max_origin_dist if origin_distances else float('nan'),
            "Min Vertex Dist (Origin)": min_origin_dist if origin_distances else float('nan'),
            "Manifold Edges": len(edge_dict) - len(boundary) - len(non_manifold),
            "Boundary Edges": len(boundary),
            "Non-Manifold Edges": len(non_manifold),
            "Detail": "; ".join(issues) if issues else ("No faces processed." if processed_face_count == 0 else ""),
        }
        
        for v_count, count in face_type_counts.items():
            stats[f"Faces ({v_count})"] = count
        for val, count in valence_distribution.items():
            stats[f"Valence ({val})"] = count
            
        return "\n".join(report), stats

    return "\n".join(report)


class OFFCheckerGUI:
# Version: 4.5

    def __init__(self, root):
        self.root = root
        self.root.title("OFF Quality Checker")
        
        # 1. Position on the left and set width to 560 to resolve the narrow text clipping
        self.root.geometry("560x750+30+30")
        
        # 2. Hide window during construction to block premature focus grabbing
        self.root.withdraw()
        self.set_icon()
        self.root.update_idletasks()

        # 3. Lower the stack priority and strip topmost triggers before displaying
        self.root.lower()
        self.root.wm_attributes("-topmost", False)

        # 4. Initialize layout components while window is withdrawn
        self.label = tk.Label(root, text="DRAG & DROP OFF FILE HERE", 
                              font=("Arial", 10, "bold"), pady=15, fg="#555")
        self.label.pack()

        self.btn = tk.Button(root, text="Browse File Manually", command=self.browse_file, padx=20)
        self.btn.pack(pady=5)

        self.batch_var = tk.BooleanVar(value=False)
        self.batch_chk = tk.Checkbutton(root, text="Batch Mode", variable=self.batch_var)
        self.batch_chk.pack(pady=5)

        self.sym_var = tk.BooleanVar(value=False)
        self.sym_chk = tk.Checkbutton(root, text="Enable Symmetry Check", variable=self.sym_var)
        self.sym_chk.pack(pady=5)

        # Adjusted text width from 80 to 68 to prevent boundary overflows
        self.text_area = scrolledtext.ScrolledText(root, width=68, height=1, font=("Consolas", 10))
        self.text_area.pack(pady=10, padx=10, expand=True, fill=tk.BOTH)

        windnd.hook_dropfiles(self.root, func=self.handle_drop)

        # 5. Display the fully realized layout
        self.root.deiconify()

        if len(sys.argv) > 1:
            self.process_file(sys.argv[1])

    def set_icon(self):
        icon_path = os.path.abspath("offchecker.ico")
        if not os.path.exists(icon_path):
            return

        try:
            # Standard Tkinter approach
            self.root.iconbitmap(icon_path)
            
            # Heavy-duty Win32 API override via ctypes
            hwnd = self.root.winfo_id()
            user32 = ctypes.windll.user32
            
            # Load icon (LR_LOADFROMFILE = 0x10, IMAGE_ICON = 1)
            hicon = user32.LoadImageW(0, icon_path, 1, 0, 0, 0x00000010)
            
            if hicon:
                # Set icon for current window (WM_SETICON = 0x80)
                user32.SendMessageW(hwnd, 0x0080, 0, hicon) # ICON_SMALL
                user32.SendMessageW(hwnd, 0x0080, 1, hicon) # ICON_BIG
                
                # Set icon for the Window Class (more persistent)
                # GCLP_HICON = -14, GCLP_HICONSM = -34
                if ctypes.sizeof(ctypes.c_void_p) == 8: # 64-bit
                    user32.SetClassLongPtrW(hwnd, -14, hicon)
                    user32.SetClassLongPtrW(hwnd, -34, hicon)
                else: # 32-bit
                    user32.SetClassLongW(hwnd, -14, hicon)
                    user32.SetClassLongW(hwnd, -34, hicon)
        except Exception:
            pass

    def handle_drop(self, files):
        if files:
            filepath = files[0].decode('gbk') 
            self.process_file(filepath)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("OFF files", "*.off"), ("All files", "*.*")])
        if file_path:
            self.process_file(file_path)

    def process_file(self, filepath):
        self.text_area.delete(1.0, tk.END)
        sym_enabled = self.sym_var.get()
        
        if self.batch_var.get():
            folder = os.path.dirname(os.path.abspath(filepath))
            off_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.off')]
            if not off_files:
                self.text_area.insert(tk.END, "No OFF files found in the directory.")
                return

            csv_path = os.path.join(folder, 'offcheck.csv')
            
            base_headers = [
                "Filename", "Vertices", "Faces", "Edges", "Genus", "V-E-F", "Gonality", "Valence",
                "Symmetry Group", "Compound Parts", "Coordinate Precision",
                "Wiener Index (Geom)",
                "Duplicate Coords", "Repeat Indices", "Crossed Faces", "Concave Faces", "Non-Planar Faces",
                "Max Planarity Error", "Zero-Area Faces", "Hemihedral Faces", "Smallest Face Area",
                "Shortest Edge Length", "Closest Vertex Distance", 
                "Max Vertex Dist (Origin)", "Min Vertex Dist (Origin)",
                "Manifold Edges", "Boundary Edges", "Non-Manifold Edges", "Detail"
            ]

            results = []
            dynamic_face_keys = set()
            dynamic_valence_keys = set()
            success_count = 0
            
            for fpath in sorted(off_files):
                res, stats = verify_off_logic(fpath, return_stats=True, run_symmetry=sym_enabled)
                if stats is None:
                    stats = {h: "" for h in base_headers}
                    stats["Filename"] = os.path.basename(fpath)
                    stats["Detail"] = res
                else:
                    for k in stats.keys():
                        if k.startswith("Faces ("):
                            dynamic_face_keys.add(k)
                        elif k.startswith("Valence ("):
                            dynamic_valence_keys.add(k)
                results.append(stats)
                success_count += 1

            def extract_num(key_str):
                m = re.search(r'\d+', key_str)
                return int(m.group()) if m else 999999

            sorted_face_headers = sorted(list(dynamic_face_keys), key=extract_num)
            sorted_valence_headers = sorted(list(dynamic_valence_keys), key=extract_num)
            
            headers = base_headers + sorted_face_headers + sorted_valence_headers

            try:
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=headers)
                    writer.writeheader()
                    for row in results:
                        row_data = {h: row.get(h, "") for h in headers}
                        writer.writerow(row_data)
                
                summary = [
                    f"BATCH MODE COMPLETE",
                    f"Folder: {folder}",
                    f"Processed {success_count} .off files.",
                    f"Results written to: {csv_path}",
                    "-" * 45,
                ]
                for row in results:
                    summary.append(f"{row['Filename']:<30} | {row['Detail']}")
                self.text_area.insert(tk.END, "\n".join(summary))
                self.text_area.configure(height=min(len(summary) + 1, 30))
            except Exception as e:
                self.text_area.insert(tk.END, f"Error writing CSV file: {e}")
        else:
            result = verify_off_logic(filepath, run_symmetry=sym_enabled)
            self.text_area.insert(tk.END, result)
            line_count = result.count('\n') + 1
            self.text_area.configure(height=line_count)

if __name__ == "__main__":
    root = tk.Tk()
    app = OFFCheckerGUI(root)
    root.mainloop()