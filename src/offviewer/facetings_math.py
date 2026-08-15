# File: facetings_math.py
# Version: 1.17.1
# Addons required: numpy, numba
#
import itertools
import math
import os
import pickle

import numpy as np
from numba import njit

SYMMETRY_TOLERANCE = 1e-7
ORDER_TOLERANCE = 1e-7
COLOR_MAP = {
    3: [0.3, 0.4, 1.0],   # blue
    4: [1.0, 0.5, 0.0],   # orange
    5: [0.1, 0.8, 0.2],   # green
    6: [0.0, 0.8, 0.8],   # cyan
    8: [0.9, 0.1, 0.9],   # magenta
    10: [1.0, 0.1, 0.1]   # red
}
DEFAULT_COLOR = [1.0, 0.9, 0.0]  # yellow
SUBGROUP_CACHE = {}
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__pycache__")
CACHE_FILE = os.path.join(CACHE_DIR, "facetings_cache.dat")

@njit(cache=True)
def perspective_matrix(fov_deg, aspect, near, far):
    fov_rad = math.radians(fov_deg)
    f = 1.0 / math.tan(fov_rad / 2.0)
    return np.array([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
        [0.0, 0.0, -1.0, 0.0]
    ], dtype=np.float32)

@njit(cache=True)
def rotation_matrix(rx, ry):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    rx_mat = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cx, -sx, 0.0],
        [0.0, sx, cx, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    ry_mat = np.array([
        [cy, 0.0, sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    return ry_mat @ rx_mat

def read_off(filepath, tol=1e-8):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
    if not lines:
        raise ValueError("Empty or invalid OFF file.")
    
    header = lines[0]
    if header.startswith('OFF'):
        if len(header) > 3:
            parts = header[3:].split()
            start_idx = 1
        else:
            parts = lines[1].split()
            start_idx = 2
    else:
        parts = lines[0].split()
        start_idx = 1

    n_verts = int(parts[0])
    n_faces = int(parts[1])

    raw_vertices = []
    for i in range(start_idx, start_idx + n_verts):
        raw_vertices.append([float(x) for x in lines[i].split()[:3]])

    raw_faces = []
    for i in range(start_idx + n_verts, start_idx + n_verts + n_faces):
        parts = lines[i].split()
        n_f_verts = int(parts[0])
        raw_faces.append([int(x) for x in parts[1:1+n_f_verts]])

    unique_vertices = []
    index_map = {}
    for idx, v in enumerate(raw_vertices):
        match_idx = -1
        for u_idx, uv in enumerate(unique_vertices):
            if math.isclose(v[0], uv[0], abs_tol=tol) and \
               math.isclose(v[1], uv[1], abs_tol=tol) and \
               math.isclose(v[2], uv[2], abs_tol=tol):
                match_idx = u_idx
                break
        if match_idx == -1:
            index_map[idx] = len(unique_vertices)
            unique_vertices.append(v)
        else:
            index_map[idx] = match_idx

    welded_faces = []
    for face in raw_faces:
        new_face = []
        for v_idx in face:
            mapped = index_map[v_idx]
            if not new_face or new_face[-1] != mapped:
                new_face.append(mapped)
        if len(new_face) > 1 and new_face[0] == new_face[-1]:
            new_face.pop()
        if len(new_face) >= 3:
            welded_faces.append(new_face)

    return np.array(unique_vertices), welded_faces

def write_off(filepath, vertices, faces):
    with open(filepath, 'w') as f:
        f.write("OFF\n")
        f.write(f"{len(vertices)} {len(faces)} 0\n")
        f.writelines(f"{v[0]:.16f} {v[1]:.16f} {v[2]:.16f}\n" for v in vertices)
        f.writelines(f"{len(face)} " + " ".join(map(str, face)) + "\n" for face in faces)

def normalize_face(face):
    if not isinstance(face, tuple):
        face = tuple(face)
    min_idx = min(face)
    shift = face.index(min_idx)
    shifted = face[shift:] + face[:shift]
    reversed_shifted = (shifted[0],) + shifted[1:][::-1]
    return min(shifted, reversed_shifted)

def find_cycles(vertices, adj, max_len, start_vertex=0, tol=1e-4):
    cycles = set()
    
    def dfs(start, curr, path, plane_normal, center, radius):
        if len(path) > max_len:
            return
            
        for neighbor in adj[curr]:
            if neighbor == start:
                if len(path) >= 3:
                    cycles.add(normalize_face(tuple(path)))
                continue
                
            if neighbor in path:
                continue
                
            if len(path) >= 3:
                v_neigh = vertices[neighbor] - vertices[start]
                if abs(np.dot(v_neigh, plane_normal)) > tol:
                    continue
                if abs(np.linalg.norm(vertices[neighbor] - center) - radius) > tol:
                    continue
                    
            elif len(path) == 2:
                p0, p1, p2 = vertices[start], vertices[path[1]], vertices[neighbor]
                v1 = p1 - p0
                v2 = p2 - p0
                cross = np.cross(v1, v2)
                denom = 2.0 * np.dot(cross, cross)
                if denom < 1e-8:
                    continue
                    
                plane_normal = cross / np.linalg.norm(cross)
                num = np.cross(cross, np.dot(v2, v2) * v1 - np.dot(v1, v1) * v2)
                center = p0 + num / denom
                radius = np.linalg.norm(center - p0)
                
            path.append(neighbor)
            dfs(start, neighbor, path, plane_normal, center, radius)
            path.pop()

    dfs(start_vertex, start_vertex, [start_vertex], None, None, None)
    return cycles

@njit(cache=True)
def ccw(a, b, c):
    val = (b[1] - a[1]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[1] - a[1])
    if abs(val) < 1e-8:
        return 0
    return 1 if val > 0 else -1

@njit(cache=True)
def intersect(p1, p2, p3, p4):
    if (p1[0] == p3[0] and p1[1] == p3[1]) or (p1[0] == p4[0] and p1[1] == p4[1]) or \
       (p2[0] == p3[0] and p2[1] == p3[1]) or (p2[0] == p4[0] and p2[1] == p4[1]):
        return False
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

# File: facetings_math.py
# Version: 1.16.4
# Addons required: numpy, numba

@njit(cache=True)
def find_planar_cycles_numba(vertices, on_plane, k):
    m = len(on_plane)
    capacity = 10000
    results = np.empty((capacity, k), dtype=np.int32)
    results_count = 0
    
    steps = 0
    max_steps = 200000
    
    for s in range(m - k + 1):
        path = np.empty(k, dtype=np.int32)
        path[0] = s
        
        visited = np.zeros(m, dtype=np.bool_)
        visited[s] = True
        
        child_idx = np.zeros(k, dtype=np.int32)
        child_idx[1] = s + 1
        
        d = 1
        while d > 0:
            steps += 1
            if steps > max_steps:
                break
                
            if d == k:
                p_prev1 = vertices[on_plane[path[k-2]]]
                p_curr1 = vertices[on_plane[path[k-1]]]
                p_next1 = vertices[on_plane[path[0]]]
                
                p_prev2 = p_curr1
                p_curr2 = p_next1
                p_next2 = vertices[on_plane[path[1]]]
                
                v1_1 = p_curr1 - p_prev1
                v2_1 = p_next1 - p_curr1
                len1_1 = np.linalg.norm(v1_1)
                len2_1 = np.linalg.norm(v2_1)
                valid = True
                if len1_1 > 1e-8 and len2_1 > 1e-8:
                    cross_1 = np.cross(v1_1 / len1_1, v2_1 / len2_1)
                    if np.linalg.norm(cross_1) < 1e-4:
                        valid = False
                else:
                    valid = False
                    
                if valid:
                    v1_2 = p_curr2 - p_prev2
                    v2_2 = p_next2 - p_curr2
                    len1_2 = np.linalg.norm(v1_2)
                    len2_2 = np.linalg.norm(v2_2)
                    if len1_2 > 1e-8 and len2_2 > 1e-8:
                        cross_2 = np.cross(v1_2 / len1_2, v2_2 / len2_2)
                        if np.linalg.norm(cross_2) < 1e-4:
                            valid = False
                    else:
                        valid = False
                        
                if valid and path[1] < path[k-1]:
                    if results_count >= len(results):
                        new_results = np.empty((len(results) * 2, k), dtype=np.int32)
                        new_results[:len(results)] = results
                        results = new_results
                    for i in range(k):
                        results[results_count, i] = on_plane[path[i]]
                    results_count += 1
                        
                d -= 1
                visited[path[d]] = False
                child_idx[d] += 1
                continue
                
            found_next = False
            for c in range(child_idx[d], m):
                if not visited[c]:
                    collinear = False
                    if d >= 2:
                        p_prev = vertices[on_plane[path[d-2]]]
                        p_curr = vertices[on_plane[path[d-1]]]
                        p_next = vertices[on_plane[c]]
                        v1 = p_curr - p_prev
                        v2 = p_next - p_curr
                        len1 = np.linalg.norm(v1)
                        len2 = np.linalg.norm(v2)
                        if len1 > 1e-8 and len2 > 1e-8:
                            cross = np.cross(v1 / len1, v2 / len2)
                            if np.linalg.norm(cross) < 1e-4:
                                collinear = True
                        else:
                            collinear = True
                            
                    if not collinear:
                        path[d] = c
                        visited[c] = True
                        child_idx[d] = c
                        found_next = True
                        break
                        
            if found_next:
                d += 1
                if d < k:
                    child_idx[d] = s + 1
            else:
                d -= 1
                if d > 0:
                    visited[path[d]] = False
                    child_idx[d] += 1
                    
        if steps > max_steps:
            break
            
    return results[:results_count]

def find_all_planar_faces(vertices, symmetries, tol=1e-4, log_callback=None):
    n = len(vertices)
    rep_planes = []
    seen_rep = set()
    v0 = 0
    
    if log_callback:
        log_callback("Detecting representative coplanar vertex subsets (fixed-vertex optimization)...")
        
    for j in range(1, n):
        for k in range(j+1, n):
            p0, p1, p2 = vertices[v0], vertices[j], vertices[k]
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < tol:
                continue
            normal = normal / norm_len
            d = np.dot(normal, p0)
            
            # Origin-passing plane exclusion optimization
            if abs(d) < tol:
                continue
            
            if normal[0] < -tol or (abs(normal[0]) < tol and normal[1] < -tol) or (abs(normal[0]) < tol and abs(normal[1]) < tol and normal[2] < -tol):
                normal = -normal
                d = -d
                
            key = (round(normal[0], 8), round(normal[1], 8), round(normal[2], 8), round(d, 8))
            if key not in seen_rep:
                seen_rep.add(key)
                rep_planes.append((normal, d))
                
    centroid = np.mean(vertices, axis=0)
    V = vertices - centroid
    
    if log_callback:
        log_callback("Precomputing symmetry vertex permutations...")
        
    vertex_permutations = []
    for g in symmetries:
        perm = []
        for v in V:
            gv = g @ v
            idx = np.argmin(np.linalg.norm(V - gv, axis=1))
            perm.append(idx)
        vertex_permutations.append(perm)

    candidate_faces = set()
    
    if log_callback:
        log_callback(f"Searching {len(rep_planes)} representative plane(s) for noble candidates...")
        
    for p_idx, (normal, d) in enumerate(rep_planes):
        on_plane = []
        for i in range(n):
            if abs(np.dot(vertices[i], normal) - d) < tol:
                on_plane.append(i)
        if len(on_plane) < 3:
            continue
                
        m = len(on_plane)
        max_face_size = min(12, m)
        
        on_plane_arr = np.array(on_plane, dtype=np.int32)
        for k in range(3, max_face_size + 1):
            cycles = find_planar_cycles_numba(vertices, on_plane_arr, k)
            for i in range(len(cycles)):
                rep_face = tuple(cycles[i])
                
                # Fast integer-set vertex coverage check
                visited_vertices = {perm[v] for v in rep_face for perm in vertex_permutations}
                    
                if len(visited_vertices) == n:
                    for perm in vertex_permutations:
                        mapped = tuple(perm[v] for v in rep_face)
                        candidate_faces.add(normalize_face(mapped))

    return list(candidate_faces)

def find_all_regular_faces(vertices, symmetries, tol=1e-4, max_k=10, log_callback=None):
    n = len(vertices)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dists.append(np.linalg.norm(vertices[i] - vertices[j]))

    unique_dists = []
    for d in sorted(dists):
        if d < tol:
            continue
        if not unique_dists or abs(d - unique_dists[-1]) > tol:
            unique_dists.append(d)

    centroid = np.mean(vertices, axis=0)
    V = vertices - centroid
    vertex_permutations = []
    for g in symmetries:
        perm = []
        for v in V:
            gv = g @ v
            idx = np.argmin(np.linalg.norm(V - gv, axis=1))
            perm.append(idx)
        vertex_permutations.append(perm)

    rep_faces = []

    for idx, a in enumerate(unique_dists):
        if log_callback:
            log_callback(f"Analyzing distance group {idx+1}/{len(unique_dists)} (edge length: {a:.4f})...")
        
        adj = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i+1, n):
                if abs(np.linalg.norm(vertices[i] - vertices[j]) - a) < tol:
                    adj[i].append(j)
                    adj[j].append(i)

        cycles = find_cycles(vertices, adj, max_k, start_vertex=0, tol=tol)

        for cycle in cycles:
            k = len(cycle)
            pts = vertices[list(cycle)]
            C = np.mean(pts, axis=0)

            r_dists = np.linalg.norm(pts - C, axis=1)
            R = r_dists[0]
            if np.any(np.abs(r_dists - R) > tol) or R < tol:
                continue

            edge_lens = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
            if np.any(np.abs(edge_lens - a) > tol):
                continue

            v1 = pts - C
            v2 = np.roll(pts, -1, axis=0) - C
            dot_products = np.sum(v1 * v2, axis=1)
            cos_thetas = dot_products / (R**2)

            if np.any(np.abs(cos_thetas - cos_thetas[0]) > tol):
                continue

            theta = np.arccos(np.clip(cos_thetas[0], -1.0, 1.0))
            m = round(theta * k / (2 * np.pi))

            if m < 1 or m >= (k + 1) // 2:
                continue
            if math.gcd(k, m) != 1:
                continue
            if abs(cos_thetas[0] - np.cos(2 * np.pi * m / k)) > tol:
                continue

            w = np.cross(v1, v2)
            w_norms = np.linalg.norm(w, axis=1)
            if np.any(w_norms < tol):
                continue
            w_unit = w / w_norms[:, None]
            dot_w = np.dot(w_unit, w_unit[0])
            if np.any(np.abs(dot_w - 1.0) > tol):
                continue

            rep_faces.append(normalize_face(cycle))

    candidate_faces = set()
    for face in rep_faces:
        for perm in vertex_permutations:
            mapped = tuple(perm[v] for v in face)
            candidate_faces.add(normalize_face(mapped))

    return list(candidate_faces)

def get_symmetry_group(vertices, tol=SYMMETRY_TOLERANCE):
    centroid = np.mean(vertices, axis=0)
    V = vertices - centroid
    n = len(V)
    if n < 3:
        return [np.eye(3)]

    v0 = V[0]
    v1_idx = -1
    for i in range(1, n):
        if np.linalg.norm(np.cross(v0, V[i])) > tol:
            v1_idx = i
            break
    if v1_idx == -1:
        return [np.eye(3)]
    v1 = V[v1_idx]

    v2_idx = -1
    normal = np.cross(v0, v1)
    for i in range(1, n):
        if abs(np.dot(V[i], normal)) > tol:
            v2_idx = i
            break

    if v2_idx != -1:
        v2 = V[v2_idx]
        M = np.column_stack((v0, v1, v2))
        M_inv = np.linalg.inv(M)
    else:
        v2 = normal / np.linalg.norm(normal)
        M = np.column_stack((v0, v1, v2))
        M_inv = np.linalg.inv(M)

    d0 = np.linalg.norm(v0)
    d1 = np.linalg.norm(v1)
    d01 = np.linalg.norm(v0 - v1)

    symmetries = []
    seen_matrices = []

    u0_candidates = [i for i in range(n) if abs(np.linalg.norm(V[i]) - d0) < tol]
    for u0_i in u0_candidates:
        u0 = V[u0_i]
        u1_candidates = [i for i in range(n) if abs(np.linalg.norm(V[i]) - d1) < tol and abs(np.linalg.norm(V[i] - u0) - d01) < tol]
        for u1_i in u1_candidates:
            u1 = V[u1_i]
            if v2_idx != -1:
                d2 = np.linalg.norm(v2)
                d02 = np.linalg.norm(v0 - v2)
                d12 = np.linalg.norm(v1 - v2)
                u2_candidates = [i for i in range(n) if abs(np.linalg.norm(V[i]) - d2) < tol and abs(np.linalg.norm(V[i] - u0) - d02) < tol and abs(np.linalg.norm(V[i] - u1) - d12) < tol]
                for u2_i in u2_candidates:
                    u2 = V[u2_i]
                    R = np.column_stack((u0, u1, u2)) @ M_inv
                    if (np.linalg.norm(R.T @ R - np.eye(3)) < tol 
                            and is_valid_symmetry(R, V, tol) 
                            and not any(np.linalg.norm(R - S) < tol for S in seen_matrices)):
                        seen_matrices.append(R)
                        symmetries.append(R)
            else:
                for sign in [1.0, -1.0]:
                    u2 = sign * np.cross(u0, u1)
                    u2 = u2 / np.linalg.norm(u2)
                    R = np.column_stack((u0, u1, u2)) @ M_inv
                    if (np.linalg.norm(R.T @ R - np.eye(3)) < tol 
                            and is_valid_symmetry(R, V, tol) 
                            and not any(np.linalg.norm(R - S) < tol for S in seen_matrices)):
                        seen_matrices.append(R)
                        symmetries.append(R)

    identity = np.eye(3)
    if not any(np.linalg.norm(identity - S) < tol for S in symmetries):
        symmetries.insert(0, identity)
    else:
        for idx, S in enumerate(symmetries):
            if np.linalg.norm(identity - S) < tol:
                symmetries.pop(idx)
                symmetries.insert(0, identity)
                break

    return symmetries

@njit(cache=True)
def is_valid_symmetry(R, V, tol):
    for i in range(len(V)):
        Rv = R @ V[i]
        min_dist = 999999.0
        for j in range(len(V)):
            dist = np.linalg.norm(V[j] - Rv)
            min_dist = min(min_dist, dist)
        if min_dist > tol:
            return False
    return True

def load_cache():
    global SUBGROUP_CACHE
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                SUBGROUP_CACHE = pickle.load(f)
        except (OSError, pickle.PickleError, AttributeError, EOFError, ValueError):
            SUBGROUP_CACHE = {}

def save_cache():
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(SUBGROUP_CACHE, f)
    except (OSError, pickle.PickleError):
        return

def find_isomorphism(table1, table2, orders1, orders2):
    n = len(table1)
    phi = {0: 0}
    rev_phi = {0: 0}
    
    groups2 = {}
    for i in range(n):
        groups2.setdefault(orders2[i], []).append(i)
        
    idx_to_map = [i for i in range(1, n)]
    
    def backtrack(step):
        if step == len(idx_to_map):
            return True
        u = idx_to_map[step]
        required_order = orders1[u]
        
        for v in groups2.get(required_order, []):
            if v in rev_phi:
                continue
                
            consistent = True
            for x, px in phi.items():
                xu = table1[x, u]
                if xu in phi and table2[px, v] != phi[xu]:
                    consistent = False
                    break
                ux = table1[u, x]
                if ux in phi and table2[v, px] != phi[ux]:
                    consistent = False
                    break
            if consistent:
                phi[u] = v
                rev_phi[v] = u
                if backtrack(step + 1):
                    return True
                del phi[u]
                del rev_phi[v]
        return False

    if backtrack(0):
        return phi
    return None

def find_all_subgroups(group_matrices, tol=SYMMETRY_TOLERANCE):
    n = len(group_matrices)
    element_orders = []
    for m in group_matrices:
        for k in range(1, 11):
            if np.linalg.norm(np.linalg.matrix_power(m, k) - np.eye(3)) < ORDER_TOLERANCE:
                element_orders.append(k)
                break
        else:
            element_orders.append(0)
            
    table = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            prod = group_matrices[i] @ group_matrices[j]
            idx = -1
            for k in range(n):
                if np.linalg.norm(group_matrices[k] - prod) < tol:
                    idx = k
                    break
            table[i, j] = idx
            
    sig = (n, tuple(sorted(element_orders)))
    
    if sig in SUBGROUP_CACHE:
        cached_subgroups, cached_table, cached_orders = SUBGROUP_CACHE[sig]
        phi = find_isomorphism(cached_table, table, cached_orders, element_orders)
        if phi is not None:
            mapped_subgroups = []
            for sg in cached_subgroups:
                mapped_subgroups.append(frozenset(phi[x] for x in sg))
            return mapped_subgroups, table

    cyclic_subgroups = set()
    for i in range(n):
        curr = {0, i}
        added = True
        while added:
            added = False
            for x in tuple(curr):
                prod = table[x, i]
                if prod not in curr:
                    curr.add(prod)
                    added = True
        cyclic_subgroups.add(frozenset(curr))

    subgroups = set(cyclic_subgroups)
    queue = list(cyclic_subgroups)
    
    head = 0
    while head < len(queue):
        curr = queue[head]
        head += 1
        for cyc in cyclic_subgroups:
            if not cyc.issubset(curr):
                union_set = set(curr).union(cyc)
                added = True
                while added:
                    added = False
                    for x in tuple(union_set):
                        for y in tuple(union_set):
                            prod = table[x, y]
                            if prod not in union_set:
                                union_set.add(prod)
                                added = True
                f_union = frozenset(union_set)
                if f_union not in subgroups:
                    subgroups.add(f_union)
                    queue.append(f_union)
                    
    SUBGROUP_CACHE[sig] = (list(subgroups), table, element_orders)
    save_cache()
    return list(subgroups), table

def classify_subgroup(sg_indices, full_matrices):
    sg_mats = [full_matrices[i] for i in sg_indices]
    N = len(sg_mats)
    
    has_inversion = False
    for m in sg_mats:
        if np.linalg.norm(m + np.eye(3)) < ORDER_TOLERANCE:
            has_inversion = True
            break
            
    rot_indices = [i for i in sg_indices if np.linalg.det(full_matrices[i]) > 0.9]
    M = len(rot_indices)
    chiral = (N == M)
    
    if M == 1:
        if N == 1:
            return "Trivial (C1)"
        elif has_inversion:
            return "Inversion (Ci)"
        else:
            return "Reflection (Cs)"
            
    max_order = 1
    for idx in rot_indices:
        m = full_matrices[idx]
        for k in range(1, 11):
            if np.linalg.norm(np.linalg.matrix_power(m, k) - np.eye(3)) < ORDER_TOLERANCE:
                max_order = max(max_order, k)
                break
                
    rot_type = "Cyclic"
    if M == 60:
        rot_type = "Icosahedral"
    elif M == 24:
        rot_type = "Octahedral"
    elif M == 12 and max_order == 3:
        rot_type = "Tetrahedral"
    elif M == 2 * max_order:
        rot_type = "Dihedral"
    else:
        rot_type = "Cyclic"
        max_order = M
        
    if chiral:
        if rot_type == "Icosahedral": return "Icosahedral chiral (I)"
        if rot_type == "Octahedral": return "Octahedral chiral (O)"
        if rot_type == "Tetrahedral": return "Tetrahedral chiral (T)"
        if rot_type == "Dihedral": return f"{max_order}-fold Dihedral chiral (D{max_order})"
        return f"{max_order}-fold Cyclic chiral (C{max_order})"
    else:
        if rot_type == "Icosahedral":
            return "Icosahedral full (Ih)"
        elif rot_type == "Octahedral":
            return "Octahedral full (Oh)"
        elif rot_type == "Tetrahedral":
            return "Pyritohedral (Th)" if has_inversion else "Tetrahedral full (Td)"
        elif rot_type == "Dihedral":
            if has_inversion:
                return f"{max_order}-fold Dihedral prismatic (D{max_order}h)" if max_order % 2 == 0 else f"{max_order}-fold Dihedral antiprismatic (D{max_order}d)"
            else:
                return f"{max_order}-fold Dihedral antiprismatic (D{max_order}d)" if max_order % 2 == 0 else f"{max_order}-fold Dihedral prismatic (D{max_order}h)"
        else:
            return f"{max_order}-fold Cyclic prismatic (C{max_order}h)" if has_inversion else f"{max_order}-fold Pyramidal (C{max_order}v)"

def filter_conjugacy_classes(subgroups, full_matrices, table):
    n = len(full_matrices)
    g_inv = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if table[i, j] == 0:
                g_inv[i] = j
                break

    unique_subgroups = []
    for sg in subgroups:
        is_conj = False
        for usg in unique_subgroups:
            if len(usg) != len(sg):
                continue
            for g in range(n):
                mapped = {table[g, table[x, g_inv[g]]] for x in usg}
                if mapped == sg:
                    is_conj = True
                    break
            if is_conj:
                break
        if not is_conj:
            unique_subgroups.append(sg)
            
    return unique_subgroups

def group_into_orbits(candidate_faces, vertices, symmetries, tol=1e-5):
    centroid = np.mean(vertices, axis=0)
    V = vertices - centroid
    
    vertex_permutations = []
    for g in symmetries:
        perm = []
        for v in V:
            gv = g @ v
            idx = np.argmin(np.linalg.norm(V - gv, axis=1))
            perm.append(idx)
        vertex_permutations.append(perm)

    face_set = set(candidate_faces)
    unvisited = set(candidate_faces)
    orbits = []

    while unvisited:
        face = next(iter(unvisited))
        orbit = set()
        for perm in vertex_permutations:
            mapped = tuple(perm[v] for v in face)
            norm_mapped = normalize_face(mapped)
            if norm_mapped in face_set:
                orbit.add(norm_mapped)
        for f in orbit:
            unvisited.discard(f)
        orbits.append(list(orbit))

    return orbits

@njit(cache=True)
def _backtrack_numba(
    num_orbits, num_edges, num_vertices, must_use_all_vertices,
    orbit_edges_flat, orbit_edges_counts, orbit_edges_offsets,
    orbit_vertices_flat, orbit_vertices_counts, orbit_vertices_offsets,
    edges_expiring_flat, edges_expiring_offsets,
    vertices_expiring_flat, vertices_expiring_offsets
):
    max_sols = 20000
    sol_orbits = np.empty((max_sols, num_orbits), dtype=np.int32)
    sol_lengths = np.zeros(max_sols, dtype=np.int32)
    sol_count = 0
    
    current_edge_counts = np.zeros(num_edges, dtype=np.int32)
    current_vertex_counts = np.zeros(num_vertices, dtype=np.int32)
    
    stack_o = np.empty(num_orbits + 1, dtype=np.int32)
    stack_state = np.zeros(num_orbits + 1, dtype=np.int32)
    
    selected = np.empty(num_orbits, dtype=np.int32)
    selected_len = 0
    
    stack_o[0] = 0
    stack_state[0] = 0
    sp = 0
    
    while sp >= 0:
        o_idx = stack_o[sp]
        state = stack_state[sp]
        
        if state == 0 and o_idx > 0:
            prev_idx = o_idx - 1
            
            start_e = edges_expiring_offsets[prev_idx]
            end_e = edges_expiring_offsets[prev_idx + 1]
            pruned = False
            for i in range(start_e, end_e):
                e_idx = edges_expiring_flat[i]
                if current_edge_counts[e_idx] == 1:
                    pruned = True
                    break
                    
            if not pruned and must_use_all_vertices:
                start_v = vertices_expiring_offsets[prev_idx]
                end_v = vertices_expiring_offsets[prev_idx + 1]
                for i in range(start_v, end_v):
                    v_idx = vertices_expiring_flat[i]
                    if current_vertex_counts[v_idx] == 0:
                        pruned = True
                        break
                        
            if pruned:
                sp -= 1
                continue
        
        if o_idx == num_orbits:
            if selected_len > 0 and sol_count < max_sols:
                sol_lengths[sol_count] = selected_len
                for i in range(selected_len):
                    sol_orbits[sol_count, i] = selected[i]
                sol_count += 1
            sp -= 1
            continue
            
        if state == 0:
            can_include = True
            start_e = orbit_edges_offsets[o_idx]
            end_e = orbit_edges_offsets[o_idx + 1]
            for i in range(start_e, end_e):
                e_idx = orbit_edges_flat[i]
                count = orbit_edges_counts[i]
                if current_edge_counts[e_idx] + count > 2:
                    can_include = False
                    break
                    
            if can_include:
                for i in range(start_e, end_e):
                    e_idx = orbit_edges_flat[i]
                    current_edge_counts[e_idx] += orbit_edges_counts[i]
                    
                start_v = orbit_vertices_offsets[o_idx]
                end_v = orbit_vertices_offsets[o_idx + 1]
                for i in range(start_v, end_v):
                    v_idx = orbit_vertices_flat[i]
                    current_vertex_counts[v_idx] += orbit_vertices_counts[i]
                    
                selected[selected_len] = o_idx
                selected_len += 1
                
                stack_state[sp] = 1
                
                sp += 1
                stack_o[sp] = o_idx + 1
                stack_state[sp] = 0
            else:
                stack_state[sp] = 1
                
        elif state == 1:
            if selected_len > 0 and selected[selected_len - 1] == o_idx:
                selected_len -= 1
                
                start_e = orbit_edges_offsets[o_idx]
                end_e = orbit_edges_offsets[o_idx + 1]
                for i in range(start_e, end_e):
                    e_idx = orbit_edges_flat[i]
                    current_edge_counts[e_idx] -= orbit_edges_counts[i]
                    
                start_v = orbit_vertices_offsets[o_idx]
                end_v = orbit_vertices_offsets[o_idx + 1]
                for i in range(start_v, end_v):
                    v_idx = orbit_vertices_flat[i]
                    current_vertex_counts[v_idx] -= orbit_vertices_counts[i]
            
            stack_state[sp] = 2
            
            sp += 1
            stack_o[sp] = o_idx + 1
            stack_state[sp] = 0
            
        else:
            sp -= 1
            
    return sol_orbits[:sol_count], sol_lengths[:sol_count]

def solve_facetings(orbits, candidate_faces, num_vertices, must_use_all_vertices=False, progress_callback=None):
    num_orbits = len(orbits)

    all_edges = set()
    for face in candidate_faces:
        for i in range(len(face)):
            all_edges.add(frozenset([face[i], face[(i+1)%len(face)]]))
    all_edges = list(all_edges)
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)

    orbit_edge_counts = np.zeros((num_orbits, num_edges), dtype=np.int32)
    orbit_vertex_counts = np.zeros((num_orbits, num_vertices), dtype=np.int32)

    for o_idx, orbit_faces in enumerate(orbits):
        for face in orbit_faces:
            for v in face:
                orbit_vertex_counts[o_idx][v] += 1
            for i in range(len(face)):
                e = frozenset([face[i], face[(i+1)%len(face)]])
                orbit_edge_counts[o_idx][edge_to_idx[e]] += 1

    max_orbit_for_edge = -np.ones(num_edges, dtype=np.int32)
    for e_idx in range(num_edges):
        for o_idx in range(num_orbits):
            if orbit_edge_counts[o_idx][e_idx] > 0:
                max_orbit_for_edge[e_idx] = o_idx

    max_orbit_for_vertex = -np.ones(num_vertices, dtype=np.int32)
    for v_idx in range(num_vertices):
        for o_idx in range(num_orbits):
            if orbit_vertex_counts[o_idx][v_idx] > 0:
                max_orbit_for_vertex[v_idx] = o_idx

    if must_use_all_vertices:
        for v_idx in range(num_vertices):
            if max_orbit_for_vertex[v_idx] == -1:
                return []

    orbit_edges_flat = []
    orbit_edges_counts_list = []
    orbit_edges_offsets = [0]
    for o_idx in range(num_orbits):
        for e_idx, m_o in enumerate(orbit_edge_counts[o_idx]):
            if m_o > 0:
                orbit_edges_flat.append(e_idx)
                orbit_edges_counts_list.append(m_o)
        orbit_edges_offsets.append(len(orbit_edges_flat))
        
    orbit_vertices_flat = []
    orbit_vertices_counts_list = []
    orbit_vertices_offsets = [0]
    for o_idx in range(num_orbits):
        for v_idx, m_o in enumerate(orbit_vertex_counts[o_idx]):
            if m_o > 0:
                orbit_vertices_flat.append(v_idx)
                orbit_vertices_counts_list.append(m_o)
        orbit_vertices_offsets.append(len(orbit_vertices_flat))

    edges_expiring_flat = []
    edges_expiring_offsets = [0]
    for o_idx in range(num_orbits):
        for e_idx, m_o in enumerate(max_orbit_for_edge):
            if m_o == o_idx:
                edges_expiring_flat.append(e_idx)
        edges_expiring_offsets.append(len(edges_expiring_flat))

    vertices_expiring_flat = []
    vertices_expiring_offsets = [0]
    for o_idx in range(num_orbits):
        for v_idx, m_o in enumerate(max_orbit_for_vertex):
            if m_o == o_idx:
                vertices_expiring_flat.append(v_idx)
        vertices_expiring_offsets.append(len(vertices_expiring_flat))

    sol_orbits, sol_lengths = _backtrack_numba(
        num_orbits, num_edges, num_vertices, must_use_all_vertices,
        np.array(orbit_edges_flat, dtype=np.int32),
        np.array(orbit_edges_counts_list, dtype=np.int32),
        np.array(orbit_edges_offsets, dtype=np.int32),
        np.array(orbit_vertices_flat, dtype=np.int32),
        np.array(orbit_vertices_counts_list, dtype=np.int32),
        np.array(orbit_vertices_offsets, dtype=np.int32),
        np.array(edges_expiring_flat, dtype=np.int32),
        np.array(edges_expiring_offsets, dtype=np.int32),
        np.array(vertices_expiring_flat, dtype=np.int32),
        np.array(vertices_expiring_offsets, dtype=np.int32)
    )

    solutions = []
    for s_idx in range(len(sol_orbits)):
        sol_faces = []
        length = sol_lengths[s_idx]
        for i in range(length):
            o_idx = sol_orbits[s_idx, i]
            sol_faces.extend(orbits[o_idx])
        solutions.append(sol_faces)
        if progress_callback:
            progress_callback()
            
    return solutions

def filter_duplicate_solutions(solutions, vertices, full_symmetries):
    centroid = np.mean(vertices, axis=0)
    V = vertices - centroid
    
    vertex_permutations = []
    for g in full_symmetries:
        perm = []
        for v in V:
            gv = g @ v
            idx = np.argmin(np.linalg.norm(V - gv, axis=1))
            perm.append(idx)
        vertex_permutations.append(perm)

    unique_solutions = []
    seen_equivalents = set()

    for sol in solutions:
        sol_set = frozenset(normalize_face(face) for face in sol)
        
        if sol_set in seen_equivalents:
            continue
        
        unique_solutions.append(sol)
        
        for perm in vertex_permutations:
            mapped_faces = []
            for face in sol_set:
                mapped_face = tuple(perm[v] for v in face)
                mapped_faces.append(normalize_face(mapped_face))
            seen_equivalents.add(frozenset(mapped_faces))
            
    return unique_solutions

def is_connected_polyhedron(faces):
    if not faces:
        return False
    
    edge_to_faces = {}
    for f_idx, face in enumerate(faces):
        for i in range(len(face)):
            u, v = face[i], face[(i+1)%len(face)]
            edge = frozenset([u, v])
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(f_idx)
            
    num_faces = len(faces)
    adj = [[] for _ in range(num_faces)]
    for edge, f_indices in edge_to_faces.items():
        for i in range(len(f_indices)):
            for j in range(i+1, len(f_indices)):
                u = f_indices[i]
                v = f_indices[j]
                adj[u].append(v)
                adj[v].append(u)
                
    visited = [False] * num_faces
    queue = [0]
    visited[0] = True
    count = 1
    
    head = 0
    while head < len(queue):
        curr = queue[head]
        head += 1
        for neighbor in adj[curr]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                count += 1
                
    return count == num_faces

def orient_faces_outwards(vertices, faces):
    poly_centroid = np.mean(vertices, axis=0)
    oriented_faces = []
    for face in faces:
        pts = vertices[list(face)]
        face_centroid = np.mean(pts, axis=0)
        v_out = face_centroid - poly_centroid
        
        normal = np.zeros(3)
        for i in range(len(face)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(face)]
            normal[0] += (p1[1] - p2[1]) * (p1[2] + p2[2])
            normal[1] += (p1[2] - p2[2]) * (p1[0] + p2[0])
            normal[2] += (p1[0] - p2[0]) * (p1[1] + p2[1])
            
        if np.dot(v_out, normal) < 0:
            oriented_faces.append(list(face)[::-1])
        else:
            oriented_faces.append(list(face))
    return oriented_faces

# File: facetings_math.py
# Version: 1.16.3
# Addons required: numpy, numba

# File: facetings_math.py
# Version: 1.16.5
# Addons required: numpy, numba

# File: facetings_math.py
# Version: 1.16.6
# Addons required: numpy, numba

# File: facetings_math.py
# Version: 1.16.7
# Addons required: numpy, numba

# File: facetings_math.py
# Version: 1.16.8
# Addons required: numpy, numba

# File: facetings_math.py
# Version: 1.17.0
# Addons required: numpy, numba

def blend_coplanar_faces_all_windings(vertices, faces, tol=1e-5):
    """Finds coplanar adjacent faces and blends them into single polygons, returning all possible windings."""
    planes = []
    for f_idx, face in enumerate(faces):
        pts = vertices[list(face)]
        normal = np.zeros(3)
        for i in range(len(face)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(face)]
            normal[0] += (p1[1] - p2[1]) * (p1[2] + p2[2])
            normal[1] += (p1[2] - p2[2]) * (p1[0] + p2[0])
            normal[2] += (p1[0] - p2[0]) * (p1[1] + p2[1])
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-7:
            continue
        normal = normal / norm_len
        d = np.dot(normal, pts[0])
        
        # Group by actual directed normal and distance to ensure only faces
        # facing the same direction are blended together.
        found = False
        for p_norm, p_d, f_list in planes:
            if abs(p_d - d) < tol and np.linalg.norm(p_norm - normal) < tol:
                f_list.append(f_idx)
                found = True
                break
        if not found:
            planes.append((normal, d, [f_idx]))
            
    def merge_loops_deterministic(loops):
        loops_copy = list(loops)
        merged_any = True
        while merged_any and len(loops_copy) > 1:
            merged_any = False
            for i in range(len(loops_copy)):
                for j in range(i+1, len(loops_copy)):
                    shared = set(loops_copy[i]) & set(loops_copy[j])
                    if shared:
                        v = next(iter(shared))
                        idx_i = loops_copy[i].index(v)
                        idx_j = loops_copy[j].index(v)
                        new_loop = loops_copy[i][:idx_i] + loops_copy[j][idx_j:] + loops_copy[j][:idx_j] + loops_copy[i][idx_i:]
                        loops_copy.pop(j)
                        loops_copy.pop(i)
                        loops_copy.append(new_loop)
                        merged_any = True
                        break
                if merged_any:
                    break
        return [loops_copy]

    all_components_alternatives = []
    
    for p_norm, p_d, f_list in planes:
        if len(f_list) == 1:
            all_components_alternatives.append([[faces[f_list[0]]]])
            continue
            
        # Force all faces in f_list to be oriented consistently with p_norm
        consistent_faces = []
        for idx in f_list:
            face = faces[idx]
            pts = vertices[list(face)]
            normal = np.zeros(3)
            for i in range(len(face)):
                p1 = pts[i]
                p2 = pts[(i + 1) % len(face)]
                normal[0] += (p1[1] - p2[1]) * (p1[2] + p2[2])
                normal[1] += (p1[2] - p2[2]) * (p1[0] + p2[0])
                normal[2] += (p1[0] - p2[0]) * (p1[1] + p2[1])
            if np.dot(normal, p_norm) < 0:
                consistent_faces.append(list(face)[::-1])
            else:
                consistent_faces.append(list(face))

        face_edges = []
        for f in consistent_faces:
            edges = {frozenset([f[i], f[(i+1)%len(f)]]) for i in range(len(f))}
            face_edges.append(edges)
            
        components = []
        unvisited = set(range(len(consistent_faces)))
        while unvisited:
            start = unvisited.pop()
            comp = [start]
            queue = [start]
            while queue:
                curr = queue.pop(0)
                curr_edges = face_edges[curr]
                neighbors = []
                for u in unvisited:
                    if curr_edges & face_edges[u]:
                        neighbors.append(u)
                for nbr in neighbors:
                    unvisited.remove(nbr)
                    comp.append(nbr)
                    queue.append(nbr)
            components.append(comp)

        for comp_faces_indices in components:
            if len(comp_faces_indices) == 1:
                all_components_alternatives.append([[consistent_faces[comp_faces_indices[0]]]])
                continue
                
            edge_counts = {}
            edge_to_face = {}
            for f_idx in comp_faces_indices:
                f = consistent_faces[f_idx]
                for i in range(len(f)):
                    u, v = f[i], f[(i+1)%len(f)]
                    edge_counts[(u, v)] = edge_counts.get((u, v), 0) + 1
                    edge_to_face[(u, v)] = f_idx
                    
            boundary_edges = set()
            for (u, v) in edge_counts:
                if (v, u) not in edge_counts:
                    boundary_edges.add((u, v))
                    
            if not boundary_edges:
                continue
                
            next_edge = {}
            for (u, v) in boundary_edges:
                f_idx = edge_to_face[(u, v)]
                curr_face = consistent_faces[f_idx]
                idx = curr_face.index(v)
                w = curr_face[(idx + 1) % len(curr_face)]
                
                curr_v, curr_w = v, w
                while (curr_v, curr_w) not in boundary_edges:
                    next_f_idx = edge_to_face[(curr_w, curr_v)]
                    next_face = consistent_faces[next_f_idx]
                    v_idx = next_face.index(curr_v)
                    curr_w = next_face[(v_idx + 1) % len(next_face)]
                    
                next_edge[(u, v)] = (curr_v, curr_w)
                
            loops = []
            unvisited_edges = set(boundary_edges)
            while unvisited_edges:
                edge = next(iter(unvisited_edges))
                unvisited_edges.remove(edge)
                
                loop_edges = [edge]
                curr_edge = edge
                while True:
                    nxt = next_edge.get(curr_edge)
                    if nxt is None or nxt == edge:
                        break
                    if nxt in unvisited_edges:
                        unvisited_edges.remove(nxt)
                        loop_edges.append(nxt)
                        curr_edge = nxt
                    else:
                        break
                        
                loop_vertices = [e[0] for e in loop_edges]
                loops.append(loop_vertices)
                
            alternatives = merge_loops_deterministic(loops)
            
            # Clean consecutive duplicates and simplify each alternative
            simplified_alternatives = []
            for alt in alternatives:
                simplified_alt = []
                for loop in alt:
                    cleaned_loop = []
                    for v in loop:
                        if not cleaned_loop or cleaned_loop[-1] != v:
                            cleaned_loop.append(v)
                    if len(cleaned_loop) > 1 and cleaned_loop[0] == cleaned_loop[-1]:
                        cleaned_loop.pop()
                        
                    if len(cleaned_loop) >= 3:
                        simplified_loop = []
                        m = len(cleaned_loop)
                        for i in range(m):
                            prev_pt = vertices[cleaned_loop[i-1]]
                            curr_pt = vertices[cleaned_loop[i]]
                            next_pt = vertices[cleaned_loop[(i+1)%m]]
                            v1 = curr_pt - prev_pt
                            v2 = next_pt - curr_pt
                            v1_len = np.linalg.norm(v1)
                            v2_len = np.linalg.norm(v2)
                            if v1_len > 1e-8 and v2_len > 1e-8:
                                v1_u = v1 / v1_len
                                v2_u = v2 / v2_len
                                cross_prod = np.cross(v1_u, v2_u)
                                if np.linalg.norm(cross_prod) > 1e-4:
                                    simplified_loop.append(cleaned_loop[i])
                        if len(simplified_loop) >= 3:
                            simplified_alt.append(simplified_loop)
                if simplified_alt:
                    simplified_alternatives.append(simplified_alt)
                    
            if simplified_alternatives:
                all_components_alternatives.append(simplified_alternatives)
            else:
                all_components_alternatives.append([[faces[idx] for idx in comp_faces_indices]])

    # Prevent combinatorial explosion across multiple components
    total_comb = 1
    for alt in all_components_alternatives:
        total_comb *= len(alt)
        
    if total_comb > 64:
        all_components_alternatives = [[alt[0]] for alt in all_components_alternatives]

    all_polyhedron_alternatives = []
    for combination in itertools.product(*all_components_alternatives):
        face_list = []
        for comp_loops in combination:
            face_list.extend(comp_loops)
        all_polyhedron_alternatives.append(face_list)
        
    return all_polyhedron_alternatives

def blend_coplanar_faces(vertices, faces, tol=1e-5):
    windings = blend_coplanar_faces_all_windings(vertices, faces, tol)
    return windings[0] if windings else faces# end


def has_adjacent_coplanar_faces(vertices, faces, tol=1e-5):
    """Returns True if any two adjacent faces share an edge and lie in the same geometric plane."""
    edge_to_faces = {}
    for f_idx, face in enumerate(faces):
        for i in range(len(face)):
            u, v = face[i], face[(i+1)%len(face)]
            edge = frozenset([u, v])
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(f_idx)
            
    normals = []
    for face in faces:
        # Determine geometric plane normal from a non-collinear triplet of vertices
        n_f = len(face)
        found_normal = False
        for i in range(n_f):
            p0 = vertices[face[i]]
            p1 = vertices[face[(i+1)%n_f]]
            p2 = vertices[face[(i+2)%n_f]]
            v1 = p1 - p0
            v2 = p2 - p0
            cross = np.cross(v1, v2)
            norm_len = np.linalg.norm(cross)
            if norm_len > 1e-6:
                normals.append(cross / norm_len)
                found_normal = True
                break
        if not found_normal:
            normals.append(np.array([0.0, 0.0, 1.0]))
        
    for edge, f_indices in edge_to_faces.items():
        if len(f_indices) > 1:
            for i in range(len(f_indices)):
                for j in range(i+1, len(f_indices)):
                    f1, f2 = f_indices[i], f_indices[j]
                    n1, n2 = normals[f1], normals[f2]
                    if np.linalg.norm(n1 - n2) < tol or np.linalg.norm(n1 + n2) < tol:
                        return True
    return False