# Version: 5.93
# Addons required: numpy, scipy, numba
# Jim McNeill / Google Gemini 2026

import heapq
import itertools
import json
import multiprocessing
import os

import numpy as np
from numba import njit
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

# ==============================================================================
# CONFIGURATION VARIABLES
# ==============================================================================
SYMMETRY_GROUP = "I"  # "I" (chiral I), "Ih" (full Ih), "I3" (chiral I3), "I3h" (full I3h), "O" (chiral O), "Oh" (full Oh), "T" (chiral T), "Td" (full Td), "Th" (pyritohedral Th)
TARGET_NGON = 5  # 3 to 6 works (6 and Ih with care) Use duals to generate anything with a gonality > 6
PRE_FILTER_TOPOLOGY = True  # Set to True for non-collapsed searches (like N=5), False to allow collapsed states (like 24-vertex Oh 4)
SUBDIVISION_FACTOR = 1  # Set > 0 (1 is usually enough) to find cases on symmetry boundaries (like 8 quads at each vertex Oh symmetry)
ADD_MIDPOINTS = (
    False  # Add extra starting seeds along the edges of the Schwarz triangle
)
DUPLICATE_CHIRAL_SEEDS = True  # Set to True to duplicate seeds with their reflected partner for chiral groups
DEBUG = False  # Export OFF files containing the convex hulls of the exact uniform generator seeds
HIGHLIGHT_ONE_FACE = True  # When True, color the first face in yellow (255, 255, 0)
N_COLOURS = "Valency"  # "Off", "Valency" (colours=valency), or "Minimal" (mathematical minimum even if > valency)
ALT_COLOURINGS = (
    True  # Set to True to search for and export all unique alternative colorings
)
USE_CENTROID = True
USE_RHOMBIREGULAR = False  # Rhombicosidodecahedron (I/Ih) / Rhombicuboctahedron (O/Oh) / Cuboctahedron (T/Td/Th)
USE_TRUNCATED_PRIMARY = False  # Truncated Icosahedron (I/Ih) / Truncated Cube (O/Oh) / Truncated Tetrahedron (T/Td/Th)
USE_TRUNCATED_SECONDARY = False  # Truncated Dodecahedron (I/Ih) / Truncated Octahedron (O/Oh) / Truncated Tetrahedron Dual (T/Td/Th)
USE_SNUB = (
    False  # Snub Dodecahedron (I/Ih) / Snub Cube (O/Oh) / Snub Tetrahedron (T/Td/Th)
)
PRODUCE_DUALS = "Some"  # "Yes", "No", "Some". "Some" only produces if dual gonality exceeds base gonality.
# Collapsed search configurations
COLLAPSED_SEARCH = (
    False  # Set to True to search directly on collapsed symmetry orbits (it's faster)
)
COLLAPSED_ORBIT = "5-fold"  # Vertices  I "5-fold" Ike, "3-fold" Dod, "2-fold" ID
#                                           O "4-fold" Oct, "3-fold" Cube, "2-fold" CO
# If collapsed edge states are being searched for then force vertex seeds.
if not PRE_FILTER_TOPOLOGY and SUBDIVISION_FACTOR == 0:
    SUBDIVISION_FACTOR = 1

# ==============================================================================
# 1. GEOMETRY AND GROUP THEORY SETUP
# ==============================================================================


def get_rotation_matrix(axis, theta):
    """Calculates the 3D rotation matrix around a given axis by a given angle
    using Rodriguez's rotation formula.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def generate_icosahedral_group_ih():
    """Generates the full 120 elements of the icosahedral symmetry group (Ih)
    by combining the 60 rotation matrices of the chiral group (I) with
    central inversion.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio
    u5 = np.array([0.0, 1.0, phi])  # 5-fold rotation axis
    u3 = np.array([1.0, 1.0, 1.0])  # 3-fold rotation axis

    R5 = get_rotation_matrix(u5, 2.0 * np.pi / 5.0)
    R3 = get_rotation_matrix(u3, 2.0 * np.pi / 3.0)

    rotations = [np.eye(3)]
    added = True
    while added:
        added = False
        for M in list(rotations):
            for generator in [R5, R3]:
                candidate = np.dot(generator, M)
                if not any(np.allclose(candidate, G, atol=1e-5) for G in rotations):
                    rotations.append(candidate)
                    added = True
                    if len(rotations) == 60:
                        break
            if len(rotations) == 60:
                break

    inversion = -np.eye(3)
    reflections = [np.dot(G, inversion) for G in rotations]
    return rotations, rotations + reflections


def generate_octahedral_group_oh():
    """Generates the full 48 elements of the octahedral symmetry group (Oh)
    by combining the 24 rotation matrices of the chiral group (O) with
    central inversion.
    """
    u4 = np.array([1.0, 0.0, 0.0])  # 4-fold rotation axis
    u3 = np.array([1.0, 1.0, 1.0])  # 3-fold rotation axis

    R4 = get_rotation_matrix(u4, 2.0 * np.pi / 4.0)
    R3 = get_rotation_matrix(u3, 2.0 * np.pi / 3.0)

    rotations = [np.eye(3)]
    added = True
    while added:
        added = False
        for M in list(rotations):
            for generator in [R4, R3]:
                candidate = np.dot(generator, M)
                if not any(np.allclose(candidate, G, atol=1e-5) for G in rotations):
                    rotations.append(candidate)
                    added = True
                    if len(rotations) == 24:
                        break
            if len(rotations) == 24:
                break

    inversion = -np.eye(3)
    reflections = [np.dot(G, inversion) for G in rotations]
    return rotations, rotations + reflections


def generate_tetrahedral_group_td():
    """Generates the 12 rotation matrices of the chiral group (T),
    the 24 matrices of the full tetrahedral group (Td) via diagonal plane reflections,
    and the 24 matrices of pyritohedral symmetry (Th) via central inversion.
    """
    u3 = np.array([1.0, 1.0, 1.0])  # 3-fold rotation axis
    u2 = np.array([1.0, 0.0, 0.0])  # 2-fold rotation axis

    R3 = get_rotation_matrix(u3, 2.0 * np.pi / 3.0)
    R2 = get_rotation_matrix(u2, np.pi)

    rotations = [np.eye(3)]
    added = True
    while added:
        added = False
        for M in list(rotations):
            for generator in [R3, R2]:
                candidate = np.dot(generator, M)
                if not any(np.allclose(candidate, G, atol=1e-5) for G in rotations):
                    rotations.append(candidate)
                    added = True
                    if len(rotations) == 12:
                        break
            if len(rotations) == 12:
                break

    # Td diagonal reflection plane (e.g. x = y)
    n = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    R_refl = np.eye(3) - 2.0 * np.outer(n, n)
    reflections = [np.dot(G, R_refl) for G in rotations]

    # Th central inversion
    inversion = -np.eye(3)
    inversions = [np.dot(G, inversion) for G in rotations]

    return rotations, rotations + reflections, rotations + inversions


# Generate symmetry operations based on configuration
if "I" in SYMMETRY_GROUP:
    GROUP_CHIRAL, GROUP_FULL = generate_icosahedral_group_ih()
elif "O" in SYMMETRY_GROUP:
    GROUP_CHIRAL, GROUP_FULL = generate_octahedral_group_oh()
else:
    GROUP_CHIRAL, GROUP_FULL_TD, GROUP_FULL_TH = generate_tetrahedral_group_td()
    if SYMMETRY_GROUP == "Th":
        GROUP_FULL = GROUP_FULL_TH
    else:
        GROUP_FULL = GROUP_FULL_TD

if SYMMETRY_GROUP in ["Ih", "I3h", "Oh", "Td", "Th"]:
    GROUP_GEN = GROUP_FULL
else:
    GROUP_GEN = GROUP_CHIRAL

GROUP_SIZE = len(GROUP_GEN)
GROUP_GEN_3D = np.ascontiguousarray(np.array(GROUP_GEN, dtype=np.float64))

# ==============================================================================
# 2. SYMBOLIC MULTIPLICATION TABLE
# ==============================================================================
MULT_TABLE = np.zeros((GROUP_SIZE, GROUP_SIZE), dtype=int)
for i in range(GROUP_SIZE):
    for j in range(GROUP_SIZE):
        prod = np.dot(GROUP_GEN[i], GROUP_GEN[j])
        best_k = 0
        best_dist = 1e9
        for k in range(GROUP_SIZE):
            dist = np.linalg.norm(prod - GROUP_GEN[k])
            if dist < best_dist:
                best_dist = dist
                best_k = k
        MULT_TABLE[i][j] = best_k

# Identify a true plane reflection (improper rotation with trace ≈ 1.0)
S_REFL = None
for G in GROUP_FULL:
    if np.linalg.det(G) < 0 and np.abs(np.trace(G) - 1.0) < 1e-4:
        S_REFL = G
        break

# Generate the automorphism on the elements of the generator group
AUT_SIGMA = np.zeros(GROUP_SIZE, dtype=int)
if SYMMETRY_GROUP in ["Ih", "I3h", "Oh", "Td", "Th"]:
    # For full groups, reflections are inner, so AUT_SIGMA can be the identity mapping
    for i in range(GROUP_SIZE):
        AUT_SIGMA[i] = i
else:
    # For chiral groups, map via S_REFL
    for i in range(GROUP_SIZE):
        M = np.dot(S_REFL, np.dot(GROUP_GEN[i], S_REFL))
        best_j = 0
        best_dist = 1e9
        for j in range(GROUP_SIZE):
            dist = np.linalg.norm(M - GROUP_GEN[j])
            if dist < best_dist:
                best_dist = dist
                best_j = j
        AUT_SIGMA[i] = best_j

# ==============================================================================
# 3. TOPOLOGICAL FILTERING (Numba Accelerated)
# ==============================================================================


@njit(cache=True)
def is_valid_topology_numba(template, mult_table, group_size):
    """Checks if a candidate face index template generates a valid, closed
    2-manifold with exactly group_size non-overlapping unique faces.
    """
    n = len(template)
    edge_counts = np.zeros(group_size * group_size, dtype=np.int8)
    for k in range(group_size):
        for i in range(n):
            u = mult_table[k, template[i]]
            v = mult_table[k, template[(i + 1) % n]]
            if u == v:
                return False  # Self-loop / degenerate face
            if u < v:
                idx = u * group_size + v
            else:
                idx = v * group_size + u
            edge_counts[idx] += 1

    for idx in range(group_size * group_size):
        count = edge_counts[idx]
        if count != 0 and count != 2:
            return False

    return True


@njit(cache=True)
def is_single_connected_component_numba(template, mult_table, group_size):
    """Filters out compounds by checking if the generated face-edge graph is
    fully connected.
    """
    n = len(template)
    adj = np.zeros((group_size, group_size), dtype=np.bool_)
    for k in range(group_size):
        for i in range(n):
            u = mult_table[k, template[i]]
            v = mult_table[k, template[(i + 1) % n]]
            adj[u, v] = True
            adj[v, u] = True

    visited = np.zeros(group_size, dtype=np.bool_)
    queue = np.zeros(group_size, dtype=np.int32)
    head = 0
    tail = 0

    queue[tail] = 0
    tail += 1
    visited[0] = True

    while head < tail:
        curr = queue[head]
        head += 1
        for neighbor in range(group_size):
            if adj[curr, neighbor] and not visited[neighbor]:
                visited[neighbor] = True
                queue[tail] = neighbor
                tail += 1

    return tail == group_size


@njit(cache=True)
def is_valid_topology_collapsed_numba(
    template, mult_table, vert_map, group_size, num_vertices
):
    """Numba-accelerated topological checker designed specifically for collapsed orbits.
    Evaluates topological validity directly on the merged physical vertex coordinates.
    """
    n = len(template)
    edge_counts = np.zeros(num_vertices * num_vertices, dtype=np.int8)
    for k in range(group_size):
        for i in range(n):
            u = vert_map[mult_table[k, template[i]]]
            v = vert_map[mult_table[k, template[(i + 1) % n]]]
            if u == v:
                return False  # Self-loop / degenerate face
            if u < v:
                idx = u * num_vertices + v
            else:
                idx = v * num_vertices + u
            edge_counts[idx] += 1

    for idx in range(num_vertices * num_vertices):
        count = edge_counts[idx]
        if count != 0 and count != 2:
            return False

    return True


@njit(cache=True)
def is_single_connected_component_collapsed_numba(
    template, mult_table, vert_map, group_size, num_vertices
):
    """Filters out compounds on collapsed orbits by checking connectivity on the merged physical graph."""
    n = len(template)
    adj = np.zeros((num_vertices, num_vertices), dtype=np.bool_)
    for k in range(group_size):
        for i in range(n):
            u = vert_map[mult_table[k, template[i]]]
            v = vert_map[mult_table[k, template[(i + 1) % n]]]
            adj[u, v] = True
            adj[v, u] = True

    visited = np.zeros(num_vertices, dtype=np.bool_)
    queue = np.zeros(num_vertices, dtype=np.int32)
    head = 0
    tail = 0

    queue[tail] = 0
    tail += 1
    visited[0] = True

    while head < tail:
        curr = queue[head]
        head += 1
        for neighbor in range(num_vertices):
            if adj[curr, neighbor] and not visited[neighbor]:
                visited[neighbor] = True
                queue[tail] = neighbor
                tail += 1

    return tail == num_vertices


@njit(cache=True)
def filter_canonical_combinations_numba(
    combinations_arr, n, mult_table, aut_sigma, group_size
):
    """Numba-accelerated combination orbit filtering. Removes combinations that are
    symmetrically equivalent under the full symmetry group, keeping only the
    lexicographically smallest representative.
    """
    num_combos = len(combinations_arr)
    g_map = np.empty(group_size, dtype=np.int32)
    for x in range(group_size):
        for g in range(group_size):
            if mult_table[g, x] == 0:
                g_map[x] = g
                break

    keep_mask = np.ones(num_combos, dtype=np.bool_)

    for i in range(num_combos):
        combo = combinations_arr[i]

        F = np.empty(n, dtype=np.int32)
        F[0] = 0
        for j in range(n - 1):
            F[j + 1] = combo[j]

        is_canonical = True
        for x_idx in range(n):
            x = F[x_idx]
            g = g_map[x]

            cand_rot = np.empty(n, dtype=np.int32)
            for j in range(n):
                cand_rot[j] = mult_table[g, F[j]]
            cand_rot.sort()

            for j in range(n - 1):
                val_cand = cand_rot[j + 1]
                val_combo = combo[j]
                if val_cand < val_combo:
                    is_canonical = False
                    break
                elif val_cand > val_combo:
                    break
            if not is_canonical:
                break

            cand_refl = np.empty(n, dtype=np.int32)
            for j in range(n):
                cand_refl[j] = aut_sigma[mult_table[g, F[j]]]
            cand_refl.sort()

            for j in range(n - 1):
                val_cand = cand_refl[j + 1]
                val_combo = combo[j]
                if val_cand < val_combo:
                    is_canonical = False
                    break
                elif val_cand > val_combo:
                    break
            if not is_canonical:
                break

        keep_mask[i] = is_canonical

    return keep_mask


# ==============================================================================
# 4. CONTINUOUS COORDINATE OPTIMIZATION (Numba Accelerated Residuals)
# ==============================================================================


@njit(cache=True)
def coplanarity_residuals_numba(params, template, group_gen_3d):
    """Numba-accelerated function returning the exact coplanarity residuals (length n - 3)
    augmented with pairwise vertex-collapse penalties to prevent the solver from
    getting trapped in degenerate high-symmetry corner collapses.
    """
    theta = params[0]
    phi_angle = params[1]

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi_angle)
    cos_phi = np.cos(phi_angle)

    P = np.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])

    n = len(template)
    vertices = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        idx = template[i]
        for r in range(3):
            val_v = 0.0
            for c in range(3):
                val_v += group_gen_3d[idx, r, c] * P[c]
            vertices[i, r] = val_v

    v0 = vertices[0]
    A = vertices[1] - v0
    B = vertices[2] - v0

    Ux = A[1] * B[2] - A[2] * B[1]
    Uy = A[2] * B[0] - A[0] * B[2]
    Uz = A[0] * B[1] - A[1] * B[0]

    norm_len = np.sqrt(Ux * Ux + Uy * Uy + Uz * Uz)
    num_pairs = (n * (n - 1)) // 2

    if norm_len < 1e-9:
        penalty = np.empty(n - 3 + num_pairs, dtype=np.float64)
        penalty.fill(1.0)
        return penalty

    n_vec = np.array([Ux / norm_len, Uy / norm_len, Uz / norm_len])
    residuals = np.zeros(n - 3 + num_pairs, dtype=np.float64)

    # 1. Standard coplanarity terms
    for i in range(3, n):
        residuals[i - 3] = np.dot(vertices[i] - v0, n_vec)

    # 2. Pairwise vertex collapse penalty terms (s = 0.02 is the collapse threshold)
    s_sq = 0.02 * 0.02
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = vertices[i, 0] - vertices[j, 0]
            dy = vertices[i, 1] - vertices[j, 1]
            dz = vertices[i, 2] - vertices[j, 2]
            d_sq = dx * dx + dy * dy + dz * dz

            # Penalty increases to 1.0 as d_sq approaches 0
            residuals[n - 3 + pair_idx] = np.exp(-d_sq / (2.0 * s_sq))
            pair_idx += 1

    return residuals


# ==============================================================================
# 5. GEOMETRIC ANALYSIS & FILTERING (Manifold & Symmetry Merge Engine)
# ==============================================================================


def get_unique_vertices_and_faces(vertices, faces, atol=1e-3):
    """Identifies and merges duplicate vertices (symmetric vertex collapses).
    Rewrites the face connectivity templates dynamically.
    Runs a full topological check to ensure the merged mesh is a valid,
    closed 2-manifold with no degenerate elements or edge-sharing anomalies.
    """
    unique_verts = []
    vert_map = {}
    for idx, v in enumerate(vertices):
        found = False
        for u_idx, uv in enumerate(unique_verts):
            if np.linalg.norm(v - uv) < atol:
                vert_map[idx] = u_idx
                found = True
                break
        if not found:
            vert_map[idx] = len(unique_verts)
            unique_verts.append(v)

    # Deduplicate overlapping faces (essential for collapsed orbits with face stabilizers)
    unique_face_sets = set()
    deduped_faces = []
    for face in faces:
        new_face = [vert_map[v_idx] for v_idx in face]
        if len(set(new_face)) < len(face):
            return None, None

        # Canonicalize face cycle (shifts and reversals) to detect identical faces
        n_face = len(new_face)
        cycle_reps = []
        for shift in range(n_face):
            rep_f = tuple(new_face[shift:] + new_face[:shift])
            cycle_reps.append(rep_f)

            rev_face = new_face[::-1]
            rep_r = tuple(rev_face[shift:] + rev_face[:shift])
            cycle_reps.append(rep_r)

        canonical_face = min(cycle_reps)
        if canonical_face not in unique_face_sets:
            unique_face_sets.add(canonical_face)
            deduped_faces.append(new_face)

    merged_faces = deduped_faces

    # Count edge sharing
    edge_counts = {}
    for face in merged_faces:
        n = len(face)
        for i in range(n):
            u = face[i]
            v = face[(i + 1) % n]
            if u == v:
                return None, None
            edge = tuple(sorted((u, v)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    # Every edge in a closed 2-manifold must be shared by exactly 2 faces
    for count in edge_counts.values():
        if count != 2:
            return None, None

    # Build face-to-face adjacency graph to verify connectivity
    adj = {}
    for face_idx, face in enumerate(merged_faces):
        n = len(face)
        for i in range(n):
            u = face[i]
            v = face[(i + 1) % n]
            edge = tuple(sorted((u, v)))
            adj.setdefault(edge, []).append(face_idx)

    face_adj = {i: set() for i in range(len(merged_faces))}
    for f_indices in adj.values():
        if len(f_indices) == 2:
            f1, f2 = f_indices
            face_adj[f1].add(f2)
            face_adj[f2].add(f1)

    # BFS connectivity check
    visited = set()
    queue = [0]
    visited.add(0)
    head = 0
    while head < len(queue):
        curr = queue[head]
        head += 1
        for neighbor in face_adj[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    if len(visited) != len(merged_faces):
        return None, None

    return np.array(unique_verts), merged_faces


def canonicalize_point_ih(P):
    """Maps generator point P to its lexicographically smallest image under the
    full symmetry group.
    """
    images = [np.dot(G, P) for G in GROUP_FULL]
    rounded_images = [np.round(img, 6) for img in images]
    rounded_images.sort(key=lambda v: (v[0], v[1], v[2]))
    return rounded_images[0]


def check_coplanar_transition(vertices, faces):
    """Identifies if the shape is in a coplanar transition state by checking
    if any adjacent faces sharing an edge are coplanar.
    """
    normals = []
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        norm = np.cross(v1 - v0, v2 - v0)
        norm_len = np.linalg.norm(norm)
        if norm_len > 1e-9:
            norm = norm / norm_len
        normals.append(norm)

    # Pre-build edge sets for each face to check true topological adjacency
    face_edges = []
    for face in faces:
        edges = set()
        n = len(face)
        for i in range(n):
            edges.add(tuple(sorted((face[i], face[(i + 1) % n]))))
        face_edges.append(edges)

    num_faces = len(faces)
    for i in range(num_faces):
        for j in range(i + 1, num_faces):
            # Check if they share at least one topological edge
            if not face_edges[i].isdisjoint(face_edges[j]):
                dot_val = np.abs(np.dot(normals[i], normals[j]))
                if dot_val > 0.9999:
                    return True
    return False


def calculate_edge_ratio_string(vertices, template):
    """Calculates the ratio of the longest to shortest edge length of the face
    and formats it strictly as "n-dddd".  No longer used.
    """
    face_v = [vertices[idx] for idx in template]
    lengths = []
    n = len(template)
    for i in range(n):
        lengths.append(np.linalg.norm(face_v[i] - face_v[(i + 1) % n]))

    ratio = max(lengths) / min(lengths)

    ratio_int = int(ratio)
    ratio_frac = round((ratio - ratio_int) * 10000)
    if ratio_frac >= 10000:
        ratio_int += 1
        ratio_frac -= 10000
    return f"{ratio_int}-{ratio_frac:04d}"


def find_n_colouring(faces, num_vertices, mode="Valency", find_all=False):
    """Finds partitionings of the faces of the polyhedron into vertex-disjoint exact covers
    or maximal packings of the vertices, dynamically configured by the mode.
    If find_all is True, returns all unique valid partitions excluding color permutations.
    """
    F = len(faces)
    if F == 0:
        return [] if find_all else None
    g = len(faces[0])
    V = num_vertices

    if mode == "Valency":
        if (F * g) % V != 0 or V % g != 0:
            return [] if find_all else None
        k = V // g  # exact target faces per color class
    elif mode == "Minimal":
        k = V // g  # maximum disjoint faces per color class
        if k == 0 or F % k != 0:
            return [] if find_all else None
    else:
        return [] if find_all else None

    face_verts = [set(f) for f in faces]
    all_partitions = []

    def get_covers(available_faces, forced_face):
        covers = []
        initial_covered = face_verts[forced_face].copy()

        if k * g == V:
            # --- Knuth's Algorithm X (Exact Cover) ---
            candidates = [
                f
                for f in available_faces
                if f != forced_face and face_verts[f].isdisjoint(initial_covered)
            ]

            def alg_x(current_candidates, current_cover, covered_verts):
                if len(current_cover) == k:
                    covers.append(current_cover)
                    return

                # Find the first uncovered vertex
                uncovered = -1
                for vert in range(V):
                    if vert not in covered_verts:
                        uncovered = vert
                        break
                if uncovered == -1:
                    return

                # We must cover 'uncovered' next. Branch only on faces containing it.
                branch_candidates = [
                    f for f in current_candidates if uncovered in face_verts[f]
                ]

                for f in branch_candidates:
                    new_covered = covered_verts | face_verts[f]
                    next_candidates = [
                        c
                        for c in current_candidates
                        if c != f and face_verts[c].isdisjoint(face_verts[f])
                    ]
                    alg_x(next_candidates, current_cover + [f], new_covered)

            alg_x(candidates, [forced_face], initial_covered)
        else:
            # --- Strictly Increasing Index Backtracking (for imperfect covers) ---
            candidates = [
                f
                for f in available_faces
                if f > forced_face and face_verts[f].isdisjoint(initial_covered)
            ]

            def search(candidates, current_cover, covered_verts):
                if len(current_cover) == k:
                    covers.append(current_cover)
                    return

                for idx, f in enumerate(candidates):
                    new_covered = covered_verts | face_verts[f]
                    next_candidates = [
                        c
                        for c in candidates[idx + 1 :]
                        if face_verts[c].isdisjoint(new_covered)
                    ]
                    search(next_candidates, current_cover + [f], new_covered)

            search(candidates, [forced_face], initial_covered)

        return covers

    def partition(available_faces, current_assignment):
        if not available_faces:
            all_partitions.append(current_assignment.copy())
            return not find_all

        forced_face = available_faces[0]
        covers = get_covers(available_faces, forced_face)

        for cover in covers:
            color_idx = (
                len(np.unique(current_assignment[current_assignment >= 0]))
                if np.any(current_assignment >= 0)
                else 0
            )
            for f in cover:
                current_assignment[f] = color_idx

            next_available = [f for f in available_faces if f not in cover]
            if partition(next_available, current_assignment):
                return True

            for f in cover:
                current_assignment[f] = -1

        return False

    initial_assignment = np.full(F, -1, dtype=np.int32)
    partition(list(range(F)), initial_assignment)

    if find_all:
        return all_partitions
    else:
        return all_partitions[0] if all_partitions else None


# ==============================================================================
# 6. OFF FILE EXPORT SYSTEM (16 DECIMAL PLACES WITH COLOR)
# ==============================================================================


def export_to_off(filename, P, template, seed_label=None, produced_filenames=None):
    """Generates the full 3D mesh and exports it to the 'noble' folder.
    Also generates and exports its topological dual with dual gonality and Wiener index in its filename.
    """
    os.makedirs("noble", exist_ok=True)

    vertices = [np.dot(G, P) for G in GROUP_GEN]
    faces = [[MULT_TABLE[k][idx] for idx in template] for k in range(GROUP_SIZE)]

    unique_verts, merged_faces = get_unique_vertices_and_faces(vertices, faces)
    if unique_verts is None:
        return False

    clean_filename = filename[:-4] if filename.lower().endswith(".off") else filename
    parts = clean_filename.split("-")

    # Calculate edge max/min ratio
    edge_set = set()
    for face in merged_faces:
        n_face = len(face)
        for j in range(n_face):
            edge_set.add(tuple(sorted((face[j], face[(j + 1) % n_face]))))

    edge_lengths = [
        np.linalg.norm(unique_verts[u] - unique_verts[v]) for u, v in edge_set
    ]
    if edge_lengths:
        max_edge = max(edge_lengths)
        min_edge = min(edge_lengths)
        edge_ratio = max_edge / min_edge if min_edge > 1e-9 else 0.0
    else:
        edge_ratio = 0.0

    color_palette = [
        "255 0 0",  # 0: Red
        "0 255 0",  # 1: Green
        "0 0 255",  # 2: Blue
        "255 255 0",  # 3: Yellow
        "255 0 255",  # 4: Magenta
        "0 255 255",  # 5: Cyan
        "255 128 0",  # 6: Orange
        "128 0 255",  # 7: Purple
        "0 255 128",  # 8: Spring Green
        "128 255 0",  # 9: Lime
        "0 128 255",  # 10: Azure
        "255 0 128",
    ]  # 11: Rose

    # 1. WRITE NORMAL SINGLE-COLOR FILE (to noble/)
    normal_filepath = os.path.join("noble", clean_filename + ".off")
    with open(normal_filepath, "w") as f:
        f.write("OFF\n")
        f.write(f"# {clean_filename}.off (Single Colour)\n")
        f.write(f"# Template: {list(template)}\n")
        if seed_label is not None:
            f.write(f"# Discovered via starting seed: {seed_label}\n")
        f.write(
            f"# Final generator coordinate (Cartesian): [{P[0]:.16f}, {P[1]:.16f}, {P[2]:.16f}]\n"
        )
        f.write(f"# Edge Max/Min Ratio: {edge_ratio:.16f}\n")
        f.write(f"{len(unique_verts)} {len(merged_faces)} 0\n")

        f.writelines(f"{v[0]:.16f} {v[1]:.16f} {v[2]:.16f}\n" for v in unique_verts)

        for idx, face in enumerate(merged_faces):
            face_str = " ".join(map(str, face))
            is_first_face = idx == 0
            show_highlight = is_first_face and HIGHLIGHT_ONE_FACE
            color_str = "255 255 0" if show_highlight else "255 0 0"
            f.write(f"{len(face)} {face_str} {color_str}\n")

    print(f"--> Saved normal to: {normal_filepath}")

    # 2. WRITE MULTICOLOR FILE (to noble/multicolour/)
    if N_COLOURS != "Off":
        if ALT_COLOURINGS:
            all_colors = find_n_colouring(
                merged_faces, len(unique_verts), mode=N_COLOURS, find_all=True
            )
        else:
            colors = find_n_colouring(
                merged_faces, len(unique_verts), mode=N_COLOURS, find_all=False
            )
            all_colors = [colors] if colors is not None else []

        for c_idx, colors in enumerate(all_colors):
            if colors is not None:
                os.makedirs(os.path.join("noble", "multicolour"), exist_ok=True)
                v_colors = len(np.unique(colors))
                if ALT_COLOURINGS and len(all_colors) > 1:
                    multicolour_filename = (
                        f"{clean_filename}_{v_colors}colour_alt{c_idx}.off"
                    )
                else:
                    multicolour_filename = f"{clean_filename}_{v_colors}colour.off"
                multicolour_filepath = os.path.join(
                    "noble", "multicolour", multicolour_filename
                )
                with open(multicolour_filepath, "w") as f:
                    f.write("OFF\n")
                    f.write(f"# {multicolour_filename}\n")
                    f.write(f"# Template: {list(template)}\n")
                    if seed_label is not None:
                        f.write(f"# Discovered via starting seed: {seed_label}\n")
                    f.write(
                        f"# Final generator coordinate (Cartesian): [{P[0]:.16f}, {P[1]:.16f}, {P[2]:.16f}]\n"
                    )
                    f.write(f"# Edge Max/Min Ratio: {edge_ratio:.16f}\n")
                    f.write(f"{len(unique_verts)} {len(merged_faces)} 0\n")

                    f.writelines(
                        f"{v[0]:.16f} {v[1]:.16f} {v[2]:.16f}\n" for v in unique_verts
                    )

                    for idx, face in enumerate(merged_faces):
                        face_str = " ".join(map(str, face))
                        color_str = color_palette[colors[idx] % len(color_palette)]
                        f.write(f"{len(face)} {face_str} {color_str}\n")
                print(f"--> Saved multicolour to: {multicolour_filepath}")
                if produced_filenames is not None:
                    produced_filenames.add(multicolour_filename)

    # --- DUAL MESH GENERATION ---
    if PRODUCE_DUALS == "No":
        return False

    # Compute dual vertices via polar reciprocation of face planes
    dual_vertices = []
    for face in merged_faces:
        normal = np.zeros(3)
        norm_len = 0.0
        n_face = len(face)
        for i in range(n_face):
            v0 = unique_verts[face[i]]
            v1 = unique_verts[face[(i + 1) % n_face]]
            v2 = unique_verts[face[(i + 2) % n_face]]

            cross_prod = np.cross(v1 - v0, v2 - v1)
            cross_len = np.linalg.norm(cross_prod)
            if cross_len > 1e-9:
                normal = cross_prod / cross_len
                norm_len = cross_len
                break

        if norm_len < 1e-9:
            return False  # Degenerate collinear face

        d = np.dot(unique_verts[face[0]], normal)
        if np.abs(d) < 1e-9:
            return False  # Dual vertex at infinity

        dual_vertices.append(normal / d)

    # Resize dual vertices to radius=1 before calculations and export
    dual_vertices = [v / np.linalg.norm(v) for v in dual_vertices]

    # Compute dual faces by ordering faces around each original vertex
    dual_faces = []
    for j in range(len(unique_verts)):
        faces_with_j = []
        for f_idx, face in enumerate(merged_faces):
            if j in face:
                faces_with_j.append(f_idx)

        if not faces_with_j:
            continue

        edge_to_faces = {}
        for f_idx in faces_with_j:
            face = merged_faces[f_idx]
            idx = face.index(j)
            prev_v = face[idx - 1]
            next_v = face[(idx + 1) % len(face)]
            for neighbor in (prev_v, next_v):
                edge = tuple(sorted((j, neighbor)))
                edge_to_faces.setdefault(edge, []).append(f_idx)

        face_adj = {f: [] for f in faces_with_j}
        for edge, f_indices in edge_to_faces.items():
            if len(f_indices) == 2:
                f1, f2 = f_indices
                face_adj[f1].append(f2)
                face_adj[f2].append(f1)

        curr = faces_with_j[0]
        visited_faces = [curr]
        visited_set = {curr}
        while len(visited_faces) < len(faces_with_j):
            next_candidates = face_adj[curr]
            found_next = False
            for cand in next_candidates:
                if cand not in visited_set:
                    visited_faces.append(cand)
                    visited_set.add(cand)
                    curr = cand
                    found_next = True
                    break
            if not found_next:
                break

        dual_faces.append(visited_faces)

    if not dual_faces:
        return False

    n_dual = len(dual_faces[0])

    # Check PRODUCE_DUALS condition
    if PRODUCE_DUALS == "Some" and n_dual <= len(template):
        return False

    # Calculate dual edge ratio string
    dual_edge_set = set()
    for face in dual_faces:
        n_f = len(face)
        for i in range(n_f):
            dual_edge_set.add(tuple(sorted((face[i], face[(i + 1) % n_f]))))

    dual_lengths = [
        np.linalg.norm(dual_vertices[u] - dual_vertices[v]) for u, v in dual_edge_set
    ]
    if dual_lengths:
        max_dual_edge = max(dual_lengths)
        min_dual_edge = min(dual_lengths)
        dual_edge_ratio = max_dual_edge / min_dual_edge if min_dual_edge > 1e-9 else 0.0
    else:
        dual_edge_ratio = 0.0

    num_dual_edges = len(dual_edge_set)

    # Compute Geometric Wiener index of the dual (using unit-radius vertices)
    adj_dual_weighted = [[] for _ in range(len(dual_vertices))]
    for u, v in dual_edge_set:
        dist_val = np.linalg.norm(dual_vertices[u] - dual_vertices[v])
        adj_dual_weighted[u].append((v, dist_val))
        adj_dual_weighted[v].append((u, dist_val))

    total_dist_dual = 0.0
    disconnected_dual = False
    for start in range(len(dual_vertices)):
        dist_map = {start: 0.0}
        pq = [(0.0, start)]
        while pq:
            d, curr = heapq.heappop(pq)
            if d > dist_map[curr]:
                continue
            for nbr, weight in adj_dual_weighted[curr]:
                new_d = d + weight
                if nbr not in dist_map or new_d < dist_map[nbr]:
                    dist_map[nbr] = new_d
                    heapq.heappush(pq, (new_d, nbr))
        if len(dist_map) < len(dual_vertices):
            disconnected_dual = True
        for d_val in dist_map.values():
            total_dist_dual += d_val

    if not disconnected_dual and len(dual_vertices) > 1:
        wiener_number_dual = total_dist_dual / 2.0
    else:
        wiener_number_dual = 0.0

    # Construct dual filenames
    if PRODUCE_DUALS == "Yes":
        dual_base = f"{parts[0]}-{n_dual}-{len(dual_vertices)}-{len(dual_faces)}-{num_dual_edges}-W{int(wiener_number_dual)}"
    else:
        dual_base = f"{parts[0]}-{n_dual}-{len(dual_vertices)}-{len(dual_faces)}-{num_dual_edges}-W{int(wiener_number_dual)}_dual"

    # 3. WRITE NORMAL DUAL FILE (to noble/)
    dual_filepath = os.path.join("noble", dual_base + ".off")
    with open(dual_filepath, "w") as f:
        f.write("OFF\n")
        f.write(f"# {dual_base}.off (Single Colour)\n")
        f.write(f"# Dual of template: {list(template)}\n")
        f.write(
            f"# Final generator coordinate (Cartesian): [{P[0]:.16f}, {P[1]:.16f}, {P[2]:.16f}]\n"
        )
        f.write(f"# Edge Max/Min Ratio: {dual_edge_ratio:.16f}\n")
        f.write(f"{len(dual_vertices)} {len(dual_faces)} 0\n")

        for v in dual_vertices:
            f.write(f"{v[0]:.16f} {v[1]:.16f} {v[2]:.16f}\n")

        for idx, face in enumerate(dual_faces):
            face_str = " ".join(map(str, face))
            is_first_face = idx == 0
            show_highlight = is_first_face and HIGHLIGHT_ONE_FACE
            color_str = "255 255 0" if show_highlight else "255 0 0"
            f.write(f"{len(face)} {face_str} {color_str}\n")

    print(f"--> Saved normal dual to: {dual_filepath}")

    # 4. WRITE MULTICOLOR DUAL FILE (to noble/multicolour/)
    if N_COLOURS != "Off":
        if ALT_COLOURINGS:
            all_dual_colors = find_n_colouring(
                dual_faces, len(dual_vertices), mode=N_COLOURS, find_all=True
            )
        else:
            dual_colors = find_n_colouring(
                dual_faces, len(dual_vertices), mode=N_COLOURS, find_all=False
            )
            all_dual_colors = [dual_colors] if dual_colors is not None else []

        for c_idx, dual_colors in enumerate(all_dual_colors):
            if dual_colors is not None:
                os.makedirs(os.path.join("noble", "multicolour"), exist_ok=True)
                v_dual_colors = len(np.unique(dual_colors))
                if ALT_COLOURINGS and len(all_dual_colors) > 1:
                    dual_multicolour_filename = (
                        f"{dual_base}_{v_dual_colors}colour_alt{c_idx}.off"
                    )
                else:
                    dual_multicolour_filename = f"{dual_base}_{v_dual_colors}colour.off"
                dual_multicolour_filepath = os.path.join(
                    "noble", "multicolour", dual_multicolour_filename
                )
                with open(dual_multicolour_filepath, "w") as f:
                    f.write("OFF\n")
                    f.write(f"# {dual_multicolour_filename}\n")
                    f.write(f"# Dual of template: {list(template)}\n")
                    f.write(
                        f"# Final generator coordinate (Cartesian): [{P[0]:.16f}, {P[1]:.16f}, {P[2]:.16f}]\n"
                    )
                    f.write(f"# Edge Max/Min Ratio: {dual_edge_ratio:.16f}\n")
                    f.write(f"{len(dual_vertices)} {len(dual_faces)} 0\n")

                    for v in dual_vertices:
                        f.write(f"{v[0]:.16f} {v[1]:.16f} {v[2]:.16f}\n")

                    for idx, face in enumerate(dual_faces):
                        face_str = " ".join(map(str, face))
                        color_str = color_palette[dual_colors[idx] % len(color_palette)]
                        f.write(f"{len(face)} {face_str} {color_str}\n")
                print(f"--> Saved multicolour dual to: {dual_multicolour_filepath}")
                if produced_filenames is not None:
                    produced_filenames.add(dual_multicolour_filename)

    if len(template) == n_dual and produced_filenames is not None:
        produced_filenames.add(dual_base + ".off")

    return True


def get_untriangulated_faces(vertices, hull):
    """Groups the triangular simplices of a ConvexHull into coplanar flat faces,
    and returns them as sorted vertex cycles.
    """
    simplices = hull.simplices
    planes = []
    for idx, simplex in enumerate(simplices):
        v0, v1, v2 = vertices[simplex[0]], vertices[simplex[1]], vertices[simplex[2]]
        norm = np.cross(v1 - v0, v2 - v0)
        norm_len = np.linalg.norm(norm)
        if norm_len > 1e-9:
            norm = norm / norm_len
        else:
            continue

        if np.dot(v0, norm) < 0:
            norm = -norm

        dist = np.dot(v0, norm)
        planes.append((norm, dist, idx))

    groups = []
    visited = set()
    for i in range(len(planes)):
        if i in visited:
            continue
        group = [planes[i][2]]
        visited.add(i)
        for j in range(i + 1, len(planes)):
            if j in visited:
                continue
            dot_val = np.dot(planes[i][0], planes[j][0])
            if dot_val > 0.9999 and np.abs(planes[i][1] - planes[j][1]) < 1e-4:
                group.append(planes[j][2])
                visited.add(j)
        groups.append(group)

    untriangulated_faces = []
    for group in groups:
        face_verts = set()
        for s_idx in group:
            for v_idx in simplices[s_idx]:
                face_verts.add(v_idx)
        face_verts = list(face_verts)

        if len(face_verts) == 3:
            untriangulated_faces.append(face_verts)
            continue

        poly_points = vertices[face_verts]
        centroid = np.mean(poly_points, axis=0)

        v0_idx = simplices[group[0]][0]
        v1_idx = simplices[group[0]][1]
        v2_idx = simplices[group[0]][2]
        norm = np.cross(
            vertices[v1_idx] - vertices[v0_idx], vertices[v2_idx] - vertices[v0_idx]
        )
        norm = norm / np.linalg.norm(norm)

        U = vertices[v0_idx] - centroid
        U = U / np.linalg.norm(U)
        V = np.cross(norm, U)
        V = V / np.linalg.norm(V)

        angles = []
        for v_idx in face_verts:
            vec = vertices[v_idx] - centroid
            x = np.dot(vec, U)
            y = np.dot(vec, V)
            angles.append(np.arctan2(y, x))

        sorted_indices = np.argsort(angles)
        sorted_face = [face_verts[idx] for idx in sorted_indices]
        untriangulated_faces.append(sorted_face)

    return untriangulated_faces


def export_convex_hull_to_off(filename, P, comment=""):
    """Generates the vertices of the point P under GROUP_GEN,
    computes its 3D convex hull, merges coplanar faces to avoid triangulation,
    and saves it as an OFF file with color formatting.
    """
    os.makedirs("noble", exist_ok=True)
    filepath = os.path.join("noble", filename)

    vertices = np.array([np.dot(G, P) for G in GROUP_GEN])

    unique_verts = []
    for v in vertices:
        if not any(np.linalg.norm(v - uv) < 1e-3 for uv in unique_verts):
            unique_verts.append(v)
    unique_verts = np.array(unique_verts)
    hull = ConvexHull(unique_verts)

    faces = get_untriangulated_faces(unique_verts, hull)

    with open(filepath, "w") as f:
        f.write("OFF\n")
        f.write(f"# {filename}\n")
        if comment:
            f.write(f"# {comment}\n")
        f.write(f"{len(unique_verts)} {len(faces)} 0\n")

        for v in unique_verts:
            f.write(f"{v[0]:.16f} {v[1]:.16f} {v[2]:.16f}\n")

        for idx, face in enumerate(faces):
            face_str = " ".join(map(str, face))
            is_first_face = idx == 0
            show_highlight = is_first_face and HIGHLIGHT_ONE_FACE
            color_str = "255 255 0" if show_highlight else "255 0 0"
            f.write(f"{len(face)} {face_str} {color_str}\n")

    print(f"--> [DEBUG] Saved convex hull to: {filepath}")


# ==============================================================================
# 7. PARALLELIZATION & GRID SEEDING UTILITIES
# ==============================================================================


def get_mirror_normals(V1, V2, V3):
    """Computes consistent normals pointing inward to the Schwarz triangle."""
    N1 = np.cross(V2, V3)
    N1 = N1 / np.linalg.norm(N1)
    if np.dot(N1, V1) < 0:
        N1 = -N1

    N2 = np.cross(V1, V3)
    N2 = N2 / np.linalg.norm(N2)
    if np.dot(N2, V2) < 0:
        N2 = -N2

    N3 = np.cross(V1, V2)
    N3 = N3 / np.linalg.norm(N3)
    if np.dot(N3, V3) < 0:
        N3 = -N3

    return N1, N2, N3


def orient_to_triangle(P, V1, V2, V3):
    """Ensures that the generator point P points into the interior of the Schwarz triangle."""
    centroid = V1 + V2 + V3
    if np.dot(P, centroid) < 0:
        return -P
    return P


def get_exact_centroid(V1, V2, V3):
    """Calculates the exact spherical centroid of the Schwarz triangle."""
    P = V1 + V2 + V3
    P = P / np.linalg.norm(P)
    return P


def get_exact_truncated_icosahedron(V1, V2, V3):
    """Solves for the exact coordinate of the regular truncated icosahedron (or truncated cube/tetrahedron).
    The point lies on the geodesic between V1 and V3.
    """
    N1, N2, N3 = get_mirror_normals(V1, V2, V3)
    P = np.cross(N2, N1 - N3)
    P = P / np.linalg.norm(P)
    return orient_to_triangle(P, V1, V2, V3)


def get_exact_truncated_dodecahedron(V1, V2, V3):
    """Solves for the exact coordinate of the regular truncated dodecahedron (or truncated octahedron/tetrahedron dual).
    The point lies on the geodesic between V2 and V3.
    """
    N1, N2, N3 = get_mirror_normals(V1, V2, V3)
    P = np.cross(N1, N2 - N3)
    P = P / np.linalg.norm(P)
    return orient_to_triangle(P, V1, V2, V3)


def get_exact_rhombicosidodecahedron(V1, V2, V3):
    """Solves for the exact coordinate of the regular rhombicosidodecahedron (or cuboctahedron).
    The point lies on the geodesic between V1 and V2.
    """
    N1, N2, N3 = get_mirror_normals(V1, V2, V3)
    P = np.cross(N3, N1 - N2)
    P = P / np.linalg.norm(P)
    return orient_to_triangle(P, V1, V2, V3)


def find_exact_snub_seed(group_chiral):
    """Identifies the exact coordinate of the regular snub dodecahedron."""
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    xi = 1.715560183177
    alpha = xi - 1.0 / xi
    beta = xi * phi_g + phi_g**2 + phi_g / xi

    candidates = [
        np.array([2.0 * alpha, 2.0, 2.0 * beta]),
        np.array([2.0 * alpha, 2.0, -2.0 * beta]),
        np.array([2.0 * alpha, -2.0, 2.0 * beta]),
        np.array([-2.0 * alpha, 2.0, 2.0 * beta]),
        np.array([2.0 * beta, 2.0 * alpha, 2.0]),
        np.array([2.0 * beta, 2.0 * alpha, 2.0]),
        np.array([2.0, 2.0 * beta, 2.0 * alpha]),
    ]

    for p_start in candidates:
        P = p_start / np.linalg.norm(p_start)
        vertices = np.array([np.dot(G, P) for G in group_chiral])

        dists = []
        for i in range(60):
            for j in range(i + 1, 60):
                dists.append(np.linalg.norm(vertices[i] - vertices[j]))
        dists.sort()

        edge_len = dists[0]
        if np.allclose(dists[:150], edge_len, atol=1e-4):
            return P

    return None


def find_exact_snub_cube_seed(group_chiral):
    """Identifies the exact coordinate of the regular snub cube."""
    xi = 1.465571231877
    candidates = [
        np.array([1.0, xi, 1.0 / xi]),
        np.array([1.0, xi, -1.0 / xi]),
        np.array([1.0, -xi, 1.0 / xi]),
        np.array([-1.0, xi, 1.0 / xi]),
    ]
    for p_start in candidates:
        P = p_start / np.linalg.norm(p_start)
        vertices = np.array([np.dot(G, P) for G in group_chiral])

        dists = []
        for i in range(24):
            for j in range(i + 1, 24):
                dists.append(np.linalg.norm(vertices[i] - vertices[j]))
        dists.sort()

        edge_len = dists[0]
        if np.allclose(dists[:60], edge_len, atol=1e-4):
            return P

    return None


def find_exact_snub_tetrahedron_seed(group_chiral):
    """Identifies the exact coordinate of the snub tetrahedron (regular icosahedron under chiral T symmetry)."""
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    return np.array([0.0, 1.0, phi_g]) / np.sqrt(2.0 + phi_g)


def generate_landmark_seeds(n_subdiv):
    """Generates a deterministic grid of starting points over a single Schwarz triangle
    on the unit sphere using barycentric coordinates.
    For starting points lying on high-symmetry boundaries or corners, both the exact
    unperturbed coordinate and a slightly perturbed interior coordinate are evaluated
    to ensure thoroughness.
    """
    if SYMMETRY_GROUP in ["I", "Ih"]:
        phi_g = (1.0 + np.sqrt(5.0)) / 2.0
        V1 = np.array([0.0, 1.0, phi_g]) / np.sqrt(2.0 + phi_g)
        V2 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        V3 = np.array([1.0, phi_g**2, phi_g]) / (2.0 * phi_g)
    elif SYMMETRY_GROUP in ["I3", "I3h"]:
        phi_g = (1.0 + np.sqrt(5.0)) / 2.0
        V1 = np.array([0.0, 1.0, phi_g]) / np.sqrt(2.0 + phi_g)
        V2 = np.array([phi_g, 0.0, 1.0]) / np.sqrt(2.0 + phi_g)
        V3 = np.array([0.0, 0.0, 1.0])
    elif "O" in SYMMETRY_GROUP:  # "O" or "Oh"
        V1 = np.array([1.0, 0.0, 0.0])
        V2 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        V3 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    else:  # "T" or "Td" or "Th"
        V1 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        V2 = np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)
        V3 = np.array([1.0, 0.0, 0.0])

    seeds = []

    if COLLAPSED_SEARCH:
        # Optimization: Only generate the single target corner corresponding to the collapsed orbit
        if COLLAPSED_ORBIT in ["5-fold", "4-fold"]:
            P = V1
            label = f"Target Corner ({COLLAPSED_ORBIT})"
        elif COLLAPSED_ORBIT == "3-fold":
            P = V2
            label = "Target Corner (3-fold)"
        else:
            P = V3
            label = "Target Corner (2-fold)"

        theta = np.arccos(P[2])
        phi = np.arctan2(P[1], P[0])
        seeds.append((theta, phi, label))
        return seeds

    if n_subdiv > 0:
        for i in range(n_subdiv + 1):
            for j in range(n_subdiv + 1 - i):
                k = n_subdiv - i - j
                w1_raw = i / n_subdiv
                w2_raw = j / n_subdiv
                w3_raw = k / n_subdiv

                # Check if the seed lies on a high-symmetry corner or boundary edge
                is_on_boundary = (
                    w1_raw == 0.0
                    or w2_raw == 0.0
                    or w3_raw == 0.0
                    or w1_raw == 1.0
                    or w2_raw == 1.0
                    or w3_raw == 1.0
                )

                # 1. Add the exact unperturbed grid point
                P_raw = w1_raw * V1 + w2_raw * w2_raw * V2 + w3_raw * V3
                norm_raw = np.linalg.norm(P_raw)
                if norm_raw > 1e-9:
                    P = P_raw / norm_raw
                    theta = np.arccos(P[2])
                    phi = np.arctan2(P[1], P[0])
                    label = f"Grid Point (w1={w1_raw:.2f}, w2={w2_raw:.2f}, w3={w3_raw:.2f})"
                    seeds.append((theta, phi, label))

                # 2. If it lies on a boundary, also add a slightly perturbed interior version
                if is_on_boundary:
                    w1 = 0.98 * w1_raw + 0.02 * (1.0 / 3.0)
                    w2 = 0.98 * w2_raw + 0.02 * (1.0 / 3.0)
                    w3 = 0.98 * w3_raw + 0.02 * (1.0 / 3.0)

                    P_perturbed = w1 * V1 + w2 * w2 * V2 + w3 * V3
                    norm_pert = np.linalg.norm(P_perturbed)
                    if norm_pert > 1e-9:
                        P = P_perturbed / norm_pert
                        theta = np.arccos(P[2])
                        phi = np.arctan2(P[1], P[0])
                        label = f"Grid Point (w1={w1_raw:.2f}, w2={w2_raw:.2f}, w3={w3_raw:.2f}) (Perturbed)"
                        seeds.append((theta, phi, label))

        if ADD_MIDPOINTS:
            for i in range(n_subdiv):
                w1_raw = (i + 0.5) / n_subdiv
                w2_raw = 1.0 - w1_raw

                # Unperturbed
                P = w1_raw * V1 + w2_raw * V2
                P = P / np.linalg.norm(P)
                label = f"Midpoint Edge 1 (w1={w1_raw:.2f}, w2={w2_raw:.2f}, w3=0.00)"
                seeds.append((np.arccos(P[2]), np.arctan2(P[1], P[0]), label))

                # Perturbed
                w1 = 0.98 * w1_raw + 0.02 * (1.0 / 3.0)
                w2 = 0.98 * w2_raw + 0.02 * (1.0 / 3.0)
                w3 = 0.02 * (1.0 / 3.0)
                P = w1 * V1 + w2 * V2 + w3 * V3
                P = P / np.linalg.norm(P)
                label = f"Midpoint Edge 1 (w1={w1_raw:.2f}, w2={w2_raw:.2f}, w3=0.00) (Perturbed)"
                seeds.append((np.arccos(P[2]), np.arctan2(P[1], P[0]), label))

            for j in range(n_subdiv):
                w2_raw = (j + 0.5) / n_subdiv
                w3_raw = 1.0 - w2_raw

                # Unperturbed
                P = w2_raw * V2 + w3_raw * V3
                P = P / np.linalg.norm(P)
                label = f"Midpoint Edge 2 (w1=0.00, w2={w2_raw:.2f}, w3={w3_raw:.2f})"
                seeds.append((np.arccos(P[2]), np.arctan2(P[1], P[0]), label))

                # Perturbed
                w1 = 0.02 * (1.0 / 3.0)
                w2 = 0.98 * w2_raw + 0.02 * (1.0 / 3.0)
                w3 = 0.98 * w3_raw + 0.02 * (1.0 / 3.0)
                P = w1 * V1 + w2 * V2 + w3 * V3
                P = P / np.linalg.norm(P)
                label = f"Midpoint Edge 2 (w1=0.00, w2={w2_raw:.2f}, w3={w3_raw:.2f}) (Perturbed)"
                seeds.append((np.arccos(P[2]), np.arctan2(P[1], P[0]), label))

            for k in range(n_subdiv):
                w3_raw = (k + 0.5) / n_subdiv
                w1_raw = 1.0 - w3_raw

                # Unperturbed
                P = w1_raw * V1 + w3_raw * V3
                P = P / np.linalg.norm(P)
                label = f"Midpoint Edge 3 (w1={w1_raw:.2f}, w2=0.00, w3={w3_raw:.2f})"
                seeds.append((np.arccos(P[2]), np.arctan2(P[1], P[0]), label))

                # Perturbed
                w1 = 0.98 * w1_raw + 0.02 * (1.0 / 3.0)
                w2 = 0.02 * (1.0 / 3.0)
                w3 = 0.98 * w3_raw + 0.02 * (1.0 / 3.0)
                P = w1 * V1 + w2 * V2 + w3 * V3
                P = P / np.linalg.norm(P)
                label = f"Midpoint Edge 3 (w1={w1_raw:.2f}, w2=0.00, w3={w3_raw:.2f}) (Perturbed)"
                seeds.append((np.arccos(P[2]), np.arctan2(P[1], P[0]), label))

    exact_points = []
    if USE_CENTROID:
        exact_points.append((get_exact_centroid(V1, V2, V3), "Exact Centroid"))

    if SYMMETRY_GROUP in ["I", "Ih"]:
        if USE_RHOMBIREGULAR:
            exact_points.append(
                (
                    get_exact_rhombicosidodecahedron(V1, V2, V3),
                    "Exact Rhombicosidodecahedron",
                )
            )
        if USE_TRUNCATED_PRIMARY:
            exact_points.append(
                (
                    get_exact_truncated_icosahedron(V1, V2, V3),
                    "Exact Truncated Icosahedron",
                )
            )
        if USE_TRUNCATED_SECONDARY:
            exact_points.append(
                (
                    get_exact_truncated_dodecahedron(V1, V2, V3),
                    "Exact Truncated Dodecahedron",
                )
            )
        if USE_SNUB:
            exact_points.append(
                (find_exact_snub_seed(GROUP_CHIRAL), "Exact Snub Dodecahedron")
            )
    elif SYMMETRY_GROUP in ["I3", "I3h"]:
        if USE_RHOMBIREGULAR:
            exact_points.append(
                (
                    get_exact_rhombicosidodecahedron(V1, V2, V3),
                    "Exact Rhombicosidodecahedron (I3)",
                )
            )
        if USE_TRUNCATED_PRIMARY:
            exact_points.append(
                (
                    get_exact_truncated_icosahedron(V1, V2, V3),
                    "Exact Truncated Icosahedron (I3)",
                )
            )
        if USE_TRUNCATED_SECONDARY:
            exact_points.append(
                (
                    get_exact_truncated_dodecahedron(V1, V2, V3),
                    "Exact Truncated Dodecahedron (I3)",
                )
            )
    elif "O" in SYMMETRY_GROUP:  # "O" or "Oh"
        if USE_RHOMBIREGULAR:
            exact_points.append(
                (
                    get_exact_rhombicosidodecahedron(V1, V2, V3),
                    "Exact Rhombicuboctahedron",
                )
            )
        if USE_TRUNCATED_PRIMARY:
            exact_points.append(
                (get_exact_truncated_icosahedron(V1, V2, V3), "Exact Truncated Cube")
            )
        if USE_TRUNCATED_SECONDARY:
            exact_points.append(
                (
                    get_exact_truncated_dodecahedron(V1, V2, V3),
                    "Exact Truncated Octahedron",
                )
            )
        if USE_SNUB:
            exact_points.append(
                (find_exact_snub_cube_seed(GROUP_CHIRAL), "Exact Snub Cube")
            )
    else:  # "T" or "Td" or "Th"
        if USE_RHOMBIREGULAR:
            exact_points.append(
                (
                    get_exact_rhombicosidodecahedron(V1, V2, V3),
                    "Exact Rhombiregular (Cuboctahedron)",
                )
            )
        if USE_TRUNCATED_PRIMARY:
            exact_points.append(
                (
                    get_exact_truncated_icosahedron(V1, V2, V3),
                    "Exact Truncated Tetrahedron (Primary)",
                )
            )
        if USE_TRUNCATED_SECONDARY:
            exact_points.append(
                (
                    get_exact_truncated_dodecahedron(V1, V2, V3),
                    "Exact Truncated Tetrahedron (Secondary)",
                )
            )
        if USE_SNUB:
            exact_points.append(
                (
                    find_exact_snub_tetrahedron_seed(GROUP_CHIRAL),
                    "Exact Snub Tetrahedron (Icosahedron)",
                )
            )

    M_triangle = np.column_stack((V1, V2, V3))

    for P_norm, label in exact_points:
        if P_norm is None:
            continue
        for G in GROUP_FULL:
            Q = np.dot(G, P_norm)
            try:
                w = np.linalg.solve(M_triangle, Q)
                if np.all(w >= -1e-5):
                    theta = np.arccos(np.clip(Q[2], -1.0, 1.0))
                    phi_angle = np.arctan2(Q[1], Q[0])
                    seeds.append((theta, phi_angle, label))
                    break
            except np.linalg.LinAlgError:
                continue

    # For chiral groups, duplicate all seeds with their reflected partner if configured
    if DUPLICATE_CHIRAL_SEEDS and SYMMETRY_GROUP in ["I", "I3", "O", "T"]:
        chiral_seeds = []
        for theta, phi_angle, label in seeds:
            chiral_seeds.append((theta, phi_angle, label))
            sin_t = np.sin(theta)
            P_vec = np.array(
                [sin_t * np.cos(phi_angle), sin_t * np.sin(phi_angle), np.cos(theta)]
            )
            P_refl = np.dot(S_REFL, P_vec)
            theta_refl = np.arccos(np.clip(P_refl[2], -1.0, 1.0))
            phi_refl = np.arctan2(P_refl[1], P_refl[0])
            chiral_seeds.append((theta_refl, phi_refl, label + " (Reflected)"))
        seeds = chiral_seeds

    return seeds


def optimize_layout_wrapper(args):
    """Worker process target for parallelized continuous optimization.
    Runs the JIT-compiled Least-Squares solver on all seeds, with an immediate
    bypass check for flat plateaus where the seed is already a perfect solution.
    Also implements a 1D sweep along the mirror boundaries if a seed lies on them.
    """
    template, seeds, group_gen_3d = args
    arr_template = np.array(template, dtype=np.int32)
    local_solutions = []

    # Standard Schwarz Triangle corner definitions
    if SYMMETRY_GROUP in ["I", "Ih"]:
        phi_g = (1.0 + np.sqrt(5.0)) / 2.0
        V1 = np.array([0.0, 1.0, phi_g]) / np.sqrt(2.0 + phi_g)
        V2 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        V3 = np.array([1.0, phi_g**2, phi_g]) / (2.0 * phi_g)
    elif SYMMETRY_GROUP in ["I3", "I3h"]:
        phi_g = (1.0 + np.sqrt(5.0)) / 2.0
        V1 = np.array([0.0, 1.0, phi_g]) / np.sqrt(2.0 + phi_g)
        V2 = np.array([phi_g, 0.0, 1.0]) / np.sqrt(2.0 + phi_g)
        V3 = np.array([0.0, 0.0, 1.0])
    elif "O" in SYMMETRY_GROUP:  # "O" or "Oh"
        V1 = np.array([1.0, 0.0, 0.0])
        V2 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        V3 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    else:  # "T" or "Td" or "Th"
        V1 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        V2 = np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)
        V3 = np.array([1.0, 0.0, 0.0])

    # Slerp helper for boundary sweeps
    def slerp(v1, v2, t):
        omega = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        sin_omega = np.sin(omega)
        if sin_omega < 1e-9:
            return (1.0 - t) * v1 + t * v2
        return (np.sin((1.0 - t) * omega) / sin_omega) * v1 + (
            np.sin(t * omega) / sin_omega
        ) * v2

    def get_coords(v):
        return np.array([np.arccos(np.clip(v[2], -1.0, 1.0)), np.arctan2(v[1], v[0])])

    # ==============================================================================
    # CRITICAL 3-POINT FLAT EDGE BYPASS (Disabled for triangles n = 3)
    # ==============================================================================
    # Evaluates the boundary edges of the Schwarz triangle at three non-degenerate
    # points (25%, 50%, 75%). If all three are flat (residual < 1e-5), the entire
    # boundary edge is a continuous line of perfect solutions. We can instantly
    # accept the midpoint (50%) as the non-degenerate representative and bypass
    # all solvers entirely.
    # ==============================================================================
    if len(template) > 3:
        edges = [
            (V1, V2, "Boundary Edge V1-V2"),
            (V2, V3, "Boundary Edge V2-V3"),
            (V3, V1, "Boundary Edge V3-V1"),
        ]

        for va, vb, edge_label in edges:
            p25 = slerp(va, vb, 0.25)
            p50 = slerp(va, vb, 0.50)
            p75 = slerp(va, vb, 0.75)

            res25 = coplanarity_residuals_numba(
                get_coords(p25), arr_template, group_gen_3d
            )
            res50 = coplanarity_residuals_numba(
                get_coords(p50), arr_template, group_gen_3d
            )
            res75 = coplanarity_residuals_numba(
                get_coords(p75), arr_template, group_gen_3d
            )

            n_coplanar = len(template) - 3
            if (
                np.all(np.abs(res25[:n_coplanar]) < 1e-5)
                and np.all(np.abs(res50[:n_coplanar]) < 1e-5)
                and np.all(np.abs(res75[:n_coplanar]) < 1e-5)
            ):
                local_solutions.append(
                    (get_coords(p50), "Flat-Edge-Bypass", edge_label)
                )

    # ==============================================================================
    # STANDARD SOLVER LOOP
    # ==============================================================================
    for seed in seeds:
        theta_val, phi_val, label = seed
        seed_coords = np.array([theta_val, phi_val])

        # 1. Direct bypass check: use 1e-5 to handle double-precision frame mismatches
        # Moved to the top to catch perfect exact seeds (including those on boundaries) immediately
        init_res = coplanarity_residuals_numba(seed_coords, arr_template, group_gen_3d)
        n_coplanar = len(template) - 3
        if np.all(np.abs(init_res[:n_coplanar]) < 1e-5) and label.startswith("Exact"):
            local_solutions.append((seed_coords, "Direct-Evaluation", label))
            continue

        # If we are running a collapsed search, we must NOT allow the solver to slide
        # off the target corner into the interior (which would generate 60 vertices instead of 20)
        if "Target Corner" in label:
            continue

        # Convert seed coordinates to 3D unit vector
        sin_theta = np.sin(theta_val)
        P_seed = np.array(
            [
                sin_theta * np.cos(phi_val),
                sin_theta * np.sin(phi_val),
                np.cos(theta_val),
            ]
        )

        # Normals of the three boundary planes of the Schwarz triangle
        N12 = np.cross(V1, V2)
        N12 = N12 / np.linalg.norm(N12)

        N23 = np.cross(V2, V3)
        N23 = N23 / np.linalg.norm(N23)

        N31 = np.cross(V3, V1)
        N31 = N31 / np.linalg.norm(N31)

        # Detect which boundary edges the seed lies on
        boundaries_to_sweep = []
        if np.abs(np.dot(P_seed, N12)) < 1e-4:
            boundaries_to_sweep.append((V1, V2))
        if np.abs(np.dot(P_seed, N23)) < 1e-4:
            boundaries_to_sweep.append((V2, V3))
        if np.abs(np.dot(P_seed, N31)) < 1e-4:
            boundaries_to_sweep.append((V3, V1))

        found_1d_sol = False
        for va, vb in boundaries_to_sweep:
            # 1D sweep along the geodesic arc between va and vb using Slerp (10 points)
            t_sweep = np.linspace(0.0, 1.0, 10)

            prev_t = 0.0
            p0 = slerp(va, vb, 0.0)
            prev_res_arr = coplanarity_residuals_numba(
                get_coords(p0), arr_template, group_gen_3d
            )
            prev_res = prev_res_arr[0]

            for t_curr in t_sweep[1:]:
                p_curr_vec = slerp(va, vb, t_curr)
                coords_curr = get_coords(p_curr_vec)
                res_curr_arr = coplanarity_residuals_numba(
                    coords_curr, arr_template, group_gen_3d
                )
                res_curr = res_curr_arr[0]

                # Check for sign change (root crossing)
                if prev_res * res_curr < 0.0:
                    # Run 1D bisection to find the exact zero-crossing in t-space
                    low_t = prev_t
                    high_t = t_curr
                    sign_low = prev_res
                    for _ in range(40):
                        mid_t = (low_t + high_t) / 2.0
                        mid_vec = slerp(va, vb, mid_t)
                        mid_coords = get_coords(mid_vec)
                        mid_res = coplanarity_residuals_numba(
                            mid_coords, arr_template, group_gen_3d
                        )[0]
                        if mid_res * sign_low < 0.0:
                            high_t = mid_t
                        else:
                            low_t = mid_t
                            sign_low = mid_res

                    final_vec = slerp(va, vb, (low_t + high_t) / 2.0)
                    final_coords = get_coords(final_vec)

                    # Run a quick local refinement on the bisection result
                    res_refine = least_squares(
                        coplanarity_residuals_numba,
                        final_coords,
                        args=(arr_template, group_gen_3d),
                        method="trf",
                        xtol=1e-15,
                        ftol=1e-15,
                    )
                    if res_refine.success and np.all(
                        np.abs(res_refine.fun[:n_coplanar]) < 1e-11
                    ):
                        final_coords = res_refine.x

                    # Validate that ALL coplanarity residuals are near-zero before accepting
                    final_residuals = coplanarity_residuals_numba(
                        final_coords, arr_template, group_gen_3d
                    )
                    if np.all(np.abs(final_residuals[:n_coplanar]) < 1e-10):
                        local_solutions.append(
                            (final_coords, "1D-Geodesic-Bisection", label)
                        )
                        found_1d_sol = True

                prev_t = t_curr
                prev_res = res_curr

        if found_1d_sol:
            continue

        # 3. Otherwise, run the Trust-Region solver (failsafe fallback for near-boundary states)
        res = least_squares(
            coplanarity_residuals_numba,
            seed_coords,
            args=(arr_template, group_gen_3d),
            method="trf",
            xtol=1e-14,
            ftol=1e-14,
        )
        if res.success and np.all(np.abs(res.fun[:n_coplanar]) < 1e-10):
            local_solutions.append((res.x, "Least-Squares", label))

    return template, local_solutions


def export_debug_hulls():
    """Generates and exports the convex hulls of the uniform generator seeds to 'noble' folder."""
    print("\n[DEBUG] Exporting convex hulls of uniform generator seeds...")
    if SYMMETRY_GROUP in ["I", "Ih"]:
        phi_g = (1.0 + np.sqrt(5.0)) / 2.0
        V1 = np.array([0.0, 1.0, phi_g]) / np.sqrt(2.0 + phi_g)
        V2 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        V3 = np.array([1.0, phi_g**2, phi_g]) / (2.0 * phi_g)

        p_rhomb = get_exact_rhombicosidodecahedron(V1, V2, V3)
        p_tr_ico = get_exact_truncated_icosahedron(V1, V2, V3)
        p_tr_dod = get_exact_truncated_dodecahedron(V1, V2, V3)
        p_snub = find_exact_snub_seed(GROUP_CHIRAL)

        targets = [
            (
                "debug_hull_rhombicosidodecahedron.off",
                p_rhomb,
                "Rhombicosidodecahedron",
            ),
            ("debug_hull_truncated_icosahedron.off", p_tr_ico, "Truncated Icosahedron"),
            (
                "debug_hull_truncated_dodecahedron.off",
                p_tr_dod,
                "Truncated Dodecahedron",
            ),
        ]
        if p_snub is not None:
            targets.append(
                ("debug_hull_snub_dodecahedron.off", p_snub, "Snub Icosahedron")
            )
    elif SYMMETRY_GROUP in ["I3", "I3h"]:
        phi_g = (1.0 + np.sqrt(5.0)) / 2.0
        V1 = np.array([0.0, 1.0, phi_g]) / np.sqrt(2.0 + phi_g)
        V2 = np.array([phi_g, 0.0, 1.0]) / np.sqrt(2.0 + phi_g)
        V3 = np.array([0.0, 0.0, 1.0])

        p_rhomb = get_exact_rhombicosidodecahedron(V1, V2, V3)
        p_tr_ico = get_exact_truncated_icosahedron(V1, V2, V3)
        p_tr_dod = get_exact_truncated_dodecahedron(V1, V2, V3)

        targets = [
            (
                "debug_hull_rhombicosidodecahedron_I3.off",
                p_rhomb,
                "Rhombicosidodecahedron (I3)",
            ),
            (
                "debug_hull_truncated_icosahedron_I3.off",
                p_tr_ico,
                "Truncated Icosahedron (I3)",
            ),
            (
                "debug_hull_truncated_dodecahedron_I3.off",
                p_tr_dod,
                "Truncated Dodecahedron (I3)",
            ),
        ]
    elif "O" in SYMMETRY_GROUP:  # "O" or "Oh"
        V1 = np.array([1.0, 0.0, 0.0])
        V2 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        V3 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)

        p_cub = get_exact_rhombicosidodecahedron(V1, V2, V3)
        p_tr_cub = get_exact_truncated_icosahedron(V1, V2, V3)
        p_tr_oct = get_exact_truncated_dodecahedron(V1, V2, V3)
        p_snub = find_exact_snub_cube_seed(GROUP_CHIRAL)

        targets = [
            ("debug_hull_cuboctahedron.off", p_cub, "Cuboctahedron"),
            ("debug_hull_truncated_cube.off", p_tr_cub, "Truncated Cube"),
            ("debug_hull_truncated_octahedron.off", p_tr_oct, "Truncated Octahedron"),
        ]
        if p_snub is not None:
            targets.append(("debug_hull_snub_cube.off", p_snub, "Snub Cube"))
    else:  # "T" or "Td" or "Th"
        V1 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        V2 = np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)
        V3 = np.array([1.0, 0.0, 0.0])

        p_cub = get_exact_rhombicosidodecahedron(V1, V2, V3)
        p_tr_pri = get_exact_truncated_icosahedron(V1, V2, V3)
        p_tr_sec = get_exact_truncated_dodecahedron(V1, V2, V3)
        p_snub = find_exact_snub_tetrahedron_seed(GROUP_CHIRAL)

        targets = [
            ("debug_hull_cuboctahedron.off", p_cub, "Cuboctahedron"),
            (
                "debug_hull_truncated_tetrahedron_primary.off",
                p_tr_pri,
                "Truncated Tetrahedron Primary",
            ),
            (
                "debug_hull_truncated_tetrahedron_secondary.off",
                p_tr_sec,
                "Truncated Tetrahedron Secondary",
            ),
        ]
        if p_snub is not None:
            targets.append(
                (
                    "debug_hull_snub_tetrahedron.off",
                    p_snub,
                    "Snub Tetrahedron (Icosahedron)",
                )
            )

    for filename, P, comment in targets:
        export_convex_hull_to_off(filename, P, comment)
    print("[DEBUG] Export complete.\n")


# ==============================================================================
# 8. MAIN SEARCH LOOP
# ==============================================================================


def canonicalize_face_cycle(coords):
    """Finds the absolute lexicographically smallest cyclic representation
    of the face coordinates (forward or backward) to preserve cycle
    ordering and avoid false faceting duplicates.
    """
    n = len(coords)
    rounded = [tuple(round(x, 5) for x in v) for v in coords]

    representations = []
    for shift in range(n):
        rep_f = rounded[shift:] + rounded[:shift]
        representations.append(tuple(rep_f))

        rev = rounded[::-1]
        rep_r = rev[shift:] + rev[:shift]
        representations.append(tuple(rep_r))

    representations.sort()
    return representations[0]


def find_and_save_noble_polyhedra():
    """Searches for unique N-gonal noble polyhedra. Uses a parallelized CPU
    worker pool and JIT-compiled topological filters for maximum performance.
    """
    SUPPRESS_COPLANAR = True

    if DEBUG:
        export_debug_hulls()

    n = TARGET_NGON
    print(
        f"Starting search for non-degenerate noble {n}-gonal polyhedra ({SYMMETRY_GROUP} symmetry)..."
    )
    print("Configured Solver: Least-Squares")

    valid_templates = []

    loaded_from_problem_file = False
    if DEBUG:
        problem_file = "nobles_problem_templates.json"
        if os.path.exists(problem_file):
            use_problems = (
                input(
                    f"\n[DEBUG] Found '{problem_file}'. Load and test ONLY these templates? (y/n): "
                )
                .strip()
                .lower()
            )
            if use_problems == "y":
                print(f"[DEBUG] Loading problem templates from '{problem_file}'...")
                try:
                    with open(problem_file, "r") as f:
                        valid_templates = json.load(f)
                    print(f"[DEBUG] Loaded {len(valid_templates)} problem templates.")
                    loaded_from_problem_file = True
                except Exception as e:
                    print(
                        f"[DEBUG] Error loading problem file: {e}. Falling back to standard generation."
                    )

    if not loaded_from_problem_file:
        if COLLAPSED_SEARCH:
            # Resolve the target corner based on COLLAPSED_ORBIT config
            if SYMMETRY_GROUP in ["I", "Ih"]:
                phi_g = (1.0 + np.sqrt(5.0)) / 2.0
                V1_axis = np.array([0.0, 1.0, phi_g]) / np.sqrt(2.0 + phi_g)
                V2_axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
                V3_axis = np.array([1.0, phi_g**2, phi_g]) / (2.0 * phi_g)
            elif SYMMETRY_GROUP in ["I3", "I3h"]:
                phi_g = (1.0 + np.sqrt(5.0)) / 2.0
                V1_axis = np.array([0.0, 1.0, phi_g]) / np.sqrt(2.0 + phi_g)
                V2_axis = np.array([phi_g, 0.0, 1.0]) / np.sqrt(2.0 + phi_g)
                V3_axis = np.array([0.0, 0.0, 1.0])
            elif "O" in SYMMETRY_GROUP:
                V1_axis = np.array([1.0, 0.0, 0.0])
                V2_axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
                V3_axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
            else:  # "T", "Td", "Th"
                V1_axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
                V2_axis = np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)
                V3_axis = np.array([1.0, 0.0, 0.0])

            V_target = (
                V1_axis
                if COLLAPSED_ORBIT in ["5-fold", "4-fold"]
                else (V2_axis if COLLAPSED_ORBIT == "3-fold" else V3_axis)
            )

            # Determine stabilizer of V_target under GROUP_GEN to match active symmetry
            stabilizer_indices = [
                i
                for i in range(len(GROUP_GEN))
                if np.allclose(np.dot(GROUP_GEN[i], V_target), V_target, atol=1e-4)
            ]

            # Partition GROUP_GEN into cosets
            visited_indices = set()
            coset_list = []
            for idx in range(GROUP_SIZE):
                if idx in visited_indices:
                    continue
                coset = [MULT_TABLE[idx][h] for h in stabilizer_indices]
                coset_list.append(coset)
                visited_indices.update(coset)

            num_cosets = len(coset_list)
            print(
                f"\n[COLLAPSED SEARCH] Orbit '{COLLAPSED_ORBIT}' has {num_cosets} unique collapsed vertices."
            )

            # Build the vert_map mapping array
            vert_map = np.zeros(GROUP_SIZE, dtype=np.int32)
            for c_idx, coset in enumerate(coset_list):
                for g in coset:
                    vert_map[g] = c_idx

            # Generate combinations directly on the coset indices
            other_cosets = list(range(1, num_cosets))
            coset_combinations = list(itertools.combinations(other_cosets, n - 1))

            # Expand coset combinations back to full GROUP_GEN representatives
            valid_templates = []
            for c_combo in coset_combinations:
                # Pick first element of each coset as representative
                rep_combo = [coset_list[c_idx][0] for c_idx in c_combo]

                # Generate all cyclic permutations of the full-group representatives
                for p in itertools.permutations(rep_combo):
                    if p[0] < p[-1]:
                        template = [0] + [int(x) for x in p]

                        # Standard pre-filters
                        if PRE_FILTER_TOPOLOGY:
                            arr_temp = np.array(template, dtype=np.int32)
                            if not is_valid_topology_collapsed_numba(
                                arr_temp, MULT_TABLE, vert_map, GROUP_SIZE, num_cosets
                            ):
                                continue
                            if not is_single_connected_component_collapsed_numba(
                                arr_temp, MULT_TABLE, vert_map, GROUP_SIZE, num_cosets
                            ):
                                continue

                        valid_templates.append(template)
        else:
            other_vertices = list(range(1, GROUP_SIZE))
            combinations = list(itertools.combinations(other_vertices, n - 1))

            print("Pre-filtering combinations under symmetry orbits...")
            combinations_arr = np.array(combinations, dtype=np.int32)

            # Use a local identity mapping for AUT_SIGMA to prevent chiral enantiomorph pruning
            local_aut_sigma = np.arange(GROUP_SIZE, dtype=np.int32)
            keep_mask = filter_canonical_combinations_numba(
                combinations_arr, n, MULT_TABLE, local_aut_sigma, GROUP_SIZE
            )
            filtered_combinations = combinations_arr[keep_mask]

            # Generate cyclic order permutations of the canonical combinations
            valid_templates = []
            for combo in filtered_combinations:
                for p in itertools.permutations(combo):
                    # Enforce cycle orientation canonicalization (p[0] < p[-1]) to eliminate reversals
                    if p[0] < p[-1]:
                        template = [0] + [int(x) for x in p]

                        # Apply topological pre-filters if enabled
                        if PRE_FILTER_TOPOLOGY:
                            arr_temp = np.array(template, dtype=np.int32)
                            if not is_valid_topology_numba(
                                arr_temp, MULT_TABLE, GROUP_SIZE
                            ):
                                continue
                            if not is_single_connected_component_numba(
                                arr_temp, MULT_TABLE, GROUP_SIZE
                            ):
                                continue

                        valid_templates.append(template)

        print(
            f"Topology generation complete. Generated {len(valid_templates)} unique physical topologies."
        )

    geom_id = 0
    dual_geom_id = 0
    solver_counts = {"Least-Squares": 0}
    produced_filenames = set()

    seeds = generate_landmark_seeds(SUBDIVISION_FACTOR)
    print(f"Using {len(seeds)} deterministic starting seeds per valid template.")

    total_viable = len(valid_templates)
    print(f"Optimizing {total_viable} layouts in parallel using Least-Squares...")
    processed_opt = 0

    opt_tasks = [(template, seeds, GROUP_GEN_3D) for template in valid_templates]
    num_cores = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_cores) as pool:
        opt_results = pool.imap_unordered(
            optimize_layout_wrapper, opt_tasks, chunksize=10
        )

        for template, local_res in opt_results:
            processed_opt += 1
            if processed_opt % 1000 == 0 or processed_opt == total_viable:
                print(
                    f"Progress: Optimized {processed_opt} / {total_viable} viable layouts..."
                )

            for x_val, solver_used, seed_label in local_res:
                theta, phi_val = x_val
                P = np.array(
                    [
                        np.sin(theta) * np.cos(phi_val),
                        np.sin(theta) * np.sin(phi_val),
                        np.cos(theta),
                    ]
                )

                vertices = [np.dot(G, P) for G in GROUP_GEN]
                faces = [
                    [MULT_TABLE[k][idx] for idx in template] for k in range(GROUP_SIZE)
                ]

                unique_verts, merged_faces = get_unique_vertices_and_faces(
                    vertices, faces
                )
                if unique_verts is None:
                    continue

                is_coplanar = check_coplanar_transition(unique_verts, merged_faces)
                if SUPPRESS_COPLANAR and is_coplanar:
                    continue

                num_vertices = len(unique_verts)
                num_faces = len(merged_faces)

                # Count unique edges
                edge_set = set()
                for face in merged_faces:
                    n_face = len(face)
                    for j in range(n_face):
                        edge_set.add(tuple(sorted((face[j], face[(j + 1) % n_face]))))
                num_edges = len(edge_set)

                # Geometric Wiener Index Calculation (using Dijkstra's algorithm)
                adj_weighted = [[] for _ in range(num_vertices)]
                for u, v in edge_set:
                    dist_val = np.linalg.norm(unique_verts[u] - unique_verts[v])
                    adj_weighted[u].append((v, dist_val))
                    adj_weighted[v].append((u, dist_val))

                total_dist = 0.0
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
                        total_dist += d_val

                if not disconnected and num_vertices > 1:
                    wiener_index = total_dist / 2.0
                else:
                    wiener_index = 0.0

                # Construct filename using Geometric Wiener Index
                filename = f"{SYMMETRY_GROUP}-{TARGET_NGON}-{num_vertices}-{num_faces}-{num_edges}-W{int(wiener_index)}.off"

                # DUPLICATE CHECK based ONLY on filename
                if filename in produced_filenames:
                    continue

                produced_filenames.add(filename)
                geom_id += 1
                solver_counts[solver_used] = solver_counts.get(solver_used, 0) + 1
                print(
                    f"\n--> Discovery {geom_id} ({solver_used}): Coordinate: [{P[0]:.16f}, {P[1]:.16f}, {P[2]:.16f}] | Filename: {filename}"
                )
                dual_written = export_to_off(
                    filename, P, template, seed_label, produced_filenames
                )
                if dual_written:
                    dual_geom_id += 1

    print(f"\nSearch complete. Generated {geom_id} unique non-degenerate models.")
    print(f"Total dual models generated: {dual_geom_id}")
    print("\nTotal discoveries by solver:")
    for s_name, count in solver_counts.items():
        print(f"- {s_name}: {count}")


if __name__ == "__main__":
    find_and_save_noble_polyhedra()
