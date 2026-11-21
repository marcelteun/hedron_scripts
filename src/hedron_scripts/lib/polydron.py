# polydron.py
# Version: 2.8.2
# Addons required: numpy, scipy, pygame, moderngl

import copy
import math
import os
import sys
import time
from functools import lru_cache

import moderngl
import numpy as np
import pygame
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from hedron_scripts.lib.polydron_utils import (
    append_face_triangles,
    append_vertex_marker,
    build_ui_buttons_data,
    choose_open_off_path,
    choose_save_off_path,
    create_divider_vao,
    create_gl_programs,
    create_help_overlay,
    get_next_output_path,
    make_look_at,
    make_perspective,
    point_to_segment_dist_2d,
    project_to_screen,
    upload_vbo)

HELP_LINES = [
    "Controls (press H to hide):",
    "  Add a polygon by selecting an edge (green) then pressing a number",
    "  Select two vertices (cyan & magenta) then press m to merge",
    "  If you have a secondary window open (via 'c' or 'j') then click one edge"
    "   in each window then 'm' to merge. (or Ctrl-m or Alt-m).",
    "  More details:",
    "  3-9, 0-2=10-12          Add polygon / attach to selected open edge",
    "  Hold num + press denominator   Attach star polygon (5/2, 7/2, 7/3, 8/3, ...)",
    "  Left-drag / Wheel       Orbit camera / zoom (either pane)",
    "  Right-drag vertex       Drag vertex (main pane)",
    "  Click vertex            Select up to 2 vertices (Cyan & Magenta)",
    "  Click edge              Select open edge or fold hinge (+5 deg)",
    "  [ or ]                  Fold selected hinge by -5 or 5 degrees",
    "  m                       Merge 2 vertices (main) or 2 open edges",
    "  Ctrl+m / Alt+m          Flip / mirror chirality before edge merge",
    "  j / c                   Open secondary and import file / Copy main to secondary",
    "  x                       Close secondary",
    "  u / Undo button         Undo last addition or merge (black box)",
    "  d / Delete button       Delete face with selected open edge (red box)",
    "  i / o                   Input .off file / Export .off mesh (16dp)",
    "  b                       Toggle black / white background",
    "  r                       Reset to blank screen & close secondary view",
    "  h / Esc                 Toggle help / Quit"]

# -----------------------------------------------------------------------------
# 1. Kinematics & Geometry Math
# -----------------------------------------------------------------------------


def rotate_vec_axis(v, axis, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def compute_polygon_normal(pts):
    norm = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    norm_len = np.linalg.norm(norm)
    return norm / norm_len if norm_len > 1e-8 else np.array([0.0, 0.0, 1.0])


@lru_cache(maxsize=64)
def canonical_regular_ngon_2d(gonality, density=1):
    """
    Returns exact 2D coordinates for a regular {gonality/density}-gon with
    unit edge length (1.0). Edge 0 -> 1 lies along the positive x-axis from
    (0, 0) to (1, 0), with circumcentre at (0.5, r_in) in the upper half-plane.
    """
    step_angle = density * math.pi / gonality
    r_in = 1.0 / (2.0 * math.tan(step_angle))
    r_circ = 1.0 / (2.0 * math.sin(step_angle))
    base_phi = -0.5 * math.pi - step_angle
    step_phi = 2.0 * step_angle

    coords = []
    for k in range(gonality):
        phi = base_phi + k * step_phi
        x = 0.5 + r_circ * math.cos(phi)
        y = r_in + r_circ * math.sin(phi)
        x = 0.0 if abs(x) < 1e-8 else x
        y = 0.0 if abs(y) < 1e-8 else y
        coords.append((x, y))
    return tuple(coords)


@lru_cache(maxsize=64)
def _canonical_ngon_array(gonality, density=1):
    arr = np.array(
        canonical_regular_ngon_2d(gonality, density), dtype=np.float64)
    arr.setflags(write=False)
    return arr


@lru_cache(maxsize=64)
def _base_ngon_array(gonality, density=1):
    """
    Returns exact centred 3D coordinates in the z=0 plane for a regular
    {gonality/density}-gon with unit edge length (1.0).
    """
    step_angle = density * math.pi / gonality
    r_circ = 1.0 / (2.0 * math.sin(step_angle))
    base_phi = -0.5 * math.pi - step_angle
    step_phi = 2.0 * step_angle

    coords = []
    for k in range(gonality):
        phi = base_phi + k * step_phi
        x = r_circ * math.cos(phi)
        y = r_circ * math.sin(phi)
        x = 0.0 if abs(x) < 1e-8 else x
        y = 0.0 if abs(y) < 1e-8 else y
        coords.append([x, y, 0.0])

    arr = np.array(coords, dtype=np.float64)
    arr.setflags(write=False)
    return arr


SHAPE_COLOURS = {
    3: [77.0 / 255.0, 102.0 / 255.0, 255.0 / 255.0],
    4: [255.0 / 255.0, 165.0 / 255.0, 0.0 / 255.0],
    5: [0.0 / 255.0, 224.0 / 255.0, 0.0 / 255.0],
    6: [112.0 / 255.0, 204.0 / 255.0, 255.0 / 255.0],
    7: [255.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0],
    8: [255.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
    9: [128.0 / 255.0, 0.0 / 255.0, 48.0 / 255.0],
    10: [255.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0],
    11: [192.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0],
    12: [0.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]}

DEFAULT_SHAPE_COLOUR = [203.0 / 255.0, 207.0 / 255.0, 210.0 / 255.0]

SHAPE_NAMES = {
    3: "Triangle",
    4: "Square",
    5: "Pentagon",
    6: "Hexagon",
    7: "Heptagon",
    8: "Octagon",
    9: "Nonagon",
    10: "Decagon",
    11: "Hendecagon",
    12: "Dodecagon"}


def get_shape_colour(gonality):
    return SHAPE_COLOURS.get(gonality, DEFAULT_SHAPE_COLOUR)


def get_shape_name(gonality, density=1):
    if density != 1:
        if gonality == 5 and density == 2:
            return "Pentagram (5/2)"
        return f"({gonality}/{density})-gon"
    return SHAPE_NAMES.get(gonality, f"{gonality}-gon")


# -----------------------------------------------------------------------------
# 2. Dynamic Polydron Model (Rigid Kinematic Formulation)
# -----------------------------------------------------------------------------


class DynamicPolydron:
    def __init__(self):
        self.history = []
        self.last_merge_map = {}
        self.reset()

    def reset(self):
        if hasattr(self, "faces") and self.faces:
            self.save_history()

        self.faces = []
        self.next_node_id = 0
        self.last_merge_map = {}

    def _record_merge_map(self, merge_map):
        if not merge_map:
            return
        if not self.last_merge_map:
            self.last_merge_map = dict(merge_map)
            return
        updated = {
            k: merge_map.get(v, v) for k, v in self.last_merge_map.items()}
        for k, v in merge_map.items():
            if k not in updated:
                updated[k] = v
        self.last_merge_map = updated

    def consume_merge_map(self):
        merge_map = self.last_merge_map
        self.last_merge_map = {}
        return merge_map

    def auto_merge_coincident(self, tol=1e-6):
        if len(self.faces) < 2:
            return {}

        node_coords = self.get_node_coords()
        nids = sorted(node_coords.keys())
        if len(nids) < 2:
            return {}

        pts_arr = np.array(
            [node_coords[nid] for nid in nids], dtype=np.float64)
        tree = cKDTree(pts_arr)
        close_pairs = tree.query_pairs(r=tol)
        if not close_pairs:
            return {}

        parent = {nid: nid for nid in nids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        face_nids_sets = [set(f["node_ids"]) for f in self.faces]
        groups = {nid: {nid} for nid in nids}
        merged_any = False

        for i, j in sorted(close_pairs):
            ra = find(nids[i])
            rb = find(nids[j])
            if ra == rb:
                continue
            ga = groups[ra]
            gb = groups[rb]
            same_face = any((ga & fs) and (gb & fs) for fs in face_nids_sets)
            if same_face:
                continue
            keep, drop = (ra, rb) if ra < rb else (rb, ra)
            parent[drop] = keep
            groups[keep] = ga | gb
            del groups[drop]
            merged_any = True

        if not merged_any:
            return {}

        merge_map = {nid: find(nid) for nid in nids if find(nid) != nid}
        for f in self.faces:
            f["node_ids"] = [merge_map.get(nid, nid) for nid in f["node_ids"]]
            if f["hinge_edge"] is not None:
                ha, hb = f["hinge_edge"]
                f["hinge_edge"] = (
                    merge_map.get(ha, ha), merge_map.get(hb, hb))

        eval_both = self._build_constraint_evaluator()
        if eval_both is not None:
            base_sub = np.array(
                [f["hinge_angle"] for f in self.faces[1:]], dtype=np.float64)
            r_chk, _ = eval_both(base_sub)
            err_sq = float(np.dot(r_chk, r_chk))
            if err_sq >= 1e-12:
                res = least_squares(
                    lambda ts: eval_both(ts)[0],
                    base_sub,
                    jac=lambda ts: eval_both(ts)[1],
                    bounds=(-np.pi, np.pi),
                    method="trf",
                    ftol=1e-15,
                    xtol=1e-15,
                    gtol=1e-15,
                    max_nfev=60)
                if 2.0 * float(res.cost) < err_sq:
                    self.faces[0]["hinge_angle"] = 0.0
                    for idx in range(1, len(self.faces)):
                        val = float(res.x[idx - 1])
                        val = math.atan2(math.sin(val), math.cos(val))
                        self.faces[idx]["hinge_angle"] = (
                            0.0 if abs(val) < 1e-8 else val)
                    self.update_world_vertices()

        self._record_merge_map(merge_map)
        return merge_map

    def add_base_polygon(self, gonality, density=1):
        self.save_history()
        base_arr = _base_ngon_array(gonality, density)
        self.faces = [{
            "id": 0,
            "gonality": gonality,
            "density": density,
            "node_ids": list(range(gonality)),
            "parent_id": None,
            "parent_edge_idx": None,
            "hinge_edge": None,
            "hinge_angle": 0.0,
            "base_pts": [base_arr[k].copy() for k in range(gonality)],
            "world_pts": [base_arr[k].copy() for k in range(gonality)]}]
        self.next_node_id = gonality
        return True

    def save_history(self, merge_edges=None):
        snapshot = {
            "faces": copy.deepcopy(self.faces),
            "next_node_id": self.next_node_id,
            "merge_edges": merge_edges}
        self.history.append(snapshot)
        if len(self.history) > 100:
            self.history.pop(0)

    def undo(self):
        if not self.history:
            return False, None
        snapshot = self.history.pop()
        self.faces = snapshot["faces"]
        self.next_node_id = snapshot["next_node_id"]
        self.last_merge_map = {}
        self.update_world_vertices()
        return True, snapshot.get("merge_edges")

    def get_open_edges(self):
        edge_count = {}
        for f in self.faces:
            n = f["gonality"]
            nids = f["node_ids"]
            for i in range(n):
                u = nids[i]
                v = nids[(i + 1) % n]
                canon = (u, v) if u < v else (v, u)
                edge_count[canon] = edge_count.get(canon, 0) + 1
        return [e for e, count in edge_count.items() if count == 1]

    def attach_polygon(self, shared_edge, gonality, density=1):
        u, v = shared_edge

        parent = None
        parent_edge_idx = -1
        u_idx, v_idx = -1, -1
        match_count = 0
        for f in self.faces:
            n = f["gonality"]
            nids = f["node_ids"]
            for i in range(n):
                a = nids[i]
                b = nids[(i + 1) % n]
                if (a == u and b == v) or (a == v and b == u):
                    match_count += 1
                    if parent is None:
                        parent = f
                        parent_edge_idx = i
                        u_idx, v_idx = a, b

        if parent is None or match_count != 1:
            return False

        self.save_history()

        new_node_ids = [u_idx, v_idx]
        for _ in range(gonality - 2):
            new_node_ids.append(self.next_node_id)
            self.next_node_id += 1

        new_face = {
            "id": len(self.faces),
            "gonality": gonality,
            "density": density,
            "node_ids": new_node_ids,
            "parent_id": parent["id"],
            "parent_edge_idx": parent_edge_idx,
            "hinge_edge": (u_idx, v_idx),
            "hinge_angle": 0.0,
            "world_pts": []}
        self.faces.append(new_face)
        self.update_world_vertices()
        self.auto_merge_coincident()
        return True

    def merge_edge_model(
            self, main_edge, sec_model, sec_edge, flip=False, mirror=False):
        if not self.faces:
            return False, "Main model is blank."
        if not sec_model.faces:
            return False, "Secondary model is blank."

        main_open = set(self.get_open_edges())
        if main_edge not in main_open:
            return False, "Main selected edge is not an unmatched edge."

        sec_open = set(sec_model.get_open_edges())
        if sec_edge not in sec_open:
            return False, "Secondary selected edge is not an unmatched edge."

        self.update_world_vertices()
        sec_model.update_world_vertices()

        main_face = None
        main_edge_idx = -1
        u_main, v_main = -1, -1
        for f in self.faces:
            n = f["gonality"]
            nids = f["node_ids"]
            for i in range(n):
                a = nids[i]
                b = nids[(i + 1) % n]
                canon = (a, b) if a < b else (b, a)
                if canon == main_edge:
                    main_face = f
                    main_edge_idx = i
                    u_main, v_main = a, b
                    break
            if main_face is not None:
                break

        if main_face is None:
            return False, "Main edge not found."

        sec_face_idx = -1
        sec_edge_idx = -1
        u_sec, v_sec = -1, -1
        for f_idx, f in enumerate(sec_model.faces):
            n = f["gonality"]
            nids = f["node_ids"]
            for i in range(n):
                a = nids[i]
                b = nids[(i + 1) % n]
                canon = (a, b) if a < b else (b, a)
                if canon == sec_edge:
                    sec_face_idx = f_idx
                    sec_edge_idx = i
                    u_sec, v_sec = a, b
                    break
            if sec_face_idx >= 0:
                break

        if sec_face_idx < 0:
            return False, "Secondary edge not found."

        m_pts = main_face["world_pts"]
        a_main = m_pts[main_edge_idx]
        b_main = m_pts[(main_edge_idx + 1) % main_face["gonality"]]
        u_m = b_main - a_main
        len_m = float(np.linalg.norm(u_m))
        if len_m <= 1e-8:
            return False, "Degenerate edge in main model."
        u_m = u_m / len_m

        c_main = np.mean(m_pts, axis=0)
        v_m = 0.5 * (a_main + b_main) - c_main
        o_m = v_m - float(np.dot(v_m, u_m)) * u_m
        len_om = float(np.linalg.norm(o_m))
        if len_om <= 1e-8:
            return False, "Degenerate face frame in main model."
        o_m = o_m / len_om
        n_m = np.cross(u_m, o_m)

        sec_coords = sec_model.get_node_coords()
        a_sec = sec_coords[u_sec]
        b_sec = sec_coords[v_sec]
        u_s = b_sec - a_sec
        len_s = float(np.linalg.norm(u_s))
        if len_s <= 1e-8:
            return False, "Degenerate edge in secondary model."
        u_s = u_s / len_s

        sec_root_face = sec_model.faces[sec_face_idx]
        c_sec = np.mean(
            [sec_coords[nid] for nid in sec_root_face["node_ids"]], axis=0)
        v_s = 0.5 * (a_sec + b_sec) - c_sec
        o_s = v_s - float(np.dot(v_s, u_s)) * u_s
        len_os = float(np.linalg.norm(o_s))
        if len_os <= 1e-8:
            return False, "Degenerate face frame in secondary model."
        o_s = o_s / len_os
        n_s = np.cross(u_s, o_s)

        mid_main = 0.5 * (a_main + b_main)
        mid_sec = 0.5 * (a_sec + b_sec)
        s_u = 1.0 if flip else -1.0
        s_o = -1.0
        s_n = (-1.0 if flip else 1.0) * (-1.0 if mirror else 1.0)

        target_u = s_u * u_m
        target_o = s_o * o_m
        target_n = s_n * n_m

        transformed_sec_pts = {}
        for nid, pt in sec_coords.items():
            d = pt - mid_sec
            new_pt = (
                mid_main
                + float(np.dot(d, u_s)) * target_u
                + float(np.dot(d, o_s)) * target_o
                + float(np.dot(d, n_s)) * target_n)
            transformed_sec_pts[nid] = np.where(
                np.abs(new_pt) < 1e-8, 0.0, new_pt)

        nid_map = {}
        if flip:
            nid_map[u_sec] = u_main
            nid_map[v_sec] = v_main
        else:
            nid_map[v_sec] = u_main
            nid_map[u_sec] = v_main

        new_next_id = self.next_node_id
        for nid in sorted(sec_coords.keys()):
            if nid not in nid_map:
                nid_map[nid] = new_next_id
                new_next_id += 1

        mapped_pts = {
            nid_map[nid]: pt for nid, pt in transformed_sec_pts.items()}

        edge_to_sec_faces = {}
        for f_idx, f in enumerate(sec_model.faces):
            k = f["gonality"]
            nids = f["node_ids"]
            for i in range(k):
                a = nid_map[nids[i]]
                b = nid_map[nids[(i + 1) % k]]
                canon = (a, b) if a < b else (b, a)
                if canon not in edge_to_sec_faces:
                    edge_to_sec_faces[canon] = []
                edge_to_sec_faces[canon].append(f_idx)

        k_root = sec_root_face["gonality"]
        raw_root_nids = sec_root_face["node_ids"]
        if flip:
            root_ordered = [
                nid_map[raw_root_nids[(sec_edge_idx + t) % k_root]]
                for t in range(k_root)]
        else:
            root_ordered = [
                nid_map[raw_root_nids[(sec_edge_idx + 1 - t) % k_root]]
                for t in range(k_root)]

        base_face_count = len(self.faces)
        added_faces = [{
            "id": base_face_count,
            "gonality": k_root,
            "density": sec_root_face.get("density", 1),
            "node_ids": root_ordered,
            "parent_id": main_face["id"],
            "parent_edge_idx": main_edge_idx,
            "hinge_edge": (u_main, v_main),
            "hinge_angle": 0.0,
            "world_pts": []}]
        visited = {sec_face_idx: base_face_count}
        queue = [sec_face_idx]

        while queue:
            cur_sec_idx = queue.pop(0)
            cur_global_id = visited[cur_sec_idx]
            cur_face = added_faces[cur_global_id - base_face_count]
            cur_nids = cur_face["node_ids"]
            cur_pts = [mapped_pts[nid] for nid in cur_nids]
            c_parent = np.mean(cur_pts, axis=0)
            k_p = len(cur_nids)

            for e_idx in range(k_p):
                a_nid = cur_nids[e_idx]
                b_nid = cur_nids[(e_idx + 1) % k_p]
                canon = (a_nid, b_nid) if a_nid < b_nid else (b_nid, a_nid)
                for nbr_sec_idx in edge_to_sec_faces.get(canon, []):
                    if nbr_sec_idx in visited:
                        continue

                    nbr_face = sec_model.faces[nbr_sec_idx]
                    k_c = nbr_face["gonality"]
                    nbr_mapped = [nid_map[nid] for nid in nbr_face["node_ids"]]
                    j = nbr_mapped.index(a_nid)
                    if nbr_mapped[(j + 1) % k_c] == b_nid:
                        ordered = [nbr_mapped[(j + t) % k_c] for t in range(k_c)]
                    else:
                        ordered = [nbr_mapped[(j - t) % k_c] for t in range(k_c)]

                    a_pt = mapped_pts[a_nid]
                    b_pt = mapped_pts[b_nid]
                    u_edge = b_pt - a_pt
                    u_len = float(np.linalg.norm(u_edge))
                    if u_len <= 1e-8:
                        continue
                    u_edge = u_edge / u_len

                    v_vec = 0.5 * (a_pt + b_pt) - c_parent
                    o_vec = v_vec - float(np.dot(v_vec, u_edge)) * u_edge
                    o_len = float(np.linalg.norm(o_vec))
                    if o_len <= 1e-8:
                        continue
                    o_vec = o_vec / o_len
                    n_vec = np.cross(u_edge, o_vec)

                    c_child = np.mean(
                        [mapped_pts[nid] for nid in ordered], axis=0)
                    w_vec = c_child - 0.5 * (a_pt + b_pt)
                    f_vec = w_vec - float(np.dot(w_vec, u_edge)) * u_edge
                    f_len = float(np.linalg.norm(f_vec))
                    if f_len <= 1e-8:
                        continue
                    f_vec = f_vec / f_len

                    theta = math.atan2(
                        float(np.dot(f_vec, n_vec)),
                        float(np.dot(f_vec, o_vec)))
                    theta = 0.0 if abs(theta) < 1e-8 else theta

                    new_global_id = base_face_count + len(added_faces)
                    visited[nbr_sec_idx] = new_global_id
                    added_faces.append({
                        "id": new_global_id,
                        "gonality": k_c,
                        "density": nbr_face.get("density", 1),
                        "node_ids": ordered,
                        "parent_id": cur_global_id,
                        "parent_edge_idx": e_idx,
                        "hinge_edge": (a_nid, b_nid),
                        "hinge_angle": theta,
                        "world_pts": []})
                    queue.append(nbr_sec_idx)

        if len(added_faces) != len(sec_model.faces):
            return False, "Cannot merge: secondary model is not edge-connected."

        self.save_history(merge_edges=(main_edge, sec_edge))
        self.faces.extend(added_faces)
        self.next_node_id = new_next_id
        self.update_world_vertices()
        self.auto_merge_coincident()

        mode_parts = []
        if flip:
            mode_parts.append("flipped")
        if mirror:
            mode_parts.append("mirrored")
        mode_str = f" ({', '.join(mode_parts)})" if mode_parts else ""
        return True, (
            f"Merged {len(added_faces)} face(s) flat{mode_str} "
            f"onto edge {main_edge}.")

    def delete_face_by_edge(self, shared_edge):
        open_edges = self.get_open_edges()
        if shared_edge not in open_edges:
            return False, "Cannot delete: Edge is not an unmatched edge."

        target_face = None
        for f in self.faces:
            n = f["gonality"]
            nids = f["node_ids"]
            for i in range(n):
                u = nids[i]
                v = nids[(i + 1) % n]
                canon = (u, v) if u < v else (v, u)
                if canon == shared_edge:
                    target_face = f
                    break
            if target_face is not None:
                break

        if target_face is None:
            return False, "Face not found."

        shape_name = get_shape_name(
            target_face["gonality"], target_face.get("density", 1))
        self.save_history()

        if len(self.faces) == 1:
            self.faces = []
            self.next_node_id = 0
            return True, f"Deleted {shape_name}; screen is now blank."

        target_id = target_face["id"]
        if target_id != 0:
            to_delete = {target_id}
            added = True
            while added:
                added = False
                for f in self.faces:
                    if f["id"] not in to_delete and f["parent_id"] in to_delete:
                        to_delete.add(f["id"])
                        added = True
            surviving = [f for f in self.faces if f["id"] not in to_delete]
        else:
            direct_children = [f for f in self.faces if f["parent_id"] == 0]
            if not direct_children:
                self.faces = []
                self.next_node_id = 0
                return True, f"Deleted {shape_name}; screen is now blank."

            def subtree_ids(root_id):
                ids = {root_id}
                added = True
                while added:
                    added = False
                    for f in self.faces:
                        if f["id"] not in ids and f["parent_id"] in ids:
                            ids.add(f["id"])
                            added = True
                return ids

            best_ids = max(
                (subtree_ids(c["id"]) for c in direct_children), key=len)
            new_root = next(c for c in direct_children if c["id"] in best_ids)
            new_root["base_pts"] = [
                pt.copy() for pt in new_root["world_pts"]]
            new_root["parent_id"] = None
            new_root["parent_edge_idx"] = None
            new_root["hinge_edge"] = None
            new_root["hinge_angle"] = 0.0
            surviving = [f for f in self.faces if f["id"] in best_ids]

        id_map = {f["id"]: idx for idx, f in enumerate(surviving)}
        for f in surviving:
            f["id"] = id_map[f["id"]]
            if f["parent_id"] is not None:
                f["parent_id"] = id_map[f["parent_id"]]

        self.faces = surviving
        self.update_world_vertices()
        return True, f"Deleted {shape_name} attached to edge {shared_edge}."

    def update_world_vertices(self, custom_angles=None):
        if not self.faces:
            return

        f0 = self.faces[0]
        if "base_pts" in f0:
            f0["world_pts"] = [pt.copy() for pt in f0["base_pts"]]
        else:
            base_arr = _base_ngon_array(f0["gonality"], f0.get("density", 1))
            f0["world_pts"] = [base_arr[k].copy() for k in range(f0["gonality"])]

        for f_idx in range(1, len(self.faces)):
            f = self.faces[f_idx]
            parent = self.faces[f["parent_id"]]
            e_idx = f["parent_edge_idx"]
            p_pts = parent["world_pts"]

            a_pt = p_pts[e_idx]
            b_pt = p_pts[(e_idx + 1) % parent["gonality"]]

            ax = float(b_pt[0] - a_pt[0])
            ay = float(b_pt[1] - a_pt[1])
            az = float(b_pt[2] - a_pt[2])
            axis_len = math.sqrt(ax * ax + ay * ay + az * az)
            if axis_len > 1e-8:
                inv_len = 1.0 / axis_len
                ux, uy, uz = ax * inv_len, ay * inv_len, az * inv_len
            else:
                ux, uy, uz = 1.0, 0.0, 0.0

            n_p = len(p_pts)
            cx = sum(float(p[0]) for p in p_pts) / n_p
            cy = sum(float(p[1]) for p in p_pts) / n_p
            cz = sum(float(p[2]) for p in p_pts) / n_p

            vx = 0.5 * float(a_pt[0] + b_pt[0]) - cx
            vy = 0.5 * float(a_pt[1] + b_pt[1]) - cy
            vz = 0.5 * float(a_pt[2] + b_pt[2]) - cz

            dot_vu = vx * ux + vy * uy + vz * uz
            px = vx - dot_vu * ux
            py = vy - dot_vu * uy
            pz = vz - dot_vu * uz
            l_out = math.sqrt(px * px + py * py + pz * pz)
            if l_out > 1e-8:
                inv_out = 1.0 / l_out
                ox, oy, oz = px * inv_out, py * inv_out, pz * inv_out
            else:
                ox, oy, oz = 0.0, -1.0, 0.0

            nx = uy * oz - uz * oy
            ny = uz * ox - ux * oz
            nz = ux * oy - uy * ox

            theta = (
                float(custom_angles[f_idx])
                if custom_angles is not None
                else f["hinge_angle"])
            ct = math.cos(theta)
            st = math.sin(theta)
            fx = ct * ox + st * nx
            fy = ct * oy + st * ny
            fz = ct * oz + st * nz

            canon_arr = _canonical_ngon_array(
                f["gonality"], f.get("density", 1))
            u_edge = np.array([ux, uy, uz], dtype=np.float64)
            u_face = np.array([fx, fy, fz], dtype=np.float64)
            pts_mat = (
                a_pt
                + canon_arr[:, 0:1] * u_edge
                + canon_arr[:, 1:2] * u_face)
            pts_mat[np.abs(pts_mat) < 1e-8] = 0.0
            f["world_pts"] = [pts_mat[k] for k in range(len(pts_mat))]

    def get_node_coords(self):
        node_pts = {}
        for f in self.faces:
            for nid, pt in zip(f["node_ids"], f["world_pts"]):
                if nid not in node_pts:
                    node_pts[nid] = []
                node_pts[nid].append(pt)
        return {nid: np.mean(pts, axis=0) for nid, pts in node_pts.items()}

    def get_centre_of_gravity(self, node_coords=None):
        if node_coords is None:
            node_coords = self.get_node_coords()
        if not node_coords:
            return np.zeros(3, dtype=np.float64)
        cog = np.mean(
            np.array(list(node_coords.values()), dtype=np.float64), axis=0)
        return np.where(np.abs(cog) < 1e-8, 0.0, cog)

    def _build_constraint_evaluator(self, extra_pair=None, track_ref=None):
        num_faces = len(self.faces)
        if num_faces <= 1:
            return None

        nid_1, nid_2 = extra_pair if extra_pair is not None else (None, None)
        parent_ids = [f["parent_id"] for f in self.faces]
        parent_edge_idxs = [f["parent_edge_idx"] for f in self.faces]
        parent_gons = [
            self.faces[pid]["gonality"] if pid is not None else 0
            for pid in parent_ids]
        face_canons = [
            _canonical_ngon_array(f["gonality"], f.get("density", 1))
            for f in self.faces]

        ancestors = [()] * num_faces
        for idx in range(1, num_faces):
            ancestors[idx] = ancestors[parent_ids[idx]] + (idx,)

        node_groups = {}
        for f_idx, f in enumerate(self.faces):
            for v_idx, nid in enumerate(f["node_ids"]):
                target = (
                    nid_1
                    if extra_pair is not None and nid == nid_2
                    else nid)
                if target not in node_groups:
                    node_groups[target] = []
                node_groups[target].append((f_idx, v_idx))

        pairs = []
        for group in node_groups.values():
            if len(group) > 1:
                fa_ref, va_ref = group[0]
                for fb, vb in group[1:]:
                    pairs.append((fb, vb, fa_ref, va_ref))

        if not pairs and track_ref is None:
            return None

        num_pairs = len(pairs)
        num_vars = num_faces - 1
        total_rows = 3 * num_pairs + (3 if track_ref is not None else 0)
        f0 = self.faces[0]
        if "base_pts" in f0:
            base_face_pts = np.array(f0["base_pts"], dtype=np.float64)
        else:
            base_face_pts = _base_ngon_array(
                f0["gonality"], f0.get("density", 1))

        last_x = None
        last_r = None
        last_j = None

        def eval_both(theta_sub):
            nonlocal last_x, last_r, last_j
            if last_x is not None and np.array_equal(theta_sub, last_x):
                return last_r, last_j

            face_pts = [None] * num_faces
            face_pts[0] = base_face_pts
            a_arr = np.empty((num_faces, 3), dtype=np.float64)
            u_edge_arr = np.empty((num_faces, 3), dtype=np.float64)

            for idx in range(1, num_faces):
                p_pts = face_pts[parent_ids[idx]]
                e_idx = parent_edge_idxs[idx]
                a_pt = p_pts[e_idx]
                b_pt = p_pts[(e_idx + 1) % parent_gons[idx]]

                ax = float(b_pt[0] - a_pt[0])
                ay = float(b_pt[1] - a_pt[1])
                az = float(b_pt[2] - a_pt[2])
                axis_len = math.sqrt(ax * ax + ay * ay + az * az)
                if axis_len > 1e-8:
                    inv_len = 1.0 / axis_len
                    ux, uy, uz = ax * inv_len, ay * inv_len, az * inv_len
                else:
                    ux, uy, uz = 1.0, 0.0, 0.0

                c_parent = p_pts.mean(axis=0)
                vx = 0.5 * float(a_pt[0] + b_pt[0]) - float(c_parent[0])
                vy = 0.5 * float(a_pt[1] + b_pt[1]) - float(c_parent[1])
                vz = 0.5 * float(a_pt[2] + b_pt[2]) - float(c_parent[2])

                dot_vu = vx * ux + vy * uy + vz * uz
                px = vx - dot_vu * ux
                py = vy - dot_vu * uy
                pz = vz - dot_vu * uz
                l_out = math.sqrt(px * px + py * py + pz * pz)
                if l_out > 1e-8:
                    inv_out = 1.0 / l_out
                    ox, oy, oz = px * inv_out, py * inv_out, pz * inv_out
                else:
                    ox, oy, oz = 0.0, -1.0, 0.0

                nx = uy * oz - uz * oy
                ny = uz * ox - ux * oz
                nz = ux * oy - uy * ox

                th = float(theta_sub[idx - 1])
                ct = math.cos(th)
                st = math.sin(th)
                fx = ct * ox + st * nx
                fy = ct * oy + st * ny
                fz = ct * oz + st * nz

                c2d = face_canons[idx]
                u_edge = np.array([ux, uy, uz], dtype=np.float64)
                u_face = np.array([fx, fy, fz], dtype=np.float64)
                face_pts[idx] = (
                    a_pt + c2d[:, 0:1] * u_edge + c2d[:, 1:2] * u_face)
                a_arr[idx] = a_pt
                u_edge_arr[idx] = u_edge

            r = np.empty(total_rows, dtype=np.float64)
            j_mat = np.zeros((total_rows, num_vars), dtype=np.float64)

            for m, (fa, va, fb, vb) in enumerate(pairs):
                pa = face_pts[fa][va]
                pb = face_pts[fb][vb]
                row = 3 * m
                r[row] = pa[0] - pb[0]
                r[row + 1] = pa[1] - pb[1]
                r[row + 2] = pa[2] - pb[2]

                for k in ancestors[fa]:
                    uex = u_edge_arr[k, 0]
                    uey = u_edge_arr[k, 1]
                    uez = u_edge_arr[k, 2]
                    dx = pa[0] - a_arr[k, 0]
                    dy = pa[1] - a_arr[k, 1]
                    dz = pa[2] - a_arr[k, 2]
                    col = k - 1
                    j_mat[row, col] += uey * dz - uez * dy
                    j_mat[row + 1, col] += uez * dx - uex * dz
                    j_mat[row + 2, col] += uex * dy - uey * dx

                for k in ancestors[fb]:
                    uex = u_edge_arr[k, 0]
                    uey = u_edge_arr[k, 1]
                    uez = u_edge_arr[k, 2]
                    dx = pb[0] - a_arr[k, 0]
                    dy = pb[1] - a_arr[k, 1]
                    dz = pb[2] - a_arr[k, 2]
                    col = k - 1
                    j_mat[row, col] -= uey * dz - uez * dy
                    j_mat[row + 1, col] -= uez * dx - uex * dz
                    j_mat[row + 2, col] -= uex * dy - uey * dx

            if track_ref is not None:
                fd, vd, t_pt = track_ref
                pd = face_pts[fd][vd]
                row = 3 * num_pairs
                r[row] = pd[0] - t_pt[0]
                r[row + 1] = pd[1] - t_pt[1]
                r[row + 2] = pd[2] - t_pt[2]

                for k in ancestors[fd]:
                    uex = u_edge_arr[k, 0]
                    uey = u_edge_arr[k, 1]
                    uez = u_edge_arr[k, 2]
                    dx = pd[0] - a_arr[k, 0]
                    dy = pd[1] - a_arr[k, 1]
                    dz = pd[2] - a_arr[k, 2]
                    col = k - 1
                    j_mat[row, col] += uey * dz - uez * dy
                    j_mat[row + 1, col] += uez * dx - uex * dz
                    j_mat[row + 2, col] += uex * dy - uey * dx

            last_x = theta_sub.copy()
            last_r = r
            last_j = j_mat
            return r, j_mat

        return eval_both

    def drag_vertex(self, nid, target_pt):
        num_faces = len(self.faces)
        if num_faces <= 1:
            return False

        if nid in self.faces[0]["node_ids"]:
            return False

        drag_ref = None
        for f_idx, f in enumerate(self.faces):
            if nid in f["node_ids"]:
                drag_ref = (f_idx, f["node_ids"].index(nid))
                break

        if drag_ref is None:
            return False

        cur_pt = np.asarray(
            self.faces[drag_ref[0]]["world_pts"][drag_ref[1]],
            dtype=np.float64)
        delta_pt = np.asarray(target_pt, dtype=np.float64) - cur_pt
        delta_len = float(np.linalg.norm(delta_pt))
        max_pull = 0.2
        if delta_len > max_pull:
            target_arr = cur_pt + (max_pull / delta_len) * delta_pt
        else:
            target_arr = cur_pt + delta_pt

        eval_track = self._build_constraint_evaluator(
            track_ref=(drag_ref[0], drag_ref[1], target_arr))
        if eval_track is None:
            return False

        num_vars = num_faces - 1
        base_sub = np.array(
            [f["hinge_angle"] for f in self.faces[1:]], dtype=np.float64)
        eye_vars = np.eye(num_vars, dtype=np.float64)

        w_loop = 500.0
        w_target = 1.0
        w_damp = 0.35
        max_step = 0.12
        lower_b = base_sub - max_step
        upper_b = base_sub + max_step

        def fun_drag_res(theta_sub):
            r_all, _ = eval_track(theta_sub)
            r_loop = r_all[:-3]
            r_tgt = r_all[-3:]
            r_damp = w_damp * (theta_sub - base_sub)
            if len(r_loop) > 0:
                return np.concatenate(
                    [w_loop * r_loop, w_target * r_tgt, r_damp])
            return np.concatenate([w_target * r_tgt, r_damp])

        def fun_drag_jac(theta_sub):
            _, j_all = eval_track(theta_sub)
            j_loop = j_all[:-3]
            j_tgt = j_all[-3:]
            j_damp = w_damp * eye_vars
            if len(j_loop) > 0:
                return np.vstack(
                    [w_loop * j_loop, w_target * j_tgt, j_damp])
            return np.vstack([w_target * j_tgt, j_damp])

        res = least_squares(
            fun_drag_res,
            base_sub.copy(),
            jac=fun_drag_jac,
            bounds=(lower_b, upper_b),
            method="trf",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=60)
        theta_cand = res.x

        eval_loop = self._build_constraint_evaluator()
        if eval_loop is not None:
            r_chk, _ = eval_loop(theta_cand)
            if float(np.dot(r_chk, r_chk)) >= 1e-12:
                w_anchor = 1e-3
                res_proj = least_squares(
                    lambda ts: np.concatenate([
                        eval_loop(ts)[0],
                        w_anchor * (ts - theta_cand)]),
                    theta_cand,
                    jac=lambda ts: np.vstack([
                        eval_loop(ts)[1],
                        w_anchor * eye_vars]),
                    bounds=(base_sub - 0.35, base_sub + 0.35),
                    method="trf",
                    ftol=1e-15,
                    xtol=1e-15,
                    gtol=1e-15,
                    max_nfev=80)
                r_proj_loop, _ = eval_loop(res_proj.x)
                if float(np.dot(r_proj_loop, r_proj_loop)) >= 1e-10:
                    return False
                theta_cand = res_proj.x

        diff = theta_cand - base_sub
        ang_diff = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))
        if float(np.max(ang_diff)) < 1e-7:
            return False

        self.faces[0]["hinge_angle"] = 0.0
        for idx in range(1, num_faces):
            val = float(theta_cand[idx - 1])
            val = math.atan2(math.sin(val), math.cos(val))
            self.faces[idx]["hinge_angle"] = 0.0 if abs(val) < 1e-8 else val
        self.update_world_vertices()
        self.auto_merge_coincident()
        return True

    def solve_merge(self, nid_1, nid_2):
        num_faces = len(self.faces)
        if num_faces <= 1 or nid_1 == nid_2:
            return False

        eval_both = self._build_constraint_evaluator(extra_pair=(nid_1, nid_2))
        if eval_both is None:
            return False

        num_vars = num_faces - 1
        base_sub = np.array(
            [f["hinge_angle"] for f in self.faces[1:]], dtype=np.float64)

        def fun_residuals(theta_sub):
            return eval_both(theta_sub)[0]

        def fun_jacobian(theta_sub):
            return eval_both(theta_sub)[1]

        res0 = least_squares(
            fun_residuals,
            base_sub.copy(),
            jac=fun_jacobian,
            bounds=(-np.pi, np.pi),
            method="trf",
            ftol=1e-15,
            xtol=1e-15,
            gtol=1e-15,
            max_nfev=150)
        best_x = res0.x
        best_cost_sq = 2.0 * float(res0.cost)

        if best_cost_sq >= 1e-12:
            candidates = []
            for step in (0.15, -0.15, 0.4, -0.4, 0.8, -0.8):
                candidates.append(np.clip(base_sub + step, -np.pi, np.pi))

            signs = np.array(
                [1.0 if idx % 2 == 1 else -1.0 for idx in range(1, num_faces)],
                dtype=np.float64)
            for s1 in (0.3, -0.3, 0.7, -0.7):
                candidates.append(np.clip(base_sub + s1 * signs, -np.pi, np.pi))

            rng = np.random.default_rng(42)
            for _ in range(6):
                candidates.append(rng.uniform(-np.pi, np.pi, size=num_vars))

            start_t = time.monotonic()
            for init in candidates:
                if time.monotonic() - start_t > 1.5:
                    break
                res = least_squares(
                    fun_residuals,
                    init,
                    jac=fun_jacobian,
                    bounds=(-np.pi, np.pi),
                    method="trf",
                    ftol=1e-11,
                    xtol=1e-11,
                    gtol=1e-11,
                    max_nfev=60)
                cost_sq = 2.0 * float(res.cost)
                if cost_sq < best_cost_sq:
                    best_cost_sq = cost_sq
                    best_x = res.x
                if best_cost_sq < 1e-12:
                    break

            if best_x is not None and 1e-12 <= best_cost_sq < 1e-2:
                res_polish = least_squares(
                    fun_residuals,
                    best_x,
                    jac=fun_jacobian,
                    bounds=(-np.pi, np.pi),
                    method="trf",
                    ftol=1e-15,
                    xtol=1e-15,
                    gtol=1e-15,
                    max_nfev=100)
                polish_cost_sq = 2.0 * float(res_polish.cost)
                if polish_cost_sq < best_cost_sq:
                    best_cost_sq = polish_cost_sq
                    best_x = res_polish.x

        if best_x is not None and best_cost_sq < 1e-10:
            self.save_history()

            self.faces[0]["hinge_angle"] = 0.0
            for idx in range(1, num_faces):
                val = float(best_x[idx - 1])
                self.faces[idx]["hinge_angle"] = (
                    0.0 if abs(val) < 1e-8 else val)
            self.update_world_vertices()

            for f in self.faces:
                f["node_ids"] = [
                    nid_1 if nid == nid_2 else nid for nid in f["node_ids"]]
                if f["hinge_edge"] is not None:
                    he = f["hinge_edge"]
                    ha = nid_1 if he[0] == nid_2 else he[0]
                    hb = nid_1 if he[1] == nid_2 else he[1]
                    f["hinge_edge"] = (ha, hb)
            self._record_merge_map({nid_2: nid_1})
            self.auto_merge_coincident()
            return True

        return False

    def fold_hinge(self, shared_edge, delta_rad):
        target_idx = None
        for idx, f in enumerate(self.faces):
            he = f["hinge_edge"]
            if he is None:
                continue
            canon_he = (he[0], he[1]) if he[0] < he[1] else (he[1], he[0])
            if canon_he == shared_edge:
                target_idx = idx
                break

        if target_idx is None:
            return False, "Selected edge is not a hinge."

        num_faces = len(self.faces)
        num_vars = num_faces - 1
        target_col = target_idx - 1
        base_sub = np.array(
            [f["hinge_angle"] for f in self.faces[1:]], dtype=np.float64)

        raw_target = float(base_sub[target_col]) + float(delta_rad)
        target_angle = math.atan2(math.sin(raw_target), math.cos(raw_target))
        target_angle = 0.0 if abs(target_angle) < 1e-8 else target_angle

        theta_trial = base_sub.copy()
        theta_trial[target_col] = target_angle

        def commit_angles(angles_vec):
            self.faces[0]["hinge_angle"] = 0.0
            for idx in range(1, num_faces):
                val = float(angles_vec[idx - 1])
                val = math.atan2(math.sin(val), math.cos(val))
                self.faces[idx]["hinge_angle"] = (
                    0.0 if abs(val) < 1e-8 else val)
            self.update_world_vertices()
            self.auto_merge_coincident()
            deg = np.degrees(self.faces[target_idx]["hinge_angle"])
            return 0.0 if abs(deg) < 1e-8 else deg

        eval_both = self._build_constraint_evaluator()
        if eval_both is None:
            deg = commit_angles(theta_trial)
            return True, (
                f"Folded hinge to {deg:.16f}\u00b0 "
                "| Press [/] or click edge to fold")

        r_trial, j_trial = eval_both(theta_trial)
        if float(np.dot(r_trial, r_trial)) < 1e-12:
            deg = commit_angles(theta_trial)
            return True, (
                f"Folded hinge to {deg:.16f}\u00b0 "
                "| Press [/] or click edge to fold")

        node_groups = {}
        for f_idx, f in enumerate(self.faces):
            for v_idx, nid in enumerate(f["node_ids"]):
                if nid not in node_groups:
                    node_groups[nid] = []
                node_groups[nid].append((f_idx, v_idx))

        pairs = []
        for group in node_groups.values():
            if len(group) > 1:
                fa_ref, va_ref = group[0]
                for fb, vb in group[1:]:
                    pairs.append((fb, vb, fa_ref, va_ref))

        _, j_base = eval_both(base_sub)
        broken_m = set()
        active_cols_per_m = []
        for m in range(len(pairs)):
            row_slice = slice(3 * m, 3 * m + 3)
            r_seg = r_trial[row_slice]
            if float(np.dot(r_seg, r_seg)) >= 1e-12:
                broken_m.add(m)
            cols_m = [
                c for c in range(num_vars)
                if float(np.linalg.norm(j_base[row_slice, c])) > 1e-8
                or float(np.linalg.norm(j_trial[row_slice, c])) > 1e-8]
            active_cols_per_m.append(cols_m)

        broken_faces = {
            c + 1 for m in broken_m for c in active_cols_per_m[m]}
        broken_faces.add(target_idx)

        allowed = set(broken_faces)
        changed = True
        while changed:
            changed = False
            for m in range(len(pairs)):
                if m in broken_m:
                    continue
                m_faces = {c + 1 for c in active_cols_per_m[m]}
                if m_faces and not m_faces.issubset(allowed):
                    overlap = m_faces & allowed
                    if overlap:
                        allowed -= overlap
                        changed = True

        def run_two_pass(opt_faces):
            opt_cols = sorted(
                idx - 1 for idx in opt_faces if idx != target_idx)
            unlock_cols = sorted(idx - 1 for idx in opt_faces)

            def expand_with(base_vec, sub_vec, cols):
                full = base_vec.copy()
                full[cols] = sub_vec
                return full

            theta_pass1 = theta_trial.copy()
            if opt_cols:
                res1 = least_squares(
                    lambda tf: eval_both(
                        expand_with(theta_trial, tf, opt_cols))[0],
                    theta_trial[opt_cols],
                    jac=lambda tf: eval_both(
                        expand_with(theta_trial, tf, opt_cols))[1][:, opt_cols],
                    method="trf",
                    ftol=1e-15,
                    xtol=1e-15,
                    gtol=1e-15,
                    max_nfev=150)
                theta_pass1 = expand_with(theta_trial, res1.x, opt_cols)
                cost_sq_1 = 2.0 * float(res1.cost)
            else:
                cost_sq_1 = float(np.dot(r_trial, r_trial))

            if cost_sq_1 < 1e-12:
                deg = commit_angles(theta_pass1)
                return True, (
                    f"Folded hinge to {deg:.16f}\u00b0 (flexed loop) "
                    "| Press [/] or click edge to fold")

            if unlock_cols:
                res2 = least_squares(
                    lambda tu: eval_both(
                        expand_with(base_sub, tu, unlock_cols))[0],
                    theta_pass1[unlock_cols],
                    jac=lambda tu: eval_both(
                        expand_with(base_sub, tu, unlock_cols))[1][
                            :, unlock_cols],
                    method="trf",
                    ftol=1e-15,
                    xtol=1e-15,
                    gtol=1e-15,
                    max_nfev=150)
                theta_pass2 = expand_with(base_sub, res2.x, unlock_cols)
                cost_sq_2 = 2.0 * float(res2.cost)
            else:
                theta_pass2 = theta_pass1
                r2, _ = eval_both(theta_pass2)
                cost_sq_2 = float(np.dot(r2, r2))

            if cost_sq_2 < 1e-10:
                diff = theta_pass2 - base_sub
                ang_diff = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))
                if float(np.max(ang_diff)) < 1e-6:
                    deg = np.degrees(self.faces[target_idx]["hinge_angle"])
                    deg = 0.0 if abs(deg) < 1e-8 else deg
                    return False, (
                        f"Rigid closed loop: relaxed back to "
                        f"{deg:.16f}\u00b0.")
                deg = commit_angles(theta_pass2)
                return True, (
                    f"Relaxed hinge to {deg:.16f}\u00b0 "
                    "| Press [/] or click edge to fold")

            return None

        if target_idx in allowed and len(allowed) > 1:
            result = run_two_pass(allowed)
            if result is not None:
                return result

        result = run_two_pass(broken_faces)
        if result is not None:
            return result

        return False, "Cannot fold: closed loop constraints cannot be satisfied."

# Version: 2.8.3

    def load_off(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                raw_lines = fh.readlines()
        except OSError as exc:
            return False, f"Could not read {filepath}: {exc}"

        tokens_lines = []
        for raw in raw_lines:
            cleaned = raw.split("#", 1)[0].strip()
            if cleaned:
                tokens_lines.append(cleaned.split())

        if not tokens_lines:
            return False, "OFF file is empty."

        idx = 0
        first = tokens_lines[0]
        if first[0].upper() in ("OFF", "COFF", "NOFF", "4OFF"):
            if len(first) >= 4:
                header_counts = first[1:4]
                idx = 1
            else:
                idx = 1
                if idx >= len(tokens_lines):
                    return False, "Invalid OFF header."
                header_counts = tokens_lines[idx][:3]
                idx += 1
        else:
            header_counts = first[:3]
            idx = 1

        try:
            n_verts = int(header_counts[0])
            n_faces = int(header_counts[1])
        except (IndexError, ValueError):
            return False, "Invalid OFF vertex/face counts."

        if n_verts < 3 or n_faces < 1:
            return False, "OFF file contains no valid faces."

        if idx + n_verts + n_faces > len(tokens_lines):
            return False, "OFF file ended unexpectedly."

        raw_verts = []
        try:
            for v_line in tokens_lines[idx:idx + n_verts]:
                raw_verts.append([
                    float(v_line[0]), float(v_line[1]), float(v_line[2])])
            idx += n_verts

            raw_faces = []
            for f_line in tokens_lines[idx:idx + n_faces]:
                k = int(f_line[0])
                if k < 3 or len(f_line) < 1 + k:
                    return False, "Invalid face record in OFF file."
                raw_faces.append([int(t) for t in f_line[1:1 + k]])
        except (IndexError, ValueError):
            return False, "Malformed vertex or face data in OFF file."

        verts_arr = np.array(raw_verts, dtype=np.float64)
        for f_nids in raw_faces:
            for nid in f_nids:
                if nid < 0 or nid >= n_verts:
                    return False, "Face vertex index out of range in OFF file."

        edge_lengths = []
        for f_nids in raw_faces:
            k = len(f_nids)
            for i in range(k):
                a, b = f_nids[i], f_nids[(i + 1) % k]
                dist = float(np.linalg.norm(verts_arr[b] - verts_arr[a]))
                if dist > 1e-8:
                    edge_lengths.append(dist)

        if not edge_lengths:
            return False, "OFF faces have zero edge length."

        edge_scale = float(np.median(edge_lengths))
        if abs(edge_scale - 1.0) >= 1e-8:
            verts_arr = verts_arr / edge_scale

        parent_uf = list(range(n_verts))

        def find_uf(x):
            while parent_uf[x] != x:
                parent_uf[x] = parent_uf[parent_uf[x]]
                x = parent_uf[x]
            return x

        tree = cKDTree(verts_arr)
        for i, j in tree.query_pairs(r=1e-6):
            ri, rj = find_uf(i), find_uf(j)
            if ri != rj:
                parent_uf[rj] = ri

        root_to_nid = {}
        merged_verts = []
        for i in range(n_verts):
            r = find_uf(i)
            if r not in root_to_nid:
                root_to_nid[r] = len(merged_verts)
                pt = verts_arr[r].copy()
                pt[np.abs(pt) < 1e-8] = 0.0
                merged_verts.append(pt)

        faces_nids = []
        for f_nids in raw_faces:
            mapped = [root_to_nid[find_uf(nid)] for nid in f_nids]
            if len(set(mapped)) != len(mapped):
                return False, "Degenerate face in OFF file."
            faces_nids.append(mapped)

        def infer_density(face_nids):
            k_len = len(face_nids)
            if k_len < 5:
                return 1
            pts = np.array(
                [merged_verts[nid] for nid in face_nids], dtype=np.float64)
            p0 = pts[0]
            p1 = pts[1]
            u_e = p1 - p0
            u_e_len = float(np.linalg.norm(u_e))
            if u_e_len <= 1e-8:
                return 1
            u_e = u_e / u_e_len
            c_pts = np.mean(pts, axis=0)
            w_v = c_pts - 0.5 * (p0 + p1)
            f_v = w_v - float(np.dot(w_v, u_e)) * u_e
            f_v_len = float(np.linalg.norm(f_v))
            if f_v_len <= 1e-8:
                return 1
            f_v = f_v / f_v_len

            best_d = 1
            best_err = float("inf")
            for d_cand in range(1, (k_len + 1) // 2):
                if math.gcd(k_len, d_cand) != 1:
                    continue
                canon = _canonical_ngon_array(k_len, d_cand)
                recon = p0 + canon[:, 0:1] * u_e + canon[:, 1:2] * f_v
                err = float(np.max(np.linalg.norm(recon - pts, axis=1)))
                if err < best_err:
                    best_err = err
                    best_d = d_cand
            return best_d

        edge_to_faces = {}
        for f_idx, f_nids in enumerate(faces_nids):
            k = len(f_nids)
            for e_idx in range(k):
                u, v = f_nids[e_idx], f_nids[(e_idx + 1) % k]
                canon = (u, v) if u < v else (v, u)
                if canon not in edge_to_faces:
                    edge_to_faces[canon] = []
                edge_to_faces[canon].append((f_idx, e_idx))

        f0_nids = faces_nids[0]
        new_faces = [{
            "id": 0,
            "gonality": len(f0_nids),
            "density": infer_density(f0_nids),
            "node_ids": list(f0_nids),
            "parent_id": None,
            "parent_edge_idx": None,
            "hinge_edge": None,
            "hinge_angle": 0.0,
            "base_pts": [merged_verts[nid].copy() for nid in f0_nids],
            "world_pts": [merged_verts[nid].copy() for nid in f0_nids]}]
        visited = {0: 0}
        queue = [0]

        while queue:
            cur_orig = queue.pop(0)
            cur_new_id = visited[cur_orig]
            cur_nids = new_faces[cur_new_id]["node_ids"]
            cur_pts = [merged_verts[nid] for nid in cur_nids]
            c_parent = np.mean(cur_pts, axis=0)
            k_p = len(cur_nids)

            for e_idx in range(k_p):
                a_nid = cur_nids[e_idx]
                b_nid = cur_nids[(e_idx + 1) % k_p]
                canon = (a_nid, b_nid) if a_nid < b_nid else (b_nid, a_nid)
                for nbr_orig, _nbr_e in edge_to_faces.get(canon, []):
                    if nbr_orig in visited:
                        continue

                    nbr_raw = faces_nids[nbr_orig]
                    k_c = len(nbr_raw)
                    j = nbr_raw.index(a_nid)
                    if nbr_raw[(j + 1) % k_c] == b_nid:
                        ordered = [nbr_raw[(j + t) % k_c] for t in range(k_c)]
                    else:
                        ordered = [nbr_raw[(j - t) % k_c] for t in range(k_c)]

                    a_pt = merged_verts[a_nid]
                    b_pt = merged_verts[b_nid]
                    u_edge = b_pt - a_pt
                    u_len = float(np.linalg.norm(u_edge))
                    if u_len <= 1e-8:
                        continue
                    u_edge = u_edge / u_len

                    v_vec = 0.5 * (a_pt + b_pt) - c_parent
                    o_vec = v_vec - float(np.dot(v_vec, u_edge)) * u_edge
                    o_len = float(np.linalg.norm(o_vec))
                    if o_len <= 1e-8:
                        continue
                    o_vec = o_vec / o_len
                    n_vec = np.cross(u_edge, o_vec)

                    c_child = np.mean(
                        [merged_verts[nid] for nid in ordered], axis=0)
                    w_vec = c_child - 0.5 * (a_pt + b_pt)
                    f_vec = w_vec - float(np.dot(w_vec, u_edge)) * u_edge
                    f_len = float(np.linalg.norm(f_vec))
                    if f_len <= 1e-8:
                        continue
                    f_vec = f_vec / f_len

                    theta = math.atan2(
                        float(np.dot(f_vec, n_vec)),
                        float(np.dot(f_vec, o_vec)))
                    theta = 0.0 if abs(theta) < 1e-8 else theta

                    new_id = len(new_faces)
                    visited[nbr_orig] = new_id
                    new_faces.append({
                        "id": new_id,
                        "gonality": k_c,
                        "density": infer_density(ordered),
                        "node_ids": ordered,
                        "parent_id": cur_new_id,
                        "parent_edge_idx": e_idx,
                        "hinge_edge": (a_nid, b_nid),
                        "hinge_angle": theta,
                        "world_pts": []})
                    queue.append(nbr_orig)

        if len(new_faces) != len(faces_nids):
            return False, "Cannot open OFF: mesh is not edge-connected."

        old_faces = self.faces
        old_next = self.next_node_id
        self.faces = new_faces
        self.next_node_id = len(merged_verts)
        self.update_world_vertices()

        max_err = 0.0
        for f in self.faces:
            for nid, wpt in zip(f["node_ids"], f["world_pts"]):
                err = float(np.linalg.norm(wpt - merged_verts[nid]))
                if err > max_err:
                    max_err = err

        if max_err > 1e-2:
            self.faces = old_faces
            self.next_node_id = old_next
            return False, "Cannot open OFF: faces are not regular polygons."

        new_snap = self.faces
        new_next = self.next_node_id
        self.faces = old_faces
        self.next_node_id = old_next
        self.save_history()
        self.faces = new_snap
        self.next_node_id = new_next
        self.last_merge_map = {}
        return True, (
            f"Opened {os.path.basename(filepath)} ({len(self.faces)} faces).")

    def export_off(self, filepath=None):
        self.update_world_vertices()

        all_pts = []
        face_vert_refs = []
        for f in self.faces:
            refs = []
            for raw_pt in f["world_pts"]:
                pt = np.array([
                    0.0 if abs(raw_pt[0]) < 1e-8 else float(raw_pt[0]),
                    0.0 if abs(raw_pt[1]) < 1e-8 else float(raw_pt[1]),
                    0.0 if abs(raw_pt[2]) < 1e-8 else float(raw_pt[2])],
                    dtype=np.float64)
                refs.append(len(all_pts))
                all_pts.append(pt)
            face_vert_refs.append(refs)

        n_total = len(all_pts)
        pts_arr = np.array(all_pts, dtype=np.float64)
        parent = list(range(n_total))

        def find(i):
            path = []
            while parent[i] != i:
                path.append(i)
                i = parent[i]
            for node in path:
                parent[node] = i
            return i

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i

        if n_total > 1:
            tree = cKDTree(pts_arr)
            for i, j in tree.query_pairs(r=1e-3):
                union(i, j)

        root_groups = {}
        for i in range(n_total):
            root = find(i)
            if root not in root_groups:
                root_groups[root] = []
            root_groups[root].append(pts_arr[i])

        cluster_map = {}
        unique_vertices = []
        for root, pts_group in root_groups.items():
            cluster_map[root] = len(unique_vertices)
            c_pt = np.mean(pts_group, axis=0)
            c_pt = np.where(np.abs(c_pt) < 1e-8, 0.0, c_pt)
            unique_vertices.append(c_pt)

        for f_idx, f in enumerate(self.faces):
            k_len = f["gonality"]
            if k_len < 4:
                continue

            v_indices = [cluster_map[find(r)] for r in face_vert_refs[f_idx]]
            p0 = unique_vertices[v_indices[0]]
            p1 = unique_vertices[v_indices[1]]
            p2 = unique_vertices[v_indices[2]]

            n_raw = np.cross(p1 - p0, p2 - p0)
            n_len = np.linalg.norm(n_raw)
            if n_len > 1e-12:
                n_hat = n_raw / n_len
                for idx in range(3, k_len):
                    v_target = v_indices[idx]
                    pj = unique_vertices[v_target]
                    dist_to_plane = float(np.dot(pj - p0, n_hat))
                    if abs(dist_to_plane) < 1e-4:
                        corrected_pt = pj - dist_to_plane * n_hat
                        unique_vertices[v_target] = np.where(
                            np.abs(corrected_pt) < 1e-8, 0.0, corrected_pt)

        lines = ["OFF", f"{len(unique_vertices)} {len(self.faces)} 0"]
        for pt in unique_vertices:
            lines.append(f"{pt[0]:.16f} {pt[1]:.16f} {pt[2]:.16f}")

        for f, refs in zip(self.faces, face_vert_refs):
            v_indices = [str(cluster_map[find(r)]) for r in refs]
            col = get_shape_colour(f["gonality"])
            r = int(round(np.clip(col[0] * 255.0, 0.0, 255.0)))
            g = int(round(np.clip(col[1] * 255.0, 0.0, 255.0)))
            b = int(round(np.clip(col[2] * 255.0, 0.0, 255.0)))
            lines.append(f"{len(v_indices)} {' '.join(v_indices)} {r} {g} {b}")

        content = "\n".join(lines) + "\n"
        if filepath:
            target_path = filepath
            if os.path.exists(target_path):
                stem, ext = os.path.splitext(target_path)
                target_path = get_next_output_path(
                    prefix=stem, ext=ext or ".off")
            with open(target_path, "w", encoding="utf-8") as fh:
                fh.write(content)
            return content, target_path
        return content, None


# -----------------------------------------------------------------------------
# 3. Main Interactive Application Loop
# -----------------------------------------------------------------------------

BTN_UNDO_RECT = (20, 16, 120, 48)
BTN_DELETE_RECT = (130, 16, 250, 48)

# Version: 2.8.4

def main():
    model = DynamicPolydron()
    sec_model = DynamicPolydron()

    pygame.init()
    width, height = 1280, 720
    pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)

    surf_prog, line_prog, help_prog = create_gl_programs(ctx)

    face_surf_vbo = ctx.buffer(reserve=120000 * 4, dynamic=True)
    face_surf_vao = ctx.vertex_array(
        surf_prog, [(face_surf_vbo, "3f 3f 3f", "in_pos", "in_norm", "in_col")])

    marker_surf_vbo = ctx.buffer(reserve=60000 * 4, dynamic=True)
    marker_surf_vao = ctx.vertex_array(
        surf_prog, [(marker_surf_vbo, "3f 3f 3f", "in_pos", "in_norm", "in_col")])

    line_vbo = ctx.buffer(reserve=30000 * 4, dynamic=True)
    line_vao = ctx.vertex_array(
        line_prog, [(line_vbo, "3f 3f", "in_pos", "in_col")])

    sec_face_surf_vbo = ctx.buffer(reserve=120000 * 4, dynamic=True)
    sec_face_surf_vao = ctx.vertex_array(
        surf_prog,
        [(sec_face_surf_vbo, "3f 3f 3f", "in_pos", "in_norm", "in_col")])

    sec_marker_surf_vbo = ctx.buffer(reserve=60000 * 4, dynamic=True)
    sec_marker_surf_vao = ctx.vertex_array(
        surf_prog,
        [(sec_marker_surf_vbo, "3f 3f 3f", "in_pos", "in_norm", "in_col")])

    sec_line_vbo = ctx.buffer(reserve=30000 * 4, dynamic=True)
    sec_line_vao = ctx.vertex_array(
        line_prog, [(sec_line_vbo, "3f 3f", "in_pos", "in_col")])

    ui_surf_vbo = ctx.buffer(reserve=1024 * 4, dynamic=True)
    ui_surf_vao = ctx.vertex_array(
        surf_prog, [(ui_surf_vbo, "3f 3f 3f", "in_pos", "in_norm", "in_col")])

    ui_line_vbo = ctx.buffer(reserve=1024 * 4, dynamic=True)
    ui_line_vao = ctx.vertex_array(
        line_prog, [(ui_line_vbo, "3f 3f", "in_pos", "in_col")])

    divider_vao = create_divider_vao(ctx, line_prog)
    help_tex, help_vao = create_help_overlay(
        ctx, help_prog, HELP_LINES, width, height)

    default_rot = np.array([
        [np.cos(-0.4), 0.0, np.sin(-0.4)],
        [0.0, 1.0, 0.0],
        [-np.sin(-0.4), 0.0, np.cos(-0.4)]]) @ np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(0.55), -np.sin(0.55)],
        [0.0, np.sin(0.55), np.cos(0.55)]])
    rot_matrix = default_rot.copy()
    sec_rot_matrix = default_rot.copy()

    cam_dist = 4.2
    sec_cam_dist = 4.2
    ident_mvp = np.eye(4, dtype=np.float32)
    ident_rot = np.eye(3, dtype=np.float32)

    split_active = False
    mouse_down = False
    orbit_pane = "main"
    last_mouse = (0, 0)
    mouse_moved_dist = 0.0
    bg_white = False
    show_help = False
    fold_hold_dir = 0
    fold_next_repeat = 0.0
    dragging_vertex = None
    drag_cog = None
    drag_ndc_z = 0.0
    drag_ndc_offset = (0.0, 0.0)
    pending_drag_pos = None
    clock = pygame.time.Clock()

    hovered_vertex = None
    hovered_edge = None
    selected_vertices = []
    selected_edge = None

    sec_hovered_edge = None
    sec_selected_edge = None

    status_msg = (
        "Press [3-9, 0-2=10-12; hold num+denom for stars] to add "
        "first polygon | [I] Open OFF | [J] Secondary OFF | [H] Help")
    last_status_msg = None

    model_dirty = True
    view_dirty = True
    pick_dirty = True
    last_highlight_key = None
    last_ui_key = None

    sec_model_dirty = True
    sec_view_dirty = True
    sec_pick_dirty = True
    sec_last_highlight_key = None

    node_coords = {}
    all_nids = []
    raw_pts_arr = np.empty((0, 3), dtype=np.float32)
    cog = np.zeros(3, dtype=np.float64)
    open_edges_set = set()
    open_nids_set = set()
    all_edges_set = set()

    sec_node_coords = {}
    sec_all_nids = []
    sec_raw_pts_arr = np.empty((0, 3), dtype=np.float32)
    sec_cog = np.zeros(3, dtype=np.float64)
    sec_open_edges_set = set()
    sec_open_nids_set = set()
    sec_all_edges_set = set()

    mvp_3d = np.eye(4, dtype=np.float32)
    rot_f32 = np.eye(3, dtype=np.float32)
    sec_mvp_3d = np.eye(4, dtype=np.float32)
    sec_rot_f32 = np.eye(3, dtype=np.float32)

    face_vert_count = 0
    marker_vert_count = 0
    line_vert_count = 0
    sec_face_vert_count = 0
    sec_marker_vert_count = 0
    sec_line_vert_count = 0
    ui_surf_vert_count = 0
    ui_line_vert_count = 0

    gon_key_map = {
        pygame.K_1: 11,
        pygame.K_2: 12,
        pygame.K_3: 3,
        pygame.K_4: 4,
        pygame.K_5: 5,
        pygame.K_6: 6,
        pygame.K_7: 7,
        pygame.K_8: 8,
        pygame.K_9: 9,
        pygame.K_0: 10}
    star_chord_map = {
        (pygame.K_5, pygame.K_2): (5, 2),
        (pygame.K_7, pygame.K_2): (7, 2),
        (pygame.K_7, pygame.K_3): (7, 3),
        (pygame.K_8, pygame.K_3): (8, 3),
        (pygame.K_9, pygame.K_2): (9, 2),
        (pygame.K_9, pygame.K_4): (9, 4),
        (pygame.K_0, pygame.K_3): (10, 3),
        (pygame.K_1, pygame.K_2): (11, 2),
        (pygame.K_1, pygame.K_3): (11, 3),
        (pygame.K_1, pygame.K_4): (11, 4),
        (pygame.K_1, pygame.K_5): (11, 5),
        (pygame.K_2, pygame.K_5): (12, 5)}
    star_numerators = {num_key for num_key, _ in star_chord_map}
    held_num_key = None
    chord_used = False
    focus_lost_evt = getattr(pygame, "WINDOWFOCUSLOST", None)

    def try_attach(gon, dens):
        nonlocal selected_edge, model_dirty, status_msg
        shape_label = get_shape_name(gon, dens)
        if not model.faces:
            model.add_base_polygon(gon, density=dens)
            status_msg = f"Created base {shape_label}."
            selected_edge = None
            selected_vertices.clear()
            model_dirty = True
            return
        if selected_edge:
            if model.attach_polygon(selected_edge, gon, density=dens):
                status_msg = f"Attached {shape_label} to edge {selected_edge}."
                selected_edge = None
                model_dirty = True
            else:
                status_msg = "Cannot attach: Edge is already closed."
        else:
            status_msg = (
                f"Click an open edge first before pressing key "
                f"for {shape_label}.")

    def perform_undo():
        nonlocal selected_edge, sec_selected_edge, model_dirty, status_msg
        ok, merge_edges = model.undo()
        if ok:
            selected_vertices.clear()
            if merge_edges is not None:
                selected_edge = merge_edges[0]
                if split_active and merge_edges[1] in sec_open_edges_set:
                    sec_selected_edge = merge_edges[1]
            else:
                selected_edge = None
            model_dirty = True
            status_msg = "Undid last addition or merge."
        else:
            status_msg = "Nothing to undo."

    def close_secondary_view():
        nonlocal split_active, sec_selected_edge, sec_hovered_edge
        nonlocal sec_model_dirty, view_dirty
        sec_model.faces = []
        sec_model.next_node_id = 0
        sec_model.history = []
        sec_selected_edge = None
        sec_hovered_edge = None
        split_active = False
        sec_model_dirty = True
        view_dirty = True

    def sync_main_model_cache():
        nonlocal node_coords, cog, all_nids, raw_pts_arr
        nonlocal open_edges_set, open_nids_set, all_edges_set, face_vert_count
        nonlocal model_dirty, view_dirty, pick_dirty, last_highlight_key
        nonlocal selected_edge, dragging_vertex, drag_cog, pending_drag_pos

        node_coords = model.get_node_coords()
        cog = model.get_centre_of_gravity(node_coords)
        all_nids = sorted(node_coords.keys())
        if all_nids:
            raw_pts_arr = np.array(
                [node_coords[nid] for nid in all_nids], dtype=np.float32)
        else:
            raw_pts_arr = np.empty((0, 3), dtype=np.float32)

        all_edges_dict = {}
        for f in model.faces:
            n = f["gonality"]
            nids = f["node_ids"]
            for i in range(n):
                u = nids[i]
                v = nids[(i + 1) % n]
                canon = (u, v) if u < v else (v, u)
                all_edges_dict[canon] = all_edges_dict.get(canon, 0) + 1

        open_edges_set = {e for e, cnt in all_edges_dict.items() if cnt == 1}
        open_nids_set = {nid for e in open_edges_set for nid in e}
        all_edges_set = set(all_edges_dict.keys())

        merge_map = model.consume_merge_map()
        if merge_map:
            if selected_edge is not None:
                ea, eb = selected_edge
                na = merge_map.get(ea, ea)
                nb = merge_map.get(eb, eb)
                if na != nb:
                    canon = (na, nb) if na < nb else (nb, na)
                    selected_edge = canon if canon in all_edges_set else None
                else:
                    selected_edge = None

            if dragging_vertex is not None:
                dragging_vertex = merge_map.get(
                    dragging_vertex, dragging_vertex)
                if dragging_vertex not in open_nids_set:
                    dragging_vertex = None
                    drag_cog = None
                    pending_drag_pos = None

            remapped_verts = []
            for nid in selected_vertices:
                mapped = merge_map.get(nid, nid)
                if mapped in open_nids_set and mapped not in remapped_verts:
                    remapped_verts.append(mapped)
            selected_vertices[:] = remapped_verts
        else:
            selected_vertices[:] = [
                nid for nid in selected_vertices if nid in open_nids_set]

        face_surf_data = []
        for f in model.faces:
            col = get_shape_colour(f["gonality"])
            norm = compute_polygon_normal(f["world_pts"])
            append_face_triangles(face_surf_data, f, norm, col)

        face_vert_count = upload_vbo(face_surf_vbo, face_surf_data, 9)
        model_dirty = False
        view_dirty = True
        pick_dirty = True
        last_highlight_key = None

    def sync_sec_model_cache():
        nonlocal sec_node_coords, sec_cog, sec_all_nids, sec_raw_pts_arr
        nonlocal sec_open_edges_set, sec_open_nids_set, sec_all_edges_set
        nonlocal sec_face_vert_count, sec_marker_vert_count
        nonlocal sec_model_dirty, sec_view_dirty, sec_pick_dirty
        nonlocal sec_last_highlight_key

        sec_node_coords = sec_model.get_node_coords()
        sec_cog = sec_model.get_centre_of_gravity(sec_node_coords)
        sec_all_nids = sorted(sec_node_coords.keys())
        if sec_all_nids:
            sec_raw_pts_arr = np.array(
                [sec_node_coords[nid] for nid in sec_all_nids],
                dtype=np.float32)
        else:
            sec_raw_pts_arr = np.empty((0, 3), dtype=np.float32)

        sec_edges_dict = {}
        for f in sec_model.faces:
            n = f["gonality"]
            nids = f["node_ids"]
            for i in range(n):
                u = nids[i]
                v = nids[(i + 1) % n]
                canon = (u, v) if u < v else (v, u)
                sec_edges_dict[canon] = sec_edges_dict.get(canon, 0) + 1

        sec_open_edges_set = {
            e for e, cnt in sec_edges_dict.items() if cnt == 1}
        sec_open_nids_set = {nid for e in sec_open_edges_set for nid in e}
        sec_all_edges_set = set(sec_edges_dict.keys())

        sec_face_data = []
        for f in sec_model.faces:
            col = get_shape_colour(f["gonality"])
            norm = compute_polygon_normal(f["world_pts"])
            append_face_triangles(sec_face_data, f, norm, col)

        sec_face_vert_count = upload_vbo(sec_face_surf_vbo, sec_face_data, 9)

        sec_marker_data = []
        for nid in sec_all_nids:
            if nid not in sec_open_nids_set:
                continue
            append_vertex_marker(
                sec_marker_data, sec_node_coords[nid], 0.035,
                [0.95, 0.95, 0.98])
        sec_marker_vert_count = upload_vbo(
            sec_marker_surf_vbo, sec_marker_data, 9)

        sec_model_dirty = False
        sec_view_dirty = True
        sec_pick_dirty = True
        sec_last_highlight_key = None

    def sync_views():
        nonlocal view_dirty, sec_view_dirty, pick_dirty, sec_pick_dirty
        nonlocal mvp_3d, rot_f32, sec_mvp_3d, sec_rot_f32
        main_w = (width // 2) if split_active else width
        sec_w = max(width - main_w, 1)

        if view_dirty:
            proj_m = make_perspective(45.0, main_w / height, 0.1, 100.0)
            view_m = make_look_at(
                np.array([0.0, 0.0, cam_dist], dtype=np.float32),
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32))
            rot_f32 = rot_matrix.astype(np.float32)
            active_cog = (
                drag_cog
                if dragging_vertex is not None and drag_cog is not None
                else cog)
            cog_f32 = active_cog.astype(np.float32)
            model_mat = np.eye(4, dtype=np.float32)
            model_mat[:3, :3] = rot_f32
            model_mat[:3, 3] = -(rot_f32 @ cog_f32)
            mvp_3d = proj_m @ view_m @ model_mat
            view_dirty = False
            pick_dirty = True

        if split_active and sec_view_dirty:
            proj_s = make_perspective(45.0, sec_w / height, 0.1, 100.0)
            view_s = make_look_at(
                np.array([0.0, 0.0, sec_cam_dist], dtype=np.float32),
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32))
            sec_rot_f32 = sec_rot_matrix.astype(np.float32)
            sec_cog_f32 = sec_cog.astype(np.float32)
            sec_model_mat = np.eye(4, dtype=np.float32)
            sec_model_mat[:3, :3] = sec_rot_f32
            sec_model_mat[:3, 3] = -(sec_rot_f32 @ sec_cog_f32)
            sec_mvp_3d = proj_s @ view_s @ sec_model_mat
            sec_view_dirty = False
            sec_pick_dirty = True

    running = True
    while running:
        clock.tick(60)
        mouse_pos = pygame.mouse.get_pos()

        if model_dirty:
            sync_main_model_cache()
        if sec_model_dirty:
            sync_sec_model_cache()
        sync_views()

        main_w = (width // 2) if split_active else width
        sec_w = max(width - main_w, 1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif focus_lost_evt is not None and event.type == focus_lost_evt:
                held_num_key = None
                chord_used = False
                fold_hold_dir = 0
                dragging_vertex = None
                drag_cog = None
                pending_drag_pos = None
                view_dirty = True
            elif event.type == pygame.KEYDOWN:
                mod_held = bool(event.mod & (pygame.KMOD_CTRL | pygame.KMOD_ALT))
                if event.key in gon_key_map and not mod_held:
                    if held_num_key is None:
                        if event.key in star_numerators:
                            held_num_key = event.key
                            chord_used = False
                        else:
                            try_attach(gon_key_map[event.key], 1)
                    elif event.key != held_num_key:
                        chord_used = True
                        spec = star_chord_map.get((held_num_key, event.key))
                        if spec is not None:
                            try_attach(spec[0], spec[1])
                        else:
                            num_gon = gon_key_map[held_num_key]
                            den_gon = gon_key_map[event.key]
                            status_msg = (
                                f"Unsupported star polygon ({num_gon}/{den_gon}).")
                elif event.key == pygame.K_m:
                    is_ctrl = bool(event.mod & pygame.KMOD_CTRL)
                    is_alt = bool(event.mod & pygame.KMOD_ALT)
                    if (
                            len(selected_vertices) == 2
                            and not is_ctrl
                            and not is_alt):
                        v1, v2 = selected_vertices
                        status_msg = (
                            f"Solving hinge merge between vertex {v1} and {v2}...")
                        pygame.display.set_caption(
                            f"Polydron Step 3 | {status_msg}")
                        pygame.event.pump()
                        if model.solve_merge(v1, v2):
                            status_msg = (
                                f"SUCCESS: Merged vertex {v1} and {v2} "
                                "to double precision!")
                            selected_vertices.clear()
                            model_dirty = True
                        else:
                            status_msg = (
                                "CANNOT MERGE: Rigid tiles cannot close that gap.")
                    elif (
                            split_active
                            and selected_edge is not None
                            and sec_selected_edge is not None):
                        ok, msg = model.merge_edge_model(
                            selected_edge,
                            sec_model,
                            sec_selected_edge,
                            flip=is_ctrl,
                            mirror=is_alt)
                        if ok:
                            selected_vertices.clear()
                            model_dirty = True
                        status_msg = msg
                    elif split_active:
                        status_msg = (
                            "Select an unmatched edge in both views "
                            "(or 2 vertices in main) and press [M] to merge.")
                    else:
                        status_msg = (
                            "Click TWO corners first (Cyan & Magenta), "
                            "then press [M] to merge.")
                elif event.key == pygame.K_u:
                    held_num_key = None
                    chord_used = False
                    fold_hold_dir = 0
                    dragging_vertex = None
                    drag_cog = None
                    pending_drag_pos = None
                    perform_undo()
                elif event.key == pygame.K_d:
                    if selected_edge:
                        success, msg = model.delete_face_by_edge(selected_edge)
                        if success:
                            selected_edge = None
                            selected_vertices.clear()
                            model_dirty = True
                        status_msg = msg
                    else:
                        status_msg = (
                            "Cannot delete: Select an unmatched (open) edge first.")
                elif event.key == pygame.K_i:
                    held_num_key = None
                    chord_used = False
                    fold_hold_dir = 0
                    dragging_vertex = None
                    drag_cog = None
                    pending_drag_pos = None
                    off_path = choose_open_off_path("Open OFF file")
                    pygame.event.pump()
                    if off_path:
                        ok, msg = model.load_off(off_path)
                        if ok:
                            selected_vertices.clear()
                            selected_edge = None
                            model_dirty = True
                        status_msg = msg
                    else:
                        status_msg = "Open OFF cancelled."
                elif event.key == pygame.K_j:
                    held_num_key = None
                    chord_used = False
                    fold_hold_dir = 0
                    dragging_vertex = None
                    drag_cog = None
                    pending_drag_pos = None
                    off_path = choose_open_off_path("Open secondary OFF file")
                    pygame.event.pump()
                    if off_path:
                        ok, msg = sec_model.load_off(off_path)
                        if ok:
                            sec_model.history = []
                            split_active = True
                            sec_selected_edge = None
                            sec_hovered_edge = None
                            sec_rot_matrix = default_rot.copy()
                            sec_cam_dist = 4.2
                            sec_model_dirty = True
                            view_dirty = True
                            sec_view_dirty = True
                            status_msg = f"Secondary: {msg}"
                        else:
                            status_msg = msg
                    else:
                        status_msg = "Open secondary OFF cancelled."
                elif event.key == pygame.K_c:
                    held_num_key = None
                    chord_used = False
                    fold_hold_dir = 0
                    dragging_vertex = None
                    drag_cog = None
                    pending_drag_pos = None
                    if not model.faces:
                        status_msg = "Main window is blank; nothing to copy."
                    else:
                        sec_model.faces = copy.deepcopy(model.faces)
                        sec_model.next_node_id = model.next_node_id
                        sec_model.history = []
                        sec_model.update_world_vertices()
                        split_active = True
                        sec_selected_edge = None
                        sec_hovered_edge = None
                        sec_rot_matrix = rot_matrix.copy()
                        sec_cam_dist = cam_dist
                        sec_model_dirty = True
                        view_dirty = True
                        sec_view_dirty = True
                        status_msg = (
                            "Copied main model into secondary view "
                            "(read-only).")
                elif event.key == pygame.K_x:
                    held_num_key = None
                    chord_used = False
                    fold_hold_dir = 0
                    dragging_vertex = None
                    drag_cog = None
                    pending_drag_pos = None
                    if split_active:
                        close_secondary_view()
                        status_msg = "Closed secondary view."
                    else:
                        status_msg = "Secondary view is already closed."
                elif event.key == pygame.K_o:
                    held_num_key = None
                    chord_used = False
                    fold_hold_dir = 0
                    dragging_vertex = None
                    drag_cog = None
                    pending_drag_pos = None
                    save_path = choose_save_off_path(
                        "Export OFF file", default_stem="output")
                    pygame.event.pump()
                    if save_path:
                        off_text, saved_path = model.export_off(save_path)
                        print(off_text)
                        status_msg = (
                            f"Exported {saved_path}: merged vertices, "
                            "planarity < 1e-15, 16dp and RGB 0-255.")
                    else:
                        status_msg = "Export OFF cancelled."
                elif event.key == pygame.K_b:
                    bg_white = not bg_white
                    bg_label = "white" if bg_white else "black"
                    status_msg = f"Background colour set to {bg_label}."
                elif event.key == pygame.K_h:
                    show_help = not show_help
                    status_msg = (
                        "Showing help (press [H] to hide)."
                        if show_help
                        else "Help hidden.")
                elif event.key == pygame.K_LEFTBRACKET:
                    if selected_edge:
                        changed, msg = model.fold_hinge(
                            selected_edge, -math.radians(5.0))
                        if changed:
                            model_dirty = True
                            fold_hold_dir = -1
                            fold_next_repeat = time.monotonic() + 0.35
                        else:
                            fold_hold_dir = 0
                        status_msg = msg
                elif event.key == pygame.K_RIGHTBRACKET:
                    if selected_edge:
                        changed, msg = model.fold_hinge(
                            selected_edge, math.radians(5.0))
                        if changed:
                            model_dirty = True
                            fold_hold_dir = 1
                            fold_next_repeat = time.monotonic() + 0.35
                        else:
                            fold_hold_dir = 0
                        status_msg = msg
                elif event.key == pygame.K_r:
                    held_num_key = None
                    chord_used = False
                    fold_hold_dir = 0
                    dragging_vertex = None
                    drag_cog = None
                    pending_drag_pos = None
                    model.reset()
                    selected_vertices.clear()
                    selected_edge = None
                    close_secondary_view()
                    model_dirty = True
                    status_msg = (
                        "Reset to blank screen and closed secondary view.")
                elif event.key == pygame.K_ESCAPE:
                    held_num_key = None
                    chord_used = False
                    fold_hold_dir = 0
                    dragging_vertex = None
                    drag_cog = None
                    pending_drag_pos = None
                    running = False
            elif event.type == pygame.KEYUP:
                if held_num_key is not None and event.key == held_num_key:
                    if not chord_used:
                        try_attach(gon_key_map[held_num_key], 1)
                    held_num_key = None
                    chord_used = False
                elif event.key == pygame.K_LEFTBRACKET and fold_hold_dir == -1:
                    fold_hold_dir = 0
                elif event.key == pygame.K_RIGHTBRACKET and fold_hold_dir == 1:
                    fold_hold_dir = 0

            elif event.type == pygame.MOUSEBUTTONDOWN:
                in_sec = split_active and event.pos[0] >= main_w
                if event.button == 1:
                    mouse_down = True
                    orbit_pane = "sec" if in_sec else "main"
                    last_mouse = event.pos
                    mouse_moved_dist = 0.0
                elif (
                        event.button == 3
                        and not in_sec
                        and hovered_vertex is not None):
                    v_pos = node_coords.get(hovered_vertex)
                    if v_pos is not None:
                        dragging_vertex = hovered_vertex
                        hovered_edge = None
                        drag_cog = cog.copy()
                        pending_drag_pos = None
                        v_h = np.array(
                            [v_pos[0], v_pos[1], v_pos[2], 1.0],
                            dtype=np.float64)
                        clip_v = mvp_3d.astype(np.float64) @ v_h
                        if abs(clip_v[3]) > 1e-8:
                            ndc_vx = float(clip_v[0] / clip_v[3])
                            ndc_vy = float(clip_v[1] / clip_v[3])
                            drag_ndc_z = float(clip_v[2] / clip_v[3])
                            mx_ndc = (event.pos[0] / main_w) * 2.0 - 1.0
                            my_ndc = 1.0 - (event.pos[1] / height) * 2.0
                            drag_ndc_offset = (
                                ndc_vx - mx_ndc, ndc_vy - my_ndc)
                            status_msg = (
                                f"Dragging vertex {dragging_vertex}...")
                        else:
                            dragging_vertex = None
                            drag_cog = None
                elif event.button == 4:
                    if in_sec:
                        sec_cam_dist = max(1.5, sec_cam_dist - 0.25)
                        sec_view_dirty = True
                    else:
                        cam_dist = max(1.5, cam_dist - 0.25)
                        view_dirty = True
                elif event.button == 5:
                    if in_sec:
                        sec_cam_dist = min(15.0, sec_cam_dist + 0.25)
                        sec_view_dirty = True
                    else:
                        cam_dist = min(15.0, cam_dist + 0.25)
                        view_dirty = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down = False
                    if mouse_moved_dist < 4.0:
                        mx, my = event.pos
                        if orbit_pane == "sec" and split_active and mx >= main_w:
                            if sec_hovered_edge is not None:
                                if sec_hovered_edge in sec_open_edges_set:
                                    if sec_selected_edge == sec_hovered_edge:
                                        sec_selected_edge = None
                                        status_msg = (
                                            "Deselected secondary edge.")
                                    else:
                                        sec_selected_edge = sec_hovered_edge
                                        status_msg = (
                                            f"Selected secondary unmatched "
                                            f"edge {sec_selected_edge} "
                                            "| Select main open edge & "
                                            "[M] / [Ctrl+M] / [Alt+M]")
                                else:
                                    status_msg = (
                                        "Secondary view is read-only: select "
                                        "an unmatched (open) edge to merge.")
                        else:
                            in_undo = (
                                BTN_UNDO_RECT[0] <= mx <= BTN_UNDO_RECT[2]
                                and BTN_UNDO_RECT[1] <= my <= BTN_UNDO_RECT[3])
                            in_delete = (
                                BTN_DELETE_RECT[0] <= mx <= BTN_DELETE_RECT[2]
                                and BTN_DELETE_RECT[1] <= my <= BTN_DELETE_RECT[3])
                            if in_undo:
                                perform_undo()
                            elif in_delete:
                                if selected_edge:
                                    success, msg = model.delete_face_by_edge(
                                        selected_edge)
                                    if success:
                                        selected_edge = None
                                        selected_vertices.clear()
                                        model_dirty = True
                                    status_msg = msg
                                else:
                                    status_msg = (
                                        "Cannot delete: Select an unmatched "
                                        "(open) edge first.")
                            elif hovered_vertex is not None:
                                if hovered_vertex in selected_vertices:
                                    selected_vertices.remove(hovered_vertex)
                                else:
                                    if len(selected_vertices) >= 2:
                                        selected_vertices.pop(0)
                                    selected_vertices.append(hovered_vertex)
                            elif hovered_edge is not None:
                                selected_edge = hovered_edge
                                is_open = hovered_edge in open_edges_set
                                if not is_open:
                                    changed, msg = model.fold_hinge(
                                        selected_edge, math.radians(5.0))
                                    if changed:
                                        model_dirty = True
                                    status_msg = msg
                                else:
                                    status_msg = (
                                        f"Selected unmatched edge "
                                        f"{selected_edge} | [3-9, 0-2=10-12; "
                                        "hold+press stars] | [D] to delete")
                elif event.button == 3 and dragging_vertex is not None:
                    finished_nid = dragging_vertex
                    dragging_vertex = None
                    drag_cog = None
                    pending_drag_pos = None
                    view_dirty = True
                    pick_dirty = True
                    status_msg = f"Finished dragging vertex {finished_nid}."
            elif event.type == pygame.MOUSEMOTION:
                if dragging_vertex is not None:
                    pending_drag_pos = event.pos
                else:
                    pick_dirty = True
                    sec_pick_dirty = True
                    if mouse_down:
                        dx = event.pos[0] - last_mouse[0]
                        dy = event.pos[1] - last_mouse[1]
                        mouse_moved_dist += math.sqrt(dx * dx + dy * dy)
                        last_mouse = event.pos

                        sens = 0.008
                        rx = np.array([
                            [1.0, 0.0, 0.0],
                            [0.0, np.cos(dy * sens), -np.sin(dy * sens)],
                            [0.0, np.sin(dy * sens), np.cos(dy * sens)]])
                        ry = np.array([
                            [np.cos(dx * sens), 0.0, np.sin(dx * sens)],
                            [0.0, 1.0, 0.0],
                            [-np.sin(dx * sens), 0.0, np.cos(dx * sens)]])
                        if orbit_pane == "sec" and split_active:
                            sec_rot_matrix = rx @ ry @ sec_rot_matrix
                            u_mat, _, vt_mat = np.linalg.svd(sec_rot_matrix)
                            sec_rot_matrix = u_mat @ vt_mat
                            sec_view_dirty = True
                        else:
                            rot_matrix = rx @ ry @ rot_matrix
                            u_mat, _, vt_mat = np.linalg.svd(rot_matrix)
                            rot_matrix = u_mat @ vt_mat
                            view_dirty = True

        if dragging_vertex is not None and pending_drag_pos is not None:
            mx_ndc = (
                (pending_drag_pos[0] / main_w) * 2.0
                - 1.0
                + drag_ndc_offset[0])
            my_ndc = (
                1.0
                - (pending_drag_pos[1] / height) * 2.0
                + drag_ndc_offset[1])
            pending_drag_pos = None
            inv_mvp = np.linalg.inv(mvp_3d.astype(np.float64))
            clip_target = np.array(
                [mx_ndc, my_ndc, drag_ndc_z, 1.0], dtype=np.float64)
            world_h = inv_mvp @ clip_target
            if abs(world_h[3]) > 1e-8:
                target_3d = world_h[:3] / world_h[3]
                if model.drag_vertex(dragging_vertex, target_3d):
                    model_dirty = True

        if not model_dirty and fold_hold_dir != 0 and selected_edge:
            now = time.monotonic()
            if now >= fold_next_repeat:
                delta = fold_hold_dir * math.radians(5.0)
                changed, msg = model.fold_hinge(selected_edge, delta)
                if changed:
                    model_dirty = True
                    fold_next_repeat = time.monotonic() + 0.08
                else:
                    fold_hold_dir = 0
                status_msg = msg

        if model_dirty:
            sync_main_model_cache()
        if sec_model_dirty:
            sync_sec_model_cache()
        sync_views()

        main_w = (width // 2) if split_active else width
        sec_w = max(width - main_w, 1)

        if dragging_vertex is not None:
            hovered_vertex = dragging_vertex
            hovered_edge = None
            pick_dirty = False
        elif pick_dirty:
            in_main_pane = (not split_active) or (mouse_pos[0] < main_w)
            if in_main_pane and len(raw_pts_arr) > 0:
                screen_pts_arr, valid_arr = project_to_screen(
                    raw_pts_arr, mvp_3d, main_w, height)
                screen_node_map = {
                    nid: screen_pts_arr[idx] for idx, nid in enumerate(all_nids)}
                valid_node_map = {
                    nid: valid_arr[idx] for idx, nid in enumerate(all_nids)}
                m_pt = np.array([mouse_pos[0], mouse_pos[1]], dtype=np.float32)

                hovered_vertex = None
                closest_v_dist = 12.0
                for nid in all_nids:
                    if nid in open_nids_set and valid_node_map[nid]:
                        d = float(
                            np.linalg.norm(m_pt - screen_node_map[nid][:2]))
                        if d < closest_v_dist:
                            closest_v_dist = d
                            hovered_vertex = nid

                hovered_edge = None
                if hovered_vertex is None:
                    closest_e_dist = 14.0
                    for e in all_edges_set:
                        ea, eb = e
                        if valid_node_map[ea] and valid_node_map[eb]:
                            d = float(point_to_segment_dist_2d(
                                m_pt,
                                screen_node_map[ea][:2],
                                screen_node_map[eb][:2]))
                            if d < closest_e_dist:
                                closest_e_dist = d
                                hovered_edge = e
            else:
                hovered_vertex = None
                hovered_edge = None
            pick_dirty = False

        if sec_pick_dirty:
            in_sec_pane = (
                split_active
                and dragging_vertex is None
                and mouse_pos[0] >= main_w)
            if in_sec_pane and len(sec_raw_pts_arr) > 0:
                sec_screen_arr, sec_valid_arr = project_to_screen(
                    sec_raw_pts_arr, sec_mvp_3d, sec_w, height)
                sec_screen_map = {
                    nid: sec_screen_arr[idx]
                    for idx, nid in enumerate(sec_all_nids)}
                sec_valid_map = {
                    nid: sec_valid_arr[idx]
                    for idx, nid in enumerate(sec_all_nids)}
                sec_m_pt = np.array(
                    [mouse_pos[0] - main_w, mouse_pos[1]], dtype=np.float32)

                sec_hovered_edge = None
                closest_se_dist = 14.0
                for e in sec_all_edges_set:
                    ea, eb = e
                    if sec_valid_map[ea] and sec_valid_map[eb]:
                        d = float(point_to_segment_dist_2d(
                            sec_m_pt,
                            sec_screen_map[ea][:2],
                            sec_screen_map[eb][:2]))
                        if d < closest_se_dist:
                            closest_se_dist = d
                            sec_hovered_edge = e
            else:
                sec_hovered_edge = None
            sec_pick_dirty = False

        if status_msg != last_status_msg:
            pygame.display.set_caption(f"Polydron Step 3 | {status_msg}")
            last_status_msg = status_msg

        highlight_key = (
            hovered_vertex,
            hovered_edge,
            tuple(selected_vertices),
            selected_edge)
        if highlight_key != last_highlight_key:
            marker_surf_data = []
            for nid in all_nids:
                if nid not in open_nids_set:
                    continue
                pos = node_coords[nid]
                if len(selected_vertices) > 0 and selected_vertices[0] == nid:
                    col = [0.0, 0.95, 1.0]
                    rad = 0.065
                elif len(selected_vertices) > 1 and selected_vertices[1] == nid:
                    col = [1.0, 0.1, 0.95]
                    rad = 0.065
                elif hovered_vertex == nid:
                    col = [1.0, 0.9, 0.0]
                    rad = 0.055
                else:
                    col = [0.95, 0.95, 0.98]
                    rad = 0.035
                append_vertex_marker(marker_surf_data, pos, rad, col)
            marker_vert_count = upload_vbo(marker_surf_vbo, marker_surf_data, 9)

            line_data = []
            for e in all_edges_set:
                ea, eb = e
                va, vb = node_coords[ea], node_coords[eb]
                is_open = e in open_edges_set

                if selected_edge == e:
                    col = [0.0, 1.0, 0.4]
                elif hovered_edge == e:
                    col = [1.0, 0.9, 0.0]
                elif is_open:
                    col = [0.25, 0.25, 0.35]
                else:
                    col = [0.08, 0.08, 0.1]

                line_data.extend([
                    float(va[0]), float(va[1]), float(va[2]), *col,
                    float(vb[0]), float(vb[1]), float(vb[2]), *col])
            line_vert_count = upload_vbo(line_vbo, line_data, 6)
            last_highlight_key = highlight_key

        if split_active:
            sec_highlight_key = (sec_hovered_edge, sec_selected_edge)
            if sec_highlight_key != sec_last_highlight_key:
                sec_line_data = []
                for e in sec_all_edges_set:
                    ea, eb = e
                    va, vb = sec_node_coords[ea], sec_node_coords[eb]
                    is_open = e in sec_open_edges_set

                    if sec_selected_edge == e:
                        col = [0.0, 1.0, 0.4]
                    elif sec_hovered_edge == e:
                        col = [1.0, 0.9, 0.0]
                    elif is_open:
                        col = [0.25, 0.25, 0.35]
                    else:
                        col = [0.08, 0.08, 0.1]

                    sec_line_data.extend([
                        float(va[0]), float(va[1]), float(va[2]), *col,
                        float(vb[0]), float(vb[1]), float(vb[2]), *col])
                sec_line_vert_count = upload_vbo(sec_line_vbo, sec_line_data, 6)
                sec_last_highlight_key = sec_highlight_key

        mx, my = mouse_pos
        hover_undo = (
            mx < main_w
            and BTN_UNDO_RECT[0] <= mx <= BTN_UNDO_RECT[2]
            and BTN_UNDO_RECT[1] <= my <= BTN_UNDO_RECT[3])
        hover_delete = (
            mx < main_w
            and BTN_DELETE_RECT[0] <= mx <= BTN_DELETE_RECT[2]
            and BTN_DELETE_RECT[1] <= my <= BTN_DELETE_RECT[3])
        can_delete = (
            selected_edge is not None and selected_edge in open_edges_set)
        ui_key = (hover_undo, hover_delete, can_delete, split_active)

        if ui_key != last_ui_key:
            ui_surf_data, ui_line_data = build_ui_buttons_data(
                main_w, height, BTN_UNDO_RECT, BTN_DELETE_RECT,
                hover_undo, hover_delete, can_delete)
            ui_surf_vert_count = upload_vbo(ui_surf_vbo, ui_surf_data, 9)
            ui_line_vert_count = upload_vbo(ui_line_vbo, ui_line_data, 6)
            last_ui_key = ui_key

        ctx.viewport = (0, 0, main_w, height)
        ctx.scissor = (0, 0, main_w, height)
        if bg_white:
            ctx.clear(1.0, 1.0, 1.0)
        else:
            ctx.clear(0.0, 0.0, 0.0)

        surf_prog["u_mvp"].write(mvp_3d.T.tobytes())
        surf_prog["u_rot"].write(rot_f32.T.tobytes())
        if face_vert_count > 0:
            face_surf_vao.render(moderngl.TRIANGLES, vertices=face_vert_count)
        if marker_vert_count > 0:
            marker_surf_vao.render(
                moderngl.TRIANGLES, vertices=marker_vert_count)

        if line_vert_count > 0:
            line_prog["u_mvp"].write(mvp_3d.T.tobytes())
            line_vao.render(moderngl.LINES, vertices=line_vert_count)

        ctx.disable(moderngl.DEPTH_TEST)
        surf_prog["u_mvp"].write(ident_mvp.tobytes())
        surf_prog["u_rot"].write(ident_rot.tobytes())
        if ui_surf_vert_count > 0:
            ui_surf_vao.render(moderngl.TRIANGLES, vertices=ui_surf_vert_count)

        if ui_line_vert_count > 0:
            line_prog["u_mvp"].write(ident_mvp.tobytes())
            ui_line_vao.render(moderngl.LINES, vertices=ui_line_vert_count)
        ctx.enable(moderngl.DEPTH_TEST)

        if split_active:
            ctx.viewport = (main_w, 0, sec_w, height)
            ctx.scissor = (main_w, 0, sec_w, height)
            ctx.clear(0.35, 0.35, 0.35)

            surf_prog["u_mvp"].write(sec_mvp_3d.T.tobytes())
            surf_prog["u_rot"].write(sec_rot_f32.T.tobytes())
            if sec_face_vert_count > 0:
                sec_face_surf_vao.render(
                    moderngl.TRIANGLES, vertices=sec_face_vert_count)
            if sec_marker_vert_count > 0:
                sec_marker_surf_vao.render(
                    moderngl.TRIANGLES, vertices=sec_marker_vert_count)

            if sec_line_vert_count > 0:
                line_prog["u_mvp"].write(sec_mvp_3d.T.tobytes())
                sec_line_vao.render(
                    moderngl.LINES, vertices=sec_line_vert_count)

        ctx.scissor = None
        ctx.viewport = (0, 0, width, height)
        ctx.disable(moderngl.DEPTH_TEST)
        if split_active:
            line_prog["u_mvp"].write(ident_mvp.tobytes())
            divider_vao.render(moderngl.LINES, vertices=2)

        if show_help:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            help_tex.use(location=0)
            help_prog["u_tex"].value = 0
            help_vao.render(moderngl.TRIANGLES)
            ctx.disable(moderngl.BLEND)

        ctx.enable(moderngl.DEPTH_TEST)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
