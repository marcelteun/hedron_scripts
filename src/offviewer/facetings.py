# File: facetings.py
# Version: 1.21.0
# Addons required: numpy, pillow, tkinterdnd2
#
import os
import re
import shutil
import threading
import multiprocessing
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    HAS_TKDND = True
except ImportError:
    HAS_TKDND = False

try:
    from PIL import Image, ImageTk

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Import modularized components
from facetings_math import (
    read_off,
    write_off,
    get_symmetry_group,
    find_all_subgroups,
    filter_conjugacy_classes,
    classify_subgroup,
    find_all_planar_faces,
    find_all_regular_faces,
    group_into_orbits,
    solve_facetings,
    filter_duplicate_solutions,
    is_connected_polyhedron,
    orient_faces_outwards,
    normalize_face,
    load_cache,
    has_adjacent_coplanar_faces,
    COLOR_MAP,
    DEFAULT_COLOR,
)
from facetings_renderer import HeadlessRenderer, render_grid_view, HAS_OPENGL_LIBS


def write_colored_off(filepath, vertices, faces):
    """Writes an OFF file containing face colors based on face side counts."""
    filename = os.path.basename(filepath)
    with open(filepath, "w") as f:
        f.write("OFF\n")
        f.write(f"# File: {filename}\n")
        f.write(f"{len(vertices)} {len(faces)} 0\n")
        for v in vertices:
            f.write(f"{v[0]:.16f} {v[1]:.16f} {v[2]:.16f}\n")
        for face in faces:
            sides = len(face)
            color = COLOR_MAP.get(sides, DEFAULT_COLOR)
            face_str = " ".join(str(idx) for idx in face)
            f.write(
                f"{len(face)} {face_str} {color[0]:.4f} {color[1]:.4f} {color[2]:.4f} 1.0\n"
            )


def get_connected_components(faces):
    """Partitions a face list into separate connected polyhedral components."""
    if not faces:
        return []
    edge_to_faces = {}
    for f_idx, face in enumerate(faces):
        for i in range(len(face)):
            u, v = face[i], face[(i + 1) % len(face)]
            edge = frozenset([u, v])
            edge_to_faces.setdefault(edge, []).append(f_idx)

    num_faces = len(faces)
    adj = [[] for _ in range(num_faces)]
    for edge, f_indices in edge_to_faces.items():
        for i in range(len(f_indices)):
            for j in range(i + 1, len(f_indices)):
                u = f_indices[i]
                v = f_indices[j]
                adj[u].append(v)
                adj[v].append(u)

    visited = [False] * num_faces
    components = []
    for start_idx in range(num_faces):
        if not visited[start_idx]:
            comp_f_indices = []
            queue = [start_idx]
            visited[start_idx] = True
            while queue:
                curr = queue.pop(0)
                comp_f_indices.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append([faces[idx] for idx in comp_f_indices])
    return components


def is_valid_or_transitive_compound(faces, vertices, symmetries):
    """Allows single polyhedra and transitive/symmetric compounds, blocking trivial compounds."""
    components = get_connected_components(faces)
    if len(components) <= 1:
        return True

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

    comp_sets = [frozenset(normalize_face(f) for f in comp) for comp in components]
    C0 = comp_sets[0]
    mapped_to = [False] * len(comp_sets)
    mapped_to[0] = True

    for perm in vertex_permutations:
        mapped_C0 = set()
        for face in C0:
            mapped_face = tuple(perm[v] for v in face)
            mapped_C0.add(normalize_face(mapped_face))
        mapped_C0 = frozenset(mapped_C0)

        for idx, C_j in enumerate(comp_sets):
            if not mapped_to[idx] and mapped_C0 == C_j:
                mapped_to[idx] = True
                break

    return all(mapped_to)


def has_vertex_adjacent_coplanar_faces(vertices, faces_list):
    """Checks if there are any vertex-adjacent coplanar faces in the solution."""
    planes = []
    for face in faces_list:
        pts = vertices[face]
        normal = np.zeros(3)
        for i in range(len(face)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(face)]
            normal[0] += (p1[1] - p2[1]) * (p1[2] + p2[2])
            normal[1] += (p1[2] - p2[2]) * (p1[0] + p2[0])
            normal[2] += (p1[0] - p2[0]) * (p1[1] + p2[1])
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-8:
            normal /= norm_len
        else:
            normal = np.array([0.0, 0.0, 1.0])
        d = np.dot(normal, pts[0])
        planes.append((normal, d))

    n_faces = len(faces_list)
    for i in range(n_faces):
        for j in range(i + 1, n_faces):
            n1, d1 = planes[i]
            n2, d2 = planes[j]
            if abs(abs(np.dot(n1, n2)) - 1.0) < 1e-5:
                same_plane = False
                if np.dot(n1, n2) > 0:
                    same_plane = abs(d1 - d2) < 1e-4
                else:
                    same_plane = abs(d1 + d2) < 1e-4

                if same_plane:
                    set_i = set(faces_list[i])
                    set_j = set(faces_list[j])
                    if not set_i.isdisjoint(set_j):
                        return True
    return False


def is_noble_polyhedron(vertices, faces, actual_symmetries):
    """Verifies if the polyhedron is both face-transitive and vertex-transitive."""
    if not faces or not actual_symmetries:
        return False

    centroid = np.mean(vertices, axis=0)
    V = vertices - centroid
    perms = []
    for g in actual_symmetries:
        perm = []
        for v in V:
            gv = g @ v
            idx = np.argmin(np.linalg.norm(V - gv, axis=1))
            perm.append(idx)
        perms.append(perm)

    norm_faces = [normalize_face(f) for f in faces]
    face_set = set(norm_faces)

    f0 = norm_faces[0]
    orbit0 = set()
    for perm in perms:
        mapped_f0 = normalize_face(tuple(perm[v] for v in f0))
        if mapped_f0 in face_set:
            orbit0.add(mapped_f0)

    if len(orbit0) != len(face_set):
        return False

    used_verts = set()
    for f in faces:
        used_verts.update(f)
    if not used_verts:
        return False

    v0 = list(used_verts)[0]
    v_orbit = set()
    for perm in perms:
        mapped_v = perm[v0]
        if mapped_v in used_verts:
            v_orbit.add(mapped_v)

    return len(v_orbit) == len(used_verts)


class FacetingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Polyhedron Faceting Generator")
        self.root.geometry("1050x620")
        self.root.minsize(1000, 550)

        # Variables
        self.off_filepath = tk.StringVar()
        self.output_directory = tk.StringVar(
            value=os.path.join(os.getcwd(), "facetings")
        )
        self.search_mode = tk.StringVar(value="Faceting")
        self.must_use_all = tk.BooleanVar(value=True)
        self.filter_compounds = tk.BooleanVar(value=True)
        self.clear_output_dir = tk.BooleanVar(value=True)
        self.filter_coplanar = tk.BooleanVar(value=True)
        self.filter_vertex_coplanar = tk.BooleanVar(value=False)
        self.strict_symmetry = tk.BooleanVar(value=False)
        self.regular_only = tk.BooleanVar(value=False)

        # Load symmetry cache prior to any generation
        load_cache()
        self.subgroups_dict = {"Full Group (default)": [np.eye(3)]}
        self.symmetry_option = tk.StringVar(value="Full Group (default)")
        self.viewer_process = None

        # Interactive preview variables
        self.preview_angle_x = 0.5
        self.preview_angle_y = 0.5
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.preview_vertices = None
        self.preview_faces = None

        # Setup headless renderer
        if HAS_PIL and HAS_OPENGL_LIBS:
            self.headless_renderer = HeadlessRenderer(400, 400)
        else:
            self.headless_renderer = None

        self.setup_ui()

        # Set up drag and drop if available
        if HAS_TKDND:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<Drop>>", self.handle_drop)

    def setup_ui(self):
        left_frame = tk.Frame(self.root)
        left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        right_frame = tk.LabelFrame(
            self.root, text="Input Polyhedron Preview (Drag to Rotate)", padx=5, pady=5
        )
        right_frame.pack(side="right", fill="both", expand=False, padx=10, pady=10)

        self.preview_label = tk.Label(right_frame, width=400, height=400, bg="#1e1e24")
        self.preview_label.pack(fill="both", expand=True)

        if not HAS_PIL or not HAS_OPENGL_LIBS or self.headless_renderer is None:
            self.preview_label.config(
                text="3D Preview Unavailable.\nPlease ensure pillow, moderngl, and pygame are installed.",
                fg="#888888",
                font=("Helvetica", 10),
            )
        else:
            self.preview_label.bind("<ButtonPress-1>", self.on_drag_start)
            self.preview_label.bind("<B1-Motion>", self.on_drag_motion)

        input_frame = tk.LabelFrame(
            left_frame, text="Files & Paths (Drag & Drop Supported)", padx=10, pady=10
        )
        input_frame.pack(fill="x", pady=5)

        tk.Label(input_frame, text="Input OFF File:").grid(
            row=0, column=0, sticky="w", pady=5
        )
        tk.Entry(input_frame, textvariable=self.off_filepath, width=45).grid(
            row=0, column=1, padx=5, pady=5
        )
        tk.Button(input_frame, text="Browse...", command=self.browse_off_file).grid(
            row=0, column=2, padx=5, pady=5
        )

        tk.Label(input_frame, text="Output Folder:").grid(
            row=1, column=0, sticky="w", pady=5
        )
        tk.Entry(input_frame, textvariable=self.output_directory, width=45).grid(
            row=1, column=1, padx=5, pady=5
        )
        tk.Button(input_frame, text="Browse...", command=self.browse_output_dir).grid(
            row=1, column=2, padx=5, pady=5
        )

        options_frame = tk.LabelFrame(
            left_frame, text="Constraints & Parameters", padx=10, pady=10
        )
        options_frame.pack(fill="x", pady=5)

        tk.Label(options_frame, text="Search Mode:").grid(
            row=0, column=0, sticky="w", pady=5
        )
        self.mode_dropdown = ttk.Combobox(
            options_frame,
            textvariable=self.search_mode,
            values=["Faceting", "Noble"],
            state="readonly",
        )
        self.mode_dropdown.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.search_mode.trace_add("write", self.update_options_state)

        tk.Label(options_frame, text="Symmetry Group:").grid(
            row=1, column=0, sticky="w", pady=5
        )
        self.symmetry_dropdown = ttk.OptionMenu(
            options_frame,
            self.symmetry_option,
            "Full Group (default)",
            "Full Group (default)",
        )
        self.symmetry_dropdown.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        self.vertex_checkbox = tk.Checkbutton(
            options_frame,
            text="Must use all vertices of the original polyhedron",
            variable=self.must_use_all,
        )
        self.vertex_checkbox.grid(row=2, column=0, columnspan=2, sticky="w", pady=2)

        tk.Checkbutton(
            options_frame,
            text="Filter out trivial compounds (allow transitive compounds)",
            variable=self.filter_compounds,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=2)
        tk.Checkbutton(
            options_frame,
            text="Clear output folder before writing output",
            variable=self.clear_output_dir,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=2)
        tk.Checkbutton(
            options_frame,
            text="Filter out edge adjacent coplanar faces",
            variable=self.filter_coplanar,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=2)
        tk.Checkbutton(
            options_frame,
            text="Filter out vertex adjacent coplanar faces",
            variable=self.filter_vertex_coplanar,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=2)
        tk.Checkbutton(
            options_frame,
            text="Strict symmetry group matches only",
            variable=self.strict_symmetry,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=2)

        self.regular_checkbox = tk.Checkbutton(
            options_frame,
            text="Regular polygonal faces only",
            variable=self.regular_only,
        )
        self.regular_checkbox.grid(row=8, column=0, columnspan=2, sticky="w", pady=2)

        self.run_button = tk.Button(
            left_frame,
            text="Generate Facetings",
            command=self.start_generation_thread,
            bg="#2b5797",
            fg="white",
            font=("Helvetica", 10, "bold"),
            height=2,
        )
        self.run_button.pack(fill="x", pady=5)

        log_frame = tk.LabelFrame(left_frame, text="Activity Log", padx=10, pady=10)
        log_frame.pack(fill="both", expand=True, pady=5)

        self.log_text = ScrolledText(
            log_frame, height=10, state="disabled", wrap="word", font=("Consolas", 9)
        )
        self.log_text.pack(fill="both", expand=True)

    def update_options_state(self, *args):
        if self.search_mode.get() == "Noble":
            self.must_use_all.set(True)
            self.vertex_checkbox.config(state="disabled")
            self.regular_checkbox.config(state="disabled")
        else:
            self.vertex_checkbox.config(state="normal")
            self.regular_checkbox.config(state="normal")

    def log(self, message):
        self.root.after(0, lambda: self._write_to_log(message + "\n"))

    def log_raw(self, message):
        self.root.after(0, lambda: self._write_to_log(message))

    def _write_to_log(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def rebuild_symmetry_dropdown(self, subgroup_names):
        menu = self.symmetry_dropdown["menu"]
        menu.delete(0, "end")
        for name in subgroup_names:
            menu.add_command(label=name, command=tk._setit(self.symmetry_option, name))
        if subgroup_names:
            self.symmetry_option.set(subgroup_names[0])

    def load_file(self, filepath):
        if not filepath or not os.path.exists(filepath):
            return

        self.run_button.config(state="disabled")
        self.root.config(cursor="watch")
        self.log(f"Loading {os.path.basename(filepath)}...")

        threading.Thread(
            target=self._async_load_file, args=(filepath,), daemon=True
        ).start()

    def _async_load_file(self, filepath):
        try:
            vertices, orig_faces = read_off(filepath)
            self.preview_vertices = vertices
            self.preview_faces = orig_faces
            self.preview_angle_x = 0.5
            self.preview_angle_y = 0.5

            self.root.after(0, self.update_preview)
            self.log(
                f"Successfully loaded {os.path.basename(filepath)} for preview ({len(vertices)} vertices)."
            )

            self.log("Detecting full symmetry group of the vertex coordinates...")
            full_symmetries = get_symmetry_group(vertices)
            self.log(f"Found full symmetry group of size: {len(full_symmetries)}")

            self.log("Determining all sub-symmetries...")
            raw_subgroups, table = find_all_subgroups(full_symmetries)

            self.log("Removing conjugate redundant sub-symmetries...")
            subgroups = filter_conjugacy_classes(raw_subgroups, full_symmetries, table)

            subgroups.sort(key=len, reverse=True)

            base_names = []
            name_counts = {}
            for sg in subgroups:
                size = len(sg)
                if size == len(full_symmetries):
                    name = f"Full Group: {classify_subgroup(sg, full_symmetries)} (size {size})"
                elif size == 1:
                    name = f"Trivial Group: {classify_subgroup(sg, full_symmetries)} (size 1)"
                else:
                    name = f"{classify_subgroup(sg, full_symmetries)} (size {size})"
                base_names.append(name)
                name_counts[name] = name_counts.get(name, 0) + 1

            seen_counts = {}
            self.subgroups_dict = {}
            subgroup_names = []

            for idx, sg in enumerate(subgroups):
                base_name = base_names[idx]
                sg_matrices = [full_symmetries[i] for i in sg]

                if name_counts[base_name] > 1:
                    seen_counts[base_name] = seen_counts.get(base_name, 0) + 1
                    unique_name = f"{base_name} #{seen_counts[base_name]}"
                else:
                    unique_name = base_name

                self.subgroups_dict[unique_name] = sg_matrices
                subgroup_names.append(unique_name)

            self.root.after(0, lambda: self.rebuild_symmetry_dropdown(subgroup_names))
            self.log(f"Identified {len(subgroups)} non-conjugate symmetry groups.")

        except Exception as e:
            self.log(f"Error loading preview for {os.path.basename(filepath)}: {e}")

        finally:
            self.root.after(0, lambda: self.root.config(cursor=""))
            self.root.after(0, lambda: self.run_button.config(state="normal"))

    def update_preview(self):
        if (
            self.headless_renderer is None
            or self.preview_vertices is None
            or self.preview_faces is None
        ):
            return
        img = self.headless_renderer.render(
            self.preview_vertices,
            self.preview_faces,
            self.preview_angle_x,
            self.preview_angle_y,
            is_noble=False,
        )
        if img:
            self.preview_photo = ImageTk.PhotoImage(img)
            self.preview_label.config(image=self.preview_photo)

    def on_drag_start(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_drag_motion(self, event):
        if self.preview_vertices is None or self.preview_faces is None:
            return
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.drag_start_x = event.x
        self.drag_start_y = event.y

        self.preview_angle_y += dx * 0.01
        self.preview_angle_x += dy * 0.01
        self.update_preview()

    def browse_off_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("OFF Files", "*.off"), ("All Files", "*.*")]
        )
        if filename:
            self.off_filepath.set(filename)
            self.load_file(filename)

    def browse_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_directory.set(directory)

    def handle_drop(self, event):
        path = event.data.strip()
        if path.startswith("{") and path.endswith("}"):
            path = path[1:-1]
        elif path.startswith('"') and path.endswith('"'):
            path = path[1:-1]

        if os.path.exists(path) and path.lower().endswith(".off"):
            self.off_filepath.set(path)
            self.load_file(path)
        else:
            self.log("Dropped file is not a valid .off file.")

    def start_generation_thread(self):
        if self.viewer_process is not None and self.viewer_process.is_alive():
            try:
                self.viewer_process.terminate()
                self.viewer_process.join()
            except Exception:  # noqa: BLE001
                pass

        thread = threading.Thread(target=self.run_generation)
        thread.daemon = True
        thread.start()

    # File: facetings.py
    # Version: 1.23.0
    # Addons required: numpy, pillow, tkinterdnd2
    #
    def run_generation(self):
        self.root.after(0, lambda: self.log_text.config(state="normal"))
        self.root.after(0, lambda: self.log_text.delete("1.0", tk.END))
        self.root.after(0, lambda: self.log_text.config(state="disabled"))

        filepath = self.off_filepath.get()
        out_dir = self.output_directory.get()
        must_all = self.must_use_all.get()
        filter_comp = self.filter_compounds.get()
        clear_out = self.clear_output_dir.get()
        sym_name = self.symmetry_option.get()
        mode = self.search_mode.get()
        reg_only = self.regular_only.get()

        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("Error", "Please select a valid input OFF file.")
            return

        self.run_button.config(state="disabled")

        try:
            self.log(f"Reading: {os.path.basename(filepath)}...")
            vertices, orig_faces = read_off(filepath)
            self.log(f"Polyhedron loaded with {len(vertices)} vertices.")

            self.log("Detecting full symmetry group of the vertex coordinates...")
            full_symmetries = get_symmetry_group(vertices)
            self.log(f"Found full symmetry group of size: {len(full_symmetries)}")

            if hasattr(self, "subgroups_dict") and sym_name in self.subgroups_dict:
                symmetries_for_generator = self.subgroups_dict[sym_name]
                self.log(f"Using symmetry group '{sym_name}' for generator orbits.")
            else:
                symmetries_for_generator = full_symmetries
                self.log("Applying full symmetry group for generator orbits.")

            max_input_k = max(len(face) for face in orig_faces) if orig_faces else 3
            max_k = max(10, max_input_k)

            if mode == "Noble":
                self.log("Searching for planar coplanar face candidates...")
                candidate_faces = find_all_planar_faces(
                    vertices, symmetries_for_generator, log_callback=self.log
                )
            else:
                if reg_only:
                    self.log(
                        f"Largest face in input has {max_input_k} vertices. Limit 'k' set to: {max_k}"
                    )
                    self.log("Searching for regular/star faces...")
                    candidate_faces = find_all_regular_faces(
                        vertices,
                        symmetries_for_generator,
                        max_k=max_k,
                        log_callback=self.log,
                    )
                else:
                    self.log(
                        "Searching for all planar coplanar face candidates (including non-regular)..."
                    )
                    candidate_faces = find_all_planar_faces(
                        vertices, symmetries_for_generator, log_callback=self.log
                    )

            self.log(f"Found {len(candidate_faces)} candidate face(s).")

            if not candidate_faces:
                self.log("Process complete: No valid faces detected.")
                self.run_button.config(state="normal")
                return

            self.log("Grouping candidate faces into orbits...")
            orbits = group_into_orbits(
                candidate_faces, vertices, symmetries_for_generator
            )
            self.log(f"Grouped into {len(orbits)} unique orbit(s).")

            if mode == "Noble":
                self.log("Running Noble Solver (filtering single orbits)...")
                noble_solutions = []
                for o_idx, orbit in enumerate(orbits):
                    edge_counts = {}
                    for face in orbit:
                        for i in range(len(face)):
                            u, v = face[i], face[(i + 1) % len(face)]
                            edge = frozenset([u, v])
                            edge_counts[edge] = edge_counts.get(edge, 0) + 1

                    if all(count == 2 for count in edge_counts.values()):
                        used_verts = set()
                        for face in orbit:
                            used_verts.update(face)
                        if len(used_verts) == len(vertices):
                            noble_solutions.append(orbit)
                raw_solutions = noble_solutions
                self.log(
                    f"Noble filtering identified {len(raw_solutions)} candidate noble orbit(s)."
                )
            else:
                self.log("Running backtracking solver...")
                raw_solutions = solve_facetings(
                    orbits,
                    candidate_faces,
                    len(vertices),
                    must_all,
                    progress_callback=lambda: self.log_raw("."),
                )
                self.log("")
                self.log(f"Solver generated {len(raw_solutions)} raw solution(s).")

            self.log(
                "Filtering out duplicates, reflections, and rotational equivalents..."
            )
            unique_solutions = filter_duplicate_solutions(
                raw_solutions, vertices, full_symmetries
            )

            if filter_comp:
                self.log(
                    "Filtering out trivial compounds (allowing transitive compounds)..."
                )
                pre_count = len(unique_solutions)
                unique_solutions = [
                    sol
                    for sol in unique_solutions
                    if is_valid_or_transitive_compound(sol, vertices, full_symmetries)
                ]
                self.log(
                    f"Removed {pre_count - len(unique_solutions)} trivial compound(s)."
                )

            if self.filter_coplanar.get():
                self.log("Filtering out results with edge adjacent coplanar faces...")
                pre_count = len(unique_solutions)
                unique_solutions = [
                    sol
                    for sol in unique_solutions
                    if not has_adjacent_coplanar_faces(vertices, sol)
                ]
                self.log(
                    f"Removed {pre_count - len(unique_solutions)} solution(s) with edge adjacent coplanar faces."
                )

            if self.filter_vertex_coplanar.get():
                self.log("Filtering out results with vertex adjacent coplanar faces...")
                pre_count = len(unique_solutions)
                unique_solutions = [
                    sol
                    for sol in unique_solutions
                    if not has_vertex_adjacent_coplanar_faces(vertices, sol)
                ]
                self.log(
                    f"Removed {pre_count - len(unique_solutions)} solution(s) with vertex adjacent coplanar faces."
                )

            # Map vertices to local coordinates for full permutations
            centroid = np.mean(vertices, axis=0)
            V = vertices - centroid
            full_permutations = []
            for g in full_symmetries:
                perm = []
                for v in V:
                    gv = g @ v
                    idx = np.argmin(np.linalg.norm(V - gv, axis=1))
                    perm.append(idx)
                full_permutations.append(perm)

            if self.strict_symmetry.get():
                self.log(
                    "Filtering out solutions that have higher symmetry than selected..."
                )
                pre_count = len(unique_solutions)
                filtered_solutions = []
                for sol in unique_solutions:
                    sol_set = {normalize_face(f) for f in sol}
                    actual_sym_count = 0
                    for perm in full_permutations:
                        mapped_sol = {
                            normalize_face(tuple(perm[v] for v in f)) for f in sol
                        }
                        if mapped_sol == sol_set:
                            actual_sym_count += 1
                    if actual_sym_count == len(symmetries_for_generator):
                        filtered_solutions.append(sol)
                unique_solutions = filtered_solutions
                self.log(
                    f"Removed {pre_count - len(unique_solutions)} solution(s) with higher symmetry."
                )

            self.log(
                "Orients faces of solutions outwards relative to polyhedron centroid..."
            )
            oriented_solutions = []
            for sol in unique_solutions:
                oriented_solutions.append(orient_faces_outwards(vertices, sol))
            unique_solutions = oriented_solutions

            # Suppress only flat solutions where all vertices in any connected component of the compound are coplanar
            non_flat_solutions = []
            for sol in unique_solutions:
                components = get_connected_components(sol)
                has_flat_part = False
                for comp in components:
                    comp_verts = list({v for face in comp for v in face})
                    if len(comp_verts) < 4:
                        has_flat_part = True
                        break
                    else:
                        pts = vertices[comp_verts]
                        centered = pts - np.mean(pts, axis=0)
                        _, s, _ = np.linalg.svd(centered)
                        if s[2] < 1e-5:
                            has_flat_part = True
                            break
                if not has_flat_part:
                    non_flat_solutions.append(sol)
            unique_solutions = non_flat_solutions

            self.log(f"Found {len(unique_solutions)} unique non-congruent faceting(s).")

            if unique_solutions:
                if clear_out and os.path.exists(out_dir):
                    self.log("Clearing output directory...")
                    for filename in os.listdir(out_dir):
                        file_path = os.path.join(out_dir, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            self.log(f"Warning: Could not delete {filename}: {e}")

                os.makedirs(out_dir, exist_ok=True)

                # Collect only edge lengths actually present in input faces or candidate faces
                used_edges_dists = []
                for face in orig_faces:
                    n_f = len(face)
                    for i in range(n_f):
                        d = np.linalg.norm(
                            vertices[face[i]] - vertices[face[(i + 1) % n_f]]
                        )
                        used_edges_dists.append(d)
                for face in candidate_faces:
                    n_f = len(face)
                    for i in range(n_f):
                        d = np.linalg.norm(
                            vertices[face[i]] - vertices[face[(i + 1) % n_f]]
                        )
                        used_edges_dists.append(d)

                # Filter, sort, and identify unique distances
                unique_dists = []
                for d in sorted(used_edges_dists):
                    if d < 1e-5:
                        continue
                    if not unique_dists or abs(d - unique_dists[-1]) > 1e-4:
                        unique_dists.append(d)

                # Map each unique distance to a letter (a, b, c, ..., z, aa, ab, etc.)
                distance_to_letter = {}
                for idx, d in enumerate(unique_dists):
                    temp_idx = idx
                    letter = ""
                    while temp_idx >= 0:
                        letter = chr(ord("a") + (temp_idx % 26)) + letter
                        temp_idx = (temp_idx // 26) - 1
                    distance_to_letter[d] = letter

                def get_face_oriented_key(face, tol=1e-5):
                    cleaned_face = []
                    for v in face:
                        if (
                            not cleaned_face
                            or np.linalg.norm(vertices[v] - vertices[cleaned_face[-1]])
                            > 1e-4
                        ):
                            cleaned_face.append(v)
                    if (
                        len(cleaned_face) > 1
                        and np.linalg.norm(
                            vertices[cleaned_face[0]] - vertices[cleaned_face[-1]]
                        )
                        <= 1e-4
                    ):
                        cleaned_face.pop()

                    n_f = len(cleaned_face)
                    if n_f < 3:
                        return None

                    letters = []
                    for i in range(n_f):
                        p1 = vertices[cleaned_face[i]]
                        p2 = vertices[cleaned_face[(i + 1) % n_f]]
                        d = np.linalg.norm(p1 - p2)

                        matched_letter = "?"
                        if distance_to_letter:
                            closest_dist = min(
                                distance_to_letter.keys(), key=lambda x: abs(x - d)
                            )
                            matched_letter = distance_to_letter[closest_dist]
                        letters.append(matched_letter)

                    cosines = []
                    for i in range(n_f):
                        prev_p = vertices[cleaned_face[i - 1]]
                        curr_p = vertices[cleaned_face[i]]
                        next_p = vertices[cleaned_face[(i + 1) % n_f]]
                        v1 = curr_p - prev_p
                        v2 = next_p - curr_p
                        v1_len = np.linalg.norm(v1)
                        v2_len = np.linalg.norm(v2)
                        if v1_len > 1e-8 and v2_len > 1e-8:
                            cos_val = np.dot(v1, v2) / (v1_len * v2_len)
                        else:
                            cos_val = 1.0
                        cosines.append(round(cos_val, 4))

                    candidates = []
                    for shift in range(n_f):
                        shifted_letters = letters[shift:] + letters[:shift]
                        shifted_cosines = cosines[shift:] + cosines[:shift]
                        candidates.append(
                            (tuple(shifted_letters), tuple(shifted_cosines))
                        )

                    return min(candidates)

                def get_face_canonical_key(face, tol=1e-5):
                    opt1 = get_face_oriented_key(face, tol)
                    if not opt1:
                        return None
                    opt2 = get_face_oriented_key(face[::-1], tol)
                    return min(opt1, opt2)

                # Collect all unique unoriented face keys from solutions and original faces
                unique_keys = set()
                for sol in unique_solutions:
                    for face in sol:
                        k_key = get_face_canonical_key(face)
                        if k_key:
                            unique_keys.add(k_key)
                for face in orig_faces:
                    k_key = get_face_canonical_key(face)
                    if k_key:
                        unique_keys.add(k_key)

                # Sort keys deterministically
                sorted_keys = sorted(list(unique_keys))

                # Map each key to its final name (e.g. bbbbb, bbbbb2, etc.)
                key_to_name = {}
                base_counts = {}
                for k_key in sorted_keys:
                    base_str = "".join(k_key[0])
                    if base_str not in base_counts:
                        base_counts[base_str] = 1
                        name = base_str
                    else:
                        base_counts[base_str] += 1
                        name = f"{base_str}{base_counts[base_str]}"
                    key_to_name[k_key] = name

                def get_solution_filename(sol, extra_suffix="", tol=1e-5):
                    type_to_oriented = {}
                    for face in sol:
                        unoriented_key = get_face_canonical_key(face, tol)
                        if not unoriented_key:
                            continue

                        oriented_key = get_face_oriented_key(face, tol)
                        unoriented_name = key_to_name.get(unoriented_key, "unknown")

                        is_forward = oriented_key == unoriented_key
                        type_to_oriented.setdefault(unoriented_name, set()).add(
                            "F" if is_forward else "R"
                        )

                    face_types = []
                    for utype, oriented_set in type_to_oriented.items():
                        if len(oriented_set) > 1:
                            face_types.append(utype + "+R")
                        else:
                            face_types.append(utype)

                    base_name = "-".join(sorted(face_types))

                    components = get_connected_components(sol)
                    if len(components) > 1:
                        base_name += f"-C{len(components)}"

                    return base_name + extra_suffix + ".off"

                def sanitize_filename(name):
                    invalid_chars = '<>:"/\\|?*'
                    for c in invalid_chars:
                        name = name.replace(c, "_")
                    name_part, ext = os.path.splitext(name)
                    if len(name_part) > 120:
                        import hashlib

                        h = hashlib.md5(name_part.encode("utf-8")).hexdigest()[:8]
                        name_part = name_part[:110] + f"_{h}"
                    return name_part + ext

                written_filenames = set()

                for idx, sol in enumerate(unique_solutions):
                    has_eac = has_adjacent_coplanar_faces(vertices, sol)
                    has_vac = has_vertex_adjacent_coplanar_faces(vertices, sol)

                    # Calculate active symmetries for this specific solution to verify if it is noble
                    sol_set = {normalize_face(f) for f in sol}
                    actual_syms = []
                    for s_idx, perm in enumerate(full_permutations):
                        mapped_sol = {
                            normalize_face(tuple(perm[v] for v in f)) for f in sol
                        }
                        if mapped_sol == sol_set:
                            actual_syms.append(full_symmetries[s_idx])

                    is_noble = is_noble_polyhedron(vertices, sol, actual_syms)

                    # Find indices of actual_syms in full_symmetries to classify actual symmetry group
                    actual_indices = []
                    for s_idx, g in enumerate(full_symmetries):
                        for s in actual_syms:
                            if np.linalg.norm(g - s) < 1e-7:
                                actual_indices.append(s_idx)
                                break

                    actual_group_desc = classify_subgroup(
                        actual_indices, full_symmetries
                    )
                    match = re.search(r"\(([^)]+)\)", actual_group_desc)
                    if match:
                        actual_sym_short = match.group(1)
                    else:
                        actual_sym_short = actual_group_desc.replace(" ", "_")

                    # Build customized filename suffix using actual symmetry
                    extra_suffix = f"-{actual_sym_short}"
                    if has_eac:
                        extra_suffix += "-EAC"
                    if has_vac:
                        extra_suffix += "-VAC"
                    if is_noble:
                        extra_suffix += "-Noble"

                    base_filename = sanitize_filename(
                        get_solution_filename(sol, extra_suffix=extra_suffix)
                    )
                    if base_filename in written_filenames:
                        name, ext = os.path.splitext(base_filename)
                        ver_idx = 1
                        while f"{name}-ver{ver_idx}{ext}" in written_filenames:
                            ver_idx += 1
                        filename = f"{name}-ver{ver_idx}{ext}"
                    else:
                        filename = base_filename
                    written_filenames.add(filename)
                    out_path = os.path.join(out_dir, filename)
                    write_colored_off(out_path, vertices, sol)

                self.log(f"Exported all files to: '{out_dir}'")

                if HAS_OPENGL_LIBS:
                    self.log(
                        "Launching ModernGL 3D Grid view window in a separate process..."
                    )
                    try:
                        self.viewer_process = multiprocessing.Process(
                            target=render_grid_view,
                            args=(vertices, unique_solutions),
                            kwargs={"is_noble": (mode == "Noble")},
                        )
                        self.viewer_process.daemon = True
                        self.viewer_process.start()
                    except Exception as e:
                        self.log(f"Failed to spawn 3D view process: {e}")
                else:
                    self.log(
                        "Could not display 3D Grid. Please verify that 'pygame' and 'moderngl' are installed."
                    )

                self.log(
                    f"Success: Saved {len(unique_solutions)} unique facetings to folder."
                )
            else:
                self.log("No valid closed polyhedral combinations satisfied the rules.")
                messagebox.showwarning(
                    "No Results", "No facetings satisfied the structural constraints."
                )

        except Exception as e:
            self.log(f"Exception encountered: {str(e)}")
            messagebox.showerror("Execution Error", f"An error occurred: {str(e)}")

        self.run_button.config(state="normal")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if HAS_TKDND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = FacetingGUI(root)
    root.mainloop()
