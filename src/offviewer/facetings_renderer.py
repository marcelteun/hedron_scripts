
# File: facetings_renderer.py
# Version: 1.20.0
# Addons required: numpy, moderngl, pygame, pillow
#
import numpy as np
import math

try:
    import pygame
    import moderngl
    HAS_OPENGL_LIBS = True
except ImportError:
    HAS_OPENGL_LIBS = False

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from math_utils import perspective_matrix, rotation_matrix, COLOR_MAP, DEFAULT_COLOR

# Base palette for the first 8 components
COMPOUND_COLORS_FLOAT = [
    [1.0, 0.0, 0.0],      # red
    [1.0, 1.0, 0.0],      # yellow
    [0.0, 1.0, 0.0],      # green
    [0.2, 0.3, 1.0],      # blue
    [1.0, 0.0, 1.0],      # magenta
    [0.0, 1.0, 1.0],      # cyan
    [1.0, 0.5, 0.0],      # orange
    [0.5, 0.0, 1.0],      # purple
]

def get_compound_color(c_idx, num_components):
    """Returns a distinct float color. Uses the base palette, or generates HSL colors if components > 8."""
    if num_components <= 8:
        return COMPOUND_COLORS_FLOAT[c_idx % len(COMPOUND_COLORS_FLOAT)]
    
    hue = c_idx / num_components
    h = hue * 6.0
    c = 0.85 * (1.0 - abs(2.0 * 0.6 - 1.0))
    x = c * (1.0 - abs(h % 2.0 - 1.0))
    m = 0.6 - c / 2.0
    if h < 1.0:   r, g, b = c, x, 0.0
    elif h < 2.0: r, g, b = x, c, 0.0
    elif h < 3.0: r, g, b = 0.0, c, x
    elif h < 4.0: r, g, b = 0.0, x, c
    elif h < 5.0: r, g, b = x, 0.0, c
    else:         r, g, b = c, 0.0, x
    return [r + m, g + m, b + m]

def get_face_component_indices(faces):
    """Partitions a face list into separate connected components and returns indices mapping."""
    if not faces:
        return [], 0
    edge_to_faces = {}
    for f_idx, face in enumerate(faces):
        for i in range(len(face)):
            u, v = face[i], face[(i+1)%len(face)]
            edge = frozenset([u, v])
            edge_to_faces.setdefault(edge, []).append(f_idx)
            
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
    comp_indices = [-1] * num_faces
    comp_count = 0
    for start_idx in range(num_faces):
        if not visited[start_idx]:
            queue = [start_idx]
            visited[start_idx] = True
            while queue:
                curr = queue.pop(0)
                comp_indices[curr] = comp_count
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            comp_count += 1
    return comp_indices, comp_count

class HeadlessRenderer:
    def __init__(self, width=400, height=400):
        self.width = width
        self.height = height
        self.ctx = None
        
        # GPU Buffer Cache Properties
        self.cache_key = None
        self.vbo_solid = None
        self.vao_solid = None
        self.vbo_lines = None
        self.vao_lines = None
        self.cached_num_solid = 0
        self.cached_num_lines = 0
        
        if HAS_OPENGL_LIBS and HAS_PIL:
            try:
                self.ctx = moderngl.create_standalone_context()
                self.fbo = self.ctx.framebuffer(
                    color_attachments=[self.ctx.texture((width, height), 3)],
                    depth_attachment=self.ctx.depth_renderbuffer((width, height))
                )
                self.setup_shaders()
            except Exception:
                self.ctx = None

    def setup_shaders(self):
        vertex_shader_solid = """
        #version 330
        in vec3 in_position;
        in vec3 in_normal;
        in vec3 in_color;
        uniform mat4 mvp;
        uniform mat4 mv;
        out vec3 v_normal_view;
        out vec3 v_color;
        void main() {
            gl_Position = mvp * vec4(in_position, 1.0);
            v_normal_view = mat3(mv) * in_normal;
            v_color = in_color;
        }
        """

        fragment_shader_solid = """
        #version 330
        in vec3 v_normal_view;
        in vec3 v_color;
        out vec4 f_color;
        void main() {
            vec3 normal = normalize(v_normal_view);
            vec3 light_dir_view = normalize(vec3(0.5, 0.5, 1.0));
            float diff = max(dot(normal, light_dir_view), 0.0) * 0.5 + 0.5;
            f_color = vec4(v_color * diff, 1.0);
        }
        """

        vertex_shader_lines = """
        #version 330
        in vec3 in_position;
        uniform mat4 mvp;
        void main() {
            gl_Position = mvp * vec4(in_position, 1.0);
        }
        """

        fragment_shader_lines = """
        #version 330
        out vec4 f_color;
        void main() {
            f_color = vec4(0.05, 0.05, 0.05, 1.0);
        }
        """

        self.prog_solid = self.ctx.program(vertex_shader=vertex_shader_solid, fragment_shader=fragment_shader_solid)
        self.prog_lines = self.ctx.program(vertex_shader=vertex_shader_lines, fragment_shader=fragment_shader_lines)

    def clear_gpu_cache(self):
        """Cleans and releases active GPU shader buffers."""
        if self.vbo_solid: self.vbo_solid.release()
        if self.vbo_lines: self.vbo_lines.release()
        if self.vao_solid: self.vao_solid.release()
        if self.vao_lines: self.vao_lines.release()
        self.vbo_solid = None
        self.vbo_lines = None
        self.vao_solid = None
        self.vao_lines = None
        self.cache_key = None

    def render(self, vertices, faces, angle_x, angle_y, colour_compounds_separately=True):
        if not self.ctx:
            return None

        # Build signature of unique mesh geometry properties
        current_key = (id(vertices), len(faces), colour_compounds_separately)

        self.fbo.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = '<='
        self.ctx.clear(0.12, 0.12, 0.14, 1.0)

        # Re-upload GPU data only if the mesh or coloring setup has changed
        if self.cache_key != current_key:
            self.clear_gpu_cache()

            centroid = np.mean(vertices, axis=0)
            shifted = vertices - centroid
            max_dist = np.max(np.linalg.norm(shifted, axis=1))
            scaled_vertices = shifted / (max_dist if max_dist > 1e-6 else 1.0) * 0.90

            tri_vertices = []
            tri_normals = []
            tri_colors = []
            line_vertices = []

            comp_indices, num_components = get_face_component_indices(faces)
            use_compound_coloring = colour_compounds_separately and (num_components > 1)

            for f_idx, face in enumerate(faces):
                pts = scaled_vertices[list(face)]
                face_centroid = np.mean(pts, axis=0)
                
                if use_compound_coloring:
                    c_idx = comp_indices[f_idx]
                    color = get_compound_color(c_idx, num_components)
                else:
                    sides = len(face)
                    raw_color = COLOR_MAP.get(sides, DEFAULT_COLOR)
                    color = [c / 255.0 for c in raw_color]

                if len(pts) >= 3:
                    norm = np.cross(pts[1] - pts[0], pts[2] - pts[0])
                    n_len = np.linalg.norm(norm)
                    norm = norm / n_len if n_len > 1e-6 else np.array([0.0, 0.0, 1.0])
                else:
                    norm = np.array([0.0, 0.0, 1.0])

                for i in range(len(face)):
                    p1 = pts[i]
                    p2 = pts[(i + 1) % len(face)]

                    tri_vertices.extend(face_centroid)
                    tri_vertices.extend(p1)
                    tri_vertices.extend(p2)
                    tri_normals.extend(norm)
                    tri_normals.extend(norm)
                    tri_normals.extend(norm)
                    tri_colors.extend(color)
                    tri_colors.extend(color)
                    tri_colors.extend(color)

                    line_vertices.extend(p1)
                    line_vertices.extend(p2)

            v_data = np.array(tri_vertices, dtype='float32')
            n_data = np.array(tri_normals, dtype='float32')
            c_data = np.array(tri_colors, dtype='float32')
            l_data = np.array(line_vertices, dtype='float32')

            vbo_array = np.zeros(len(v_data) // 3 * 9, dtype='float32')
            for i in range(len(v_data) // 3):
                vbo_array[9*i : 9*i+3] = v_data[3*i : 3*i+3]
                vbo_array[9*i+3 : 9*i+6] = n_data[3*i : 3*i+3]
                vbo_array[9*i+6 : 9*i+9] = c_data[3*i : 3*i+3]

            self.vbo_solid = self.ctx.buffer(vbo_array.tobytes())
            self.vao_solid = self.ctx.vertex_array(self.prog_solid, [(self.vbo_solid, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])

            self.vbo_lines = self.ctx.buffer(l_data.tobytes())
            self.vao_lines = self.ctx.simple_vertex_array(self.prog_lines, self.vbo_lines, 'in_position')
            
            self.cache_key = current_key
            self.cached_num_solid = len(v_data) // 3
            self.cached_num_lines = len(l_data) // 3

        aspect = self.width / self.height
        proj = perspective_matrix(45.0, aspect, 0.1, 10.0)
        view = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -3.0],
            [0, 0, 0, 1]
        ], dtype='float32')

        model = rotation_matrix(angle_x, angle_y)
        mvp = proj @ view @ model
        mv = view @ model

        # Update matrices in shaders & render (VBO binding is skipped)
        self.prog_solid['mvp'].write(mvp.T.copy().tobytes())
        self.prog_solid['mv'].write(mv.T.copy().tobytes())
        self.vao_solid.render(moderngl.TRIANGLES)

        self.prog_lines['mvp'].write(mvp.T.copy().tobytes())
        self.vao_lines.render(moderngl.LINES)

        raw_data = self.fbo.read(components=3, dtype='f1')
        img = Image.frombytes('RGB', (self.width, self.height), raw_data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        return img


def render_grid_view(vertices, solutions, tk_update_callback=None, control_flags=None, colour_compounds_separately=True):
    if not HAS_OPENGL_LIBS:
        print("Required 3D libraries (pygame, moderngl) are missing.")
        return

    centroid = np.mean(vertices, axis=0)
    shifted = vertices - centroid
    max_dist = np.max(np.linalg.norm(shifted, axis=1))
    scaled_vertices = shifted / (max_dist if max_dist > 1e-6 else 1.0) * 0.90

    pygame.init()
    screen_w, screen_h = 1024, 768
    pygame.display.set_mode((screen_w, screen_h), pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Polyhedron Facetings Grid View")

    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.depth_func = '<='
    ctx.disable(moderngl.CULL_FACE)

    vertex_shader_solid = """
    #version 330
    in vec3 in_position;
    in vec3 in_normal;
    in vec3 in_color;
    uniform mat4 mvp;
    uniform mat4 mv;
    out vec3 v_normal_view;
    out vec3 v_color;
    void main() {
        gl_Position = mvp * vec4(in_position, 1.0);
        v_normal_view = mat3(mv) * in_normal;
        v_color = in_color;
    }
    """

    fragment_shader_solid = """
    #version 330
    in vec3 v_normal_view;
    in vec3 v_color;
    out vec4 f_color;
    void main() {
        vec3 normal = normalize(v_normal_view);
        vec3 light_dir_view = normalize(vec3(0.5, 0.5, 1.0));
        float diff = max(dot(normal, light_dir_view), 0.0) * 0.5 + 0.5;
        f_color = vec4(v_color * diff, 1.0);
    }
    """

    vertex_shader_lines = """
    #version 330
    in vec3 in_position;
    uniform mat4 mvp;
    void main() {
        gl_Position = mvp * vec4(in_position, 1.0);
    }
    """

    fragment_shader_lines = """
    #version 330
    out vec4 f_color;
    void main() {
        f_color = vec4(0.05, 0.05, 0.05, 1.0);
    }
    """

    prog_solid = ctx.program(vertex_shader=vertex_shader_solid, fragment_shader=fragment_shader_solid)
    prog_lines = ctx.program(vertex_shader=vertex_shader_lines, fragment_shader=fragment_shader_lines)

    vaos = []
    for sol in solutions[:64]:
        tri_vertices = []
        tri_normals = []
        tri_colors = []
        line_vertices = []

        comp_indices, num_components = get_face_component_indices(sol)
        use_compound_coloring = colour_compounds_separately and (num_components > 1)

        for f_idx, face in enumerate(sol):
            pts = scaled_vertices[list(face)]
            face_centroid = np.mean(pts, axis=0)
            
            if use_compound_coloring:
                c_idx = comp_indices[f_idx]
                color = get_compound_color(c_idx, num_components)
            else:
                sides = len(face)
                raw_color = COLOR_MAP.get(sides, DEFAULT_COLOR)
                color = [c / 255.0 for c in raw_color]
            
            if len(pts) >= 3:
                norm = np.cross(pts[1] - pts[0], pts[2] - pts[0])
                n_len = np.linalg.norm(norm)
                norm = norm / n_len if n_len > 1e-6 else np.array([0.0, 0.0, 1.0])
            else:
                norm = np.array([0.0, 0.0, 1.0])
                
            for i in range(len(face)):
                p1 = pts[i]
                p2 = pts[(i + 1) % len(face)]
                
                tri_vertices.extend(face_centroid)
                tri_vertices.extend(p1)
                tri_vertices.extend(p2)
                tri_normals.extend(norm)
                tri_normals.extend(norm)
                tri_normals.extend(norm)
                tri_colors.extend(color)
                tri_colors.extend(color)
                tri_colors.extend(color)
                
                line_vertices.extend(p1)
                line_vertices.extend(p2)

        v_data = np.array(tri_vertices, dtype='float32')
        n_data = np.array(tri_normals, dtype='float32')
        c_data = np.array(tri_colors, dtype='float32')
        l_data = np.array(line_vertices, dtype='float32')
        
        vbo_array = np.zeros(len(v_data) // 3 * 9, dtype='float32')
        for i in range(len(v_data) // 3):
            vbo_array[9*i : 9*i+3] = v_data[3*i : 3*i+3]
            vbo_array[9*i+3 : 9*i+6] = n_data[3*i : 3*i+3]
            vbo_array[9*i+6 : 9*i+9] = c_data[3*i : 3*i+3]

        vbo_solid = ctx.buffer(vbo_array.tobytes())
        vao_solid = ctx.vertex_array(prog_solid, [(vbo_solid, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])
        
        vbo_lines = ctx.buffer(l_data.tobytes())
        vao_lines = ctx.simple_vertex_array(prog_lines, vbo_lines, 'in_position')
        
        vaos.append((vao_solid, len(v_data) // 3, vao_lines, len(l_data) // 3))

    num_sols = len(vaos)
    cols = int(math.ceil(math.sqrt(num_sols))) if num_sols > 0 else 1
    rows = int(math.ceil(num_sols / cols)) if num_sols > 0 else 1

    running = True
    clock = pygame.time.Clock()
    angle_x = 0.0
    angle_y = 0.0

    while running:
        if control_flags is not None and not control_flags.get('running', True):
            running = False
            break

        dt = clock.tick(60) / 1000.0
        angle_x += 0.4 * dt
        angle_y += 0.6 * dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        ctx.clear(0.12, 0.12, 0.14, 1.0)
        
        cell_w = screen_w // cols
        cell_h = screen_h // rows

        for idx, (vao_solid, num_solid, vao_lines, num_lines) in enumerate(vaos):
            r = idx // cols
            c = idx % cols
            
            vx = c * cell_w
            vy = (rows - 1 - r) * cell_h
            ctx.viewport = (vx, vy, cell_w, cell_h)

            aspect = cell_w / cell_h if cell_h > 0 else 1.0
            proj = perspective_matrix(45.0, aspect, 0.1, 10.0)
            
            view = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, -3.0],
                [0, 0, 0, 1]
            ], dtype='float32')

            model = rotation_matrix(angle_x, angle_y)
            mvp = proj @ view @ model
            mv = view @ model

            prog_solid['mvp'].write(mvp.T.copy().tobytes())
            prog_solid['mv'].write(mv.T.copy().tobytes())
            vao_solid.render(moderngl.TRIANGLES)
            
            prog_lines['mvp'].write(mvp.T.copy().tobytes())
            vao_lines.render(moderngl.LINES)

        pygame.display.flip()

        if tk_update_callback:
            try:
                tk_update_callback()
            except Exception:
                running = False

    pygame.quit()
    if control_flags is not None:
        control_flags['running'] = False

def render_grid_view(vertices, solutions, tk_update_callback=None, control_flags=None, is_noble=False, colour_compounds_separately=True):
    if not HAS_OPENGL_LIBS:
        print("Required 3D libraries (pygame, moderngl) are missing.")
        return

    centroid = np.mean(vertices, axis=0)
    shifted = vertices - centroid
    max_dist = np.max(np.linalg.norm(shifted, axis=1))
    scaled_vertices = shifted / (max_dist if max_dist > 1e-6 else 1.0) * 0.90

    pygame.init()
    screen_w, screen_h = 1024, 768
    pygame.display.set_mode((screen_w, screen_h), pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Polyhedron Facetings Grid View")

    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.depth_func = '<='
    ctx.disable(moderngl.CULL_FACE)

    vertex_shader_solid = """
    #version 330
    in vec3 in_position;
    in vec3 in_normal;
    in vec3 in_color;
    uniform mat4 mvp;
    uniform mat4 mv;
    out vec3 v_normal_view;
    out vec3 v_color;
    void main() {
        gl_Position = mvp * vec4(in_position, 1.0);
        v_normal_view = mat3(mv) * in_normal;
        v_color = in_color;
    }
    """

    fragment_shader_solid = """
    #version 330
    in vec3 v_normal_view;
    in vec3 v_color;
    out vec4 f_color;
    void main() {
        vec3 normal = normalize(v_normal_view);
        vec3 light_dir_view = normalize(vec3(0.5, 0.5, 1.0));
        float diff = max(dot(normal, light_dir_view), 0.0) * 0.5 + 0.5;
        f_color = vec4(v_color * diff, 1.0);
    }
    """

    vertex_shader_lines = """
    #version 330
    in vec3 in_position;
    uniform mat4 mvp;
    void main() {
        gl_Position = mvp * vec4(in_position, 1.0);
    }
    """

    fragment_shader_lines = """
    #version 330
    out vec4 f_color;
    void main() {
        f_color = vec4(0.05, 0.05, 0.05, 1.0);
    }
    """

    prog_solid = ctx.program(vertex_shader=vertex_shader_solid, fragment_shader=fragment_shader_solid)
    prog_lines = ctx.program(vertex_shader=vertex_shader_lines, fragment_shader=fragment_shader_lines)

    vaos = []
    for sol in solutions[:64]:
        tri_vertices = []
        tri_normals = []
        tri_colors = []
        line_vertices = []
        
        if is_noble:
            face_color_indices = compute_noble_face_colors(sol, len(vertices))
            max_c = max(face_color_indices) + 1 if face_color_indices else 1

        comp_indices, num_components = get_face_component_indices(sol)
        use_compound_coloring = colour_compounds_separately and (num_components > 1)

        for f_idx, face in enumerate(sol):
            pts = scaled_vertices[list(face)]
            face_centroid = np.mean(pts, axis=0)
            
            if use_compound_coloring:
                c_idx = comp_indices[f_idx]
                color = get_compound_color(c_idx, num_components)
            elif is_noble:
                color = get_palette_color(face_color_indices[f_idx], max_c)
            else:
                sides = len(face)
                raw_color = COLOR_MAP.get(sides, DEFAULT_COLOR)
                color = [c / 255.0 for c in raw_color]
            
            if len(pts) >= 3:
                norm = np.cross(pts[1] - pts[0], pts[2] - pts[0])
                n_len = np.linalg.norm(norm)
                norm = norm / n_len if n_len > 1e-6 else np.array([0.0, 0.0, 1.0])
            else:
                norm = np.array([0.0, 0.0, 1.0])
                
            for i in range(len(face)):
                p1 = pts[i]
                p2 = pts[(i + 1) % len(face)]
                
                tri_vertices.extend(face_centroid)
                tri_vertices.extend(p1)
                tri_vertices.extend(p2)
                tri_normals.extend(norm)
                tri_normals.extend(norm)
                tri_normals.extend(norm)
                tri_colors.extend(color)
                tri_colors.extend(color)
                tri_colors.extend(color)
                
                line_vertices.extend(p1)
                line_vertices.extend(p2)

        v_data = np.array(tri_vertices, dtype='float32')
        n_data = np.array(tri_normals, dtype='float32')
        c_data = np.array(tri_colors, dtype='float32')
        l_data = np.array(line_vertices, dtype='float32')
        
        vbo_array = np.zeros(len(v_data) // 3 * 9, dtype='float32')
        for i in range(len(v_data) // 3):
            vbo_array[9*i : 9*i+3] = v_data[3*i : 3*i+3]
            vbo_array[9*i+3 : 9*i+6] = n_data[3*i : 3*i+3]
            vbo_array[9*i+6 : 9*i+9] = c_data[3*i : 3*i+3]

        vbo_solid = ctx.buffer(vbo_array.tobytes())
        vao_solid = ctx.vertex_array(prog_solid, [(vbo_solid, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])
        
        vbo_lines = ctx.buffer(l_data.tobytes())
        vao_lines = ctx.simple_vertex_array(prog_lines, vbo_lines, 'in_position')
        
        vaos.append((vao_solid, len(v_data) // 3, vao_lines, len(l_data) // 3))

    num_sols = len(vaos)
    cols = int(math.ceil(math.sqrt(num_sols))) if num_sols > 0 else 1
    rows = int(math.ceil(num_sols / cols)) if num_sols > 0 else 1

    running = True
    clock = pygame.time.Clock()
    angle_x = 0.0
    angle_y = 0.0

    while running:
        if control_flags is not None and not control_flags.get('running', True):
            running = False
            break

        dt = clock.tick(60) / 1000.0
        angle_x += 0.4 * dt
        angle_y += 0.6 * dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        ctx.clear(0.12, 0.12, 0.14, 1.0)
        
        cell_w = screen_w // cols
        cell_h = screen_h // rows

        for idx, (vao_solid, num_solid, vao_lines, num_lines) in enumerate(vaos):
            r = idx // cols
            c = idx % cols
            
            vx = c * cell_w
            vy = (rows - 1 - r) * cell_h
            ctx.viewport = (vx, vy, cell_w, cell_h)

            aspect = cell_w / cell_h if cell_h > 0 else 1.0
            proj = perspective_matrix(45.0, aspect, 0.1, 10.0)
            
            view = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, -3.0],
                [0, 0, 0, 1]
            ], dtype='float32')

            model = rotation_matrix(angle_x, angle_y)
            mvp = proj @ view @ model
            mv = view @ model

            prog_solid['mvp'].write(mvp.T.copy().tobytes())
            prog_solid['mv'].write(mv.T.copy().tobytes())
            vao_solid.render(moderngl.TRIANGLES)
            
            prog_lines['mvp'].write(mvp.T.copy().tobytes())
            vao_lines.render(moderngl.LINES)

        pygame.display.flip()

        if tk_update_callback:
            try:
                tk_update_callback()
            except Exception:
                running = False

    pygame.quit()
    if control_flags is not None:
        control_flags['running'] = False