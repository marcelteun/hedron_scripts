# Hedron Scripts contributor guidance

## Commands

This is a Python 3.12+ package managed with `uv`. Install the locked runtime and development dependencies with:

```sh
uv sync
```

Use the installed console scripts during development:

```sh
uv run offviewer [directory]
uv run offchecker [path-to-file.off]
uv run facetings
uv run leonardo [path-to-file.off-or-vrml]
```

Run linting and formatting checks with Ruff:

```sh
uv run ruff check .
uv run ruff format --check .
```

Run either command on a single changed module by passing its path, for example:

```sh
uv run ruff check src/hedron_scripts/offviewer.py
uv run ruff format --check src/hedron_scripts/offviewer.py
```

Build the distributable package with:

```sh
uv build
```

There is no automated test suite. Exercise GUI changes through the relevant console script using representative `.off` input files.

## Architecture

- `pyproject.toml` defines four console scripts: `offviewer`, `offchecker`, `facetings`, and `leonardo`. Their same-named modules directly under `src/hedron_scripts/` are deliberately thin entry-point wrappers; each imports and invokes `main()` from `hedron_scripts.lib`.
- `src/hedron_scripts/lib/` contains the externally delivered applications. `offviewer` is a Pygame/ModernGL folder browser and renderer; it runs `offcheck.verify_off_logic()` in the background to populate sortable model metrics. `offcheck` provides both the Tk GUI and the reusable OFF validation/metrics function.
- `facetings` is a Tk GUI for generating faceted polyhedra. It uses `math_utils` for OFF parsing, symmetry groups, face search, and output geometry, and `facetings_renderer` for previews and a separate-process OpenGL grid. `math_utils` also owns the Numba-compiled geometry routines and the subgroup cache under `lib/__pycache__/`.
- `leonardo` is an independent GLFW/ImGui/ModernGL viewer that imports OFF or supported VRML geometry, constructs a Leonardo-style frame mesh, and exports an OFF file beside the source.
- `lib/external_tools.json` supplies Offviewer’s right-click integrations. It is JSON data using Windows-oriented example paths; keep its expected `prompt`, `program`, and optional `args` structure when editing it.

## Repository conventions

- Treat modules in `src/hedron_scripts/lib/` as an upstream mirror: keep changes there minimal and localized to reduce conflicts when new deliveries are integrated. Put packaging, entry-point, and repository-integration work in the top-level package modules or root configuration where possible.
- The `lib` modules intentionally support two execution modes. Preserve package imports such as `from hedron_scripts.lib import math_utils` and their `ModuleNotFoundError` standalone fallbacks; upstream code is also run directly outside this package.
- OFF files may contain comments, inline or separate `OFF` headers, polygon faces, and optional face colors. Reuse `math_utils.read_off()`/`write_off()` for Facetings workflows rather than adding a competing parser; preserve vertex welding and face-index cleanup behavior.
- Geometry calculations use NumPy arrays and numerical tolerances. Keep existing tolerance constants and the face-normalization/orientation flow consistent across symmetry, faceting, and rendering paths; seemingly equivalent changes can alter duplicate detection or group classification.
- GUI applications manage optional platform dependencies explicitly: Windows-only drag-and-drop uses `windnd`, Facetings degrades when TkinterDnD/Pillow/OpenGL are unavailable, and `facetings_renderer.HAS_OPENGL_LIBS` gates 3D previews. Preserve those capability checks instead of making imports unconditional.
