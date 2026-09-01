# About
These scripts were written by Jim McNeill using AI tools. I decided it would be
a good idea to add them to a git repo. These files come as is and without any
guaratees and I cannot promise that any problems will be solved any time soon.

## Using the Repository from GitHub

After cloning the repository, install [uv](https://docs.astral.sh/uv/getting-started/installation/)
and run a command with:

```sh
uv run <command> [arguments]
```

## Using a Released Version

A released version can be installed with:

```sh
pip install hedron_scripts
```

It is wise to use a virtual Python environment, e.g:

```
python -m venv path_to_venv
. path_to_venv/bin/activate
```

## Command-Line Tools

### `offviewer [path-to-folder-or-file.off]`

Opens an interactive 3D browser for the OFF files in a folder. Pass a folder to
open its grid view, or an OFF file to open that file in the enlarged view. The
viewer calculates model metrics in the background; those metrics can be used
for sorting and for identifying duplicate values. Until this calculation
finishes, sorting is unavailable. Equal sort values are highlighted in pink;
**Delete duplicates** keeps the oldest file for each value, using a tolerance
for floating-point metrics.

```sh
uv run offviewer models/
offviewer path/to/model.off
```

#### Using the viewer

- In the grid view, use `+`/`=` and `-` to change cell size, and left-drag or
  use the arrow keys to rotate a model. Avoid auto-rotation until metric
  analysis has finished, as it slows the calculation.
- Double-click a model to open the enlarged view. Use the left and right arrow
  keys to navigate between files; `f`, `v`, `e`, and `c` cycle face, vertex,
  edge, and compound-part displays respectively; `x` returns to the grid.
- Right-click a model for file actions, including rename, delete, viewing its
  source, and any configured external tools.

### `offchecker [path-to-file.off]`

Opens the OFF checker GUI. Passing an OFF file processes it immediately;
otherwise use the GUI to choose a file. The checker reports geometric and
topological properties, including faces, edges, genus, planarity, and
symmetry. Its batch mode writes `offcheck.csv` in the selected file's folder.

```sh
uv run offchecker path/to/model.off
offchecker
```

### `facetings`

Opens the Facetings GUI for generating valid faceted polyhedra from an input
OFF file. Choose the input file in the application, configure the search and
symmetry options, then export the generated OFF files. A 3D grid preview is
available when the required OpenGL libraries are installed.

```sh
uv run facetings
facetings
```

### `leonardo [path-to-file.off-or-vrml]`

Opens the Leonardo-style frame-mesh viewer. Pass an OFF file or a supported
VRML file, or drag one onto the application window. Adjust frame width and
depth in the control panel; **Export OFF** writes a `*_leonardo.off` file
alongside the source model.

```sh
uv run leonardo path/to/model.off
leonardo path/to/model.wrl
```

## License

This package is published under the GNU Public License version 2; see
[LICENSE.txt](LICENSE.txt).
