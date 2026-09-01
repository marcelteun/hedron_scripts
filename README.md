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
for sorting and for identifying duplicate values.

```sh
uv run offviewer models/
offviewer path/to/model.off
```

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

## The Off File Viewer
The off file viewer is called `offviewer`. It works as a file browser and shows
the 3D interactive images for files living in a folder. It is also possible to
get some properties for the polyhedra modelled by the OFF file.

When you open a grid window the sort mode will be greyed out while it
calculates the metrics for the off files.  Once it has done this and you select
a sort order, duplicate values of the sort parameter are highlighted in pink.
The delete duplicates button will keep the oldest file with any one value.
Float parameters have a small tolerance built in.

- Auto rotate is fun but wait until the sort button blacks up or you'll slow it down.
- + (or =) and - will zoom the cells on the grid screen-
- Any polyhedron can be rotated by left dragging with the mouse ot=r by using the arrow keys
- Double click on an image to get to the enlarged screen.
    - The left and right arrows are previous/next file.
    - 'f' on the enlarged screen will cycle through the face types one by one.
    - 'v' cycles through the vertices
    - 'e' does some edge options (also on the grid view)
    - 'x' to return to the original view
- The grid view also has some functions if you right click: delete, rename, view source, open in Stella.
- The offcheck.py file is also a standalone checker.  If you change the metrics that this is calculating they should flow through into offviewer automatically.
- The path and 'search' at the top of the grid screen don't work (yet) and will probably be removed

## License

This package is published under the GNU Public License version 2, see the LICENSE.txt file.
