# About
These scripts were written by Jim McNeill using AI tools. I decided it would be
a good idea to add them to a git repo. These files come as is and without any
guaratees and I cannot promise that any problems will be solved any time soon.

# Using Repository from Github

After cloning the the repository the offviewer can be run from the directory through

`uv run offviewer [dir]`

This requires that you have uv installed, see [uv](https://docs.astral.sh/uv/getting-started/installation/)


# Using a Released Version

A released version can be installed by
`pip install offviewer`

It is wise to use a virtual Python environment, e.g. on Linux:
```
python -m venv path_to_venv
. path_to_venv
```

Then the script can be run by:
`offviewer [path-to-folder]`

There is also a script `offchecker`, which only works on MS Windows.

# The Off File Viewer
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
    - 'f' on the enlarged screen will cycle through the faces one by one.
    - 'v' does the same with the vertices
    - 'e' does some edge options
    - 'x' to return to the original view
- The grid view also has some functions if you right click.
- The offcheck.py file is also a standalone checker.  If you change the metrics that this is calculating they should flow through into offviewer automatically.
- The path and 'search' at the top of the grid screen don't work (yet) and will probably be removed
