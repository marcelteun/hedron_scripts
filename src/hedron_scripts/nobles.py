import argparse

from hedron_scripts.lib import nobles as nobles_lib


def main():
    parser = argparse.ArgumentParser(
        description="Search for and export noble polyhedra as OFF files."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="noble",
        help="Directory in which to save generated OFF files (default: noble)",
    )
    parser.add_argument(
        "-s",
        "--symmetry",
        choices=("I", "Ih", "I3", "I3h", "O", "Oh", "T", "Td", "Th"),
        default=nobles_lib.SYMMETRY_GROUP,
        help="Symmetry group to search (default: %(default)s)",
    )
    args = parser.parse_args()
    nobles_lib.find_and_save_noble_polyhedra(
        output_dir=args.output, symmetry_group=args.symmetry
    )


if __name__ == "__main__":
    main()
