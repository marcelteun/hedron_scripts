import argparse

from hedron_scripts.lib.nobles import find_and_save_noble_polyhedra


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
    args = parser.parse_args()
    find_and_save_noble_polyhedra(args.output)


if __name__ == "__main__":
    main()
