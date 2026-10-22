"""Import a Steam CSV through the same partial-upsert ingestion path."""

import argparse
from pathlib import Path

from src.gamerec.db import init_db

from . import load_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    load_csv.CSV_PATH = args.input
    init_db()
    load_csv.load_csv()


if __name__ == "__main__":
    main()
