"""Serve DuckDB's local UI with the game catalog attached read-only."""

import argparse
import time
from pathlib import Path

import duckdb

from src.gamerec.catalog import is_duckdb
from src.gamerec.config import settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=settings.get_db_path())
    parser.add_argument(
        "--workspace", type=Path, default=settings.DATA_DIR / "catalog-ui.duckdb"
    )
    args = parser.parse_args()
    catalog, workspace = args.catalog.resolve(), args.workspace.resolve()
    if not catalog.is_file() or not is_duckdb(catalog):
        raise ValueError("Choose an existing DuckDB catalog.")
    if catalog == workspace:
        raise ValueError("The UI workspace must be separate from the catalog.")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(workspace)) as connection:
        # ATTACH takes a SQL literal; escape quotes in Windows/user-supplied paths.
        path_literal = str(catalog).replace("'", "''")
        connection.execute(f"ATTACH '{path_literal}' AS catalog (READ_ONLY)")
        connection.execute("INSTALL ui")
        connection.execute("LOAD ui")
        connection.execute("CALL start_ui_server()")
        print("DuckDB browser: http://localhost:4213", flush=True)
        print(
            "Read-only tables: catalog.main; notebooks: separate UI workspace.",
            flush=True,
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            connection.execute("CALL stop_ui_server()")


if __name__ == "__main__":
    main()
