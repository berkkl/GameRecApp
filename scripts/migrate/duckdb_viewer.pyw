"""Windows file opener for DuckDB's local UI, with read-only catalog access."""

import argparse
import json
import logging
import os
import socket
import time
import uuid
import webbrowser
from pathlib import Path

import duckdb


def attach_catalog(connection, catalog):
    path = Path(catalog).resolve()
    if not path.is_file() or path.suffix.lower() != ".duckdb":
        raise ValueError("Select an existing .duckdb database.")
    escaped = str(path).replace("'", "''")
    connection.execute(f"ATTACH '{escaped}' AS catalog (READ_ONLY)")


def available_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    session = (
        Path(__file__).resolve().parents[2] / "data" / "duckdb-ui" / uuid.uuid4().hex
    )
    session.mkdir(parents=True)
    logging.basicConfig(
        filename=str(session / "viewer.log"), level=logging.INFO, encoding="utf-8"
    )
    try:
        with duckdb.connect(str(session / "workspace.duckdb")) as connection:
            attach_catalog(connection, args.catalog)
            connection.execute("INSTALL ui")
            connection.execute("LOAD ui")
            # Pre-attach a private notebook database under the UI's expected
            # alias, avoiding the globally shared notebook file's process lock.
            ui_store = str(session / "notebooks.duckdb").replace("'", "''")
            connection.execute(f"ATTACH '{ui_store}' AS _duckdb_ui")
            port = args.port if args.port is not None else available_port()
            if not 1 <= port <= 65535:
                raise ValueError("Invalid viewer port.")
            connection.execute(f"SET ui_local_port = {port}")
            connection.execute("CALL start_ui_server()")
            url = f"http://localhost:{port}"
            (session / "status.json").write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "url": url,
                        "catalog": str(args.catalog.resolve()),
                        "read_only": True,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            logging.info("Viewer running at %s", url)
            if not args.no_browser:
                webbrowser.open(url)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                connection.execute("CALL stop_ui_server()")
    except Exception as exc:
        logging.exception("DuckDB viewer failed")
        if args.no_browser:
            raise
        # pythonw has no console; show a useful local error when double-click fails.
        import ctypes

        ctypes.windll.user32.MessageBoxW(
            None, f"Could not open the database:\n{exc}", "DuckDB Local UI", 0x10
        )


if __name__ == "__main__":
    main()
