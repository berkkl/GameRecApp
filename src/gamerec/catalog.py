"""Read SQLite catalogs or immutable DuckDB snapshots through mapping rows."""

import hashlib
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb


def is_duckdb(path):
    return Path(path).suffix.lower() == ".duckdb"


def catalog_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DuckRows:
    def __init__(self, cursor):
        self.cursor = cursor
        self.names = [column[0] for column in cursor.description]

    def mapping(self, row):
        result = dict(zip(self.names, row))
        for name, value in result.items():
            if isinstance(value, datetime):
                result[name] = (
                    value.astimezone(timezone.utc).isoformat()
                    if value.tzinfo
                    else value.isoformat()
                )
            elif isinstance(value, date):
                result[name] = value.isoformat()
        return result

    def fetchone(self):
        row = self.cursor.fetchone()
        return self.mapping(row) if row is not None else None

    def fetchall(self):
        try:
            return [self.mapping(row) for row in self.cursor.fetchall()]
        finally:
            self.cursor.close()

    def __iter__(self):
        try:
            while rows := self.cursor.fetchmany(256):
                for row in rows:
                    yield self.mapping(row)
        finally:
            self.cursor.close()


class DuckCatalog:
    def __init__(self, path):
        self.connection = duckdb.connect(str(Path(path).resolve()), read_only=True)

    def execute(self, sql, parameters=()):
        # Independent cursors allow a review lookup while iterating games.
        cursor = self.connection.cursor()
        try:
            return DuckRows(cursor.execute(sql, parameters))
        except Exception:
            cursor.close()
            raise

    def close(self):
        self.connection.close()


def open_catalog(path):
    path = Path(path)
    if is_duckdb(path):
        return DuckCatalog(path)
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def has_catalog_table(connection, name):
    if isinstance(connection, DuckCatalog):
        row = connection.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema='main' AND table_name=?",
            (name,),
        ).fetchone()
    else:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
    return row is not None
