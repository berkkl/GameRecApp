"""Enable a validated serving pointer through an atomic local config update."""

import argparse
import json
import os
import uuid
from pathlib import Path

from dotenv import set_key

from scripts.migrate.activate_catalog import validate_pair


def configure_bundle(pointer, env_file):
    pointer, env_file = Path(pointer).resolve(), Path(env_file)
    bundle = json.loads(pointer.read_text(encoding="utf-8"))
    report = validate_pair(bundle["catalog"], bundle["index"])
    if (
        report["model_revision"] != bundle["model_revision"]
        or report["catalog_sha256"] != bundle["catalog_sha256"]
    ):
        raise ValueError("Serving pointer identity does not match the validated pair.")
    before = env_file.read_bytes() if env_file.exists() else b""
    temporary = env_file.with_name(env_file.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(before)
        set_key(str(temporary), "ACTIVE_CATALOG_MANIFEST", pointer.as_posix())
        current = env_file.read_bytes() if env_file.exists() else b""
        if current != before:
            raise ValueError("Local config changed during bundle configuration.")
        os.replace(temporary, env_file)
    finally:
        if temporary.exists():
            temporary.unlink()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    configure_bundle(args.pointer, args.env_file)
    print("Configured ACTIVE_CATALOG_MANIFEST. Restart the API to load the bundle.")
