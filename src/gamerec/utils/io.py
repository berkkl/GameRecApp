import json
from typing import Any, Dict


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
