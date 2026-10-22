"""Reject private workspace files from the tracked public source tree."""

import re
import subprocess
from pathlib import Path

ALLOWED_MARKDOWN = {
    "README.md",
    "DEVELOPER_GUIDE.md",
    "DEPLOYMENT_GCP.md",
    "DESIGN.md",
    "docs/RECOMMENDATION_ARCHITECTURE.md",
    "docs/STEAM_REFRESH.md",
}
PRIVATE_PARTS = {
    "handoffs",
    ".agents",
    ".agent",
    ".codex",
    ".claude",
    ".gemini",
    ".cursor",
    ".firecrawl",
    ".firebase",
    "secrets",
    "node_modules",
    ".venv",
}
PRIVATE_SUFFIXES = {
    ".db",
    ".duckdb",
    ".sqlite",
    ".sqlite3",
    ".npy",
    ".npz",
    ".index",
    ".ids",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".log",
    ".jsonl",
    ".har",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".zip",
    ".gz",
    ".tar",
}
# This public, upstream sentiment lexicon is an application dependency.
ALLOWED_BINARY = {"nltk_data/sentiment/vader_lexicon.zip"}
PRIVATE_TEXT = re.compile(
    r"/"
    r"Users/[^/\s]+/|"
    r"/"
    r"home/(?!runner/|user/)[^/\s]+/|"
    r"[A-Za-z]:[\\/]Users[\\/][^\\/\s]+[\\/]"
)


def main():
    root = Path(__file__).resolve().parents[1]
    names = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    failures = []
    for name in filter(None, names):
        path = Path(name)
        if name in ALLOWED_BINARY:
            continue
        blocked = (
            bool(set(path.parts) & PRIVATE_PARTS)
            or (path.suffix.lower() == ".md" and name not in ALLOWED_MARKDOWN)
            or path.suffix.lower() in PRIVATE_SUFFIXES
            or (path.name.startswith(".env") and name != ".env.example")
            or path.name.endswith(".env")
            or path.name.startswith(
                ("id_rsa", "id_ed25519", "credentials", "service-account")
            )
            or name in {".firebaserc", "public/benchmark.html"}
            or (
                name.startswith("data/")
                and name not in {"data/sample.json", "data/.gitkeep"}
            )
        )
        if blocked:
            failures.append(f"{name}: private or unreviewed file category")
            continue
        content = root / name
        if content.is_symlink():
            failures.append(f"{name}: symlink needs explicit review")
            continue
        try:
            value = content.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"{name}: unreviewed binary")
            continue
        if PRIVATE_TEXT.search(value):
            failures.append(f"{name}: personal filesystem path")
    if failures:
        print("Public file check failed (values redacted):")
        print("\n".join(failures))
        return 1
    print(f"Public file check passed: {len(list(filter(None, names)))} tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
