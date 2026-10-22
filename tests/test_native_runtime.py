"""Exercise FAISS and PyTorch together in a fresh native-runtime process."""

import subprocess
import sys
from pathlib import Path


def test_faiss_search_followed_by_torch_inference(tmp_path):
    script = """
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from src.gamerec.search.faiss_store import FAISSStore
import numpy as np
import torch
import sys

store = FAISSStore(Path(sys.argv[1]) / 'games.index', Path(sys.argv[1]) / 'games.ids')
store.add(np.ones((50000, 384), dtype=np.float32), list(range(1, 50001)))
assert len(store.search(np.ones(384, dtype=np.float32), k=2)) == 2
torch.set_num_threads(4)
with torch.inference_mode():
    output = torch.nn.Embedding(1000, 384)(torch.ones((32, 128), dtype=torch.long))
assert output.shape == (32, 128, 384)
assert torch.isfinite(output).all()
def worker():
    assert len(store.search(np.ones(384, dtype=np.float32), k=400)) == 400
    with torch.inference_mode():
        result = torch.ones((128, 384)) @ torch.ones((384, 128))
    assert torch.isfinite(result).all()
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(lambda _: worker(), range(4)))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
