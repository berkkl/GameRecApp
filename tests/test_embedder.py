import numpy as np
import pytest

from src.gamerec.text.embedder import embed_text, load_embedder


@pytest.mark.model
def test_embedder_properties():
    load_embedder()

    text = "This is a test sentence."
    embedding = embed_text(text)

    # Check shape
    assert embedding.shape == (384,)

    # Check normalization (L2 norm should be approx 1.0)
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-5)

    # Check stability
    embedding2 = embed_text(text)
    assert np.allclose(embedding, embedding2)

    # Check empty text handling (should probably return zero vector or error, checking current behavior)
    # Based on implementation, it might embed empty string.
    empty_emb = embed_text("")
    assert empty_emb.shape == (384,)
