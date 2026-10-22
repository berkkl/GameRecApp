import numpy as np
import pytest

from scripts.index.build_rag import chunks, game_documents
from src.gamerec.search.rag_store import RAGStore
from src.gamerec.text.embedder import Embedder


def test_chunking_preserves_end_and_review_provenance():
    text = " ".join(str(i) for i in range(400))
    parts = list(chunks(text))
    assert parts[-1].endswith("399")
    assert set(text.split()) == {word for part in parts for word in part.split()}
    docs = list(
        game_documents(
            {"app_id": 1, "name": "Fixture", "summary": text, "tags_steam": ["Puzzle"]},
            [
                {
                    "review_id": 99,
                    "text": "An enjoyable puzzle game with difficult but fair levels.",
                    "language": "english",
                    "voted_up": False,
                }
            ],
        )
    )
    assert docs[-1]["review_id"] == 99
    assert docs[-1]["voted_up"] is False


def test_rag_groups_chunks_and_keeps_winning_evidence(tmp_path):
    store = RAGStore(tmp_path / "rag.index", tmp_path / "rag.ids", "fixture")
    docs = [
        {"app_id": 1, "text": "summary"},
        {"app_id": 1, "text": "winning review"},
        {"app_id": 2, "text": "other game"},
    ]
    store.add_documents(np.array([[0.9, 0.1], [1, 0], [0, 1]], dtype=np.float32), docs)
    store.save()
    loaded = RAGStore(store.index_path, store.id_map_path, "fixture")
    loaded.load()
    results, evidence = loaded.search_with_evidence(
        np.array([1, 0], dtype=np.float32), 2
    )
    assert [i for i, _ in results] == [1, 2]
    assert evidence[1]["text"] == "winning review"
    assert loaded.score_game(np.array([1, 0]), 1)[1]["text"] == "winning review"


@pytest.mark.model
def test_multilingual_retrieval():
    model = Embedder("intfloat/multilingual-e5-small")
    docs = model.embed(
        [
            "A peaceful farming game where you grow crops and look after animals.",
            "A competitive first person shooter with guns and military battles.",
            "A difficult racing game with fast cars.",
        ]
    )
    english = model.embed_queries(["I want a relaxing farming game."])[0]
    turkish = model.embed_queries(["Rahatlatıcı bir çiftçilik oyunu istiyorum."])[0]
    assert int(np.argmax(docs @ english)) == 0
    assert int(np.argmax(docs @ turkish)) == 0
