"""Optional multilingual query/passage cross-encoder for the shortlist."""

from threading import RLock

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from ..text.cleaners import clean_text
from ..utils.metadata import game_tags


class SemanticReranker:
    def __init__(self, model_name, revision, device="cpu", top_n=40):
        if not revision:
            raise ValueError(
                "Pin RERANK_REVISION to a model commit before enabling it."
            )
        self.model = CrossEncoder(
            model_name,
            revision=revision,
            device=device,
            max_length=512,
            trust_remote_code=False,
        )
        self.top_n = top_n
        self.lock = RLock()

    def rerank(self, query, games):
        head = games[: self.top_n]
        if not head:
            return games
        # Catalog descriptions anchor mechanics. A retrieved negative review
        # mentioning an unwanted feature is not evidence that it satisfies it.
        pairs = [
            (
                query,
                f"{g['name']}. Tags: {', '.join(game_tags(g))}. "
                + clean_text(g.get("summary"))[:2200],
            )
            for g in head
        ]
        with self.lock:
            logits = np.asarray(
                self.model.predict(
                    pairs,
                    batch_size=8,
                    show_progress_bar=False,
                    activation_fn=torch.nn.Identity(),
                )
            ).reshape(-1)
        if len(logits) != len(head) or not np.isfinite(logits).all():
            raise ValueError("Cross-encoder returned invalid relevance scores.")
        semantic_order = sorted(
            range(len(head)), key=lambda i: (-float(logits[i]), head[i]["app_id"])
        )
        ranks = {i: rank for rank, i in enumerate(semantic_order)}
        scored = []
        for i, game in enumerate(head):
            # Fuse ranks; raw logits/sigmoid are not calibrated probabilities.
            score = 0.7 / (20 + ranks[i] + 1) + 0.3 / (20 + i + 1)
            copy = {
                **game,
                "semantic_rank_score": score,
                "cross_encoder_logit": float(logits[i]),
            }
            scored.append(copy)
        scored.sort(key=lambda g: (-g["semantic_rank_score"], g["app_id"]))
        return scored + games[self.top_n :]
