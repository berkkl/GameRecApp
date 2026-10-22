"""Train the optional tag classifier using observed labels and held-out games."""

import argparse
import sqlite3
from collections import Counter

from src.gamerec.config import settings
from src.gamerec.ml_models.tag_classifier import TagClassifier
from src.gamerec.text.cleaners import clean_text
from src.gamerec.utils.metadata import string_list


def load_training_data(db_path, max_games=25000, min_tag_frequency=30):
    if max_games <= 0 or min_tag_frequency <= 0:
        raise ValueError("Sample and frequency limits must be positive.")
    conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        games = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM games ORDER BY ((app_id * 1103515245 + 12345) & 2147483647) LIMIT ?",
                (max_games,),
            )
        ]
    finally:
        conn.close()

    def observed(game):
        return string_list(game.get("tags_steam")) + string_list(game.get("genres"))

    frequencies = Counter(tag for game in games for tag in set(observed(game)))
    allowed = {tag for tag, count in frequencies.items() if count >= min_tag_frequency}
    texts, labels, groups = [], [], []
    for game in games:
        tags = sorted(set(observed(game)) & allowed)
        summary = clean_text(game.get("summary") or "")
        if not tags or len(summary.split()) < 8:
            continue
        # All variants of one game stay on the same side of the evaluation split.
        for text in (f"{game['name']}. {summary}", f"games like {game['name']}"):
            texts.append(text)
            labels.append(tags)
            groups.append(game["app_id"])
    return texts, labels, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["current", "huggingface"], default=settings.DATASET_SOURCE
    )
    parser.add_argument(
        "--max-games", "--max-samples", dest="max_games", type=int, default=25000
    )
    parser.add_argument("--min-tag-frequency", type=int, default=30)
    args = parser.parse_args()
    texts, labels, groups = load_training_data(
        settings.get_db_path(args.dataset), args.max_games, args.min_tag_frequency
    )
    if len(set(groups)) < 100:
        raise ValueError(
            "At least 100 labeled games are required for this training run."
        )
    classifier = TagClassifier()
    metrics = classifier.train(texts, labels, groups=groups)
    classifier.save()
    print(metrics)


if __name__ == "__main__":
    main()
