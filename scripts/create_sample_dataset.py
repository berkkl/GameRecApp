import json
import os
import random
from pathlib import Path

DATA_DIR = Path("data/sample")
GAMES_FILE = DATA_DIR / "games.jsonl"
REVIEWS_DIR = DATA_DIR / "reviews"
SUMMARIES_DIR = DATA_DIR / "summaries"


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REVIEWS_DIR, exist_ok=True)
    os.makedirs(SUMMARIES_DIR, exist_ok=True)


def generate_games():
    random.seed(42)
    games = []

    # Game name components for variety
    prefixes = [
        "Cosmic",
        "Shadow",
        "Crystal",
        "Dark",
        "Ancient",
        "Mystic",
        "Epic",
        "Royal",
        "Dragon",
        "Phoenix",
        "Eternal",
        "Legendary",
        "Cyber",
        "Neon",
        "Steel",
        "Blood",
        "Thunder",
        "Storm",
        "Fire",
        "Ice",
        "Golden",
        "Silver",
        "Crimson",
        "Azure",
        "Emerald",
        "Savage",
        "Wild",
        "Lost",
        "Forgotten",
        "Hidden",
        "Secret",
        "Ultimate",
        "Supreme",
        "Divine",
        "Infernal",
        "Celestial",
        "Galactic",
        "Quantum",
        "Nuclear",
        "Atomic",
        "Mythic",
        "Arcane",
        "Runic",
        "Stellar",
        "Lunar",
        "Solar",
        "Void",
        "Chaos",
        "Order",
        "Primal",
    ]

    suffixes = [
        "Quest",
        "Adventure",
        "Chronicles",
        "Saga",
        "Legends",
        "Tales",
        "Stories",
        "Realm",
        "Kingdom",
        "Empire",
        "Dynasty",
        "Legacy",
        "Destiny",
        "Prophecy",
        "Warriors",
        "Heroes",
        "Knights",
        "Defenders",
        "Guardians",
        "Hunters",
        "Frontier",
        "Odyssey",
        "Journey",
        "Expedition",
        "Conquest",
        "Crusade",
        "Revolution",
        "Rebellion",
        "Uprising",
        "Arena",
        "Tournament",
        "Showdown",
        "Battleground",
        "Warzone",
        "Conflict",
        "Siege",
        "Invasion",
        "Defense",
        "Survival",
        "Escape",
        "Rescue",
        "Discovery",
        "Exploration",
        "Mystery",
        "Puzzle",
        "Challenge",
        "Trial",
        "Master",
        "Simulator",
        "Tycoon",
    ]

    # Genres and tags
    genres = [
        "Action",
        "Adventure",
        "RPG",
        "Strategy",
        "Simulation",
        "Sports",
        "Puzzle",
        "Shooter",
        "Platformer",
        "Fighting",
        "Racing",
        "Indie",
    ]

    moods = ["Relaxing", "Intense", "Dark", "Funny", "Emotional", "Challenging", "Cozy"]

    mechanics = [
        "Open World",
        "Turn-based",
        "Real-time",
        "Co-op",
        "Multiplayer",
        "Singleplayer",
        "Crafting",
        "Survival",
        "Roguelike",
        "Deckbuilding",
    ]

    themes = [
        "Sci-Fi",
        "Fantasy",
        "Horror",
        "Historical",
        "Cyberpunk",
        "Post-apocalyptic",
    ]

    developers = [
        "Indie Studios",
        "GameDev Co",
        "PixelCraft",
        "CodeForge",
        "Digital Dreams",
        "ByteWorks",
        "PlayMakers",
        "StudioX",
    ]

    publishers = [
        "Indie Pub",
        "GamesCorp",
        "Digital Distribution",
        "Global Games",
        "PlayNow Publishing",
        "NextGen Games",
        "Arcade Publishing",
    ]

    owner_bands = [
        "0-20k",
        "20k-50k",
        "50k-100k",
        "100k-200k",
        "200k-500k",
        "500k-1M",
        "1M-2M",
        "2M-5M",
    ]

    # Generate 100 games
    for i in range(100):
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        title = f"{prefix} {suffix}"

        # Randomize release year
        year = random.randint(2018, 2024)
        month = random.randint(1, 12)
        day = random.randint(1, 28)

        # Determine price (some free games)
        is_free = random.random() < 0.15  # 15% chance of free
        if is_free:
            price = 0.0
        else:
            price = round(
                random.choice(
                    [4.99, 9.99, 14.99, 19.99, 24.99, 29.99, 39.99, 49.99, 59.99]
                ),
                2,
            )

        # Select 3-5 tags from different categories
        num_tags = random.randint(3, 5)
        all_tags = genres + moods + mechanics + themes
        selected_tags = random.sample(all_tags, k=num_tags)

        game = {
            "app_id": i + 1,
            "name": title,
            "release_date": f"{year}-{month:02d}-{day:02d}",
            "developers": random.sample(developers, k=random.randint(1, 2)),
            "publishers": random.sample(publishers, k=1),
            "price": price,
            "owners_band": random.choice(owner_bands),
            "rating_pct": round(random.uniform(45, 98), 2),
            "review_count": random.randint(250, 5000),
            "tags_steam": selected_tags,
            "tags_ml": [],
            "summary": f"{title} is a fictional demo game featuring {', '.join(selected_tags)}.",
        }
        games.append(game)

    with open(GAMES_FILE, "w") as f:
        for game in games:
            f.write(json.dumps(game) + "\n")

    return games


def generate_reviews(games):
    positive_comments = [
        "Amazing game! Highly recommend.",
        "One of the best games I've played in years.",
        "Great gameplay, fantastic graphics!",
        "Love this game, can't stop playing!",
        "Absolutely worth every penny.",
        "Brilliant design and execution.",
        "This game exceeded all my expectations.",
        "Perfect blend of challenge and fun.",
        "Masterpiece! Can't wait for a sequel.",
        "Hours of entertainment, totally worth it.",
        "The mechanics are so satisfying!",
        "Best game in its genre.",
        "Incredible experience from start to finish.",
        "10/10 would recommend to everyone.",
        "This is what gaming is all about!",
    ]

    negative_comments = [
        "Disappointing. Not what I expected.",
        "Too many bugs, needs more polish.",
        "Not worth the price unfortunately.",
        "Great concept but poor execution.",
        "Gets boring after a while.",
        "Technical issues ruined the experience.",
        "Not as good as previous titles.",
        "Repetitive gameplay, lacks depth.",
        "Too difficult, not fun at all.",
        "Refunded it after an hour.",
        "Feels incomplete and rushed.",
        "Network issues make it unplayable.",
        "Lacking content for the price.",
        "Poor optimization, runs badly.",
        "Expected more from this developer.",
    ]

    for game in games:
        app_id = game["app_id"]
        reviews = []
        num_reviews = random.randint(15, 30)  # 15-30 reviews per game

        for j in range(num_reviews):
            is_positive = random.random() < (
                game["rating_pct"] / 100
            )  # Match game rating
            review = {
                "review_id": app_id * 1000 + j,
                "app_id": app_id,
                "language": "english",
                "votes_up": (
                    random.randint(0, 50) if is_positive else random.randint(0, 10)
                ),
                "votes_funny": random.randint(0, 15),
                "voted_up": is_positive,
                "timestamp": 1672531200 + j * 86400,
                "text": random.choice(
                    positive_comments if is_positive else negative_comments
                ),
            }
            reviews.append(review)

        with open(REVIEWS_DIR / f"{app_id}.jsonl", "w") as f:
            for review in reviews:
                f.write(json.dumps(review) + "\n")


def generate_summaries(games):
    for game in games:
        app_id = game["app_id"]
        with open(SUMMARIES_DIR / f"{app_id}.txt", "w") as f:
            f.write(game["summary"])


def main():
    print("Generating sample data...")
    ensure_dirs()
    games = generate_games()
    generate_reviews(games)
    generate_summaries(games)
    print(f"Generated {len(games)} games in {DATA_DIR}")


if __name__ == "__main__":
    main()
