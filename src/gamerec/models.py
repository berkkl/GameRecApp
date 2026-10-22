from typing import Any, Dict, List

from pydantic import BaseModel


class Game(BaseModel):
    id: str
    title: str
    description: str
    tags: List[str] = []
    metadata: Dict[str, Any] = {}


class Recommendation(BaseModel):
    game: Game
    score: float
    reason: str
