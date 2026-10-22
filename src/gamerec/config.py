import json
from pathlib import Path
from typing import Dict, Literal, Optional

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")
    PROJECT_NAME: str = "GameRecApp"
    DATASET_SOURCE: Literal["current", "huggingface"] = "current"
    DATA_DIR: Path = PROJECT_ROOT / "data"
    DB_PATH: Path = PROJECT_ROOT / "data/games.db"
    INDEX_PATH: Path = PROJECT_ROOT / "data/games.index"
    DB_PATH_HF: Path = PROJECT_ROOT / "data/games_hf.db"
    INDEX_PATH_HF: Path = PROJECT_ROOT / "data/games_hf.index"
    TAGS_PATH: Path = PROJECT_ROOT / "src/gamerec/tags/tags.yml"
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBED_THREADS: int = Field(default=4, ge=1, le=64)
    OPENROUTER_API_KEY: Optional[SecretStr] = None
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[SecretStr] = None
    FUSION_WEIGHTS: Dict[str, float] = {"reviews": 0.6, "tags": 0.3, "summary": 0.1}
    TOP_K: int = 100
    MAX_RETRIEVAL_CANDIDATES: int = Field(default=1600, ge=100, le=10000)
    RERANK_MODEL: Optional[str] = None
    RERANK_REVISION: Optional[str] = None
    RERANK_DEVICE: str = "cpu"
    RERANK_TOP_N: int = Field(default=40, ge=10, le=100)
    SOCIAL_COSINE_WEIGHT: float = Field(default=0.95, ge=0, le=1)
    EMBED_REVISION: Optional[str] = None
    EMBED_DEVICE: str = "cpu"
    ACTIVE_CATALOG_MANIFEST: Optional[Path] = None
    STEAM_WEB_API_KEY: Optional[SecretStr] = None
    ENABLE_SEARCH_HISTORY: bool = False
    ENABLE_SEARCH_LOGS: bool = False
    ENABLE_CLOUD_HISTORY: bool = False
    ENABLE_TRANSLATION: bool = False
    ENABLE_AI_FEEDBACK: bool = False
    DEMO_CATALOG: bool = False
    CATALOG_NOTICE: Optional[str] = None
    CORS_ORIGINS: list[str] = []
    HISTORY_TOKEN: Optional[SecretStr] = None

    @field_validator(
        "ACTIVE_CATALOG_MANIFEST", "EMBED_REVISION", "RERANK_REVISION", mode="before"
    )
    @classmethod
    def empty_optional_setting(cls, value):
        # An explicit empty environment override lets isolated tools bypass .env.
        return None if value == "" else value

    @model_validator(mode="after")
    def serving_bundle(self):
        if self.ACTIVE_CATALOG_MANIFEST:
            bundle = json.loads(
                self.ACTIVE_CATALOG_MANIFEST.read_text(encoding="utf-8")
            )
            if bundle.get("schema_version") != 1:
                raise ValueError("Unsupported serving catalog manifest.")
            self.DB_PATH = Path(bundle["catalog"])
            self.INDEX_PATH = Path(bundle["index"])
            self.EMBED_MODEL = bundle["model"]
            self.EMBED_REVISION = bundle.get("model_revision")
        return self

    def get_db_path(self, source: Optional[str] = None) -> Path:
        source = source or self.DATASET_SOURCE
        if source not in {"current", "huggingface"}:
            raise ValueError(f"Unknown dataset: {source}")
        return self.DB_PATH_HF if source == "huggingface" else self.DB_PATH

    def get_index_path(self, source: Optional[str] = None) -> Path:
        source = source or self.DATASET_SOURCE
        if source not in {"current", "huggingface"}:
            raise ValueError(f"Unknown dataset: {source}")
        return self.INDEX_PATH_HF if source == "huggingface" else self.INDEX_PATH


settings = Settings()
