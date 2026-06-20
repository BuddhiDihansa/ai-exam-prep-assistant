from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # MongoDB
    MONGODB_URI: str
    DB_NAME: str = "unisage"

    # Groq
    GROQ_API_KEY: str
    GROQ_MODEL: str = "openai/gpt-oss-20b"

    # CORS - comma separated string in .env, parsed to list here
    CORS_ORIGINS: str = "http://localhost:5173"

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Ingestion
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_FILE_SIZE_MB: int = 10

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


# Single shared instance - import this everywhere
settings = Settings()