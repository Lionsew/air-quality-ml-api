from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Air Quality & Weather ML Predictor"
    # Por defecto usa SQLite local. Si defines DATABASE_URL en .env o en la nube, usará Postgres.
    DATABASE_URL: str = "sqlite:///./air_quality.db"
    OPENAQ_API_KEY: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()