from pathlib import Path

import pydantic


class _Settings(pydantic.BaseSettings):
    environment: str = "dev"
    csv_cache: Path = Path("csvs/")
    param_dir: Path = Path("params/")
    results_dir: Path = Path("results/")

    class Config:
        env_prefix: str = "bt_"
        env_file = Path(__file__).parent / ".env"
        env_file_encoding: str = "utf-8"


def get() -> _Settings:
    return _Settings()