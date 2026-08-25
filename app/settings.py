from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8080
    job_dir: Path = Path("data/jobs")
    model_cache: Path = Path("data/models")

    device: str = "cpu"
    enable_florence: bool = True
    enable_sam: bool = True
    enable_vl_critic: bool = False
    max_iters: int = 6
    overlay_alpha: float = 0.45

    florence_id: str = "microsoft/Florence-2-base"
    sam_weights: str = "sam2.1_t.pt"
    vl_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"


settings = Settings()
