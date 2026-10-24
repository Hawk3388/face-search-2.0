import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DLIB_MODEL = "dlib-128-large-v1"


@dataclass(frozen=True)
class Settings:
    database: Path
    backend: str = "json"
    metric: str = "cosine"
    tolerance: float = 0.07
    token: str = ""
    allow_unauthenticated: bool = False
    encoder: str = "dlib"
    model_path: str = ""
    max_upload_bytes: int = 10 * 1024 * 1024
    max_pixels: int = 20_000_000
    max_faces: int = 20
    max_results: int = 100
    num_jitters: int = 10
    requests_per_minute: int = 120

    def __post_init__(self):
        import math

        if self.backend not in ("json", "sqlite") or self.metric not in ("euclidean", "cosine"):
            raise ValueError("Invalid storage backend or search metric")
        if not math.isfinite(self.tolerance) or not 0 <= self.tolerance <= 2:
            raise ValueError("SEARCH_TOLERANCE must be between 0 and 2")
        for name in ("max_upload_bytes", "max_pixels", "max_faces", "max_results", "num_jitters"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.requests_per_minute < 0:
            raise ValueError("REQUESTS_PER_MINUTE must not be negative")

    @classmethod
    def from_env(cls, backend="json"):
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        metric = os.getenv("SEARCH_METRIC", "cosine")
        if metric not in ("euclidean", "cosine"):
            raise ValueError("SEARCH_METRIC must be euclidean or cosine")
        filename = "face_embeddings.db" if backend == "sqlite" else "face_embeddings.json"
        return cls(
            database=Path(os.getenv("DATABASE_PATH", str(ROOT / filename))).resolve(),
            backend=backend,
            metric=metric,
            tolerance=float(os.getenv("SEARCH_TOLERANCE", "0.07" if metric == "cosine" else "0.5")),
            token=os.getenv("TOKEN", ""),
            allow_unauthenticated=os.getenv("ALLOW_UNAUTHENTICATED", "0") == "1",
            encoder=os.getenv("FACE_ENCODER", "dlib"),
            model_path=os.getenv("MODEL_PATH", ""),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))),
            max_pixels=int(os.getenv("MAX_IMAGE_PIXELS", "20000000")),
            max_faces=int(os.getenv("MAX_FACES", "20")),
            max_results=int(os.getenv("MAX_RESULTS", "100")),
            num_jitters=int(os.getenv("NUM_JITTERS", "10")),
            requests_per_minute=int(os.getenv("REQUESTS_PER_MINUTE", "120")),
        )
