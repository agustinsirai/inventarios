import os
from pathlib import Path

# Repo root = web_version/
BASE_DIR = Path(__file__).resolve().parent.parent

# Assets
TIFF_DIR = Path(os.getenv("TIFF_DIR", BASE_DIR / "assets_tiff")).resolve()

# Storage (runtime outputs)
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", BASE_DIR / "storage")).resolve()
STORAGE_PREVIEWS = STORAGE_DIR / "previews"
STORAGE_JSON = STORAGE_DIR / "json"
STORAGE_TMP = STORAGE_DIR / "_tmp"

# Optional (por si más adelante volvés a usar DB)
DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "inventarios.sqlite")).resolve()

# Feature flags
ENABLE_DOWNLOAD = os.getenv("ENABLE_DOWNLOAD", "1").strip().lower() not in ("0", "false", "no", "off")

# Ensure folders exist
for p in (STORAGE_PREVIEWS, STORAGE_JSON, STORAGE_TMP):
    p.mkdir(parents=True, exist_ok=True)