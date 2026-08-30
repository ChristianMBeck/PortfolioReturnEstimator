"""Application-wide constants and paths."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
DB_PATH: Path = DATA_DIR / "portfolio.db"
DEFAULT_TICKERS: list[str] = ["SPY", "AGG", "GLD"]
