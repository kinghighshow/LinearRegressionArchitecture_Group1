from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from .config import ExperimentConfig
except ImportError:
    ExperimentConfig = None  # type: ignore

REQUIRED_COLUMNS = ["Time"] + [f"Axis #{i}" for i in range(1, 9)]

# database_service.get_records() returns raw tuples in this column order
# (see robot_data's CREATE TABLE / dashboard.py's use of the same order).
DB_COLUMNS = ["id", "trait"] + [f"axis_{i}" for i in range(1, 9)] + ["year", "month", "day", "time"]


class DataExtractionAnalysis:
    """Loads robot measurements from CSV or the Neon DB, validates the
    schema, and profiles them. Both loaders return the same column
    layout so DataPreparation never needs to know which source was used.
    """

    def __init__(self, config: Optional["ExperimentConfig"] = None,
                 logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger("DataExtractionAnalysis")
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

    def detect_source(self) -> str:
        source_type = self.config.get("data", "source_type", default="csv") if self.config else "csv"
        self.logger.info("Data source type: %s", source_type)
        return source_type

    # ---- CSV -----------------------------------------------------------
    def load_csv(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Robot data CSV not found: {path}")
        df = pd.read_csv(path)
        df["Time"] = pd.to_datetime(df["Time"])
        self.logger.info("Loaded %d rows from %s", len(df), path)
        return df

    # ---- Neon PostgreSQL (reuses the team's database_service.py) -------
    def load_database(self) -> pd.DataFrame:
        """Pulls every row from `robot_data` via the team's existing
        connect_db()/get_records() and reshapes it to match the CSV
        schema: axis_1 -> 'Axis #1', trait -> 'Trait', and
        year/month/day/time -> a single 'Time' column (the same
        reconstruction dashboard.py does for its live view)."""
        try:
            from .database_service import get_records
        except ImportError as exc:
            raise ImportError(
                "load_database() needs psycopg and python-dotenv installed, "
                "and src/database_service.py on the path."
            ) from exc

        try:
            records = get_records()
        except Exception as exc:
            raise ConnectionError(
                "Could not read from the Neon database. Check that a .env "
                "file with DATABASE_URL is present (see database_service.py)."
            ) from exc

        if not records:
            self.logger.warning("No rows returned from robot_data.")
            return pd.DataFrame(columns=REQUIRED_COLUMNS + ["Trait"])

        df = pd.DataFrame(records, columns=DB_COLUMNS)

        df["Time"] = pd.to_datetime(
            df["year"].astype(str) + "-" +
            df["month"].astype(str) + "-" +
            df["day"].astype(str) + " " +
            df["time"].astype(str),
            utc=True,
        )

        rename_map = {f"axis_{i}": f"Axis #{i}" for i in range(1, 9)}
        rename_map["trait"] = "Trait"
        df = df.rename(columns=rename_map)
        df = df.drop(columns=["id", "year", "month", "day", "time"])

        self.logger.info("Loaded %d rows from the Neon robot_data table", len(df))
        return df

    # ---- Shared -----------------------------------------------------------
    def validate_schema(self, df: pd.DataFrame) -> bool:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            self.logger.error("Missing required columns: %s", missing)
            return False
        return True

    def profile(self, df: pd.DataFrame) -> dict:
        axis_cols = [c for c in df.columns if c.startswith("Axis")]
        profile = {
            "rows": len(df),
            "time_range": (df["Time"].min(), df["Time"].max()),
            "missing_values": df[axis_cols].isna().sum().to_dict(),
            "axis_stats": df[axis_cols].describe().T[["mean", "std", "max"]].to_dict(orient="index"),
        }
        self.logger.info("Profiled %d rows across %d axes", profile["rows"], len(axis_cols))
        return profile

    def extract(self, csv_path: Optional[str | Path] = None) -> pd.DataFrame:
        """Orchestrates the full extraction step; returns the validated
        dataset from whichever source `data.source_type` (or an explicit
        csv_path override) points at."""
        source_type = self.detect_source()

        if csv_path is not None:
            df = self.load_csv(csv_path)
        elif source_type == "database":
            df = self.load_database()
        else:
            path = self.config.data.get("csv_path") if self.config else None
            if path is None:
                raise ValueError("No CSV path supplied (pass csv_path or set data.csv_path in config).")
            df = self.load_csv(path)

        if not self.validate_schema(df):
            raise ValueError("Robot data failed schema validation.")

        self.profile(df)
        return df
