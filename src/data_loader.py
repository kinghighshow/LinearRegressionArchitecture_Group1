"""
data_loader.py
---------------
All data-acquisition logic in one place, isolated from preprocessing and
modeling (separation of concerns). Wraps the three source types required
by Part 1: CSV, REST API, and a relational (SQLite) database.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

try:
    from .config import ExperimentConfig
except ImportError:  # allows running the module standalone
    ExperimentConfig = None  # type: ignore


class DataLoader:
    def __init__(self, config: Optional["ExperimentConfig"] = None):
        self.config = config

    # ---- CSV -------------------------------------------------------
    def load_csv(self, path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV not found: {path}")
        return pd.read_csv(path, **read_csv_kwargs)

    # ---- REST API ----------------------------------------------------
    def load_api(
        self,
        url: str,
        params: Optional[dict] = None,
        timeout: int = 30,
        record_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch JSON from a REST/CKAN-style API and return it as a DataFrame.

        record_path: dotted path to the list of records inside the JSON
        body, e.g. 'result.records' for CKAN's datastore_search response.
        """
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()

        if record_path:
            for key in record_path.split("."):
                payload = payload[key]

        return pd.DataFrame(payload)

    # ---- Relational database -----------------------------------------
    def load_database(self, db_path: str | Path, query: str) -> pd.DataFrame:
        db_path = Path(db_path)
        if not db_path.is_file():
            raise FileNotFoundError(f"Database not found: {db_path}")
        with sqlite3.connect(db_path) as connection:
            return pd.read_sql_query(query, connection)

    # ---- Config-driven convenience ------------------------------------
    def load_from_config(self) -> dict[str, pd.DataFrame]:
        """Load every source listed in configs/experiment_config.yaml at once.

        Returns a dict keyed by source name, e.g.
        {'california': df, 'toronto': df, 'api': df, 'database': df}
        """
        if self.config is None:
            raise ValueError("DataLoader has no ExperimentConfig to read from.")

        results: dict[str, pd.DataFrame] = {}

        for name, path in self.config.csv_paths.items():
            try:
                results[name] = self.load_csv(path)
            except FileNotFoundError as exc:
                print(f"Skipping CSV source '{name}': {exc}")

        api_cfg = self.config.api
        if api_cfg.get("url"):
            try:
                results["api"] = self.load_api(
                    api_cfg["url"],
                    params=api_cfg.get("params"),
                    record_path=api_cfg.get("record_path"),
                )
            except requests.RequestException as exc:
                print(f"Skipping API source: {exc}")

        db_cfg = self.config.database
        if db_cfg.get("path") and db_cfg.get("query"):
            try:
                results["database"] = self.load_database(db_cfg["path"], db_cfg["query"])
            except FileNotFoundError as exc:
                print(f"Skipping database source: {exc}")

        return results
