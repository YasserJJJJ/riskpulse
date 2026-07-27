from pathlib import Path
from typing import Any, cast

from riskpulse.persistence.database import Database


def test_database_creates_parent_directory_and_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / "riskpulse.db"
    database = Database(f"sqlite:///{database_path}")

    database.create_schema()

    assert database_path.is_file()
    assert database.is_ready() is True
    database.close()


def test_database_reports_failed_connection(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'riskpulse.db'}")
    real_engine = database.engine

    class BrokenEngine:
        def connect(self) -> None:
            raise RuntimeError("database unavailable")

    database.engine = cast(Any, BrokenEngine())
    assert database.is_ready() is False
    database.engine = real_engine
    database.close()
