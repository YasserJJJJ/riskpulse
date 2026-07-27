from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


def test_initial_migration_upgrades_and_downgrades(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database_url = f"sqlite:///{tmp_path / 'migration.db'}"
    monkeypatch.setenv("RISKPULSE_DATABASE_URL", database_url)
    config = Config("alembic.ini")

    command.upgrade(config, "head")

    engine = create_engine(database_url)
    inspector = inspect(engine)
    assert "scoring_events" in inspector.get_table_names()
    assert {
        "ix_scoring_events_idempotency_key",
        "ix_scoring_events_pending_reviews",
        "ix_scoring_events_scored_at",
    } <= {index["name"] for index in inspector.get_indexes("scoring_events")}

    command.downgrade(config, "base")
    assert "scoring_events" not in inspect(engine).get_table_names()
    engine.dispose()
