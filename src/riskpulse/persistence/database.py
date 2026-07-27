from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from riskpulse.persistence.models import Base, ScoringEvent


class Database:
    """Own the SQLAlchemy engine and request-scoped session factory."""

    def __init__(self, url: str, *, echo: bool = False) -> None:
        connect_args: dict[str, bool] = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        self.engine: Engine = create_engine(
            url,
            echo=echo,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        database_path = self.engine.url.database
        if self.engine.url.get_backend_name() == "sqlite" and database_path not in {
            None,
            "",
            ":memory:",
        }:
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self._session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
        )

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def sessions(self) -> Iterator[Session]:
        with self._session_factory() as session:
            yield session

    def is_ready(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(select(ScoringEvent.transaction_id).limit(1))
        except Exception:
            return False
        return True

    def close(self) -> None:
        self.engine.dispose()
