"""
Shared DB engine for every stage.

Defaults to SQLite (zero setup, matches the original prototype). Set
DATABASE_URL to a PostgreSQL DSN (e.g. a free Neon/Supabase instance) to
switch every stage to Postgres instead — the schema in normalize.py is
written for both backends, so no other code needs to change.
"""
from __future__ import annotations
import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DATABASE_URL = os.environ.get("DATABASE_URL")
IS_POSTGRES = bool(DATABASE_URL)

AGG_DB = os.environ.get("AGG_DB", os.path.join(ROOT, "data", "aggregator.db"))


def get_engine() -> Engine:
    if IS_POSTGRES:
        url = DATABASE_URL
        if url.startswith("postgres://"):  # SQLAlchemy wants the explicit dialect
            url = url.replace("postgres://", "postgresql+psycopg2://", 1)
        return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)
    os.makedirs(os.path.dirname(AGG_DB), exist_ok=True)
    return create_engine(f"sqlite:///{AGG_DB}")


ENGINE = get_engine()
