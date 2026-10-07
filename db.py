"""Shared Postgres helpers. The connection string lives in .env (never committed)."""

import os
import re
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parent
SQL_DIR = ROOT / "sql"


def normalize_url(url: str) -> str:
    """Plain libpq form: Heroku-style postgres:// and SQLAlchemy driver
    suffixes (postgresql+psycopg2://) are both rewritten to postgresql://."""
    url = url.strip()
    scheme, sep, rest = url.partition("://")
    if sep and (scheme in ("postgres", "postgresql") or scheme.startswith("postgresql+")):
        return "postgresql://" + rest
    return url


def database_url() -> str:
    load_dotenv(ROOT / ".env")
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        sys.exit("DATABASE_URL is not set. Copy .env.example to .env and paste your "
                 "Postgres connection string (Neon / Supabase / local).")
    return normalize_url(url)


def get_engine():
    """SQLAlchemy engine, used for pandas.read_sql.

    The driver is pinned to psycopg2: SQLAlchemy 2.1 made psycopg (v3) the
    default for bare postgresql:// URLs, and this project ships psycopg2-binary.
    """
    url = "postgresql+psycopg2://" + database_url().partition("://")[2]
    return create_engine(url, pool_pre_ping=True)


def connect():
    """Plain psycopg2 connection, used for bulk upserts and running .sql files."""
    return psycopg2.connect(database_url(), connect_timeout=30)


_SETTING_NAME = re.compile(r"^[a-z_]+\.[a-z_]+$")
_SETTING_VALUE = re.compile(r"^[A-Za-z0-9_]+$")


def run_sql_file(conn, path: Path, search_path: str | None = None,
                 settings: dict[str, str] | None = None) -> None:
    """Execute a whole .sql file. The file controls its own BEGIN / COMMIT.

    search_path, if given, is set in the same round trip as the file, so a
    connection pooler cannot run the file against a different schema.
    settings (e.g. {"backtest.fundamentals_mode": "snapshot"}) are applied with
    SET LOCAL right after the file's first BEGIN, so they last exactly one
    transaction and can't leak to other sessions through a pooler.
    """
    sql = Path(path).read_text(encoding="utf-8")
    if settings:
        for name, value in settings.items():
            if not (_SETTING_NAME.match(name) and _SETTING_VALUE.match(value)):
                raise ValueError(f"refusing unsafe setting {name}={value!r}")
        if "BEGIN;" not in sql:
            raise ValueError(f"{path} has no BEGIN; to scope settings to")
        local = "".join(f"\nSET LOCAL {name} = '{value}';" for name, value in settings.items())
        sql = sql.replace("BEGIN;", "BEGIN;" + local, 1)
    if search_path:
        sql = f"SET search_path TO {search_path};\n" + sql
    previous = conn.autocommit
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(sql)
            except Exception:
                # A failure inside the file's BEGIN leaves the server in an aborted
                # transaction that psycopg2 (in autocommit) doesn't know about.
                cur.execute("ROLLBACK")
                raise
    finally:
        conn.autocommit = previous
