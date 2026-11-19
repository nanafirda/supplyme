"""Connexion SQLite, création du schéma, écriture et lecture des tables."""
from __future__ import annotations

import gzip
import logging
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

import config as C

log = logging.getLogger(__name__)


def preparer_base(db_path: Path = C.DB_PATH) -> None:
    """Version en ligne : la base est stockée compressée et découpée (limite GitHub de 25 Mo par fichier)."""
    if db_path.exists():
        return
    parts = sorted(db_path.parent.glob(db_path.name + ".gz.part*"))
    if parts:
        with open(db_path.with_suffix(".db.gz"), "wb") as out:
            for p in parts:
                out.write(p.read_bytes())
    gz = db_path.with_suffix(".db.gz")
    if gz.exists():
        with gzip.open(gz, "rb") as src, open(db_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


@contextmanager
def connect(db_path: Path = C.DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path = C.DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executescript(C.SCHEMA_PATH.read_text(encoding="utf-8"))


def ecrire(df: pd.DataFrame, table: str, db_path: Path = C.DB_PATH, vider: bool = False) -> None:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    with connect(db_path) as conn:
        if vider:
            conn.execute(f"DELETE FROM {table}")
        out.to_sql(table, conn, if_exists="append", index=False, chunksize=50_000)
    log.info("%-17s : %s lignes", table, f"{len(out):,}")


def query(sql: str, params=(), db_path: Path = C.DB_PATH, parse_dates=None) -> pd.DataFrame:
    with connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params, parse_dates=parse_dates)


def metadata_set(values: dict, db_path: Path = C.DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executemany("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                         [(k, str(v)) for k, v in values.items()])


def metadata_get(db_path: Path = C.DB_PATH) -> dict:
    with connect(db_path) as conn:
        return dict(conn.execute("SELECT key, value FROM metadata").fetchall())


def controles(db_path: Path = C.DB_PATH) -> dict:
    with connect(db_path) as conn:
        q = lambda s: conn.execute(s).fetchone()[0]  # noqa: E731
        return {t: q(f"SELECT COUNT(*) FROM {t}") for t in ("customers", "products", "orders", "order_items")} | {
            "violations_cles_etrangeres": len(conn.execute("PRAGMA foreign_key_check").fetchall()),
            "commandes_sans_produit": q("SELECT COUNT(*) FROM orders o WHERE NOT EXISTS "
                                        "(SELECT 1 FROM order_items i WHERE i.order_id = o.order_id)")}
