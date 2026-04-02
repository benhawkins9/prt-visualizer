"""
SQLite cache layer.

Schema
------
websites     — one row per client URL (id = PRT url_id)
terms        — one row per tracked keyword under a website (id = url_term_id)
rank_history — daily rank records per term
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "prt_cache.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS websites (
                id            INTEGER PRIMARY KEY,
                url           TEXT NOT NULL,
                business_name TEXT,
                note          TEXT
            );

            CREATE TABLE IF NOT EXISTS terms (
                id         INTEGER PRIMARY KEY,   -- url_term_id
                website_id INTEGER NOT NULL REFERENCES websites(id),
                keyword    TEXT NOT NULL,
                term_type  TEXT
            );

            CREATE TABLE IF NOT EXISTS rank_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                term_id      INTEGER NOT NULL REFERENCES terms(id),
                checked_date TEXT NOT NULL,
                rank         INTEGER,
                UNIQUE(term_id, checked_date)
            );

            CREATE INDEX IF NOT EXISTS idx_rh_term_date
                ON rank_history(term_id, checked_date);
        """)
        # Migrate existing DBs that predate the term_type column
        cols = {r[1] for r in conn.execute("PRAGMA table_info(terms)")}
        if "term_type" not in cols:
            conn.execute("ALTER TABLE terms ADD COLUMN term_type TEXT")


# ── display name helper ────────────────────────────────────────────────────

def display_name(website: dict) -> str:
    """business_name if set, otherwise the URL."""
    name = (website.get("business_name") or "").strip()
    return name if name else website.get("url", f"ID {website['id']}")


# ── writers ────────────────────────────────────────────────────────────────

def upsert_websites(websites: list[dict]):
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO websites(id, url, business_name, note)
               VALUES(:id, :url, :business_name, :note)
               ON CONFLICT(id) DO UPDATE SET
                   url=excluded.url,
                   business_name=excluded.business_name,
                   note=excluded.note""",
            [
                {
                    "id": int(w["id"]),
                    "url": w.get("url", ""),
                    "business_name": w.get("business_name") or "",
                    "note": w.get("note") or "",
                }
                for w in websites
            ],
        )


def upsert_terms_and_history(website_id: int, terms: list[dict]):
    """
    Upsert all terms and their rank history for a website.
    Each term is expected to have:
        url_term_id, name, rankhistory: [{checked, rank}, ...]
    """
    with get_conn() as conn:
        for term in terms:
            term_id = int(term["url_term_id"])
            keyword = term.get("name", "")
            term_type = term.get("term_type") or ""
            conn.execute(
                """INSERT INTO terms(id, website_id, keyword, term_type)
                   VALUES(?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       keyword=excluded.keyword,
                       term_type=excluded.term_type""",
                (term_id, website_id, keyword, term_type),
            )
            history = term.get("rankhistory") or []
            is_snack_pack = term_type == "snack_pack"
            conn.executemany(
                """INSERT INTO rank_history(term_id, checked_date, rank)
                   VALUES(?, ?, ?)
                   ON CONFLICT(term_id, checked_date) DO UPDATE SET rank=excluded.rank""",
                [
                    (
                        term_id,
                        h["checked"],
                        # snack_pack terms expose the A/B/C map pack position as
                        # `maprank`; the `rank` field on these rows is the organic
                        # position, which is not what we want for local pack analysis.
                        h.get("maprank") if is_snack_pack else h.get("rank"),
                    )
                    for h in history
                    if h.get("checked")
                ],
            )


# ── readers ────────────────────────────────────────────────────────────────

def load_websites() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM websites ORDER BY business_name, url"
        ).fetchall()
    return [dict(r) for r in rows]


def load_all_history(date_from: str, date_to: str) -> list[dict]:
    """
    Flat rows: website_id, website_name, term_id, keyword, checked_date, rank
    across all clients for the given date range.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT w.id          AS website_id,
                      w.url         AS website_url,
                      w.business_name,
                      t.id          AS term_id,
                      t.keyword,
                      t.term_type,
                      h.checked_date,
                      h.rank
               FROM websites w
               JOIN terms t ON t.website_id = w.id
               JOIN rank_history h ON h.term_id = t.id
               WHERE h.checked_date BETWEEN ? AND ?
               ORDER BY w.id, t.keyword, h.checked_date""",
            (date_from, date_to),
        ).fetchall()
    return [dict(r) for r in rows]


def load_history_for_website(website_id: int, date_from: str, date_to: str) -> list[dict]:
    """Flat rows: term_id, keyword, checked_date, rank for one client."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT t.id AS term_id,
                      t.keyword,
                      t.term_type,
                      h.checked_date,
                      h.rank
               FROM terms t
               JOIN rank_history h ON h.term_id = t.id
               WHERE t.website_id = ?
                 AND h.checked_date BETWEEN ? AND ?
               ORDER BY t.keyword, h.checked_date""",
            (website_id, date_from, date_to),
        ).fetchall()
    return [dict(r) for r in rows]
