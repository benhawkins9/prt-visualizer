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


# ── Aggregated readers (memory-efficient — do NOT load full history) ────────
#
# load_all_history() returns 3M+ rows and costs ~640 MB RAM on cloud.
# The functions below aggregate in SQL and return ~75K rows total.

def load_snapshots(date_from: str, date_to: str) -> tuple[list[dict], list[dict]]:
    """
    Returns (latest_per_term, earliest_per_term) in two queries.
    Each row: website_id, business_name, website_url, term_id, keyword,
              term_type, rank, checked_date
    ~12K rows each instead of 3.3M.
    """
    sql = """
        SELECT w.id  AS website_id,
               w.business_name,
               w.url AS website_url,
               t.id  AS term_id,
               t.keyword,
               t.term_type,
               h.rank,
               h.checked_date
        FROM rank_history h
        JOIN (
            SELECT term_id, {agg}(checked_date) AS target_date
            FROM rank_history
            WHERE checked_date BETWEEN ? AND ?
            GROUP BY term_id
        ) m ON m.term_id = h.term_id AND h.checked_date = m.target_date
        JOIN terms    t ON t.id    = h.term_id
        JOIN websites w ON w.id    = t.website_id
    """
    with get_conn() as conn:
        latest   = [dict(r) for r in conn.execute(
            sql.format(agg="MAX"), (date_from, date_to)).fetchall()]
        baseline = [dict(r) for r in conn.execute(
            sql.format(agg="MIN"), (date_from, date_to)).fetchall()]
    return latest, baseline


def load_monthly_bucket_counts(date_from: str, date_to: str) -> list[dict]:
    """
    Pre-aggregate rank observations into 3 buckets by month.
    Columns: website_id, term_type, month, bucket, cnt
    ~30K rows instead of 3.3M.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT w.id AS website_id,
                      t.term_type,
                      substr(h.checked_date, 1, 7) AS month,
                      CASE
                          WHEN h.rank IS NULL OR h.rank = 0 OR h.rank >= 101
                               THEN 'Not Ranking'
                          WHEN h.rank <= 3 THEN 'Top 3'
                          ELSE 'Ranking'
                      END AS bucket,
                      COUNT(*) AS cnt
               FROM websites w
               JOIN terms        t ON t.website_id = w.id
               JOIN rank_history h ON h.term_id    = t.id
               WHERE h.checked_date BETWEEN ? AND ?
               GROUP BY w.id, t.term_type, month, bucket""",
            (date_from, date_to),
        ).fetchall()
    return [dict(r) for r in rows]


def load_monthly_organic_dist(date_from: str, date_to: str) -> list[dict]:
    """5-bucket organic distribution by month. ~25K rows."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT substr(h.checked_date, 1, 7) AS month,
                      CASE
                          WHEN h.rank IS NULL OR h.rank = 0 OR h.rank >= 101
                               THEN 'Not Ranking'
                          WHEN h.rank <= 3  THEN 'Top 3'
                          WHEN h.rank <= 10 THEN 'Top 10'
                          WHEN h.rank <= 30 THEN 'Top 30'
                          ELSE 'Top 100'
                      END AS bucket,
                      COUNT(*) AS cnt
               FROM rank_history h
               JOIN terms t ON t.id = h.term_id
               WHERE h.checked_date BETWEEN ? AND ?
                 AND t.term_type IN ('organic', 'mobile')
               GROUP BY month, bucket""",
            (date_from, date_to),
        ).fetchall()
    return [dict(r) for r in rows]


def load_monthly_local_dist(date_from: str, date_to: str) -> list[dict]:
    """4-bucket local pack distribution by month. ~15K rows."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT substr(h.checked_date, 1, 7) AS month,
                      CASE
                          WHEN h.rank = 1 THEN 'Position 1 (A)'
                          WHEN h.rank = 2 THEN 'Position 2 (B)'
                          WHEN h.rank = 3 THEN 'Position 3 (C)'
                          ELSE 'Not in Pack'
                      END AS bucket,
                      COUNT(*) AS cnt
               FROM rank_history h
               JOIN terms t ON t.id = h.term_id
               WHERE h.checked_date BETWEEN ? AND ?
                 AND t.term_type IN ('snack_pack', 'local_finder')
               GROUP BY month, bucket""",
            (date_from, date_to),
        ).fetchall()
    return [dict(r) for r in rows]


def load_active_term_ids(date_from: str, date_to: str) -> tuple[frozenset, frozenset]:
    """
    Returns (active_ids, all_term_ids).
    Active = ever ranked ≤50 organic OR ≤3 local pack.
    Pure SQL — never loads full history.
    """
    with get_conn() as conn:
        active_rows = conn.execute(
            """SELECT DISTINCT t.id AS term_id
               FROM terms t
               JOIN rank_history h ON h.term_id = t.id
               WHERE h.checked_date BETWEEN ? AND ?
                 AND h.rank IS NOT NULL AND h.rank > 0 AND h.rank < 101
                 AND (
                     (t.term_type IN ('organic','mobile')      AND h.rank <= 50)
                  OR (t.term_type IN ('snack_pack','local_finder') AND h.rank <= 3)
                 )""",
            (date_from, date_to),
        ).fetchall()
        all_rows = conn.execute("SELECT id FROM terms").fetchall()
    return frozenset(r[0] for r in active_rows), frozenset(r[0] for r in all_rows)


def load_monthly_avg_vis_score(
    date_from: str, date_to: str, ranking_only: bool = False
) -> list[dict]:
    """
    Monthly average organic visibility score across all clients.
    Uses the same CTR-weighted formula as visibility_score() in app.py.
    ranking_only=True: excludes observations where rank is 0/NULL/>=101,
      so the average reflects only keywords that were actually ranking that month.
    ~24 rows.
    """
    rank_filter = "AND h.rank IS NOT NULL AND h.rank > 0 AND h.rank < 101" if ranking_only else ""
    with get_conn() as conn:
        rows = conn.execute(
            f"""SELECT substr(h.checked_date, 1, 7) AS month,
                      AVG(CASE
                          WHEN h.rank IS NULL OR h.rank = 0 OR h.rank >= 101 THEN 0
                          WHEN h.rank = 1  THEN 100
                          WHEN h.rank = 2  THEN 85
                          WHEN h.rank = 3  THEN 75
                          WHEN h.rank <= 5 THEN 60
                          WHEN h.rank <= 10 THEN 40
                          WHEN h.rank <= 20 THEN 20
                          WHEN h.rank <= 50 THEN 5
                          ELSE 1
                      END) AS avg_vis
               FROM rank_history h
               JOIN terms t ON t.id = h.term_id
               WHERE h.checked_date BETWEEN ? AND ?
                 AND t.term_type IN ('organic', 'mobile')
                 {rank_filter}
               GROUP BY month
               ORDER BY month""",
            (date_from, date_to),
        ).fetchall()
    return [dict(r) for r in rows]


def load_all_history(date_from: str, date_to: str) -> list[dict]:
    """
    Full raw history — 3M+ rows, ~640 MB RAM.
    ONLY used by get_client_history() for per-client drill-down.
    Do NOT call this for the main dashboard.
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
