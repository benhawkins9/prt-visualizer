"""
PRT Visualizer — Streamlit Dashboard
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date

from db import init_db, load_websites, load_all_history, load_history_for_website, display_name

# ── Constants ─────────────────────────────────────────────────────────────────
DATE_FROM  = "2024-02-01"
TODAY      = date.today().isoformat()
NOT_RANKED = 101

ORGANIC_TYPES   = {"organic", "mobile"}
LOCAL_MAP_TYPES = {"snack_pack", "local_finder"}

GREEN       = "#22c55e"
YELLOW      = "#eab308"
RED         = "#ef4444"
BLUE        = "#3b82f6"

# Organic distribution — 5 buckets
ORG_BUCKETS  = ["Top 3", "Top 10", "Top 30", "Top 100", "Not Ranking"]
ORG_COLORS   = ["#166534", "#86efac", "#eab308", "#f97316", "#ef4444"]

# Local pack distribution — 4 buckets
LOC_BUCKETS  = ["Position 1 (A)", "Position 2 (B)", "Position 3 (C)", "Not in Pack"]
LOC_COLORS   = ["#166534", "#22c55e", "#86efac", "#ef4444"]

st.set_page_config(page_title="PRT Visualizer", page_icon="📈", layout="wide")
init_db()

if "selected_client" not in st.session_state:
    st.session_state.selected_client = None


# ── Rank / score helpers ──────────────────────────────────────────────────────

def cap_rank(r) -> int:
    """
    Normalize PRT's unranked sentinels (0, None, NaN, 501+) → 101.
    Uses try/int() so that float('nan') and None both route to NOT_RANKED
    rather than crashing on NaN comparisons (NaN >= 101 is always False).
    """
    try:
        ri = int(r)
    except (TypeError, ValueError):
        return NOT_RANKED
    if ri == 0 or ri >= NOT_RANKED:
        return NOT_RANKED
    return ri


def visibility_score(rank_capped: int) -> int:
    """
    Weighted visibility score reflecting the CTR curve.
    Higher = better.  Scale 0–100.
    Position  1  → 100 pts
    Position  2  →  85 pts
    Position  3  →  75 pts
    Position 4-5 →  60 pts
    Position 6-10→  40 pts
    Position 11-20→  20 pts
    Position 21-50→   5 pts
    Position 51-100→  1 pt
    Not ranking  →   0 pts
    """
    if rank_capped >= NOT_RANKED: return 0
    if rank_capped == 1:          return 100
    if rank_capped == 2:          return 85
    if rank_capped == 3:          return 75
    if rank_capped <= 5:          return 60
    if rank_capped <= 10:         return 40
    if rank_capped <= 20:         return 20
    if rank_capped <= 50:         return 5
    return 1  # 51–100


def bucket_rank(rank_capped: int) -> str:
    """Coarse 3-bucket for the main stacked bar (used across all types)."""
    if rank_capped <= 3:          return "Top 3"
    if rank_capped < NOT_RANKED:  return "Ranking"
    return "Not Ranking"


def organic_bucket(rank_capped: int) -> str:
    """Fine 5-bucket for organic distribution charts."""
    if rank_capped >= NOT_RANKED: return "Not Ranking"
    if rank_capped <= 3:          return "Top 3"
    if rank_capped <= 10:         return "Top 10"
    if rank_capped <= 30:         return "Top 30"
    return "Top 100"


def local_bucket(rank_capped: int) -> str:
    """4-bucket for local pack distribution charts."""
    if rank_capped == 1:          return "Position 1 (A)"
    if rank_capped == 2:          return "Position 2 (B)"
    if rank_capped == 3:          return "Position 3 (C)"
    return "Not in Pack"


def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rank_capped"] = df["rank"].apply(cap_rank)
    df["vis_score"]   = df["rank_capped"].apply(visibility_score)
    df["bucket"]      = df["rank_capped"].apply(bucket_rank)
    return df


# ── DataFrame utilities ───────────────────────────────────────────────────────

def latest_per_term(df: pd.DataFrame) -> pd.DataFrame:
    """Most recent row for each term_id."""
    return df.sort_values("checked_date").groupby("term_id").last().reset_index()


def earliest_per_term(df: pd.DataFrame) -> pd.DataFrame:
    """Earliest row for each term_id (used as Feb-2024 baseline)."""
    return df.sort_values("checked_date").groupby("term_id").first().reset_index()


def monthly_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each calendar month, compute % in Top 3 / Ranking / Not Ranking
    across all rank observations in that month.
    """
    d = df.copy()
    d["month"] = d["checked_date"].dt.to_period("M").astype(str)
    grouped = d.groupby(["month", "bucket"]).size().unstack(fill_value=0)
    for col in ["Top 3", "Ranking", "Not Ranking"]:
        if col not in grouped.columns:
            grouped[col] = 0
    grouped["total"]          = grouped[["Top 3", "Ranking", "Not Ranking"]].sum(axis=1)
    grouped["pct_top3"]       = grouped["Top 3"]       / grouped["total"] * 100
    grouped["pct_ranking"]    = grouped["Ranking"]     / grouped["total"] * 100
    grouped["pct_not_ranking"]= grouped["Not Ranking"] / grouped["total"] * 100
    return grouped.reset_index().sort_values("month")


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_all_history() -> pd.DataFrame:
    rows = load_all_history(DATE_FROM, TODAY)
    if not rows:
        # Include ALL computed columns so downstream code never hits KeyError
        # on rank_capped / vis_score / bucket even when the DB is empty.
        return pd.DataFrame(columns=[
            "website_id", "website_url", "business_name", "client_name",
            "term_id", "keyword", "term_type", "checked_date", "rank",
            "rank_capped", "vis_score", "bucket",
        ])
    df = pd.DataFrame(rows)
    df["checked_date"] = pd.to_datetime(df["checked_date"])
    df["rank"]         = pd.to_numeric(df["rank"], errors="coerce")
    df["term_type"]    = df["term_type"].fillna("").str.lower()
    df["client_name"]  = df.apply(
        lambda r: r["business_name"].strip()
        if r["business_name"] and r["business_name"].strip()
        else r["website_url"],
        axis=1,
    )
    return prep_df(df)


@st.cache_data(show_spinner=False)
def get_client_history(website_id: int) -> pd.DataFrame:
    rows = load_history_for_website(website_id, DATE_FROM, TODAY)
    if not rows:
        return pd.DataFrame(columns=[
            "term_id", "keyword", "term_type", "checked_date", "rank",
            "rank_capped", "vis_score", "bucket",
        ])
    df = pd.DataFrame(rows)
    df["checked_date"] = pd.to_datetime(df["checked_date"])
    df["rank"]         = pd.to_numeric(df["rank"], errors="coerce")
    df["term_type"]    = df["term_type"].fillna("").str.lower()
    return prep_df(df)


@st.cache_data(show_spinner=False)
def classify_terms(df: pd.DataFrame) -> tuple[frozenset, frozenset]:
    """
    Classify every term_id as active or moonshot based on its full history.

    Active  — has ranked ≤ 50 organic at any point, OR appeared in the local
              pack (rank 1-3) at any point since Feb 2024.
    Moonshot — never hit either threshold in the entire tracked period.

    Returns (active_ids, moonshot_ids) as frozensets.
    """
    if df.empty or "rank_capped" not in df.columns:
        return frozenset(), frozenset()

    active: set = set()

    org = df[df["term_type"].isin(ORGANIC_TYPES)]
    active.update(org.loc[org["rank_capped"] <= 50, "term_id"].unique())

    loc = df[df["term_type"].isin(LOCAL_MAP_TYPES)]
    active.update(loc.loc[loc["rank_capped"] <= 3, "term_id"].unique())

    all_ids   = frozenset(df["term_id"].unique())
    active_fs = frozenset(active)
    return active_fs, all_ids - active_fs


@st.cache_data(show_spinner=False)
def all_client_stats(df_all: pd.DataFrame) -> dict:
    """
    Precompute per-client card stats in a single pass so the client grid
    doesn't re-filter the full DataFrame 104 times.
    Returns { website_id: stats_dict | None }
    """
    result = {}
    for wid, df_site in df_all.groupby("website_id"):
        df_org = df_site[df_site["term_type"].isin(ORGANIC_TYPES)]
        df_loc = df_site[df_site["term_type"].isin(LOCAL_MAP_TYPES)]

        stats: dict = {"has_organic": False, "has_local": False}

        if not df_org.empty:
            latest   = latest_per_term(df_org)
            baseline = earliest_per_term(df_org)
            stats.update({
                "has_organic":  True,
                "vis_now":      latest["vis_score"].mean(),
                "vis_then":     baseline["vis_score"].mean(),
                "pct_top10":    (latest["rank_capped"] <= 10).mean() * 100,
            })
            stats["vis_change"] = stats["vis_now"] - stats["vis_then"]

        if not df_loc.empty:
            latest   = latest_per_term(df_loc)
            baseline = earliest_per_term(df_loc)
            n_top3      = int((latest["rank_capped"] <= 3).sum())
            n_top3_then = int((baseline["rank_capped"] <= 3).sum())
            n_total     = len(latest)
            stats.update({
                "has_local":    True,
                "n_top3":       n_top3,
                "n_top3_then":  n_top3_then,
                "n_local_total":n_total,
                "pct_top3":     n_top3 / n_total * 100 if n_total else 0,
            })

        result[wid] = stats if (stats["has_organic"] or stats["has_local"]) else None
    return result


# ── Stacked bar chart helper ──────────────────────────────────────────────────

def distribution_stacked_bar(
    df: pd.DataFrame,
    bucket_fn,
    buckets: list[str],
    colors: list[str],
    title: str,
) -> go.Figure:
    """
    100% stacked bar by month using a custom bucket function.
    bucket_fn(rank_capped) → bucket label string
    """
    d = df.copy()
    d["month"]  = d["checked_date"].dt.to_period("M").astype(str)
    d["bucket"] = d["rank_capped"].apply(bucket_fn)

    grouped = d.groupby(["month", "bucket"]).size().unstack(fill_value=0)
    for b in buckets:
        if b not in grouped.columns:
            grouped[b] = 0
    grouped = grouped[buckets]  # enforce order
    totals  = grouped.sum(axis=1)
    pct     = grouped.div(totals, axis=0) * 100

    months = pct.index.tolist()
    fig = go.Figure()
    for bucket, color in zip(buckets, colors):
        fig.add_trace(go.Bar(
            name=bucket,
            x=months,
            y=pct[bucket].round(1),
            marker_color=color,
            hovertemplate=f"{bucket}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Month",
        yaxis=dict(range=[0, 100], ticksuffix="%", title="% of Keywords"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=360,
        margin=dict(t=60),
    )
    return fig


def distribution_donuts(
    df_base: pd.DataFrame,
    df_latest: pd.DataFrame,
    bucket_fn,
    buckets: list[str],
    colors: list[str],
    label_base: str = "Feb 2024",
    label_latest: str = "Today",
) -> go.Figure:
    """
    Side-by-side donut charts: before (df_base) vs after (df_latest).
    Each df is a latest/earliest snapshot (one row per term).
    """
    from plotly.subplots import make_subplots

    def _counts(snap):
        d = snap.copy()
        d["bucket"] = d["rank_capped"].apply(bucket_fn)
        counts = d.groupby("bucket").size().reindex(buckets, fill_value=0)
        return counts.values.tolist()

    vals_base   = _counts(df_base)
    vals_latest = _counts(df_latest)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=[label_base, label_latest],
    )
    fig.add_trace(go.Pie(
        labels=buckets, values=vals_base,
        hole=0.55, marker_colors=colors,
        textinfo="percent", showlegend=True,
        name=label_base,
    ), row=1, col=1)
    fig.add_trace(go.Pie(
        labels=buckets, values=vals_latest,
        hole=0.55, marker_colors=colors,
        textinfo="percent", showlegend=False,
        name=label_latest,
    ), row=1, col=2)
    fig.update_layout(height=300, margin=dict(t=40, b=20))
    return fig


def stacked_bar_chart(df: pd.DataFrame, title: str) -> go.Figure:
    m = monthly_buckets(df)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Top 3", x=m["month"], y=m["pct_top3"].round(1),
        marker_color=GREEN,
        hovertemplate="%{y:.1f}% in Top 3<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Ranking (not top 3)", x=m["month"], y=m["pct_ranking"].round(1),
        marker_color=YELLOW,
        hovertemplate="%{y:.1f}% ranking, not top 3<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Not Ranking", x=m["month"], y=m["pct_not_ranking"].round(1),
        marker_color=RED,
        hovertemplate="%{y:.1f}% not ranking<extra></extra>",
    ))
    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Month",
        yaxis=dict(range=[0, 100], ticksuffix="%", title="% of Keywords"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=380,
        margin=dict(t=60),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("PRT Visualizer")
    st.markdown("---")

    if st.button("🔄 Refresh Data", use_container_width=True):
        from sync import sync_all
        status_box = st.empty()
        with st.spinner("Syncing with ProRankTracker…"):
            try:
                failures = sync_all(progress_callback=lambda msg: status_box.info(msg))
                st.cache_data.clear()
                if failures:
                    st.warning(f"Sync finished with {len(failures)} failure(s):")
                    for f in failures:
                        st.caption(f)
                else:
                    st.success("Sync complete — all clients updated.")
            except Exception as e:
                st.error(f"Sync failed: {e}")

    st.markdown("---")

    keyword_segment = st.radio(
        "Keywords",
        ["Active Only", "All Keywords", "Moonshots Only"],
        index=0,
        help=(
            "**Active** — ranked top 50 organic or appeared in local pack at any point.\n\n"
            "**Moonshot** — never hit either threshold; potential future opportunities."
        ),
    )

    st.markdown("---")

    if st.session_state.selected_client:
        if st.button("← All Clients", use_container_width=True):
            st.session_state.selected_client = None
            st.rerun()


# ── Load data ─────────────────────────────────────────────────────────────────

df_all_raw = get_all_history()
websites   = load_websites()

# Segment filter — computed once, applied everywhere
active_ids, moonshot_ids = classify_terms(df_all_raw)
n_active   = len(active_ids)
n_moonshot = len(moonshot_ids)

if keyword_segment == "Active Only":
    df_all = df_all_raw[df_all_raw["term_id"].isin(active_ids)]
elif keyword_segment == "Moonshots Only":
    df_all = df_all_raw[df_all_raw["term_id"].isin(moonshot_ids)]
else:
    df_all = df_all_raw


# ═══════════════════════════════════════════════════════════════════════════════
#  CLIENT DRILL-DOWN VIEW
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.selected_client:
    client = st.session_state.selected_client
    name   = display_name(client)

    st.header(f"📊 {name}")
    st.caption(client.get("url", ""))

    df_c_raw = get_client_history(client["id"])

    # Apply the same segment filter as the main dashboard
    if keyword_segment == "Active Only":
        df_c = df_c_raw[df_c_raw["term_id"].isin(active_ids)]
    elif keyword_segment == "Moonshots Only":
        df_c = df_c_raw[df_c_raw["term_id"].isin(moonshot_ids)]
    else:
        df_c = df_c_raw

    if df_c.empty:
        st.info("No ranking data in cache for this client. Click **Refresh Data** to sync.")
        st.stop()

    df_c_org = df_c[df_c["term_type"].isin(ORGANIC_TYPES)]
    df_c_loc = df_c[df_c["term_type"].isin(LOCAL_MAP_TYPES)]
    has_org  = not df_c_org.empty
    has_loc  = not df_c_loc.empty

    # ── Mini scorecards ───────────────────────────────────────────────────────
    metric_cols = []
    if has_org:
        l = latest_per_term(df_c_org)
        b = earliest_per_term(df_c_org)
        metric_cols += [
            ("Visibility Score", f"{l['vis_score'].mean():.0f}/100",
             f"{l['vis_score'].mean() - b['vis_score'].mean():+.0f} since Feb 2024",
             "normal" if l["vis_score"].mean() >= b["vis_score"].mean() else "inverse"),
            ("Organic Top 10", f"{(l['rank_capped'] <= 10).mean()*100:.0f}%",
             f"{(l['rank_capped'] <= 10).sum()}/{len(l)} keywords", "off"),
        ]
    if has_loc:
        l = latest_per_term(df_c_loc)
        b = earliest_per_term(df_c_loc)
        n3 = int((l["rank_capped"] <= 3).sum())
        d3 = n3 - int((b["rank_capped"] <= 3).sum())
        metric_cols += [
            ("Local Pack Top 3", f"{n3} / {len(l)}",
             f"{d3:+d} since Feb 2024",
             "normal" if d3 >= 0 else "inverse"),
        ]

    if metric_cols:
        cols = st.columns(len(metric_cols))
        for col, (lbl, val, delta, dc) in zip(cols, metric_cols):
            with col:
                st.metric(lbl, val, delta=delta, delta_color=dc)

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_names = (["All"] if has_org and has_loc else []) + \
                (["Organic"] if has_org else []) + \
                (["Local Pack / Maps"] if has_loc else [])

    for tab_obj, tab_name in zip(st.tabs(tab_names), tab_names):
        with tab_obj:
            df_tab = (
                df_c_org if tab_name == "Organic" else
                df_c_loc if tab_name == "Local Pack / Maps" else
                df_c
            )

            st.plotly_chart(
                stacked_bar_chart(df_tab, "Monthly Ranking Distribution"),
                use_container_width=True,
            )

            # Visibility score line (organic tabs only)
            if tab_name in ("All", "Organic") and has_org:
                vis_src = df_c_org if tab_name == "All" else df_tab
                avg_vis = (
                    vis_src.groupby("checked_date")["vis_score"]
                    .mean().reset_index()
                    .rename(columns={"vis_score": "avg_vis"})
                )
                fig_vis = px.line(
                    avg_vis, x="checked_date", y="avg_vis",
                    labels={"checked_date": "Date",
                            "avg_vis": "Avg. Visibility Score (0–100)"},
                    title="Organic Visibility Score Over Time",
                )
                fig_vis.update_yaxes(range=[0, 100])
                fig_vis.update_traces(line_color=BLUE, line_width=2)
                st.caption(
                    "ℹ️ Visibility Score is a weighted metric based on click-through rate "
                    "estimates. Position 1 = 100 pts, position 10 = 40 pts, not ranking = 0 pts. "
                    "Higher is always better."
                )
                st.plotly_chart(fig_vis, use_container_width=True)

            # Keyword summary table
            rows_kw = []
            for tid, sub in df_tab.groupby("term_id"):
                sub = sub.sort_values("checked_date")
                first, last = sub.iloc[0], sub.iloc[-1]
                chg = int(last["rank_capped"]) - int(first["rank_capped"])
                rows_kw.append({
                    "Keyword":      last["keyword"],
                    "Type":         last["term_type"],
                    "Start":        "NR" if first["rank_capped"] >= NOT_RANKED else str(first["rank_capped"]),
                    "Current":      "NR" if last["rank_capped"]  >= NOT_RANKED else str(last["rank_capped"]),
                    "Change":       f"{-chg:+d}" if chg != 0 else "—",
                    "Visibility":   f"{last['vis_score']}/100",
                    "_sort":        last["vis_score"],
                })
            if rows_kw:
                kw_df = (pd.DataFrame(rows_kw)
                           .sort_values("_sort", ascending=False)
                           .drop(columns=["_sort"]))
                st.dataframe(kw_df, use_container_width=True, hide_index=True)

    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

st.title("SEO Rankings Dashboard")

# Segment badge + moonshot note
_seg_labels = {
    "Active Only":    f"Showing: Active keywords only ({n_active:,} keywords)",
    "All Keywords":   f"Showing: All keywords ({n_active + n_moonshot:,} total)",
    "Moonshots Only": f"Showing: Moonshot keywords only ({n_moonshot:,} keywords)",
}
_seg_caption = _seg_labels[keyword_segment]
if keyword_segment == "Active Only" and n_moonshot > 0:
    _seg_caption += (
        f" · **{n_moonshot:,} moonshot keywords excluded** "
        f"(never ranked top 50 organic or appeared in local pack) — "
        f"toggle in sidebar to view them as future opportunities."
    )
st.caption(f"Data: {DATE_FROM} → {TODAY}  ·  {_seg_caption}")

if df_all.empty:
    st.warning("No data yet — click **Refresh Data** to sync.")
    st.stop()

df_organic = df_all[df_all["term_type"].isin(ORGANIC_TYPES)]
df_local   = df_all[df_all["term_type"].isin(LOCAL_MAP_TYPES)]

# ── Scorecard tabs ─────────────────────────────────────────────────────────────

try:
    tab_all, tab_org, tab_loc = st.tabs(["🌐  All", "🔍  Organic", "📍  Local Pack / Maps"])

    with tab_all:
        l_all = latest_per_term(df_all)
        b_all = earliest_per_term(df_all)
        vis_now  = l_all["vis_score"].mean()
        vis_then = b_all["vis_score"].mean()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Keywords Tracked", f"{df_all['term_id'].nunique():,}")
        with c2:
            st.metric(
                "Overall Visibility Score",
                f"{vis_now:.0f} / 100",
                delta=f"{vis_now - vis_then:+.0f} since Feb 2024",
                delta_color="normal" if vis_now >= vis_then else "inverse",
                help="Weighted score (0–100) across all keyword types. "
                     "Reflects estimated click-through rate. Higher is better.",
            )
        with c3:
            pct_t3 = (l_all["rank_capped"] <= 3).mean() * 100
            st.metric(
                "Keywords in Top 3",
                f"{pct_t3:.1f}%",
                delta=f"{int((l_all['rank_capped'] <= 3).sum()):,} keywords",
            )

    with tab_org:
        if df_organic.empty:
            st.info("No organic keyword data available.")
        else:
            l_org = latest_per_term(df_organic)
            b_org = earliest_per_term(df_organic)
            vis_org_now  = l_org["vis_score"].mean()
            vis_org_then = b_org["vis_score"].mean()
            pct10_now    = (l_org["rank_capped"] <= 10).mean() * 100
            pct10_then   = (b_org["rank_capped"] <= 10).mean() * 100

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Organic Visibility Score",
                    f"{vis_org_now:.0f} / 100",
                    delta=f"{vis_org_now - vis_org_then:+.0f} since Feb 2024",
                    delta_color="normal" if vis_org_now >= vis_org_then else "inverse",
                    help="Weighted score reflecting estimated CTR. "
                         "Position 1 = 100 pts · Position 10 = 40 pts · Not ranking = 0 pts.",
                )
            with c2:
                st.metric(
                    "Keywords in Top 10",
                    f"{pct10_now:.1f}%",
                    delta=f"{pct10_now - pct10_then:+.1f} pp since Feb 2024",
                    delta_color="normal" if pct10_now >= pct10_then else "inverse",
                )
            with c3:
                st.metric(
                    "Ranking Top 10",
                    f"{int((l_org['rank_capped'] <= 10).sum()):,} keywords",
                    delta=f"of {len(l_org):,} tracked",
                )

    with tab_loc:
        if df_local.empty:
            st.info("No local pack / maps data available.")
        else:
            l_loc = latest_per_term(df_local)
            b_loc = earliest_per_term(df_local)
            n3_now  = int((l_loc["rank_capped"] <= 3).sum())
            n3_then = int((b_loc["rank_capped"] <= 3).sum())
            pct3_now  = n3_now  / len(l_loc) * 100
            pct3_then = n3_then / len(b_loc) * 100

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Keywords in Top 3 (Map Pack A–C)",
                    f"{pct3_now:.1f}%",
                    delta=f"{pct3_now - pct3_then:+.1f} pp since Feb 2024",
                    delta_color="normal" if pct3_now >= pct3_then else "inverse",
                    help="A ranking of 1, 2, or 3 in the local pack (A/B/C) counts as a WIN.",
                )
            with c2:
                st.metric(
                    "Keywords in Top 3",
                    f"{n3_now:,}",
                    delta=f"{n3_now - n3_then:+d} since Feb 2024",
                    delta_color="normal" if n3_now >= n3_then else "inverse",
                )
            with c3:
                n_ranking = int((l_loc["rank_capped"] < NOT_RANKED).sum())
                st.metric(
                    "Ranking (Any Position)",
                    f"{n_ranking:,} keywords",
                    delta=f"of {len(l_loc):,} tracked",
                )

except Exception as _exc:
    st.error(f"Scorecard error: {_exc}")
    st.exception(_exc)

st.markdown("---")

# ── Main stacked bar chart ─────────────────────────────────────────────────────

try:
    chart_view = st.radio(
        "View by:",
        ["All", "Organic", "Local Pack / Maps"],
        horizontal=True,
        key="chart_view",
    )

    df_chart = (
        df_organic if chart_view == "Organic" else
        df_local   if chart_view == "Local Pack / Maps" else
        df_all
    )

    chart_title = {
        "All":               "Monthly Ranking Distribution — All Keywords",
        "Organic":           "Monthly Ranking Distribution — Organic",
        "Local Pack / Maps": "Monthly Ranking Distribution — Local Pack / Maps",
    }[chart_view]

    if df_chart.empty:
        st.info("No data for this filter.")
    else:
        st.plotly_chart(stacked_bar_chart(df_chart, chart_title), use_container_width=True)

    # Organic visibility score trend (shown under All or Organic view)
    if chart_view in ("All", "Organic") and not df_organic.empty:
        avg_vis = (
            df_organic.groupby("checked_date")["vis_score"]
            .mean().reset_index()
            .rename(columns={"vis_score": "avg_vis"})
        )
        fig_vis = px.line(
            avg_vis, x="checked_date", y="avg_vis",
            labels={"checked_date": "Date", "avg_vis": "Avg. Visibility Score"},
            title="Organic Visibility Score Over Time — All Clients",
        )
        fig_vis.update_yaxes(range=[0, 100])
        fig_vis.update_traces(line_color=BLUE, line_width=2)
        st.caption(
            "ℹ️ **Visibility Score** is a weighted metric based on estimated click-through rates. "
            "Position 1 = 100 pts, position 10 = 40 pts, not ranking = 0 pts. "
            "Higher is always better."
        )
        st.plotly_chart(fig_vis, use_container_width=True)

except Exception as _exc:
    st.error(f"Chart error: {_exc}")
    st.exception(_exc)

st.markdown("---")

# ── Rank distribution charts ──────────────────────────────────────────────────

try:
    st.subheader("Keyword Rank Distribution Over Time")
    st.caption(
        "Each bar shows the full distribution of where keywords were ranking that month. "
        "Green growing and red shrinking = the story you want to tell clients."
    )

    dist_tab_org, dist_tab_loc = st.tabs(["🔍 Organic Distribution", "📍 Local Pack Distribution"])

    with dist_tab_org:
        if df_organic.empty:
            st.info("No organic data available.")
        else:
            st.plotly_chart(
                distribution_stacked_bar(
                    df_organic, organic_bucket, ORG_BUCKETS, ORG_COLORS,
                    "Organic Ranking Distribution by Month",
                ),
                use_container_width=True,
            )

            b_org_snap = earliest_per_term(df_organic)
            l_org_snap = latest_per_term(df_organic)
            st.markdown("**Before vs. After Snapshot**")
            st.plotly_chart(
                distribution_donuts(
                    b_org_snap, l_org_snap,
                    organic_bucket, ORG_BUCKETS, ORG_COLORS,
                    label_base="Feb 2024", label_latest="Today",
                ),
                use_container_width=True,
            )

    with dist_tab_loc:
        if df_local.empty:
            st.info("No local pack data available.")
        else:
            st.plotly_chart(
                distribution_stacked_bar(
                    df_local, local_bucket, LOC_BUCKETS, LOC_COLORS,
                    "Local Pack Ranking Distribution by Month",
                ),
                use_container_width=True,
            )

            b_loc_snap = earliest_per_term(df_local)
            l_loc_snap = latest_per_term(df_local)
            st.markdown("**Before vs. After Snapshot**")
            st.plotly_chart(
                distribution_donuts(
                    b_loc_snap, l_loc_snap,
                    local_bucket, LOC_BUCKETS, LOC_COLORS,
                    label_base="Feb 2024", label_latest="Today",
                ),
                use_container_width=True,
            )

except Exception as _exc:
    st.error(f"Distribution chart error: {_exc}")
    st.exception(_exc)

st.markdown("---")

# ── Client grid ───────────────────────────────────────────────────────────────

try:
    st.subheader("Client Performance — click a card to drill in")

    if not websites:
        st.info("No clients found. Click **Refresh Data** to sync.")
    else:
        # Precompute all card stats in one pass, then sort
        c_stats  = all_client_stats(df_all)
        site_map = {s["id"]: s for s in websites}

        def _sort_key(wid):
            s = c_stats.get(int(wid))
            if s is None:
                return -1
            if s["has_local"]:
                return s["pct_top3"]
            if s["has_organic"]:
                return s["vis_now"]
            return -1

        sorted_ids = sorted(c_stats.keys(), key=_sort_key, reverse=True)
        tracked_ids = set(sorted_ids)
        for site in websites:
            if site["id"] not in tracked_ids and int(site["id"]) not in tracked_ids:
                sorted_ids.append(site["id"])

        COLS = 3
        cols = st.columns(COLS)

        for idx, wid in enumerate(sorted_ids):
            site  = site_map.get(wid) or site_map.get(str(wid))
            if site is None:
                continue
            stats = c_stats.get(wid) or c_stats.get(str(wid))
            name  = display_name(site)

            with cols[idx % COLS]:
                with st.container(border=True):
                    st.markdown(f"**{name}**")
                    if site.get("url") and site.get("url") != name:
                        st.caption(site["url"])

                    if not stats:
                        st.caption("⚪ No data")
                    else:
                        # Local pack block
                        if stats["has_local"]:
                            pct3  = stats["pct_top3"]
                            n3    = stats["n_top3"]
                            ntot  = stats["n_local_total"]
                            delta = stats["n_top3"] - stats["n_top3_then"]

                            if pct3 >= 50:
                                badge = "🟢"
                            elif pct3 > 0:
                                badge = "🟡"
                            else:
                                badge = "🔴"

                            delta_str = f"{'↑' if delta >= 0 else '↓'} {abs(delta):+d} keywords since Feb '24"
                            st.markdown(f"{badge} **Local Pack: {n3} / {ntot} keywords in Top 3 (A/B/C)**")
                            st.caption(f"{pct3:.0f}% in pack  ·  {delta_str}")

                        # Organic block — vis_score is organic-only, never mixed with local pack
                        if stats["has_organic"]:
                            vis    = stats["vis_now"]
                            vis_ch = stats["vis_change"]
                            arrow  = "↑" if vis_ch > 1 else ("↓" if vis_ch < -1 else "→")
                            color  = GREEN if vis_ch > 1 else (RED if vis_ch < -1 else YELLOW)
                            st.markdown(
                                f"🔍 **Organic Visibility Score: {vis:.0f}/100** "
                                f"<span style='color:{color}'>{arrow} {abs(vis_ch):.0f} pts since Feb '24</span>",
                                unsafe_allow_html=True,
                            )

                    if st.button("Details →", key=f"drill_{site['id']}", use_container_width=True):
                        st.session_state.selected_client = site
                        st.rerun()

except Exception as _exc:
    st.error(f"Client grid error: {_exc}")
    st.exception(_exc)
