"""
PRT Visualizer — Streamlit Dashboard
Run with:  streamlit run app.py

Shareable client report:  /?client=<slug>
"""

import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date

from db import (
    init_db, load_websites, load_history_for_website, display_name,
    load_snapshots, load_monthly_bucket_counts,
    load_monthly_organic_dist, load_monthly_local_dist,
    load_active_term_ids, load_monthly_avg_vis_score,
)

# ── Constants ─────────────────────────────────────────────────────────────────
DATE_FROM  = "2024-02-01"
TODAY      = date.today().isoformat()
NOT_RANKED = 101

ORGANIC_TYPES   = {"organic", "mobile"}
LOCAL_MAP_TYPES = {"snack_pack", "local_finder"}

GREEN  = "#22c55e"
YELLOW = "#eab308"
RED    = "#ef4444"
BLUE   = "#3b82f6"

ORG_BUCKETS = ["Top 3", "Top 10", "Top 30", "Top 100", "Not Ranking"]
ORG_COLORS  = ["#166534", "#86efac", "#eab308", "#f97316", "#ef4444"]

LOC_BUCKETS = ["Position 1 (A)", "Position 2 (B)", "Position 3 (C)", "Not in Pack"]
LOC_COLORS  = ["#166534", "#22c55e", "#86efac", "#ef4444"]

st.set_page_config(page_title="PRT Visualizer", page_icon="📈", layout="wide")
init_db()

if "selected_client" not in st.session_state:
    st.session_state.selected_client = None


# ── Rank / score helpers ──────────────────────────────────────────────────────

def cap_rank(r) -> int:
    try:
        ri = int(r)
    except (TypeError, ValueError):
        return NOT_RANKED
    if ri == 0 or ri >= NOT_RANKED:
        return NOT_RANKED
    return ri


def visibility_score(rank_capped: int) -> int:
    if rank_capped >= NOT_RANKED: return 0
    if rank_capped == 1:          return 100
    if rank_capped == 2:          return 85
    if rank_capped == 3:          return 75
    if rank_capped <= 5:          return 60
    if rank_capped <= 10:         return 40
    if rank_capped <= 20:         return 20
    if rank_capped <= 50:         return 5
    return 1


def bucket_rank(rank_capped: int) -> str:
    if rank_capped <= 3:         return "Top 3"
    if rank_capped < NOT_RANKED: return "Ranking"
    return "Not Ranking"


def organic_bucket(rank_capped: int) -> str:
    if rank_capped >= NOT_RANKED: return "Not Ranking"
    if rank_capped <= 3:          return "Top 3"
    if rank_capped <= 10:         return "Top 10"
    if rank_capped <= 30:         return "Top 30"
    return "Top 100"


def local_bucket(rank_capped: int) -> str:
    if rank_capped == 1: return "Position 1 (A)"
    if rank_capped == 2: return "Position 2 (B)"
    if rank_capped == 3: return "Position 3 (C)"
    return "Not in Pack"


def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rank_capped"] = df["rank"].apply(cap_rank)
    df["vis_score"]   = df["rank_capped"].apply(visibility_score)
    df["bucket"]      = df["rank_capped"].apply(bucket_rank)
    return df


def make_slug(name: str) -> str:
    """Convert a display name to a URL-safe slug."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# ── DataFrame utilities (drill-down view only) ────────────────────────────────

def latest_per_term(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("checked_date").groupby("term_id").last().reset_index()


def earliest_per_term(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("checked_date").groupby("term_id").first().reset_index()


def monthly_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly % breakdown — used only in drill-down / report stacked bar charts."""
    d = df.copy()
    d["month"] = d["checked_date"].dt.to_period("M").astype(str)
    grouped = d.groupby(["month", "bucket"]).size().unstack(fill_value=0)
    for col in ["Top 3", "Ranking", "Not Ranking"]:
        if col not in grouped.columns:
            grouped[col] = 0
    grouped["total"]           = grouped[["Top 3", "Ranking", "Not Ranking"]].sum(axis=1)
    grouped["pct_top3"]        = grouped["Top 3"]       / grouped["total"] * 100
    grouped["pct_ranking"]     = grouped["Ranking"]     / grouped["total"] * 100
    grouped["pct_not_ranking"] = grouped["Not Ranking"] / grouped["total"] * 100
    return grouped.reset_index().sort_values("month")


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_snapshots(date_from: str, date_to: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per term for most-recent and earliest observations. ~12K rows each."""
    latest_rows, baseline_rows = load_snapshots(date_from, date_to)
    _empty = pd.DataFrame(columns=[
        "website_id", "business_name", "website_url", "term_id", "keyword",
        "term_type", "rank", "checked_date", "rank_capped", "vis_score",
        "bucket", "client_name",
    ])
    if not latest_rows:
        return _empty, _empty

    def _make_df(rows):
        df = pd.DataFrame(rows)
        df["rank"]      = pd.to_numeric(df["rank"], errors="coerce")
        df["term_type"] = df["term_type"].fillna("").str.lower()
        df["client_name"] = df.apply(
            lambda r: r["business_name"].strip()
            if r["business_name"] and r["business_name"].strip()
            else r["website_url"], axis=1,
        )
        return prep_df(df)

    return _make_df(latest_rows), _make_df(baseline_rows)


@st.cache_data(show_spinner=False)
def get_monthly_counts(date_from: str, date_to: str) -> pd.DataFrame:
    rows = load_monthly_bucket_counts(date_from, date_to)
    if not rows:
        return pd.DataFrame(columns=["website_id", "term_type", "month", "bucket", "cnt"])
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_monthly_organic_dist_df(date_from: str, date_to: str) -> pd.DataFrame:
    rows = load_monthly_organic_dist(date_from, date_to)
    if not rows:
        return pd.DataFrame(columns=["month", "bucket", "cnt"])
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_monthly_local_dist_df(date_from: str, date_to: str) -> pd.DataFrame:
    rows = load_monthly_local_dist(date_from, date_to)
    if not rows:
        return pd.DataFrame(columns=["month", "bucket", "cnt"])
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_monthly_avg_vis_df(date_from: str, date_to: str, ranking_only: bool = False) -> pd.DataFrame:
    rows = load_monthly_avg_vis_score(date_from, date_to, ranking_only=ranking_only)
    if not rows:
        return pd.DataFrame(columns=["month", "avg_vis"])
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_active_ids(date_from: str, date_to: str) -> tuple[frozenset, frozenset]:
    return load_active_term_ids(date_from, date_to)


@st.cache_data(show_spinner=False)
def get_client_history(website_id: int, date_from: str, date_to: str) -> pd.DataFrame:
    """Full per-client history — small (hundreds of rows), safe to load."""
    rows = load_history_for_website(website_id, date_from, date_to)
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
def all_client_stats(df_latest: pd.DataFrame, df_baseline: pd.DataFrame) -> dict:
    """Per-client card stats from snapshot DataFrames (one row per term_id each)."""
    result = {}
    for wid in df_latest["website_id"].unique():
        l     = df_latest[df_latest["website_id"] == wid]
        b     = df_baseline[df_baseline["website_id"] == wid]
        l_org = l[l["term_type"].isin(ORGANIC_TYPES)]
        b_org = b[b["term_type"].isin(ORGANIC_TYPES)]
        l_loc = l[l["term_type"].isin(LOCAL_MAP_TYPES)]
        b_loc = b[b["term_type"].isin(LOCAL_MAP_TYPES)]

        stats: dict = {"has_organic": False, "has_local": False}

        if not l_org.empty:
            vis_now  = l_org["vis_score"].mean()
            vis_then = b_org["vis_score"].mean() if not b_org.empty else vis_now
            stats.update({
                "has_organic": True,
                "vis_now":     vis_now,
                "vis_then":    vis_then,
                "vis_change":  vis_now - vis_then,
                "pct_top10":   (l_org["rank_capped"] <= 10).mean() * 100,
            })

        if not l_loc.empty:
            n_top3      = int((l_loc["rank_capped"] <= 3).sum())
            n_top3_then = int((b_loc["rank_capped"] <= 3).sum()) if not b_loc.empty else 0
            n_total     = len(l_loc)
            stats.update({
                "has_local":     True,
                "n_top3":        n_top3,
                "n_top3_then":   n_top3_then,
                "n_local_total": n_total,
                "pct_top3":      n_top3 / n_total * 100 if n_total else 0,
            })

        result[wid] = stats if (stats["has_organic"] or stats["has_local"]) else None
    return result


# ── Chart helpers ─────────────────────────────────────────────────────────────

def stacked_bar_from_counts(
    counts_df: pd.DataFrame,
    term_types: set | None,
    title: str,
    wins_focus: bool = False,
) -> go.Figure:
    """
    100% stacked bar from pre-aggregated monthly bucket counts.
    wins_focus: if True, 'Not Ranking' rows are expected to already be excluded from
                counts_df; the trace is omitted and the Y-axis label reflects that.
    """
    d = counts_df.copy()
    if term_types:
        d = d[d["term_type"].isin(term_types)]
    grouped = d.groupby(["month", "bucket"])["cnt"].sum().unstack(fill_value=0)
    for b in ["Top 3", "Ranking"]:
        if b not in grouped.columns:
            grouped[b] = 0
    cols = ["Top 3", "Ranking"] if wins_focus else ["Top 3", "Ranking", "Not Ranking"]
    if not wins_focus and "Not Ranking" not in grouped.columns:
        grouped["Not Ranking"] = 0
    grouped = grouped[cols]
    totals  = grouped.sum(axis=1)
    pct     = grouped.div(totals, axis=0) * 100
    months  = pct.index.tolist()

    y_label = "% of Ranking Keywords" if wins_focus else "% of Keywords"
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Top 3", x=months, y=pct["Top 3"].round(1),
                         marker_color=GREEN,
                         hovertemplate="Top 3: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Bar(name="Ranking (not top 3)", x=months, y=pct["Ranking"].round(1),
                         marker_color=YELLOW,
                         hovertemplate="Ranking (not top 3): %{y:.1f}%<extra></extra>"))
    if not wins_focus:
        fig.add_trace(go.Bar(name="Not Ranking", x=months, y=pct["Not Ranking"].round(1),
                             marker_color=RED,
                             hovertemplate="Not Ranking: %{y:.1f}%<extra></extra>"))
    fig.update_layout(
        barmode="stack", title=title, xaxis_title="Month",
        yaxis=dict(range=[0, 100], ticksuffix="%", title=y_label),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=380, margin=dict(t=60),
    )
    return fig


def dist_bar_from_counts(
    dist_df: pd.DataFrame,
    buckets: list[str],
    colors: list[str],
    title: str,
    benchmark_pct: float | None = None,
    benchmark_label: str = "",
) -> go.Figure:
    """
    100% stacked bar from pre-aggregated monthly distribution counts.
    benchmark_pct: if set, draws a dashed reference line at that % value.
    """
    grouped = dist_df.groupby(["month", "bucket"])["cnt"].sum().unstack(fill_value=0)
    for b in buckets:
        if b not in grouped.columns:
            grouped[b] = 0
    grouped = grouped[buckets]
    totals  = grouped.sum(axis=1)
    pct     = grouped.div(totals, axis=0) * 100
    months  = pct.index.tolist()

    fig = go.Figure()
    for bucket, color in zip(buckets, colors):
        fig.add_trace(go.Bar(
            name=bucket, x=months, y=pct[bucket].round(1),
            marker_color=color,
            hovertemplate=f"{bucket}: %{{y:.1f}}%<extra></extra>",
        ))

    if benchmark_pct is not None:
        fig.add_hline(
            y=benchmark_pct,
            line_dash="dash",
            line_color="rgba(255,255,255,0.7)",
            line_width=2,
            annotation_text=benchmark_label or f"{benchmark_pct}%",
            annotation_position="top right",
            annotation_font_color="rgba(255,255,255,0.85)",
        )

    fig.update_layout(
        barmode="stack", title=title, xaxis_title="Month",
        yaxis=dict(range=[0, 100], ticksuffix="%", title="% of Keywords"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=360, margin=dict(t=60),
    )
    return fig


def distribution_donuts(
    df_base: pd.DataFrame,
    df_latest: pd.DataFrame,
    bucket_fn,
    buckets: list[str],
    colors: list[str],
    label_base: str = "Start of period",
    label_latest: str = "End of period",
) -> go.Figure:
    """
    Side-by-side donuts with count + percentage in each slice.
    Hover shows "N keywords (X%)" and each donut center shows total keyword count.
    """
    from plotly.subplots import make_subplots

    def _counts(snap):
        d = snap.copy()
        d["bucket"] = d["rank_capped"].apply(bucket_fn)
        return d.groupby("bucket").size().reindex(buckets, fill_value=0).values.tolist()

    vals_base   = _counts(df_base)
    vals_latest = _counts(df_latest)
    total_base   = sum(vals_base)
    total_latest = sum(vals_latest)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=[label_base, label_latest],
    )

    for col_idx, (vals, total, name) in enumerate(
        [(vals_base, total_base, label_base), (vals_latest, total_latest, label_latest)], 1
    ):
        fig.add_trace(go.Pie(
            labels=buckets,
            values=vals,
            hole=0.55,
            marker_colors=colors,
            texttemplate="%{percent:.0%}<br>(%{value:,})",
            hovertemplate="%{label}: %{value:,} keywords (%{percent:.1%})<extra></extra>",
            showlegend=(col_idx == 1),
            name=name,
        ), row=1, col=col_idx)

        # Total count in the donut hole
        x_center = 0.225 if col_idx == 1 else 0.775
        fig.add_annotation(
            text=f"<b>{total:,}</b><br><span style='font-size:11px'>keywords</span>",
            x=x_center, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=13),
        )

    fig.update_layout(height=320, margin=dict(t=50, b=20))
    return fig


def stacked_bar_chart(df: pd.DataFrame, title: str) -> go.Figure:
    """Per-client stacked bar — used in drill-down and report views (small DataFrames)."""
    m = monthly_buckets(df)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Top 3", x=m["month"], y=m["pct_top3"].round(1),
                         marker_color=GREEN,
                         hovertemplate="%{y:.1f}% in Top 3<extra></extra>"))
    fig.add_trace(go.Bar(name="Ranking (not top 3)", x=m["month"], y=m["pct_ranking"].round(1),
                         marker_color=YELLOW,
                         hovertemplate="%{y:.1f}% ranking, not top 3<extra></extra>"))
    fig.add_trace(go.Bar(name="Not Ranking", x=m["month"], y=m["pct_not_ranking"].round(1),
                         marker_color=RED,
                         hovertemplate="%{y:.1f}% not ranking<extra></extra>"))
    fig.update_layout(
        barmode="stack", title=title, xaxis_title="Month",
        yaxis=dict(range=[0, 100], ticksuffix="%", title="% of Keywords"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=380, margin=dict(t=60),
    )
    return fig


# ── Available months for the date slider ──────────────────────────────────────

_all_months = pd.date_range(start=DATE_FROM, end=TODAY, freq="MS").strftime("%Y-%m").tolist()
if not _all_months:
    _all_months = [TODAY[:7]]


# ═══════════════════════════════════════════════════════════════════════════════
#  SHAREABLE CLIENT REPORT  (/?client=<slug>)
# ═══════════════════════════════════════════════════════════════════════════════

_client_slug = st.query_params.get("client", "")

if _client_slug:
    # Collapse the sidebar entirely in report mode
    st.markdown("""
    <style>
    [data-testid="stSidebar"]      { display: none; }
    [data-testid="collapsedControl"]{ display: none; }
    </style>
    """, unsafe_allow_html=True)

    websites = load_websites()
    matched  = None
    for site in websites:
        site_name = display_name(site)
        if (make_slug(site_name) == _client_slug
                or make_slug(site.get("url", "")) == _client_slug):
            matched = site
            break

    if matched is None:
        st.error(f"Client not found: `{_client_slug}`")
        st.markdown("[← Back to Dashboard](?)")
        st.stop()

    client_name = display_name(matched)

    # ── Report header ──────────────────────────────────────────────────────────
    st.markdown(
        f"<h1 style='margin:0.5rem 0 0'>📊 {client_name}</h1>"
        f"<p style='color:gray;margin:0.2rem 0 0.5rem'>{matched.get('url', '')}</p>",
        unsafe_allow_html=True,
    )
    st.caption(f"SEO Rankings Report  ·  {DATE_FROM} → {TODAY}  ·  [← Back to Dashboard](?)")
    st.markdown("---")

    df_c_raw = get_client_history(matched["id"], DATE_FROM, TODAY)

    if df_c_raw.empty:
        st.info("No ranking data available for this client.")
        st.stop()

    # Reports are always in Wins Focus — client-facing, lead with positive story
    _report_df = df_c_raw[df_c_raw["rank_capped"] < NOT_RANKED]
    df_c_org = _report_df[_report_df["term_type"].isin(ORGANIC_TYPES)]
    df_c_loc = _report_df[_report_df["term_type"].isin(LOCAL_MAP_TYPES)]
    has_org  = not df_c_org.empty
    has_loc  = not df_c_loc.empty

    # ── Scorecards ────────────────────────────────────────────────────────────
    metric_cols = []
    if has_org:
        l_o = latest_per_term(df_c_org)
        b_o = earliest_per_term(df_c_org)
        metric_cols += [
            ("Visibility Score",
             f"{l_o['vis_score'].mean():.0f}/100",
             f"{l_o['vis_score'].mean() - b_o['vis_score'].mean():+.0f} since Feb 2024",
             "normal" if l_o["vis_score"].mean() >= b_o["vis_score"].mean() else "inverse"),
            ("Organic Top 10",
             f"{(l_o['rank_capped'] <= 10).mean()*100:.0f}%",
             f"{(l_o['rank_capped'] <= 10).sum()}/{len(l_o)} keywords", "off"),
        ]
    if has_loc:
        l_lc = latest_per_term(df_c_loc)
        b_lc = earliest_per_term(df_c_loc)
        n3   = int((l_lc["rank_capped"] <= 3).sum())
        d3   = n3 - int((b_lc["rank_capped"] <= 3).sum())
        metric_cols += [
            ("Local Pack Top 3",
             f"{n3} / {len(l_lc)}",
             f"{d3:+d} since Feb 2024",
             "normal" if d3 >= 0 else "inverse"),
        ]
    if metric_cols:
        for col, (lbl, val, delta, dc) in zip(st.columns(len(metric_cols)), metric_cols):
            with col:
                st.metric(lbl, val, delta=delta, delta_color=dc)

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    if has_org:
        st.subheader("Organic Rankings Over Time")
        st.plotly_chart(
            stacked_bar_chart(df_c_org, "Organic Ranking Distribution by Month"),
            use_container_width=True,
        )
        avg_vis_c = (
            df_c_org.groupby("checked_date")["vis_score"]
            .mean().reset_index().rename(columns={"vis_score": "avg_vis"})
        )
        fig_vis_c = px.line(avg_vis_c, x="checked_date", y="avg_vis",
                            labels={"checked_date": "Date",
                                    "avg_vis": "Avg. Visibility Score (0–100)"},
                            title="Organic Visibility Score Over Time")
        fig_vis_c.update_yaxes(range=[0, 100])
        fig_vis_c.update_traces(line_color=BLUE, line_width=2)
        st.plotly_chart(fig_vis_c, use_container_width=True)
        st.caption(
            "Visibility Score weights positions by estimated CTR. "
            "Position 1 = 100 pts · Position 10 = 40 pts · Not ranking = 0 pts."
        )

    if has_loc:
        st.subheader("Local Pack Rankings Over Time")
        st.plotly_chart(
            stacked_bar_chart(df_c_loc, "Local Pack Ranking Distribution by Month"),
            use_container_width=True,
        )
        st.info(
            "**Industry context:** Appearing in the local pack top 3 (A/B/C) for 10–20% of "
            "tracked keywords is competitive for most local businesses. Many markets see 5–10% "
            "as a strong starting point."
        )

    # ── Before / after donuts ─────────────────────────────────────────────────
    if has_org or has_loc:
        st.markdown("---")
        st.subheader("Before vs. After Snapshot")

    if has_org:
        st.markdown("**Organic Distribution**")
        st.plotly_chart(
            distribution_donuts(
                earliest_per_term(df_c_org), latest_per_term(df_c_org),
                organic_bucket, ORG_BUCKETS, ORG_COLORS,
                label_base="Feb 2024", label_latest="Today",
            ),
            use_container_width=True,
        )

    if has_loc:
        st.markdown("**Local Pack Distribution**")
        st.plotly_chart(
            distribution_donuts(
                earliest_per_term(df_c_loc), latest_per_term(df_c_loc),
                local_bucket, LOC_BUCKETS, LOC_COLORS,
                label_base="Feb 2024", label_latest="Today",
            ),
            use_container_width=True,
        )

    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  NORMAL DASHBOARD FLOW
# ═══════════════════════════════════════════════════════════════════════════════

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

    # ── Date range slider ──────────────────────────────────────────────────────
    st.markdown("**Date Range**")
    if len(_all_months) >= 2:
        sel_from, sel_to = st.select_slider(
            "date_range",
            options=_all_months,
            value=(_all_months[0], _all_months[-1]),
            format_func=lambda m: pd.to_datetime(m + "-01").strftime("%b %Y"),
            label_visibility="collapsed",
        )
    else:
        sel_from = sel_to = _all_months[0]

    filter_from = sel_from + "-01"
    filter_to   = sel_to   + "-31"   # generous upper bound works with SQLite string BETWEEN

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

    show_mode = st.radio(
        "Show Mode",
        ["Wins Focus", "Full Picture"],
        index=0,
        help=(
            "**Wins Focus** — hides 'Not Ranking' from charts; metrics reflect "
            "actively ranking keywords only.\n\n"
            "**Full Picture** — shows everything including not-ranking keywords."
        ),
    )

    st.markdown("---")

    if st.session_state.selected_client:
        if st.button("← All Clients", use_container_width=True):
            st.session_state.selected_client = None
            st.rerun()


# ── Load data (all pre-aggregated — no full history loaded here) ──────────────

df_latest, df_baseline = get_snapshots(filter_from, filter_to)
websites               = load_websites()
active_ids, all_ids    = get_active_ids(filter_from, filter_to)

n_active     = len(active_ids)
moonshot_ids = all_ids - active_ids
n_moonshot   = len(moonshot_ids)

if keyword_segment == "Active Only":
    df_snap = df_latest[df_latest["term_id"].isin(active_ids)]
    df_base = df_baseline[df_baseline["term_id"].isin(active_ids)]
elif keyword_segment == "Moonshots Only":
    df_snap = df_latest[df_latest["term_id"].isin(moonshot_ids)]
    df_base = df_baseline[df_baseline["term_id"].isin(moonshot_ids)]
else:
    df_snap = df_latest
    df_base = df_baseline

# Wins Focus: strip not-ranking observations so every downstream view reflects
# only keywords that are actively ranking.
_wins = show_mode == "Wins Focus"
if _wins:
    df_snap = df_snap[df_snap["rank_capped"] < NOT_RANKED]
    df_base = df_base[df_base["rank_capped"] < NOT_RANKED]

df_organic  = df_snap[df_snap["term_type"].isin(ORGANIC_TYPES)]
df_local    = df_snap[df_snap["term_type"].isin(LOCAL_MAP_TYPES)]
df_org_base = df_base[df_base["term_type"].isin(ORGANIC_TYPES)]
df_loc_base = df_base[df_base["term_type"].isin(LOCAL_MAP_TYPES)]

monthly_counts   = get_monthly_counts(filter_from, filter_to)
monthly_org_dist = get_monthly_organic_dist_df(filter_from, filter_to)
monthly_loc_dist = get_monthly_local_dist_df(filter_from, filter_to)
monthly_avg_vis  = get_monthly_avg_vis_df(filter_from, filter_to, ranking_only=_wins)

# Wins Focus: drop not-ranking rows from monthly aggregates before charting
_mc  = monthly_counts[monthly_counts["bucket"] != "Not Ranking"]  if _wins else monthly_counts
_mod = monthly_org_dist[monthly_org_dist["bucket"] != "Not Ranking"] if _wins else monthly_org_dist
_mld = monthly_loc_dist[monthly_loc_dist["bucket"] != "Not in Pack"] if _wins else monthly_loc_dist

# Bucket lists for distribution charts — drop the "nothing" bucket in Wins Focus
_org_buckets = ORG_BUCKETS[:-1] if _wins else ORG_BUCKETS   # drop "Not Ranking"
_org_colors  = ORG_COLORS[:-1]  if _wins else ORG_COLORS
_loc_buckets = LOC_BUCKETS[:-1] if _wins else LOC_BUCKETS   # drop "Not in Pack"
_loc_colors  = LOC_COLORS[:-1]  if _wins else LOC_COLORS


# ═══════════════════════════════════════════════════════════════════════════════
#  CLIENT DRILL-DOWN VIEW
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.selected_client:
    client = st.session_state.selected_client
    name   = display_name(client)

    st.header(f"📊 {name}")
    st.caption(client.get("url", ""))

    slug      = make_slug(name)
    share_url = f"?client={slug}"
    st.markdown(
        f"[🔗 Shareable report for this client]({share_url}) "
        f"— send this link to share only this client's data.",
    )

    df_c_raw = get_client_history(client["id"], filter_from, filter_to)

    if keyword_segment == "Active Only":
        df_c = df_c_raw[df_c_raw["term_id"].isin(active_ids)]
    elif keyword_segment == "Moonshots Only":
        df_c = df_c_raw[df_c_raw["term_id"].isin(moonshot_ids)]
    else:
        df_c = df_c_raw

    if _wins:
        df_c = df_c[df_c["rank_capped"] < NOT_RANKED]

    if df_c.empty:
        st.info("No ranking data in cache for this client. Click **Refresh Data** to sync.")
        st.stop()

    df_c_org = df_c[df_c["term_type"].isin(ORGANIC_TYPES)]
    df_c_loc = df_c[df_c["term_type"].isin(LOCAL_MAP_TYPES)]
    has_org  = not df_c_org.empty
    has_loc  = not df_c_loc.empty

    metric_cols = []
    if has_org:
        l = latest_per_term(df_c_org)
        b = earliest_per_term(df_c_org)
        metric_cols += [
            ("Visibility Score", f"{l['vis_score'].mean():.0f}/100",
             f"{l['vis_score'].mean() - b['vis_score'].mean():+.0f} vs start of period",
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
             f"{d3:+d} vs start of period",
             "normal" if d3 >= 0 else "inverse"),
        ]
    if metric_cols:
        for col, (lbl, val, delta, dc) in zip(st.columns(len(metric_cols)), metric_cols):
            with col:
                st.metric(lbl, val, delta=delta, delta_color=dc)

    st.markdown("---")

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

            if tab_name in ("All", "Organic") and has_org:
                vis_src = df_c_org if tab_name == "All" else df_tab
                avg_vis = (
                    vis_src.groupby("checked_date")["vis_score"]
                    .mean().reset_index()
                    .rename(columns={"vis_score": "avg_vis"})
                )
                fig_vis = px.line(avg_vis, x="checked_date", y="avg_vis",
                                  labels={"checked_date": "Date",
                                          "avg_vis": "Avg. Visibility Score (0–100)"},
                                  title="Organic Visibility Score Over Time")
                fig_vis.update_yaxes(range=[0, 100])
                fig_vis.update_traces(line_color=BLUE, line_width=2)
                st.caption(
                    "ℹ️ Visibility Score weights positions by estimated CTR. "
                    "Position 1 = 100 pts · Position 10 = 40 pts · Not ranking = 0 pts."
                )
                st.plotly_chart(fig_vis, use_container_width=True)

            rows_kw = []
            for tid, sub in df_tab.groupby("term_id"):
                sub = sub.sort_values("checked_date")
                first, last = sub.iloc[0], sub.iloc[-1]
                chg = int(last["rank_capped"]) - int(first["rank_capped"])
                rows_kw.append({
                    "Keyword":    last["keyword"],
                    "Type":       last["term_type"],
                    "Start":      "NR" if first["rank_capped"] >= NOT_RANKED else str(first["rank_capped"]),
                    "Current":    "NR" if last["rank_capped"]  >= NOT_RANKED else str(last["rank_capped"]),
                    "Change":     f"{-chg:+d}" if chg != 0 else "—",
                    "Visibility": f"{last['vis_score']}/100",
                    "_sort":      last["vis_score"],
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


if df_snap.empty:
    st.warning("No data yet — click **Refresh Data** to sync.")
    st.stop()

# ── Scorecard tabs ─────────────────────────────────────────────────────────────

try:
    tab_all, tab_org, tab_loc = st.tabs(["🌐  All", "🔍  Organic", "📍  Local Pack / Maps"])

    with tab_all:
        vis_now  = df_snap["vis_score"].mean()
        vis_then = df_base["vis_score"].mean() if not df_base.empty else vis_now
        c1, c2, c3 = st.columns(3)
        with c1:
            _kw_label = "Actively Ranking Keywords" if _wins else "Total Keywords Tracked"
            st.metric(_kw_label, f"{df_snap['term_id'].nunique():,}")
        with c2:
            st.metric(
                "Overall Visibility Score",
                f"{vis_now:.0f} / 100",
                delta=f"{vis_now - vis_then:+.0f} pts vs start of period",
                delta_color="normal" if vis_now >= vis_then else "inverse",
                help="Weighted score (0–100) across all keyword types. Higher is better.",
            )
        with c3:
            pct_t3 = (df_snap["rank_capped"] <= 3).mean() * 100
            st.metric(
                "Keywords in Top 3",
                f"{pct_t3:.1f}%",
                delta=f"{int((df_snap['rank_capped'] <= 3).sum()):,} keywords",
            )

    with tab_org:
        if df_organic.empty:
            st.info("No organic keyword data available.")
        else:
            vis_org_now  = df_organic["vis_score"].mean()
            vis_org_then = df_org_base["vis_score"].mean() if not df_org_base.empty else vis_org_now
            pct10_now    = (df_organic["rank_capped"] <= 10).mean() * 100
            pct10_then   = (df_org_base["rank_capped"] <= 10).mean() * 100 if not df_org_base.empty else pct10_now
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Organic Visibility Score",
                    f"{vis_org_now:.0f} / 100",
                    delta=f"{vis_org_now - vis_org_then:+.0f} pts vs start of period",
                    delta_color="normal" if vis_org_now >= vis_org_then else "inverse",
                    help="Weighted score reflecting estimated CTR. Position 1 = 100 · Not ranking = 0.",
                )
            with c2:
                st.metric(
                    "Keywords in Top 10",
                    f"{pct10_now:.1f}%",
                    delta=f"{pct10_now - pct10_then:+.1f} pp vs start of period",
                    delta_color="normal" if pct10_now >= pct10_then else "inverse",
                )
            with c3:
                st.metric(
                    "Ranking Top 10",
                    f"{int((df_organic['rank_capped'] <= 10).sum()):,} keywords",
                    delta=f"of {len(df_organic):,} tracked",
                )

    with tab_loc:
        if df_local.empty:
            st.info("No local pack / maps data available.")
        else:
            n3_now    = int((df_local["rank_capped"] <= 3).sum())
            n3_then   = int((df_loc_base["rank_capped"] <= 3).sum()) if not df_loc_base.empty else n3_now
            pct3_now  = n3_now / len(df_local) * 100
            pct3_then = n3_then / len(df_loc_base) * 100 if not df_loc_base.empty else pct3_now
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Keywords in Top 3 (Map Pack A–C)",
                    f"{pct3_now:.1f}%",
                    delta=f"{pct3_now - pct3_then:+.1f} pp vs start of period",
                    delta_color="normal" if pct3_now >= pct3_then else "inverse",
                    help="Ranking 1, 2, or 3 in the local pack (A/B/C) = a WIN.",
                )
            with c2:
                st.metric(
                    "Keywords in Top 3",
                    f"{n3_now:,}",
                    delta=f"{n3_now - n3_then:+d} vs start of period",
                    delta_color="normal" if n3_now >= n3_then else "inverse",
                )
            with c3:
                n_ranking = int((df_local["rank_capped"] < NOT_RANKED).sum())
                st.metric(
                    "Ranking (Any Position)",
                    f"{n_ranking:,} keywords",
                    delta=f"of {len(df_local):,} tracked",
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
    _suffix = " — Ranking Only" if _wins else ""
    chart_title = {
        "All":               f"Monthly Ranking Distribution — All Keywords{_suffix}",
        "Organic":           f"Monthly Ranking Distribution — Organic{_suffix}",
        "Local Pack / Maps": f"Monthly Ranking Distribution — Local Pack / Maps{_suffix}",
    }[chart_view]
    term_types_filter = (
        ORGANIC_TYPES   if chart_view == "Organic" else
        LOCAL_MAP_TYPES if chart_view == "Local Pack / Maps" else
        None
    )

    if _mc.empty:
        st.info("No data for the selected period.")
    else:
        st.plotly_chart(
            stacked_bar_from_counts(_mc, term_types_filter, chart_title, wins_focus=_wins),
            use_container_width=True,
        )

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

    _org_title = f"Organic Ranking Distribution by Month{_suffix}"
    _loc_title = f"Local Pack Ranking Distribution by Month{_suffix}"

    with dist_tab_org:
        if _mod.empty:
            st.info("No organic data available.")
        else:
            st.plotly_chart(
                dist_bar_from_counts(_mod, _org_buckets, _org_colors, _org_title),
                use_container_width=True,
            )
            if not df_org_base.empty and not df_organic.empty:
                st.markdown("**Before vs. After Snapshot**")
                st.plotly_chart(
                    distribution_donuts(
                        df_org_base, df_organic,
                        organic_bucket, _org_buckets, _org_colors,
                        label_base="Start of period", label_latest="End of period",
                    ),
                    use_container_width=True,
                )

    with dist_tab_loc:
        if _mld.empty:
            st.info("No local pack data available.")
        else:
            st.plotly_chart(
                dist_bar_from_counts(
                    _mld, _loc_buckets, _loc_colors, _loc_title,
                    benchmark_pct=15 if not _wins else None,
                    benchmark_label="~15% industry avg (top 3 combined)",
                ),
                use_container_width=True,
            )
            if not _wins:
                st.info(
                    "**Industry context:** The dashed line marks ~15% — appearing in the local pack "
                    "top 3 (A/B/C) for 10–20% of keywords is **competitive for most local businesses**. "
                    "Many new clients start well below 5% and build from there."
                )
            if not df_loc_base.empty and not df_local.empty:
                st.markdown("**Before vs. After Snapshot**")
                st.plotly_chart(
                    distribution_donuts(
                        df_loc_base, df_local,
                        local_bucket, _loc_buckets, _loc_colors,
                        label_base="Start of period", label_latest="End of period",
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
        # ── Search box ────────────────────────────────────────────────────────
        client_search = st.text_input(
            "Search clients",
            placeholder="Filter by name or URL…",
            label_visibility="collapsed",
        )

        c_stats  = all_client_stats(df_snap, df_base)
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

        # Apply search filter
        if client_search.strip():
            q = client_search.strip().lower()
            sorted_ids = [
                wid for wid in sorted_ids
                if (s := site_map.get(wid) or site_map.get(str(wid))) is not None
                and (q in display_name(s).lower() or q in s.get("url", "").lower())
            ]
            if not sorted_ids:
                st.info(f'No clients match "{client_search}".')

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
                        if stats["has_local"]:
                            pct3  = stats["pct_top3"]
                            n3    = stats["n_top3"]
                            ntot  = stats["n_local_total"]
                            delta = stats["n_top3"] - stats["n_top3_then"]
                            badge = "🟢" if pct3 >= 50 else ("🟡" if pct3 > 0 else "🔴")
                            delta_str = f"{'↑' if delta >= 0 else '↓'} {abs(delta):+d} keywords since Feb '24"
                            st.markdown(
                                f"{badge} **Local Pack: {n3} / {ntot} keywords in Top 3 (A/B/C)**"
                            )
                            st.caption(f"{pct3:.0f}% in pack  ·  {delta_str}")

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
