"""
Module 7 — Streamlit Dashboard
Premium dark-themed dashboard with donut chart, fan charts, event waterfall,
heatmap, rebalancing panel, efficient frontier, and PDF export.
"""

import io
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEFAULT_MC_PATHS,
    DEFAULT_N_AGENTS,
    FX_API_URL,
    GEOPOLITICAL_EVENTS,
    PORTFOLIO,
    REPORTS_DIR,
    RISK_PROFILE,
    SECTOR_COLOURS,
    TICKERS,
)

logger = logging.getLogger(__name__)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_ingestion() -> Dict[str, Any]:
    from modules.data_ingestion import run_ingestion
    return run_ingestion()


@st.cache_data(ttl=1800, show_spinner=False)
def cached_sentiment(news_payload: Dict[str, Any]) -> Dict[str, Any]:
    from modules.sentiment_engine import run_sentiment_analysis
    return run_sentiment_analysis(news_payload)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_event_generation(horizon: int, enabled_events: tuple[str, ...]) -> Dict[str, Any]:
    from modules.event_generator import run_event_generation
    return run_event_generation(
        n_days=horizon,
        enabled_events=list(enabled_events) if enabled_events else None,
        seed=42,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def cached_forecasting(
    ohlcv_data: Dict[str, Any],
    composite_scores: Dict[str, float],
    macro_data: Dict[str, Any],
    rolling_sentiment: Dict[str, Any],
    gdelt_data: Any,
    horizon: int,
    dry_run: bool,
) -> Dict[str, Any]:
    from modules.ml_forecasting import run_forecasting
    return run_forecasting(
        ohlcv_data=ohlcv_data,
        sentiment_scores=composite_scores,
        macro_data=macro_data,
        rolling_sentiment=rolling_sentiment,
        gdelt_data=gdelt_data,
        horizons=[horizon],
        dry_run=dry_run,
        reuse_models=True,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def cached_agent_simulation(
    ohlcv_data: Dict[str, Any],
    composite_scores: Dict[str, float],
    event_timelines: Dict[str, Any],
    rolling_sentiment: Dict[str, Any],
    macro_data: Dict[str, Any],
    n_agents: int,
    horizon: int,
    dry_run: bool,
) -> Dict[str, Any]:
    from modules.agent_simulation import run_agent_simulation
    return run_agent_simulation(
        ohlcv_data=ohlcv_data,
        sentiment_scores=composite_scores,
        event_timelines=event_timelines,
        rolling_sentiment=rolling_sentiment,
        macro_data=macro_data,
        n_agents=n_agents,
        n_steps=horizon,
        dry_run=dry_run,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def cached_analytics(
    ohlcv_data: Dict[str, Any],
    forecast_data: Dict[str, Any],
    agent_data: Dict[str, Any],
    _event_data: Dict[str, Any],
    fx_rate: float,
    n_mc_paths: int,
    horizon: int,
    dry_run: bool,
) -> Dict[str, Any]:
    from modules.portfolio_analytics import run_portfolio_analytics
    return run_portfolio_analytics(
        ohlcv_data=ohlcv_data,
        forecast_data=forecast_data,
        agent_sim_data=agent_data,
        event_data=_event_data,
        fx_rate=fx_rate,
        n_paths=n_mc_paths,
        horizons=[horizon],
        dry_run=dry_run,
    )

# ---------------------------------------------------------------------------
# Page config & theming
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio World Simulation",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium dark theme CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-0: #08111f;
    --bg-1: #0f172a;
    --bg-2: #111c30;
    --panel: rgba(15, 23, 42, 0.82);
    --panel-strong: rgba(15, 23, 42, 0.96);
    --border: rgba(148, 163, 184, 0.14);
    --line: rgba(59, 130, 246, 0.28);
    --text: #e6edf7;
    --muted: #96a8c3;
    --accent: #58a6ff;
    --accent-2: #1d4ed8;
    --positive: #4ade80;
    --negative: #fb7185;
    --warning: #fbbf24;
    --shadow: 0 24px 70px rgba(2, 8, 23, 0.45);
}

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

body {
    background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.12), transparent 30%),
        radial-gradient(circle at top right, rgba(99, 102, 241, 0.10), transparent 24%),
        linear-gradient(180deg, var(--bg-0) 0%, #09101d 100%);
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(circle at 12% 10%, rgba(37, 99, 235, 0.18), transparent 18%),
        radial-gradient(circle at 88% 8%, rgba(56, 189, 248, 0.10), transparent 16%),
        linear-gradient(180deg, #08111f 0%, #0b1324 100%);
}

/* Header gradient bar */
.stApp > header {
    background: linear-gradient(90deg, rgba(8,17,31,0.75) 0%, rgba(15,23,42,0.75) 50%, rgba(11,19,36,0.75) 100%);
    backdrop-filter: blur(10px);
}

/* Main area */
.main .block-container {
    padding-top: 1.4rem;
    max-width: 1480px;
    padding-bottom: 4rem;
}

div[data-testid="stVerticalBlock"] > div:has(> .section-shell) {
    margin-bottom: 1rem;
}

/* Hero */
.hero-shell {
    position: relative;
    overflow: hidden;
    background:
        linear-gradient(135deg, rgba(13, 27, 45, 0.95) 0%, rgba(17, 28, 48, 0.94) 55%, rgba(10, 20, 36, 0.94) 100%);
    border: 1px solid var(--border);
    border-radius: 26px;
    padding: 1.5rem 1.6rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.2rem;
}
.hero-shell:before {
    content: "";
    position: absolute;
    inset: auto -10% -55% 38%;
    height: 250px;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.22), transparent 60%);
    pointer-events: none;
}
.hero-kicker {
    color: #8bb8ff;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.72rem;
    font-weight: 700;
    margin-bottom: 0.65rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text);
    font-size: clamp(2rem, 3vw, 3.2rem);
    line-height: 1.02;
    font-weight: 700;
    margin-bottom: 0.7rem;
    letter-spacing: -0.04em;
}
.hero-subtitle {
    max-width: 860px;
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.7;
}

/* Cards */
.metric-card {
    background: linear-gradient(180deg, rgba(17, 28, 48, 0.88) 0%, rgba(10, 18, 32, 0.92) 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.15rem 1.2rem;
    margin: 0.3rem 0;
    box-shadow: 0 18px 40px rgba(2, 8, 23, 0.28);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    min-height: 140px;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(88, 166, 255, 0.28);
    box-shadow: 0 24px 50px rgba(2, 8, 23, 0.38);
}
.metric-card h3 {
    color: var(--muted);
    font-size: 0.76rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.metric-card .value {
    color: var(--text);
    font-size: 1.9rem;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.04em;
}
.metric-card .delta-positive { color: var(--positive); }
.metric-card .delta-negative { color: var(--negative); }
.metric-card .delta-warning { color: var(--warning); }
.metric-card .micro-note {
    color: var(--muted);
    font-size: 0.83rem;
    margin-top: 0.8rem;
    line-height: 1.45;
}

.mini-stat-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.9rem;
    margin-top: 1rem;
}

.mini-stat {
    background: rgba(8, 15, 28, 0.56);
    border: 1px solid rgba(148, 163, 184, 0.08);
    border-radius: 16px;
    padding: 0.85rem 1rem;
}

.mini-stat .label {
    color: var(--muted);
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.mini-stat .number {
    color: var(--text);
    font-size: 1.2rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
}

/* Section headers */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 1rem;
    padding-bottom: 0;
    border-bottom: none;
    display: inline-block;
    letter-spacing: -0.03em;
}

.section-shell {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.76) 0%, rgba(9, 16, 29, 0.86) 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 1.05rem 1.1rem 0.7rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}

.section-subtitle {
    color: var(--muted);
    font-size: 0.92rem;
    margin: -0.25rem 0 1rem;
    line-height: 1.6;
}

.dataframe-shell {
    background: rgba(8, 15, 28, 0.42);
    border-radius: 18px;
    padding: 0.45rem;
}

/* Table styling */
table {
    border-collapse: collapse;
    width: 100%;
}
th {
    background: rgba(17, 28, 48, 0.95);
    color: var(--muted);
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
td, th {
    padding: 0.6rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.08);
}
tr:hover td {
    background: rgba(59,130,246,0.05);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(8,17,31,0.98) 0%, rgba(15,23,42,0.98) 55%, rgba(17,28,48,0.98) 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.08);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--muted);
}
section[data-testid="stSidebar"] [data-baseweb="slider"] {
    padding-top: 0.3rem;
    padding-bottom: 0.8rem;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(8, 15, 28, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 14px;
}
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: white;
    border: none;
    border-radius: 14px;
    font-weight: 700;
    letter-spacing: 0.02em;
    box-shadow: 0 18px 30px rgba(29, 78, 216, 0.28);
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
}
section[data-testid="stSidebar"] [data-testid="stCheckbox"] {
    background: rgba(8, 15, 28, 0.44);
    border: 1px solid rgba(148, 163, 184, 0.07);
    border-radius: 12px;
    padding: 0.35rem 0.5rem;
    margin-bottom: 0.35rem;
}
.sidebar-shell {
    background: linear-gradient(180deg, rgba(9,16,29,0.7) 0%, rgba(15,23,42,0.72) 100%);
    border: 1px solid rgba(148,163,184,0.08);
    border-radius: 18px;
    padding: 0.95rem 1rem;
    margin-bottom: 1rem;
}
.sidebar-title {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text);
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}
.sidebar-copy {
    color: var(--muted);
    font-size: 0.85rem;
    line-height: 1.55;
}
.chart-shell {
    background: linear-gradient(180deg, rgba(11, 18, 32, 0.72) 0%, rgba(8, 15, 28, 0.72) 100%);
    border: 1px solid rgba(148, 163, 184, 0.08);
    border-radius: 20px;
    padding: 0.45rem;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--text);
}

/* Warning badge */
.risk-badge {
    background: linear-gradient(135deg, #f97316, #dc2626);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

/* FX banner */
.fx-banner {
    background: linear-gradient(90deg, rgba(10,20,36,0.86), rgba(16,30,52,0.86));
    border: 1px solid rgba(88, 166, 255, 0.22);
    border-radius: 18px;
    padding: 0.85rem 1rem;
    color: #a8cdfd;
    font-size: 0.92rem;
    font-weight: 500;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    box-shadow: 0 16px 36px rgba(2, 8, 23, 0.25);
}

.event-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(8, 15, 28, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 999px;
    padding: 0.55rem 0.8rem;
    margin: 0.22rem 0.35rem 0.22rem 0;
    color: var(--text);
    font-size: 0.84rem;
}

.legend-shell {
    background: rgba(8, 15, 28, 0.48);
    border: 1px solid rgba(148, 163, 184, 0.08);
    border-radius: 18px;
    padding: 0.9rem 1rem;
    margin-top: 0.8rem;
}

@media (max-width: 1100px) {
    .mini-stat-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 720px) {
    .hero-shell {
        padding: 1.2rem;
        border-radius: 20px;
    }
    .mini-stat-grid {
        grid-template-columns: 1fr;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def fetch_live_fx():
    """Get live USD/EUR rate."""
    try:
        resp = requests.get(FX_API_URL, timeout=5)
        return resp.json()["rates"]["EUR"]
    except Exception:
        return 0.92


def render_section_header(title: str, subtitle: str = "") -> None:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def close_section_shell() -> None:
    st.markdown('</div>', unsafe_allow_html=True)


def render_metric_card(title: str, value: str, tone: str = "neutral", note: str = "") -> None:
    tone_class = "delta-positive" if tone == "positive" else "delta-negative" if tone == "negative" else "delta-warning" if tone == "warning" else ""
    note_html = f'<div class="micro-note">{note}</div>' if note else ""
    st.markdown(
        f'<div class="metric-card"><h3>{title}</h3><div class="value {tone_class}">{value}</div>{note_html}</div>',
        unsafe_allow_html=True,
    )


def create_donut_chart(holdings: dict) -> go.Figure:
    """Create portfolio donut chart with sector colours."""
    tickers = list(holdings.keys())
    weights = [holdings[t]["weight"] for t in tickers]
    sectors = [holdings[t]["sector"] for t in tickers]
    colours = [SECTOR_COLOURS.get(s, "#6366f1") for s in sectors]

    fig = go.Figure(go.Pie(
        labels=[f"{t}<br>{w:.0%}" for t, w in zip(tickers, weights)],
        values=weights,
        hole=0.55,
        marker=dict(colors=colours, line=dict(color="#0f172a", width=2)),
        textinfo="label",
        textfont=dict(size=12, color="white"),
        hovertemplate="<b>%{label}</b><br>Weight: %{value:.1%}<br>€%{customdata:.2f}<extra></extra>",
        customdata=[holdings[t]["invested_eur"] for t in tickers],
    ))
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
        annotations=[dict(
            text=f"<b>€{PORTFOLIO['total_invested']:.0f}</b><br>invested",
            x=0.5, y=0.5, font_size=16, font_color="#e2e8f0",
            showarrow=False,
        )],
    )
    return fig


def create_fan_chart(ticker: str, paths_data: dict, horizon: int) -> go.Figure:
    """Create fan chart showing price path percentile bands."""
    if not paths_data or horizon not in paths_data:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    data = paths_data[horizon]
    paths = data.get("paths", np.array([]))
    if paths.size == 0:
        return go.Figure()

    days = np.arange(paths.shape[1])
    p10 = np.percentile(paths, 10, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p90 = np.percentile(paths, 90, axis=0)

    fig = go.Figure()
    # 10-90 band
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([p90, p10[::-1]]),
        fill="toself",
        fillcolor="rgba(88,166,255,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="10-90%",
        hoverinfo="skip",
    ))
    # 25-75 band
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill="toself",
        fillcolor="rgba(88,166,255,0.24)",
        line=dict(color="rgba(0,0,0,0)"),
        name="25-75%",
        hoverinfo="skip",
    ))
    # Median line
    fig.add_trace(go.Scatter(
        x=days, y=p50,
        line=dict(color="#7dd3fc", width=3),
        name="Median",
    ))

    fig.update_layout(
        title=dict(text=f"{ticker} — T+{horizon}d Price Forecast", font=dict(size=15, color="#e8eef8", family="Space Grotesk"), x=0.03),
        xaxis_title="Trading Days",
        yaxis_title="Price ($)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,15,28,0.76)",
        font=dict(color="#94a3b8", family="Inter"),
        height=300,
        margin=dict(t=48, b=40, l=50, r=20),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.18, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.10)", zerolinecolor="rgba(148,163,184,0.08)", linecolor="rgba(148,163,184,0.14)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.10)", zerolinecolor="rgba(148,163,184,0.08)", linecolor="rgba(148,163,184,0.14)"),
    )
    return fig


def create_event_waterfall(scenario_impact: dict) -> go.Figure:
    """Waterfall chart showing EUR portfolio impact per event."""
    if not scenario_impact:
        return go.Figure()

    names = []
    p50_values = []
    colours = []

    for event_key, data in scenario_impact.items():
        name = data.get("emoji", "") + " " + data.get("name", event_key)
        pnl = data.get("portfolio_pnl", {}).get("p50", 0)
        names.append(name[:30])
        p50_values.append(pnl)
        colours.append("#4ade80" if pnl >= 0 else "#f87171")

    fig = go.Figure(go.Bar(
        x=names,
        y=p50_values,
        marker_color=colours,
        marker_line=dict(color="rgba(255,255,255,0.08)", width=1),
        text=[f"€{v:+.2f}" for v in p50_values],
        textposition="outside",
        textfont=dict(size=11, color="#e2e8f0"),
    ))
    fig.update_layout(
        title=dict(text="Event Impact on Portfolio (Median EUR P&L)", font=dict(size=15, color="#e8eef8", family="Space Grotesk"), x=0.03),
        xaxis_title="",
        yaxis_title="EUR Impact",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,15,28,0.76)",
        font=dict(color="#94a3b8", family="Inter"),
        height=350,
        margin=dict(t=40, b=80, l=50, r=20),
        xaxis=dict(tickangle=-25, gridcolor="rgba(148,163,184,0.08)", linecolor="rgba(148,163,184,0.14)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.10)", linecolor="rgba(148,163,184,0.14)"),
    )
    return fig


def create_heatmap(scenario_impact: dict) -> go.Figure:
    """Heatmap: ticker × event → impact magnitude."""
    if not scenario_impact:
        return go.Figure()

    events = list(scenario_impact.keys())
    event_names = [scenario_impact[e].get("emoji", "") + " " + scenario_impact[e].get("name", e)[:20] for e in events]

    z = []
    for ticker in TICKERS:
        row = []
        for event_key in events:
            data = scenario_impact[event_key]
            imp = data.get("ticker_impact", {}).get(ticker, {})
            row.append(imp.get("p50_eur", 0))
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=event_names,
        y=TICKERS,
        colorscale=[
            [0, "#7f1d1d"],
            [0.22, "#fb7185"],
            [0.5, "#0f172a"],
            [0.78, "#67e8f9"],
            [1, "#22c55e"],
        ],
        zmid=0,
        text=[[f"€{v:+.1f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="EUR", tickfont=dict(color="#94a3b8")),
    ))
    fig.update_layout(
        title=dict(text="Impact Heatmap (Ticker × Event, EUR)", font=dict(size=15, color="#e8eef8", family="Space Grotesk"), x=0.03),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,15,28,0.76)",
        font=dict(color="#94a3b8", family="Inter"),
        height=350,
        margin=dict(t=40, b=80, l=60, r=20),
        xaxis=dict(tickangle=-25),
    )
    return fig


def create_efficient_frontier(ef_data: dict) -> go.Figure:
    """Plot efficient frontier with current portfolio marked."""
    if not ef_data:
        return go.Figure()

    fig = go.Figure()

    # Random portfolios
    random_ports = ef_data.get("random_portfolios", [])
    if random_ports:
        vols = [p["volatility"] for p in random_ports]
        rets = [p["return"] for p in random_ports]
        sharpes = [p["sharpe"] for p in random_ports]

        fig.add_trace(go.Scatter(
            x=vols, y=rets,
            mode="markers",
            marker=dict(
                size=3,
                color=sharpes,
                colorscale=[
                    [0, "#1d4ed8"],
                    [0.5, "#38bdf8"],
                    [1, "#4ade80"],
                ],
                showscale=True,
                colorbar=dict(title="Sharpe", tickfont=dict(color="#94a3b8")),
                opacity=0.78,
            ),
            name="Random Portfolios",
            hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
        ))

    # Current portfolio
    current = ef_data.get("current_portfolio", {})
    if current:
        fig.add_trace(go.Scatter(
            x=[current.get("volatility", 0)],
            y=[current.get("expected_return", 0)],
            mode="markers+text",
            marker=dict(size=16, color="#ef4444", symbol="star", line=dict(width=2, color="white")),
            text=["Your Portfolio"],
            textposition="top center",
            textfont=dict(color="#f87171", size=12, family="Inter"),
            name="Current Portfolio",
        ))

    # Optimal portfolios
    for port in ef_data.get("frontier_portfolios", []):
        label = port.get("label", f"Sharpe {port.get('sharpe', 0):.1f}")
        fig.add_trace(go.Scatter(
            x=[port.get("volatility", 0)],
            y=[port.get("expected_return", 0)],
            mode="markers+text",
            marker=dict(size=12, color="#22c55e", symbol="diamond"),
            text=[label],
            textposition="bottom center",
            textfont=dict(color="#4ade80", size=10),
            name=label,
        ))

    fig.update_layout(
        title=dict(text="Efficient Frontier", font=dict(size=15, color="#e8eef8", family="Space Grotesk"), x=0.03),
        xaxis_title="Annualised Volatility",
        yaxis_title="Annualised Return",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,15,28,0.76)",
        font=dict(color="#94a3b8", family="Inter"),
        height=400,
        margin=dict(t=40, b=50, l=60, r=20),
        xaxis=dict(tickformat=".0%", gridcolor="rgba(148,163,184,0.10)", linecolor="rgba(148,163,184,0.14)"),
        yaxis=dict(tickformat=".0%", gridcolor="rgba(148,163,184,0.10)", linecolor="rgba(148,163,184,0.14)"),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def generate_pdf_report(analytics: dict, fx_rate: float) -> bytes:
    """Generate PDF summary report using reportlab."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError:
        return b"reportlab not installed"

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=20, textColor=colors.HexColor("#1e40af"))
    elements.append(Paragraph("Portfolio World Simulation — Report", title_style))
    elements.append(Spacer(1, 5 * mm))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | USD/EUR: {fx_rate:.4f}", styles["Normal"]))
    elements.append(Spacer(1, 8 * mm))

    # Portfolio summary
    elements.append(Paragraph("Portfolio Holdings", styles["Heading2"]))
    table_data = [["Ticker", "Sector", "Weight", "Invested (EUR)"]]
    for t in TICKERS:
        h = PORTFOLIO["holdings"][t]
        table_data.append([t, h["sector"], f"{h['weight']:.0%}", f"€{h['invested_eur']:.2f}"])
    table_data.append(["TOTAL", "", "100%", f"€{PORTFOLIO['total_invested']:.2f}"])

    t = Table(table_data, colWidths=[60, 180, 60, 80])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f1f5f9")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 8 * mm))

    # Portfolio metrics
    pm = analytics.get("portfolio_metrics", {})
    if pm:
        elements.append(Paragraph("Portfolio Metrics (Monte Carlo)", styles["Heading2"]))
        for horizon, metrics in pm.items():
            elements.append(Paragraph(f"<b>T+{horizon} days</b>", styles["Normal"]))
            metric_data = [
                ["Metric", "Value"],
                ["Expected Return (EUR)", f"€{metrics.get('expected_return_eur', 0):+.2f}"],
                ["VaR 95% (EUR)", f"€{metrics.get('var_95_eur', 0):.2f}"],
                ["VaR 99% (EUR)", f"€{metrics.get('var_99_eur', 0):.2f}"],
                ["CVaR 95% (EUR)", f"€{metrics.get('cvar_95_eur', 0):.2f}"],
            ]
            mt = Table(metric_data, colWidths=[160, 120])
            mt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ]))
            elements.append(mt)
            elements.append(Spacer(1, 4 * mm))

    # Risk warnings
    conc = analytics.get("concentration", {})
    if conc.get("warning"):
        elements.append(Paragraph(
            f"⚠ CONCENTRATION RISK: Semiconductor cluster correlation = {conc.get('semi_correlation', 0):.3f} (>0.85 threshold)",
            ParagraphStyle("Warning", parent=styles["Normal"], textColor=colors.red, fontSize=10),
        ))
        elements.append(Spacer(1, 4 * mm))

    # Rebalancing
    rebal = analytics.get("rebalancing", {}).get("recommendations", {})
    if rebal:
        elements.append(Paragraph("Rebalancing Recommendations", styles["Heading2"]))
        rebal_data = [["Ticker", "Current", "Suggested", "Delta (EUR)", "Action"]]
        for t in TICKERS:
            r = rebal.get(t, {})
            rebal_data.append([
                t,
                f"{r.get('current_weight', 0):.0%}",
                f"{r.get('suggested_weight', 0):.0%}",
                f"€{r.get('delta_eur', 0):+.2f}",
                r.get("action", "hold").upper(),
            ])
        rt = Table(rebal_data, colWidths=[50, 60, 60, 80, 70])
        rt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        elements.append(rt)

    doc.build(elements)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------------------------

def run_dashboard():
    """Main Streamlit dashboard entry point."""

    # --- Top FX Banner ---
    fx_rate = fetch_live_fx()
    st.markdown(
        f"""<div class="fx-banner">
            <span>💱 <b>Live FX Rate</b></span>
            <span>USD/EUR: <b>{fx_rate:.4f}</b> | Updated: {datetime.now().strftime('%H:%M:%S')}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # Title
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">Portfolio Intelligence Studio</div>
            <div class="hero-title">Stock Portfolio World Simulation</div>
            <div class="hero-subtitle">
                Scenario-driven portfolio intelligence for your real holdings, combining machine learning,
                agent-based market behavior, and geopolitical stress testing in one interface.
            </div>
            <div class="mini-stat-grid">
                <div class="mini-stat"><div class="label">Portfolio Size</div><div class="number">8 holdings</div></div>
                <div class="mini-stat"><div class="label">Capital</div><div class="number">EUR {PORTFOLIO['total_invested']:.0f}</div></div>
                <div class="mini-stat"><div class="label">Simulation Stack</div><div class="number">ML + Agents + Events</div></div>
                <div class="mini-stat"><div class="label">FX Anchor</div><div class="number">USD/EUR {fx_rate:.4f}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-shell">
                <div class="sidebar-title">Simulation Controls</div>
                <div class="sidebar-copy">Tune scale, horizon, and active scenarios before running the portfolio engine.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        n_agents = st.slider("Number of Agents", 100, 100_000, DEFAULT_N_AGENTS, step=1000,
                             help="Market participants in the agent-based simulation")
        n_mc_paths = st.slider("Monte Carlo Paths", 100, 20_000, DEFAULT_MC_PATHS, step=500,
                               help="Number of simulation paths for portfolio analytics")
        horizon = st.selectbox("Time Horizon (days)", [30, 60, 90, 365], index=3)

        st.markdown(
            """
            <div class="sidebar-shell">
                <div class="sidebar-title">Event Regimes</div>
                <div class="sidebar-copy">Choose which geopolitical and macro narratives can enter the simulation run.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        enabled_events = []
        for event_key, event_spec in GEOPOLITICAL_EVENTS.items():
            emoji = event_spec.get("emoji", "📌")
            name = event_spec["name"]
            prob = event_spec["probability_per_quarter"]
            if st.checkbox(f"{emoji} {name} ({prob:.0%}/qtr)", value=True, key=event_key):
                enabled_events.append(event_key)

        run_sim = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
        dry_run = st.checkbox("🧪 Dry run (fast, reduced agents)", value=True)

        st.markdown(
            """
            <div class="sidebar-shell">
                <div class="sidebar-title">Portfolio Risk Profile</div>
                <div class="sidebar-copy">Structural concentration, narrative dependency, and macro sensitivity already baked into the engine.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for key, info in RISK_PROFILE.items():
            with st.expander(key.replace("_", " ").title()):
                st.caption(info.get("note", ""))

    # --- Main Content ---

    # Load results from session state (or show empty state)
    if "analytics" not in st.session_state:
        st.session_state.analytics = None
        st.session_state.ticker_paths = None
        st.session_state.forecast_data = None

    if run_sim:
        with st.spinner("🔄 Running full simulation pipeline..."):
            try:
                # Module 1: Data Ingestion
                st.toast("📥 Fetching market data...")
                ingestion_data = cached_ingestion()

                # Module 2: Sentiment Analysis
                st.toast("🧠 Analysing sentiment...")
                sentiment_data = cached_sentiment(ingestion_data["news"])

                # Module 5: Event Generation
                st.toast("🌐 Generating events...")
                event_data = cached_event_generation(horizon, tuple(enabled_events))

                # Module 3: ML Forecasting
                st.toast("📈 Training ML models...")
                ml_data = cached_forecasting(
                    ohlcv_data=ingestion_data["ohlcv"],
                    composite_scores=sentiment_data["composite_scores"],
                    macro_data=ingestion_data["macro"],
                    rolling_sentiment=sentiment_data["rolling_sentiment"],
                    gdelt_data=ingestion_data["gdelt"],
                    horizon=horizon,
                    dry_run=dry_run,
                )

                # Module 4: Agent Simulation
                st.toast("🤖 Running agent simulation...")
                agent_data = cached_agent_simulation(
                    ohlcv_data=ingestion_data["ohlcv"],
                    composite_scores=sentiment_data["composite_scores"],
                    event_timelines=event_data["timeline_data"]["per_ticker"],
                    rolling_sentiment=sentiment_data["rolling_sentiment"],
                    macro_data=ingestion_data["macro"],
                    n_agents=n_agents,
                    horizon=horizon,
                    dry_run=dry_run,
                )

                # Module 6: Portfolio Analytics
                st.toast("📊 Computing analytics...")
                analytics = cached_analytics(
                    ohlcv_data=ingestion_data["ohlcv"],
                    forecast_data=ml_data,
                    agent_data=agent_data,
                    _event_data=event_data,
                    fx_rate=fx_rate,
                    n_mc_paths=n_mc_paths,
                    horizon=horizon,
                    dry_run=dry_run,
                )

                st.session_state.analytics = analytics
                st.session_state.ticker_paths = analytics.get("ticker_paths", {})
                st.session_state.scenario_impact = analytics.get("scenario_impact", {})
                st.session_state.forecast_data = ml_data
                st.session_state.horizon = horizon
                st.session_state.event_data = event_data
                st.session_state.sentiment_data = sentiment_data
                st.session_state.fx_rate = fx_rate

                st.toast("✅ Simulation complete!", icon="🎉")
                st.rerun()

            except Exception as e:
                st.error(f"Simulation failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # --- Display Results ---
    analytics = st.session_state.analytics

    # Always show portfolio donut
    col1, col2 = st.columns([1, 2])

    with col1:
        render_section_header("📊 Portfolio Allocation", "Current portfolio composition by ticker, sector, and capital weight.")
        fig_donut = create_donut_chart(PORTFOLIO["holdings"])
        st.plotly_chart(fig_donut, use_container_width=True)

        # Sector legend
        st.markdown('<div class="legend-shell">', unsafe_allow_html=True)
        for sector, colour in SECTOR_COLOURS.items():
            tickers_in_sector = [t for t in TICKERS if PORTFOLIO["holdings"][t]["sector"] == sector]
            if tickers_in_sector:
                st.markdown(
                    f'<span style="color:{colour};font-size:0.86rem;">● {sector}</span> '
                    f'<span style="color:#64748b;font-size:0.8rem;">({", ".join(tickers_in_sector)})</span><br>',
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)
        close_section_shell()

    with col2:
        if analytics:
            horizon = st.session_state.get("horizon", 90)
            pm = analytics.get("portfolio_metrics", {}).get(horizon, {})

            render_section_header("📈 Portfolio Metrics", "Core return and risk outputs from the active simulation horizon.")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                val = pm.get("expected_return_eur", 0)
                render_metric_card(f"Expected Return (T+{horizon})", f"EUR {val:+.2f}", "positive" if val >= 0 else "negative", f"Median path: EUR {pm.get('percentiles_eur', {}).get('p50', 0):+.2f}")
            with m2:
                render_metric_card("VaR 95%", f"EUR {pm.get('var_95_eur', 0):.2f}", "negative", "One-tailed downside estimate")
            with m3:
                render_metric_card("CVaR 95%", f"EUR {pm.get('cvar_95_eur', 0):.2f}", "negative", "Average loss inside the worst tail")
            with m4:
                conc = analytics.get("concentration", {})
                corr = conc.get("semi_correlation", 0)
                warning = conc.get("warning", False)
                badge = ' <span class="risk-badge">HIGH</span>' if warning else ""
                render_metric_card(f"Semi Cluster Corr{badge}", f"{corr:.3f}", "warning" if warning else "neutral", "Semiconductor concentration watch")

            # Triggered events
            events_log = st.session_state.get("event_data", {}).get("event_log", [])
            if events_log:
                st.markdown("<div class='section-subtitle' style='margin-top:0.9rem;'>Triggered event regime for this run.</div>", unsafe_allow_html=True)
                for ev in events_log:
                    st.markdown(f"<span class='event-pill'>{ev.get('emoji', '📌')} {ev['name']} · Day {ev['start_day']}-{ev['end_day']}</span>", unsafe_allow_html=True)
            else:
                st.info("No geopolitical events triggered in this run.")
            close_section_shell()
        else:
            st.markdown("### 👈 Configure & run simulation from the sidebar")
            st.info("Click **Run Simulation** in the sidebar to begin the analysis pipeline.")

    if not analytics:
        return

    st.markdown("---")

    render_section_header("🧪 Forecast Validation", "Reality check for forecast quality using simple walk-forward metrics.")
    forecast_data = st.session_state.get("forecast_data", {}) or {}
    validation_rows = []
    for ticker in TICKERS:
        ticker_validation = forecast_data.get(ticker, {}).get("validation", {})
        metrics = ticker_validation.get(horizon, {})
        if metrics:
            validation_rows.append({
                "Ticker": ticker,
                "Direction": f"{metrics.get('directional_accuracy', 0.0):.0%}",
                "MAE": f"{metrics.get('mae_pct', 0.0):.1%}",
                "RMSE": f"{metrics.get('rmse_pct', 0.0):.1%}",
                "Samples": int(metrics.get("sample_size", 0)),
            })
    if validation_rows:
        st.markdown('<div class="dataframe-shell">', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(validation_rows), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Validation metrics are not available for this run.")
    close_section_shell()

    st.markdown("---")

    render_section_header("📏 Benchmark Comparison", "Compare the active simulation against simpler reference baselines.")
    benchmark_data = analytics.get("benchmarks", {}).get(horizon, {})
    if benchmark_data:
        sim = analytics.get("portfolio_metrics", {}).get(horizon, {})
        benchmark_rows = [
            {
                "Mode": "Simulation",
                "Expected PnL": f"EUR {sim.get('expected_return_eur', 0.0):+.2f}",
                "VaR 95": f"EUR {sim.get('var_95_eur', 0.0):.2f}",
                "P10 / P50 / P90": (
                    f"EUR {sim.get('percentiles_eur', {}).get('p10', 0.0):+.2f} / "
                    f"EUR {sim.get('percentiles_eur', {}).get('p50', 0.0):+.2f} / "
                    f"EUR {sim.get('percentiles_eur', {}).get('p90', 0.0):+.2f}"
                ),
            },
            {
                "Mode": "Buy & Hold Baseline",
                "Expected PnL": f"EUR {benchmark_data.get('buy_and_hold', {}).get('expected_return_eur', 0.0):+.2f}",
                "VaR 95": f"EUR {benchmark_data.get('buy_and_hold', {}).get('var_95_eur', 0.0):.2f}",
                "P10 / P50 / P90": (
                    f"EUR {benchmark_data.get('buy_and_hold', {}).get('percentiles_eur', {}).get('p10', 0.0):+.2f} / "
                    f"EUR {benchmark_data.get('buy_and_hold', {}).get('percentiles_eur', {}).get('p50', 0.0):+.2f} / "
                    f"EUR {benchmark_data.get('buy_and_hold', {}).get('percentiles_eur', {}).get('p90', 0.0):+.2f}"
                ),
            },
            {
                "Mode": "Historical Bootstrap",
                "Expected PnL": f"EUR {benchmark_data.get('historical_bootstrap', {}).get('expected_return_eur', 0.0):+.2f}",
                "VaR 95": f"EUR {benchmark_data.get('historical_bootstrap', {}).get('var_95_eur', 0.0):.2f}",
                "P10 / P50 / P90": (
                    f"EUR {benchmark_data.get('historical_bootstrap', {}).get('percentiles_eur', {}).get('p10', 0.0):+.2f} / "
                    f"EUR {benchmark_data.get('historical_bootstrap', {}).get('percentiles_eur', {}).get('p50', 0.0):+.2f} / "
                    f"EUR {benchmark_data.get('historical_bootstrap', {}).get('percentiles_eur', {}).get('p90', 0.0):+.2f}"
                ),
            },
        ]
        st.markdown('<div class="dataframe-shell">', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(benchmark_rows), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Benchmark comparison is not available for this run.")
    close_section_shell()

    st.markdown("---")

    # Fan Charts
    render_section_header("🎯 Price Forecast Fan Charts", "Distribution bands for each simulated ticker under the current horizon.")
    ticker_paths = st.session_state.get("ticker_paths", {})
    horizon = st.session_state.get("horizon", 90)

    cols = st.columns(4)
    for i, ticker in enumerate(TICKERS):
        with cols[i % 4]:
            fig = create_fan_chart(ticker, ticker_paths.get(ticker), horizon)
            st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    close_section_shell()

    st.markdown("---")

    # Event Impact Analysis
    col_water, col_heat = st.columns(2)
    scenario_impact = st.session_state.get("scenario_impact", {})

    with col_water:
        render_section_header("🌊 Event Impact Waterfall", "Median portfolio impact of each event scenario in EUR.")
        fig_water = create_event_waterfall(scenario_impact)
        st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
        st.plotly_chart(fig_water, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        close_section_shell()

    with col_heat:
        render_section_header("🔥 Impact Heatmap", "Ticker-level event sensitivity map across the active scenario set.")
        fig_heat = create_heatmap(scenario_impact)
        st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        close_section_shell()

    st.markdown("---")

    # Rebalancing & Efficient Frontier
    col_rebal, col_ef = st.columns(2)

    with col_rebal:
        render_section_header("⚖️ Rebalancing Recommendations", "Suggested tilts after risk-adjusted portfolio review.")
        rebal = analytics.get("rebalancing", {}).get("recommendations", {})
        if rebal:
            rebal_df = pd.DataFrame([
                {
                    "Ticker": t,
                    "Current": f"{r.get('current_weight', 0):.0%}",
                    "Suggested": f"{r.get('suggested_weight', 0):.0%}",
                    "Delta EUR": f"€{r.get('delta_eur', 0):+.2f}",
                    "Action": r.get("action", "hold").upper(),
                }
                for t, r in rebal.items()
            ])
            st.markdown('<div class="dataframe-shell">', unsafe_allow_html=True)
            st.dataframe(rebal_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Visual comparison
            fig_rebal = go.Figure()
            fig_rebal.add_trace(go.Bar(
                name="Current",
                x=list(rebal.keys()),
                y=[rebal[t]["current_weight"] for t in rebal],
                marker_color="#3b82f6",
            ))
            fig_rebal.add_trace(go.Bar(
                name="Suggested",
                x=list(rebal.keys()),
                y=[rebal[t]["suggested_weight"] for t in rebal],
                marker_color="#22c55e",
            ))
            fig_rebal.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.8)",
                font=dict(color="#94a3b8"),
                height=250,
                margin=dict(t=10, b=30, l=50, r=20),
                yaxis=dict(tickformat=".0%", gridcolor="#1e293b"),
                xaxis=dict(gridcolor="#1e293b"),
                legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15),
            )
            st.plotly_chart(fig_rebal, use_container_width=True)
        close_section_shell()

    with col_ef:
        render_section_header("📐 Efficient Frontier", "Where the current portfolio sits relative to diversified alternatives.")
        ef_data = analytics.get("efficient_frontier", {})
        fig_ef = create_efficient_frontier(ef_data)
        st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
        st.plotly_chart(fig_ef, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        close_section_shell()

    # Sentiment summary
    st.markdown("---")
    render_section_header("🧠 Sentiment Summary", "Headline-derived directional tone for each holding.")
    sentiment_data = st.session_state.get("sentiment_data", {})
    scores = sentiment_data.get("composite_scores", {})
    if scores:
        sent_cols = st.columns(len(TICKERS))
        for i, ticker in enumerate(TICKERS):
            with sent_cols[i]:
                val = scores.get(ticker, 0)
                colour = "#4ade80" if val > 0.1 else "#f87171" if val < -0.1 else "#fbbf24"
                st.markdown(
                    f'<div style="text-align:center;padding:0.5rem;">'
                    f'<div style="font-weight:600;color:#94a3b8;font-size:0.8rem;">{ticker}</div>'
                    f'<div style="font-size:1.3rem;font-weight:700;color:{colour};">{val:+.2f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    close_section_shell()

    # PDF Export
    st.markdown("---")
    if st.button("📄 Export PDF Report", type="secondary"):
        with st.spinner("Generating PDF..."):
            pdf_bytes = generate_pdf_report(analytics, fx_rate)
            if pdf_bytes and pdf_bytes != b"reportlab not installed":
                report_path = REPORTS_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                report_path.write_bytes(pdf_bytes)
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                )
                st.success(f"Report saved to `{report_path}`")
            else:
                st.warning("Install `reportlab` for PDF export: `pip install reportlab`")


if __name__ == "__main__":
    run_dashboard()
