"""
main.py — Orchestrator for the Stock Portfolio World Simulation Engine.
Runs the full pipeline: data ingestion → sentiment → ML → agents → events → analytics.
Supports --dry-run for quick testing.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DEFAULT_HORIZONS,
    DEFAULT_MC_PATHS,
    DEFAULT_N_AGENTS,
    PORTFOLIO,
    RANDOM_SEED,
    REPORTS_DIR,
    TICKERS,
)


class SafeConsoleHandler(logging.StreamHandler):
    """Console handler that degrades Unicode safely on non-UTF-8 terminals."""

    TRANSLATIONS = str.maketrans({
        "✓": "[OK]",
        "✗": "[X]",
        "▶": ">",
        "→": "->",
        "—": "-",
        "–": "-",
        "€": "EUR ",
        "🌍": "[WORLD]",
        "🧪": "[DRY RUN]",
        "💱": "[FX]",
        "📊": "[SUMMARY]",
        "🌐": "[EVENTS]",
        "⚖": "[REBALANCE]",
        "⏱": "[TIME]",
        "💊": "[DRUG]",
        "⚡": "[NUCLEAR]",
        "📌": "[*]",
    })

    @classmethod
    def _sanitize(cls, text: str) -> str:
        return text.translate(cls.TRANSLATIONS)

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            terminator = self.terminator
            try:
                stream.write(msg + terminator)
            except UnicodeEncodeError:
                encoding = getattr(stream, "encoding", None) or "ascii"
                safe_message = self._sanitize(msg + terminator)
                safe_message = safe_message.encode(encoding, errors="replace").decode(encoding)
                stream.write(safe_message)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            SafeConsoleHandler(sys.stdout),
            logging.FileHandler(REPORTS_DIR / "pipeline.log", mode="w", encoding="utf-8"),
        ],
    )


def run_pipeline(
    dry_run: bool = False,
    n_agents: int = DEFAULT_N_AGENTS,
    n_mc_paths: int = DEFAULT_MC_PATHS,
    horizons: list[int] | None = None,
    enabled_events: list[str] | None = None,
    seed: int = RANDOM_SEED,
    skip_ml: bool = False,
):
    """Run the complete simulation pipeline.

    Args:
        dry_run: If True, use reduced parameters for fast testing.
        n_agents: Number of agents for simulation.
        n_mc_paths: Number of Monte Carlo paths.
        horizons: Forecast horizons in days.
        enabled_events: List of event keys to enable (None = all).
        seed: Random seed.
        skip_ml: Skip ML training (use dummy forecasts).
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()

    if horizons is None:
        horizons = DEFAULT_HORIZONS
    if dry_run:
        n_agents = min(n_agents, 1000)
        n_mc_paths = min(n_mc_paths, 500)
        logger.info("🧪 DRY RUN MODE — reduced parameters for fast testing")

    logger.info("=" * 70)
    logger.info("  🌍 STOCK PORTFOLIO WORLD SIMULATION ENGINE")
    logger.info("=" * 70)
    logger.info(f"  Portfolio: {len(TICKERS)} tickers | €{PORTFOLIO['total_invested']:.2f} invested")
    logger.info(f"  Agents: {n_agents:,} | MC paths: {n_mc_paths:,}")
    logger.info(f"  Horizons: T+{'/'.join(map(str, horizons))} days")
    logger.info(f"  Seed: {seed} | Dry run: {dry_run}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # MODULE 1 — Data Ingestion
    # ------------------------------------------------------------------
    from modules.data_ingestion import run_ingestion
    ingestion_data = run_ingestion()

    fx_rate = ingestion_data["fx_rate"]
    logger.info(f"\n  💱 USD/EUR: {fx_rate:.4f}")

    # ------------------------------------------------------------------
    # MODULE 2 — Sentiment Analysis
    # ------------------------------------------------------------------
    from modules.sentiment_engine import run_sentiment_analysis
    sentiment_data = run_sentiment_analysis(ingestion_data["news"])

    # ------------------------------------------------------------------
    # MODULE 5 — Event Generation (before ML, so events can inform features)
    # ------------------------------------------------------------------
    from modules.event_generator import run_event_generation
    event_data = run_event_generation(
        n_days=max(horizons),
        enabled_events=enabled_events,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # MODULE 3 — ML Forecasting
    # ------------------------------------------------------------------
    ml_data = {}
    if not skip_ml:
        from modules.ml_forecasting import run_forecasting
        ml_data = run_forecasting(
            ohlcv_data=ingestion_data["ohlcv"],
            sentiment_scores=sentiment_data["composite_scores"],
            macro_data=ingestion_data["macro"],
            rolling_sentiment=sentiment_data["rolling_sentiment"],
            gdelt_data=ingestion_data["gdelt"],
            horizons=horizons,
            dry_run=dry_run,
            reuse_models=True,
        )
    else:
        logger.info("\n⏭ ML training skipped (--skip-ml flag)")

    # ------------------------------------------------------------------
    # MODULE 4 — Agent-Based Simulation
    # ------------------------------------------------------------------
    from modules.agent_simulation import run_agent_simulation
    agent_data = run_agent_simulation(
        ohlcv_data=ingestion_data["ohlcv"],
        sentiment_scores=sentiment_data["composite_scores"],
        event_timelines=event_data["timeline_data"]["per_ticker"],
        rolling_sentiment=sentiment_data["rolling_sentiment"],
        macro_data=ingestion_data["macro"],
        n_agents=n_agents,
        n_steps=max(horizons),
        seed=seed,
        dry_run=dry_run,
    )

    # ------------------------------------------------------------------
    # MODULE 6 — Portfolio Analytics
    # ------------------------------------------------------------------
    from modules.portfolio_analytics import run_portfolio_analytics
    analytics = run_portfolio_analytics(
        ohlcv_data=ingestion_data["ohlcv"],
        forecast_data=ml_data,
        agent_sim_data=agent_data,
        event_data=event_data,
        fx_rate=fx_rate,
        n_paths=n_mc_paths,
        horizons=horizons,
        seed=seed,
        dry_run=dry_run,
    )

    # ------------------------------------------------------------------
    # Summary Output
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("  📊 SIMULATION RESULTS SUMMARY")
    logger.info("=" * 70)

    # Portfolio metrics
    for h in horizons:
        pm = analytics.get("portfolio_metrics", {}).get(h, {})
        if pm:
            logger.info(f"\n  T+{h} days:")
            logger.info(f"    Expected Return: €{pm.get('expected_return_eur', 0):+.2f} "
                         f"({pm.get('expected_return_pct', 0)*100:+.1f}%)")
            logger.info(f"    VaR 95%:  €{pm.get('var_95_eur', 0):.2f}")
            logger.info(f"    VaR 99%:  €{pm.get('var_99_eur', 0):.2f}")
            logger.info(f"    CVaR 95%: €{pm.get('cvar_95_eur', 0):.2f}")
            pct = pm.get("percentiles_eur", {})
            logger.info(f"    P&L distribution: "
                         f"p10=€{pct.get('p10', 0):+.2f} | "
                         f"p50=€{pct.get('p50', 0):+.2f} | "
                         f"p90=€{pct.get('p90', 0):+.2f}")

    # Concentration risk
    conc = analytics.get("concentration", {})
    if conc.get("warning"):
        logger.info(f"\n  ⚠ CONCENTRATION WARNING: Semiconductor cluster correlation = {conc['semi_correlation']:.3f}")
    else:
        logger.info(f"\n  ✓ Semiconductor correlation: {conc.get('semi_correlation', 0):.3f} (below 0.85 threshold)")

    # Events
    events = event_data.get("event_log", [])
    if events:
        logger.info(f"\n  🌐 Events triggered ({len(events)}):")
        for ev in events:
            logger.info(f"    {ev.get('emoji', '📌')} {ev['name']} (Day {ev['start_day']}–{ev['end_day']})")

    # Rebalancing
    rebal = analytics.get("rebalancing", {}).get("recommendations", {})
    actions = [(t, r) for t, r in rebal.items() if r.get("action") != "hold"]
    if actions:
        logger.info(f"\n  ⚖ Rebalancing suggestions:")
        for t, r in actions:
            logger.info(f"    {r['action'].upper()} {t}: €{r['delta_eur']:+.2f} "
                         f"({r['current_weight']:.0%} → {r['suggested_weight']:.0%})")

    logger.info(f"\n  ⏱ Pipeline completed in {elapsed:.1f}s")
    logger.info("=" * 70)
    logger.info("  ▶ Launch dashboard: python -m streamlit run modules/dashboard.py")
    logger.info("=" * 70)

    return {
        "ingestion": ingestion_data,
        "sentiment": sentiment_data,
        "events": event_data,
        "ml": ml_data,
        "agents": agent_data,
        "analytics": analytics,
        "fx_rate": fx_rate,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stock Portfolio World Simulation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --dry-run          # Quick test with reduced parameters
  python main.py                    # Full pipeline (50K agents, 10K MC paths)
  python main.py --agents 10000     # Custom agent count
  python main.py --skip-ml          # Skip ML training for faster iteration
        """,
    )
    parser.add_argument("--dry-run", action="store_true", help="Fast mode with reduced parameters")
    parser.add_argument("--agents", type=int, default=DEFAULT_N_AGENTS, help="Number of agents")
    parser.add_argument("--mc-paths", type=int, default=DEFAULT_MC_PATHS, help="Monte Carlo paths")
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS, help="Forecast horizons")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML model training")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    setup_logging(args.verbose)
    run_pipeline(
        dry_run=args.dry_run,
        n_agents=args.agents,
        n_mc_paths=args.mc_paths,
        horizons=args.horizons,
        seed=args.seed,
        skip_ml=args.skip_ml,
    )


if __name__ == "__main__":
    main()
