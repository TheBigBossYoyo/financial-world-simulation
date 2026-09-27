"""
Module 5 — Geopolitical & Macro Event Generator
Stochastically triggers 7 hardcoded geopolitical/macro events during simulation,
applies impact shocks per ticker from uniform distributions, handles overlapping events.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEOPOLITICAL_EVENTS, RANDOM_SEED, TICKERS

logger = logging.getLogger(__name__)


class EventGenerator:
    """Stochastic geopolitical & macro event generator."""

    def __init__(self, seed: int = RANDOM_SEED, enabled_events: Optional[List[str]] = None):
        self.rng = np.random.RandomState(seed)
        self.events = GEOPOLITICAL_EVENTS
        if enabled_events is not None:
            self.events = {k: v for k, v in self.events.items() if k in enabled_events}
        self.active_events = []

    def _should_trigger(self, prob_per_quarter: float, n_days: int) -> Tuple[bool, int]:
        """Determine if an event triggers during the simulation period.

        Scales the quarterly probability to the simulated horizon, then samples once.
        Returns (triggered, start_day).
        """
        horizon_scale = min(max(np.sqrt(n_days / 63), 0.75), 1.8)
        horizon_prob = 1 - (1 - prob_per_quarter) ** horizon_scale
        if self.rng.random() >= horizon_prob:
            return False, -1
        return True, int(self.rng.randint(0, max(1, n_days)))

    def _sample_impact(self, impact_spec: Dict) -> float:
        """Sample impact magnitude from the event specification."""
        direction = impact_spec.get("direction", 0)
        low = impact_spec.get("low", 0)
        high = impact_spec.get("high", 0)

        if direction == 0:
            # Neutral — sample from the range directly (can be + or -)
            return self.rng.uniform(low, high)
        else:
            magnitude = self.rng.uniform(abs(low), abs(high))
            return direction * magnitude

    def _sample_duration(self, duration_range: Tuple[int, int]) -> int:
        """Sample event duration from uniform distribution."""
        return self.rng.randint(duration_range[0], duration_range[1] + 1)

    def generate_event_timeline(
        self,
        n_days: int,
        pe_data: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Generate complete event timeline for a simulation run.

        Args:
            n_days: Number of simulation trading days.
            pe_data: Optional dict with forward P/E ratios for conditional triggers.

        Returns:
            Dict with:
              - 'event_log': list of triggered events with details
              - 'per_ticker': {ticker: {'timeline': bool array, 'impacts': float array}}
        """
        event_log = []
        # Per-ticker timeline arrays
        per_ticker = {
            ticker: {
                "timeline": np.zeros(n_days, dtype=bool),
                "impacts": np.zeros(n_days),
            }
            for ticker in TICKERS
        }

        max_events = 2 if n_days <= 90 else 3 if n_days <= 180 else 4
        for event_key, event_spec in sorted(self.events.items(), key=lambda item: item[1]["probability_per_quarter"], reverse=True):
            if len(event_log) >= max_events:
                break
            prob = event_spec["probability_per_quarter"]

            # Check conditional trigger (AI Bubble Burst)
            if "trigger_condition" in event_spec:
                if pe_data:
                    nvda_pe = pe_data.get("NVDA", 0)
                    avgo_pe = pe_data.get("AVGO", 0)
                    condition_met = nvda_pe > 40 and avgo_pe > 35
                    if condition_met:
                        prob = min(prob * 1.5, 0.6)
                    else:
                        prob *= 0.2
                else:
                    prob *= 0.15

            triggered, start_day = self._should_trigger(prob, n_days)
            if not triggered:
                continue

            duration = self._sample_duration(event_spec["duration_days"])
            end_day = min(start_day + duration, n_days)

            impacts = {}
            for ticker in TICKERS:
                impact_spec = event_spec["portfolio_impact"].get(ticker, {})
                if impact_spec:
                    impact = self._sample_impact(impact_spec)
                    impacts[ticker] = impact

                    # Apply impact to timeline
                    # Spread impact over duration (gradual, not instant)
                    active_days = end_day - start_day
                    if active_days > 0:
                        per_ticker[ticker]["timeline"][start_day:end_day] = True
                        # Impact ramps up linearly, then holds
                        ramp_len = min(active_days // 3, 10)
                        impact_profile = np.ones(active_days)
                        if ramp_len > 0:
                            impact_profile[:ramp_len] = np.linspace(0.2, 1.0, ramp_len)
                        daily_impact = impact * impact_profile / active_days
                        per_ticker[ticker]["impacts"][start_day:end_day] += daily_impact
                        per_ticker[ticker]["impacts"] = np.clip(per_ticker[ticker]["impacts"], -0.03, 0.03)

            event_entry = {
                "event_key": event_key,
                "name": event_spec["name"],
                "emoji": event_spec.get("emoji", ""),
                "start_day": start_day,
                "end_day": end_day,
                "duration": end_day - start_day,
                "impacts": impacts,
            }
            event_log.append(event_entry)
            logger.info(
                f"  {event_spec.get('emoji', '📌')} Event triggered: {event_spec['name']} "
                f"(day {start_day}–{end_day}, {end_day - start_day}d)"
            )

        return {
            "event_log": event_log,
            "per_ticker": per_ticker,
        }

    def generate_scenario_impacts(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Generate impact table for each event at 10th/50th/90th percentile.

        Used for the scenario impact analysis (no stochastic triggering).
        Returns: {event_key: {ticker: {p10, p50, p90}}}
        """
        n_samples = 10000
        scenario_table = {}

        for event_key, event_spec in self.events.items():
            ticker_impacts = {}
            for ticker in TICKERS:
                impact_spec = event_spec["portfolio_impact"].get(ticker, {})
                if not impact_spec:
                    ticker_impacts[ticker] = {"p10": 0.0, "p50": 0.0, "p90": 0.0}
                    continue

                samples = np.array([self._sample_impact(impact_spec) for _ in range(n_samples)])
                ticker_impacts[ticker] = {
                    "p10": float(np.percentile(samples, 10)),
                    "p50": float(np.percentile(samples, 50)),
                    "p90": float(np.percentile(samples, 90)),
                }
            scenario_table[event_key] = {
                "name": event_spec["name"],
                "emoji": event_spec.get("emoji", ""),
                "probability": event_spec["probability_per_quarter"],
                "impacts": ticker_impacts,
            }

        return scenario_table


# ---------------------------------------------------------------------------
# Master event generation function
# ---------------------------------------------------------------------------

def run_event_generation(
    n_days: int = 90,
    pe_data: Optional[Dict[str, float]] = None,
    enabled_events: Optional[List[str]] = None,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Generate event timelines and scenario impact tables.

    Returns:
        Dict with 'timeline_data', 'scenario_table', 'generator'.
    """
    logger.info("=" * 60)
    logger.info("MODULE 5 — Geopolitical & Macro Event Generator")
    logger.info("=" * 60)

    generator = EventGenerator(seed=seed, enabled_events=enabled_events)

    # Generate stochastic timeline for this simulation run
    logger.info(f"\n  Generating event timeline for {n_days} days...")
    timeline_data = generator.generate_event_timeline(n_days, pe_data)

    n_events = len(timeline_data["event_log"])
    logger.info(f"  ✓ {n_events} event(s) triggered in this run")

    if n_events == 0:
        logger.info("  (No events triggered — benign macro environment)")

    # Generate deterministic scenario impact table (for analytics)
    logger.info(f"\n  Computing scenario impact table (10K samples per event)...")
    scenario_table = generator.generate_scenario_impacts()
    logger.info(f"  ✓ Scenario table generated for {len(scenario_table)} events")

    result = {
        "timeline_data": timeline_data,
        "scenario_table": scenario_table,
        "generator": generator,
        "event_log": timeline_data["event_log"],
    }

    logger.info("\n✓ Event generation complete.")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_event_generation(n_days=90)
    print(f"\nTriggered events: {[e['name'] for e in result['event_log']]}")
    for event_key, data in result["scenario_table"].items():
        print(f"\n{data['emoji']} {data['name']} (p={data['probability']:.0%}):")
        for ticker in TICKERS:
            imp = data["impacts"][ticker]
            print(f"  {ticker}: p10={imp['p10']:+.1%} | p50={imp['p50']:+.1%} | p90={imp['p90']:+.1%}")
