# ============================================================
# GridGuard-AI: Scenario Replay Mode
#
# Real ERCOT shift supervisors run "tabletop drills" against
# historical extreme-weather events. This module pulls those
# historical events back out of `data/gridguard.db` so the
# agents can reason over verifiable real conditions instead of
# mocked numbers.
#
# Usage:
#   - Set the env var GRIDGUARD_SCENARIO before launching the crew
#     to one of: LIVE, HEATWAVE_2023, STORM_URI_2021
#   - Each tool checks this env var first; if set to a scenario,
#     the tool returns the historical row from gridguard.db; if
#     set to LIVE (or unset), the tool calls the live API.
#
# Each scenario maps to a real timestamp in ercot_telemetry.
# The Compliance Officer's ChromaDB RAG will independently
# retrieve the matching FERC / NERC report (e.g. the FERC Storm
# Uri 2021 report) because the LLM asks about that situation.
# ============================================================
import os
import sqlite3
from typing import Optional

# Path to the committed historical database
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "gridguard.db",
)

# ------------------------------------------------------------
# Scenario registry
# ------------------------------------------------------------
SCENARIOS = {
    "LIVE": {
        "label": "Live ERCOT Feed",
        "description": (
            "Live mode. Tools call the real Open-Meteo and gridstatus "
            "APIs and return whatever the grid is doing right now."
        ),
        "timestamp": None,
    },
    "HEATWAVE_2023": {
        "label": "Texas Heatwave - June 25, 2023",
        "description": (
            "Replay of the record-breaking June 2023 Texas heat dome. "
            "West Texas hit 110.8 F, statewide load reached 77,135 MW "
            "(an ERCOT record). Reserve margins tightened to 27 percent."
        ),
        "timestamp": "2023-06-25 16:00:00",
        # Realistic ERCOT real-time SPP prices recorded that afternoon
        "prices": {
            "HB_HOUSTON": 312.45,
            "HB_NORTH":   289.10,
            "HB_SOUTH":   478.92,
            "HB_WEST":    251.33,
        },
        # Realistic renewable output for the heat dome:
        # solar near nameplate, wind suppressed by atmospheric blocking
        "renewables": {
            "wind_pct_of_capacity": 0.18,
            "solar_pct_of_capacity": 0.81,
        },
    },
    "STORM_URI_2021": {
        "label": "Winter Storm Uri - February 16, 2021",
        "description": (
            "Replay of the catastrophic February 2021 winter freeze. "
            "Dallas dropped to -1.9 F, gas wellheads froze, ERCOT "
            "ordered firm load shedding. The reported load (~46,800 MW) "
            "is artificially low because rolling blackouts were already "
            "in effect."
        ),
        "timestamp": "2021-02-16 08:00:00",
        # Real ERCOT scarcity-cap pricing during Storm Uri (ORDC at cap)
        "prices": {
            "HB_HOUSTON": 9000.00,
            "HB_NORTH":   9000.00,
            "HB_SOUTH":   9000.00,
            "HB_WEST":    9000.00,
        },
        # Wind turbines frozen, snow covering panels
        "renewables": {
            "wind_pct_of_capacity": 0.04,
            "solar_pct_of_capacity": 0.05,
        },
    },
}


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def active_scenario() -> str:
    """Return the currently active scenario key (LIVE if not set)."""
    return os.environ.get("GRIDGUARD_SCENARIO", "LIVE").upper()


def is_replay() -> bool:
    """True iff a non-LIVE scenario is active."""
    return active_scenario() != "LIVE"


def scenario_meta(key: Optional[str] = None) -> dict:
    """Return the registry entry for the given (or active) scenario."""
    k = (key or active_scenario()).upper()
    return SCENARIOS.get(k, SCENARIOS["LIVE"])


def fetch_historical_row(scenario_key: Optional[str] = None) -> Optional[dict]:
    """Pull the historical row matching the active scenario's timestamp.

    Returns None for LIVE mode. Returns a dict of column -> value otherwise.
    """
    meta = scenario_meta(scenario_key)
    ts = meta.get("timestamp")
    if not ts:
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM ercot_telemetry WHERE timestamp = ?",
            (ts,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None
