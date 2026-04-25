# ============================================================
# GridGuard-AI: Market Analyst Agent
# Agent built by: Yoana Cook
# Course: ITAI 2376 - Deep Learning in AI
# ============================================================

import os
import requests
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai import LLM
from crewai.tools import tool

from src.scenarios import is_replay, scenario_meta

load_dotenv()

# ============================================================
# The Brain
# ============================================================
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0
)

# ============================================================
# The Tool: Fetch Live ERCOT Energy Prices
# ============================================================
@tool("fetch_realtime_prices")
def fetch_realtime_prices(query: str) -> str:
    """Fetches real-time energy prices from the four major
    ERCOT trading hubs and identifies price spike risks."""

    # ERCOT's 4 major trading hubs
    hubs = ["HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST"]

    # ---- Scenario Replay Mode -----------------------------------------
    # When a historical scenario is active, replay the canonical hub
    # prices for that event (e.g. ORDC-cap $9000/MWh during Storm Uri).
    if is_replay():
        meta = scenario_meta()
        scenario_prices = meta.get("prices")
        if scenario_prices:
            report_lines = []
            spike_detected = False
            highest_price = 0
            highest_hub = ""
            for hub in hubs:
                price = float(scenario_prices.get(hub, 0))
                if price > highest_price:
                    highest_price = price
                    highest_hub = hub
                if price > 100:
                    spike_detected = True
                report_lines.append(f"{hub}: ${price:.2f}/MWh")
            summary = "\n".join(report_lines)
            status = "[WARNING] PRICE SPIKE DETECTED" if spike_detected else "[OK] Prices Normal"
            return (
                f"ERCOT PRICE REPORT (scenario replay: {meta['label']})\n"
                f"=========================================================\n"
                f"Source: documented hub prices from {meta['timestamp']}\n\n"
                f"{summary}\n\n"
                f"Highest Price: ${highest_price:.2f}/MWh at {highest_hub}\n"
                f"Status: {status}\n"
                f"Note: Prices above $100/MWh signal emerging grid stress."
            )
    # ---- Live mode (default) ------------------------------------------

    # We'll use the ERCOT public API for real-time prices
    url = "https://api.ercot.com/api/public-reports/np6-905-cd/act_sys_load_by_wzn"

    try:
        import gridstatus
        ercot = gridstatus.Ercot()
        # gridstatus >= 0.25 requires the `market` kwarg. REAL_TIME_15_MIN
        # is the live 15-minute settlement-point-price feed used by ERCOT.
        prices = ercot.get_spp(date="latest", market="REAL_TIME_15_MIN")

        report_lines = []
        spike_detected = False
        highest_price = 0
        highest_hub = ""

        for hub in hubs:
            hub_data = prices[prices["Location"] == hub]
            if not hub_data.empty:
                price = float(hub_data["SPP"].iloc[-1])
                if price > highest_price:
                    highest_price = price
                    highest_hub = hub
                if price > 100:
                    spike_detected = True
                report_lines.append(f"{hub}: ${price:.2f}/MWh")
            else:
                report_lines.append(f"{hub}: No data")

        summary = "\n".join(report_lines)
        status = "[WARNING] PRICE SPIKE DETECTED" if spike_detected else "[OK] Prices Normal"

        return (
            f"ERCOT REAL-TIME PRICE REPORT (live gridstatus SPP feed)\n"
            f"========================================================\n"
            f"{summary}\n\n"
            f"Highest Price: ${highest_price:.2f}/MWh at {highest_hub}\n"
            f"Status: {status}\n"
            f"Note: Prices above $100/MWh signal emerging grid stress."
        )

    except Exception as e:
        # Fallback: deterministic illustrative prices if the live SPP feed is
        # unavailable (e.g. gridstatus version drift, ERCOT 403 from a cloud IP,
        # or network failure). Honest labeling: this is NOT live data.
        import random
        base_prices = {
            "HB_HOUSTON": round(random.uniform(28, 145), 2),
            "HB_NORTH":   round(random.uniform(25, 132), 2),
            "HB_SOUTH":   round(random.uniform(30, 138), 2),
            "HB_WEST":    round(random.uniform(22, 98), 2),
        }

        report_lines = []
        spike_detected = False
        highest_price = 0
        highest_hub = ""

        for hub, price in base_prices.items():
            if price > highest_price:
                highest_price = price
                highest_hub = hub
            if price > 100:
                spike_detected = True
            report_lines.append(f"{hub}: ${price:.2f}/MWh")

        summary = "\n".join(report_lines)
        status = "[WARNING] PRICE SPIKE DETECTED" if spike_detected else "[OK] Prices Normal"

        return (
            f"ERCOT PRICE REPORT (illustrative fallback - live SPP feed unavailable: {type(e).__name__})\n"
            f"================================================================\n"
            f"{summary}\n\n"
            f"Highest Price: ${highest_price:.2f}/MWh at {highest_hub}\n"
            f"Status: {status}\n"
            f"Note: Prices above $100/MWh signal emerging grid stress."
        )

# ============================================================
# The Agent
# ============================================================
market_analyst = Agent(
    role="Chief Market Intelligence Officer",
    goal=(
        "Monitor real-time energy prices across all four ERCOT "
        "trading hubs and identify price spikes that signal "
        "emerging grid stress before it becomes a crisis."
    ),
    backstory=(
        "You are a veteran energy trader who has watched the Texas "
        "power market for 15 years. You know that price spikes are "
        "the earliest warning signal of grid distress — they appear "
        "before physical reserves formally drop. When Houston hub "
        "prices cross $100/MWh, you've learned to start making calls."
    ),
    tools=[fetch_realtime_prices],
    llm=llm,
    verbose=True,
    cache=False
)

# ============================================================
# The Task
# ============================================================
market_task = Task(
    description=(
        "Fetch current real-time energy prices across all ERCOT "
        "trading hubs. Identify any price spikes above $100/MWh. "
        "Assess whether current pricing indicates emerging grid "
        "stress. Provide a market intelligence report with specific "
        "prices and a risk verdict."
    ),
    expected_output=(
        "A market intelligence report containing: current price "
        "per hub in $/MWh, identification of any spikes, and a "
        "final market risk assessment (NORMAL / ELEVATED / CRITICAL) "
        "with trading context and justification."
    ),
    agent=market_analyst
)

# ============================================================
# The Crew
# ============================================================
crew = Crew(
    agents=[market_analyst],
    tasks=[market_task],
    verbose=True
)

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("GRIDGUARD-AI: MARKET ANALYST AGENT")
    print("="*60 + "\n")
    result = crew.kickoff()
    print("\n" + "="*60)
    print("FINAL REPORT:")
    print("="*60)
    print(result)