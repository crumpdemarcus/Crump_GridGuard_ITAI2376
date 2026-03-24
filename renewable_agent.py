# ============================================================
# GridGuard-AI: Renewable Forecaster Agent
# Agent built by: Yoana Cook
# Course: ITAI 2376 - Deep Learning in AI
# ============================================================

import os
import requests
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

# Load your API key from the .env file
load_dotenv()
# ============================================================
# The Brain: Connect to Llama3 via Groq
# ============================================================
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# ============================================================
# The Tool: Fetch Live Weather & Calculate Renewable Output
# ============================================================
@tool("estimate_renewable_output")
def estimate_renewable_output(query: str) -> str:
    """Fetches live solar and wind data for Texas and estimates
    current renewable energy output in MW."""

    regions = [
        {"name": "West Texas",    "lat": 31.99, "lon": -102.08},
        {"name": "South Texas",   "lat": 27.80, "lon": -97.40},
        {"name": "Houston",       "lat": 29.76, "lon": -95.37},
        {"name": "Dallas",        "lat": 32.78, "lon": -96.80},
    ]

    # Texas grid capacity (approximate)
    WIND_CAPACITY_MW = 40000   # Texas has ~40,000 MW of wind capacity
    SOLAR_CAPACITY_MW = 8000   # Texas has ~8,000 MW of solar capacity

    total_wind_mw = 0
    total_solar_mw = 0
    report_lines = []

    for region in regions:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={region['lat']}&longitude={region['lon']}"
            f"&current=windspeed_10m,direct_radiation"
            f"&windspeed_unit=mph"
        )
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()["current"]
            wind_mph = data["windspeed_10m"]
            solar_wm2 = data["direct_radiation"]

            # Estimate MW: wind power scales with cube of wind speed
            wind_factor = min((wind_mph / 30) ** 3, 1.0)
            solar_factor = min(solar_wm2 / 1000, 1.0)

            region_wind_mw = (WIND_CAPACITY_MW / len(regions)) * wind_factor
            region_solar_mw = (SOLAR_CAPACITY_MW / len(regions)) * solar_factor

            total_wind_mw += region_wind_mw
            total_solar_mw += region_solar_mw

            report_lines.append(
                f"{region['name']}: Wind {wind_mph:.1f} mph "
                f"({region_wind_mw:.0f} MW) | "
                f"Solar {solar_wm2:.0f} W/m² "
                f"({region_solar_mw:.0f} MW)"
            )
        except Exception as e:
            report_lines.append(f"{region['name']}: ERROR - {e}")

    total_mw = total_wind_mw + total_solar_mw
    pct_of_capacity = (total_mw / (WIND_CAPACITY_MW + SOLAR_CAPACITY_MW)) * 100

    summary = "\n".join(report_lines)
    return (
        f"RENEWABLE OUTPUT REPORT\n"
        f"=======================\n"
        f"{summary}\n\n"
        f"Total Wind: {total_wind_mw:.0f} MW\n"
        f"Total Solar: {total_solar_mw:.0f} MW\n"
        f"Combined Output: {total_mw:.0f} MW "
        f"({pct_of_capacity:.1f}% of total capacity)\n"
        f"Status: {'⚠️ LOW OUTPUT - Grid stress risk' if pct_of_capacity < 30 else '✅ Normal output'}"
    )
# ============================================================
# The Agent: Renewable Forecaster
# ============================================================
renewable_forecaster = Agent(
    role="Chief Renewable Energy Forecaster",
    goal=(
        "Analyze current wind and solar conditions across Texas "
        "and assess whether renewable energy output poses a risk "
        "to grid stability."
    ),
    backstory=(
        "You are a senior energy analyst specializing in renewable "
        "integration for the Texas power grid. You understand that "
        "sudden drops in wind or solar output can destabilize ERCOT "
        "and trigger cascading failures. Your job is to catch these "
        "drops before they hit the physical grid."
    ),
    tools=[estimate_renewable_output],
    llm=llm,
    verbose=True
)

# ============================================================
# The Task: What we're asking the agent to do
# ============================================================
renewable_task = Task(
    description=(
        "Fetch the current renewable energy output across Texas. "
        "Analyze wind and solar conditions for each region. "
        "Determine if current output levels represent a risk to "
        "grid stability. Provide a clear assessment with specific "
        "MW numbers and a final risk verdict."
    ),
    expected_output=(
        "A renewable energy report containing: current output by "
        "region in MW, total combined output, percentage of capacity, "
        "and a final risk assessment (LOW RISK / ELEVATED RISK / "
        "CRITICAL RISK) with justification."
    ),
    agent=renewable_forecaster
)

# ============================================================
# The Crew: Put it all together and run it
# ============================================================
crew = Crew(
    agents=[renewable_forecaster],
    tasks=[renewable_task],
    verbose=True
)

# ============================================================
# Run the agent
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("GRIDGUARD-AI: RENEWABLE FORECASTER AGENT")
    print("="*60 + "\n")
    result = crew.kickoff()
    print("\n" + "="*60)
    print("FINAL REPORT:")
    print("="*60)
    print(result)