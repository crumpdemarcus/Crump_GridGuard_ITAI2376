# ============================================================
# GridGuard-AI: Weather Analyst Agent
# Agent built by: DeMarcus Crump
# Course: ITAI 2376 - Deep Learning in AI
# ============================================================

import os
import requests
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai import LLM
from crewai.tools import tool

load_dotenv()

# ============================================================
# The Brain
# ============================================================
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0
)

# ============================================================
# The Tool: Fetch Texas Weather
# ============================================================
@tool("fetch_texas_weather")
def fetch_texas_weather(query: str) -> str:
    """Fetches real-time weather conditions for 5 major Texas regions
    including temperature and wind speed to assess physical grid risk."""

    regions = [
        {"name": "Houston",     "lat": 29.76, "lon": -95.37},
        {"name": "Dallas",      "lat": 32.78, "lon": -96.80},
        {"name": "Austin",      "lat": 30.27, "lon": -97.74},
        {"name": "West Texas",  "lat": 31.99, "lon": -102.08},
        {"name": "South Texas", "lat": 27.80, "lon": -97.40},
    ]

    report_lines = []
    risk_detected = False

    for region in regions:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={region['lat']}&longitude={region['lon']}"
            f"&current=temperature_2m,windspeed_10m"
            f"&temperature_unit=fahrenheit&windspeed_unit=mph"
        )
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()["current"]
            temp_f = data["temperature_2m"]
            wind_mph = data["windspeed_10m"]

            severity = ""
            # Extreme cold or heat can cause grid failures
            if temp_f < 20 or temp_f > 100:
                risk_detected = True
                severity = " [HIGH RISK]"

            report_lines.append(
                f"{region['name']}: Temp {temp_f:.1f}°F, Wind {wind_mph:.1f} mph{severity}"
            )
        except Exception as e:
            report_lines.append(f"{region['name']}: ERROR - {e}")

    summary = "\n".join(report_lines)
    status = "[WARNING] PHYSICAL GRID RISK DETECTED" if risk_detected else "[OK] Weather Normal"

    return (
        f"TEXAS WEATHER RISK REPORT\n"
        f"=========================\n"
        f"{summary}\n\n"
        f"Status: {status}\n"
        f"Note: Extreme temperatures (<20F or >100F) signal physical stress on infrastructure."
    )

# ============================================================
# The Agent
# ============================================================
weather_analyst = Agent(
    role="Chief Climate Risk Officer",
    goal=(
        "Monitor real-time weather conditions across Texas and flag "
        "meteorological threats such as extreme temperatures or storms "
        "that pose a physical risk to the power grid."
    ),
    backstory=(
        "You are an expert meteorologist who has studied the devastating "
        "impacts of extreme weather on power infrastructure in Texas. "
        "You isolate physical meteorological risks before they cause "
        "cascading failures."
    ),
    tools=[fetch_texas_weather],
    llm=llm,
    verbose=True
)

# ============================================================
# The Task
# ============================================================
weather_task = Task(
    description=(
        "Fetch the current weather conditions for the 5 major Texas "
        "regions. Identify any extreme temperature risks. "
        "Provide a weather physical risk report with specific temperatures "
        "and a risk verdict."
    ),
    expected_output=(
        "A weather report containing: current temperatures and wind "
        "speeds per region, identification of any weather extremes, "
        "and a final physical risk assessment (NORMAL / ELEVATED / CRITICAL) "
        "with meteorological context."
    ),
    agent=weather_analyst
)

# ============================================================
# The Crew
# ============================================================
crew = Crew(
    agents=[weather_analyst],
    tasks=[weather_task],
    verbose=True
)

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("GRIDGUARD-AI: WEATHER ANALYST AGENT")
    print("="*60 + "\n")
    result = crew.kickoff()
    print("\n" + "="*60)
    print("FINAL REPORT:")
    print("="*60)
    print(result)
