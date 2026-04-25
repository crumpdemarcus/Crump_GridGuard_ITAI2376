# ============================================================
# GridGuard-AI: Orchestrator
# Course: ITAI 2376 - Deep Learning in AI
# ============================================================

from crewai import Crew, Process

# Import all agents and tasks
from src.weather_agent import weather_analyst, weather_task
from src.compliance_agent import compliance_officer, compliance_task
from src.renewable_agent import renewable_forecaster, renewable_task
from src.market_agent import market_analyst, market_task
from src.grid_monitor_agent import grid_monitor, monitor_task
from src.grid_operator_agent import grid_operator, operator_task

# ============================================================
# Architecture Configuration (Fan-Out / Fan-In)
# ============================================================

# Fan-Out Phase 1 & 2: Set gathering tasks to run concurrently.
weather_task.async_execution = True
compliance_task.async_execution = True
renewable_task.async_execution = True
monitor_task.async_execution = True
market_task.async_execution = True

# Fan-In Phase 3: The decision-maker explicitly waits for context.
operator_task.context = [
    weather_task,
    compliance_task,
    renewable_task,
    monitor_task,
    market_task
]

# ============================================================
# Unified Crew Kickoff
# ============================================================

gridguard_crew = Crew(
    agents=[
        weather_analyst,
        compliance_officer,
        renewable_forecaster,
        market_analyst,
        grid_monitor,
        grid_operator
    ],
    tasks=[
        weather_task,
        compliance_task,
        renewable_task,
        market_task,
        monitor_task,
        operator_task
    ],
    # Even though it's set to sequential, the async_execution config
    # hijacks the flow to execute parallel threads until the Operator.
    process=Process.sequential,
    verbose=True,
    # Disable tool-result caching: scenario replay mode changes what each
    # tool returns based on the GRIDGUARD_SCENARIO env var. With cache=True
    # (the CrewAI default) a tool called with identical arguments in a
    # later run returns the cached result from the previous run, which
    # silently leaks Live data into Heatwave runs and Heatwave data into
    # Storm Uri runs. Disabling the cache forces every run to re-execute
    # tools so they pick up the current scenario.
    cache=False,
)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("INITIALIZING GRIDGUARD-AI EXPERT ORCHESTRATION (FAN-OUT/FAN-IN)")
    print("="*80 + "\n")
    
    # Kicking off the Crew will automatically start Phase 1 and 2 in parallel.
    final_output = gridguard_crew.kickoff()
    
    print("\n" + "="*80)
    print("ORCHESTRATION COMPLETED. FINAL SYSTEM REPORT:")
    print("="*80)
    print(final_output)
