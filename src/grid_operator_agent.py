# ============================================================
# GridGuard-AI: Grid Operator Agent
# Agent built by: DeMarcus Crump
# Course: ITAI 2376 - Deep Learning in AI
# ============================================================

import os
import json
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field, ValidationError
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
# The Pydantic Schema (Physical & Regulatory Constraints)
# ------------------------------------------------------------
# Bounds are anchored to the ERCOT EEA Load Shed Tables (EEA3
# controlled outages begin at reserves < 1,750 MW, with load-shed
# blocks typically capped at ~5,000 MW per dispatch event). Positive
# target_mw = emergency generation deployment. Negative target_mw =
# load shedding. Pydantic mathematically clamps LLM output to these
# bounds so the Grid Operator cannot hallucinate a dispatch that
# violates federal reliability thresholds.
# ============================================================
_MAX_EMERGENCY_DEPLOY_MW = 5000
_MAX_LOAD_SHED_MW = -5000


class DispatchSchema(BaseModel):
    action: str = Field(
        ...,
        description=(
            "The dispatch action. Allowed values include "
            "'deploy_reserves', 'load_shed', 'emergency_dispatch', "
            "'monitor', 'declare_eea1', 'declare_eea2', 'declare_eea3'."
        ),
    )
    target_mw: int = Field(
        ...,
        description=(
            f"MW to dispatch. Positive for emergency generation, negative "
            f"for load shed. Clamped to [{_MAX_LOAD_SHED_MW}, "
            f"{_MAX_EMERGENCY_DEPLOY_MW}] per ERCOT Load Shed Tables."
        ),
        ge=_MAX_LOAD_SHED_MW,
        le=_MAX_EMERGENCY_DEPLOY_MW,
    )
    compliance_verified: bool = Field(
        ...,
        description="True iff action was cleared against the Compliance Officer's NERC/ERCOT findings.",
    )
    notes: Optional[str] = Field(
        default="",
        description="Plain-English justification referencing the specific NERC/ERCOT standard(s) that triggered this action.",
    )

# ============================================================
# The Tool: Save Dispatch Report
# ============================================================
@tool("save_dispatch_report")
def save_dispatch_report(dispatch_data: str) -> str:
    """Validates and saves the final dispatch commands as a structured JSON.
    Provide the dispatch string in valid JSON format including 'action',
    'target_mw', and 'compliance_verified'."""
    
    try:
        data = json.loads(dispatch_data)
        
        # Pydantic STRICT validation (this clamps the LLM output mathematically)
        validated_data = DispatchSchema(**data)
        
        # Save to file at repo root
        with open("final_dispatch.json", "w") as f:
            json.dump(validated_data.model_dump(), f, indent=4)
            
        return "[SUCCESS] DISPATCH REPORT SUCCESSFULLY VALIDATED AND SAVED AS JSON."
        
    except json.JSONDecodeError:
        return "ERROR: Invalid JSON format provided. Please format as strict JSON."
    except ValidationError as e:
        # If the LLM tries to surpass the ERCOT Load Shed Table bounds, Pydantic halts and re-prompts the LLM
        return f"PYDANTIC VALIDATION ERROR: The dispatch data violates infrastructure constraints:\n{e.json()}"

# ============================================================
# The Agent
# ============================================================
grid_operator = Agent(
    role="Chief Energy Trading Strategist & Dispatcher",
    goal=(
        "Synthesize reports from Weather, Compliance, Renewable, "
        "Market, and Grid Monitor agents. Make the final deterministic "
        "decision on grid dispatch actions and output them in a strict JSON format."
    ),
    backstory=(
        "You are the final authority in the control room. You do not "
        "gather raw API data; you listen to your Chief Officers. "
        "If they report wind drops, tight reserves, falling temps, "
        "and spike prices, you pull the trigger on emergency dispatch. "
        "Your decisions must strictly adhere strictly to the Compliance Officer's limits."
    ),
    tools=[save_dispatch_report],
    llm=llm,
    verbose=True,
    cache=False
)

# ============================================================
# The Task
# ============================================================
operator_task = Task(
    description=(
        "Given the hypothetical inputs from other agents (Renewables are DROPPING, "
        "Load is SPIKING past LSTM predictions, ERCOT reserves at 2100 MW, "
        "Compliance requires EEA2 protocols), synthesize a final dispatch action.\n"
        "You MUST use the save_dispatch_report tool and provide a JSON string "
        "containing `action`, `target_mw`, and `compliance_verified`."
    ),
    expected_output=(
        "A confirmation of the saved JSON dispatch report, followed by a "
        "plain-english justification of the final decision."
    ),
    agent=grid_operator
)

# ============================================================
# The Crew
# ============================================================
crew = Crew(
    agents=[grid_operator],
    tasks=[operator_task],
    verbose=True
)

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("GRIDGUARD-AI: GRID OPERATOR AGENT")
    print("="*60 + "\n")
    result = crew.kickoff()
    print("\n" + "="*60)
    print("FINAL DISPATCH EXECUTED:")
    print("="*60)
    print(result)
