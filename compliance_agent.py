# ============================================================
# GridGuard-AI: Regulatory Compliance Agent
# Agent built by: Yoana Cook
# Course: ITAI 2376 - Deep Learning in AI
# ============================================================

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

load_dotenv()

# ============================================================
# The Brain
# ============================================================
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# ============================================================
# The Knowledge Base: NERC & ERCOT Regulatory Standards
# This simulates what ChromaDB + RAG does — stores regulatory
# knowledge the agent can search and retrieve from
# ============================================================
REGULATORY_KNOWLEDGE_BASE = {
    "BAL-001": {
        "title": "Real Power Balancing Control Performance",
        "standard": "NERC BAL-001",
        "description": "Requires balancing authorities to maintain "
            "generation-load balance. Reserve margin must stay above "
            "13.75% at all times. Violation triggers emergency protocols.",
        "threshold": "Reserve margin < 13.75%",
        "action": "Activate emergency demand response immediately"
    },
    "BAL-002": {
        "title": "Disturbance Control Performance",
        "standard": "NERC BAL-002",
        "description": "Requires recovery from disturbances within "
            "15 minutes. ERCOT must restore frequency to 59.95-60.05 Hz "
            "within 15 minutes of any disturbance event.",
        "threshold": "Frequency deviation > 0.05 Hz for > 15 minutes",
        "action": "Deploy spinning reserves within 10 minutes"
    },
    "EOP-011": {
        "title": "Emergency Operations",
        "standard": "NERC EOP-011",
        "description": "Requires ERCOT to have emergency operating "
            "procedures for conditions threatening grid reliability. "
            "Includes load shedding protocols when reserves fall "
            "below operational thresholds.",
        "threshold": "Reserves < 2,300 MW (ERCOT minimum)",
        "action": "Initiate controlled load shedding by region"
    },
    "IRO-006": {
        "title": "Reliability Coordination",
        "standard": "NERC IRO-006",
        "description": "Requires reliability coordinators to take "
            "actions to prevent instability, uncontrolled separation, "
            "or cascading outages. ERCOT must act within 30 minutes "
            "of identifying an emerging reliability threat.",
        "threshold": "Any condition threatening N-1 contingency",
        "action": "Issue reliability coordinator directives within 30 min"
    },
    "URI_LESSONS": {
        "title": "Winter Storm Uri Post-Mortem",
        "standard": "FERC/NERC Joint Report 2021",
        "description": "February 2021: 4.5M homes lost power. Root causes: "
            "inadequate weatherization of natural gas facilities, "
            "over-reliance on weather-sensitive generation without "
            "backup, and failure to act on prior winterization warnings "
            "from 2011 review. Key lesson: act on early warning signals "
            "before cascade begins.",
        "threshold": "Temperature forecast below 20°F statewide",
        "action": "Require generator weatherization verification 72hrs prior"
    },
    "ERCOT_PROTOCOLS": {
        "title": "ERCOT Emergency Operating Procedures",
        "standard": "ERCOT Nodal Protocols Section 6",
        "description": "ERCOT declares Energy Emergency Alerts (EEA) "
            "in three levels: EEA1 (reserves 2300-3000 MW), "
            "EEA2 (reserves 1750-2300 MW), EEA3 (< 1750 MW, "
            "controlled outages begin). Public conservation requests "
            "issued at EEA1.",
        "threshold": "Reserves below 3,000 MW",
        "action": "Declare EEA1, request public conservation"
    }
}

# ============================================================
# The Tool: Search Compliance Database
# ============================================================
@tool("search_compliance_database")
def search_compliance_database(query: str) -> str:
    """Searches the NERC and ERCOT regulatory knowledge base
    for relevant compliance standards, thresholds, and required
    actions based on current grid conditions."""

    query_lower = query.lower()
    relevant_standards = []

    # Search logic — match query keywords to knowledge base
    keyword_map = {
        "reserve": ["BAL-001", "EOP-011", "ERCOT_PROTOCOLS"],
        "frequency": ["BAL-002"],
        "emergency": ["EOP-011", "ERCOT_PROTOCOLS", "IRO-006"],
        "winter": ["URI_LESSONS"],
        "temperature": ["URI_LESSONS"],
        "wind": ["URI_LESSONS", "BAL-001"],
        "solar": ["BAL-001"],
        "renewable": ["BAL-001", "URI_LESSONS"],
        "price": ["ERCOT_PROTOCOLS"],
        "spike": ["ERCOT_PROTOCOLS", "IRO-006"],
        "load": ["EOP-011", "ERCOT_PROTOCOLS"],
        "cascade": ["IRO-006", "URI_LESSONS"],
        "balance": ["BAL-001", "BAL-002"],
        "disturbance": ["BAL-002", "IRO-006"],
    }

    matched_ids = set()
    for keyword, standard_ids in keyword_map.items():
        if keyword in query_lower:
            matched_ids.update(standard_ids)

    # Default: return top 3 most relevant if no keyword match
    if not matched_ids:
        matched_ids = ["BAL-001", "EOP-011", "ERCOT_PROTOCOLS"]

    for std_id in matched_ids:
        if std_id in REGULATORY_KNOWLEDGE_BASE:
            std = REGULATORY_KNOWLEDGE_BASE[std_id]
            relevant_standards.append(
                f"Standard: {std['standard']}\n"
                f"Title: {std['title']}\n"
                f"Rule: {std['description']}\n"
                f"Trigger Threshold: {std['threshold']}\n"
                f"Required Action: {std['action']}\n"
            )

    result = "\n---\n".join(relevant_standards)
    return (
        f"REGULATORY COMPLIANCE SEARCH RESULTS\n"
        f"Query: '{query}'\n"
        f"=====================================\n"
        f"{result}\n"
        f"=====================================\n"
        f"Retrieved {len(relevant_standards)} relevant standards."
    )

# ============================================================
# The Agent
# ============================================================
compliance_officer = Agent(
    role="Chief Regulatory Compliance Officer",
    goal=(
        "Ensure all GridGuard-AI dispatch recommendations comply "
        "with NERC reliability standards and ERCOT operating "
        "protocols. Identify which regulations apply to current "
        "grid conditions and what actions are legally required."
    ),
    backstory=(
        "You are a former NERC auditor with 20 years of experience "
        "reviewing grid reliability incidents. You were on the "
        "investigation team after Winter Storm Uri and you know "
        "exactly which warning signs were ignored. Your job is to "
        "make sure that never happens again — by identifying the "
        "exact regulatory requirements that apply to any emerging "
        "grid condition before it becomes a crisis."
    ),
    tools=[search_compliance_database],
    llm=llm,
    verbose=True
)

# ============================================================
# The Task
# ============================================================
compliance_task = Task(
    description=(
        "Current grid conditions: renewable output is at ELEVATED "
        "RISK (6,353 MW, 13.2% of capacity). Energy prices show "
        "ELEVATED risk with spikes detected at multiple hubs. "
        "Search the regulatory database and identify: which NERC "
        "standards apply to these conditions, what thresholds have "
        "been crossed or are at risk of being crossed, and what "
        "actions ERCOT is legally required to take right now."
    ),
    expected_output=(
        "A compliance report identifying: applicable NERC/ERCOT "
        "standards for current conditions, specific thresholds at "
        "risk, legally required actions with timeframes, and a "
        "compliance risk level (COMPLIANT / WARNING / VIOLATION)."
    ),
    agent=compliance_officer
)

# ============================================================
# The Crew
# ============================================================
crew = Crew(
    agents=[compliance_officer],
    tasks=[compliance_task],
    verbose=True
)

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("GRIDGUARD-AI: COMPLIANCE OFFICER AGENT")
    print("="*60 + "\n")
    result = crew.kickoff()
    print("\n" + "="*60)
    print("FINAL COMPLIANCE REPORT:")
    print("="*60)
    print(result)