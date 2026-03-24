cat > README.md << 'EOF'
# GridGuard-AI: Autonomous Power Grid Monitoring Agents

**ITAI 2376 - Deep Learning in Artificial Intelligence | Spring 2026**
**Built by: Yoana Cook**
**Houston Community College**

## Overview
GridGuard-AI is an autonomous multi-agent system that monitors the Texas (ERCOT) power grid in real time. Inspired by the February 2021 Winter Storm Uri disaster that left 4.5 million Texans without power, this system uses AI agents to detect emerging grid stress before it becomes a crisis.

## Agents
- **renewable_agent.py** — Fetches live wind/solar data, calculates MW output, assesses grid stability risk
- **market_agent.py** — Monitors ERCOT energy prices across 4 hubs, detects price spikes signaling grid stress  
- **compliance_agent.py** — Searches NERC/ERCOT regulatory database, identifies required actions for current conditions

## Tech Stack
- Framework: CrewAI | LLM: Llama 3.3 70B via Groq | Python 3.11

## Setup
pip install crewai langchain-groq requests python-dotenv
Add GROQ_API_KEY to .env file and run any agent.
EOF