# ============================================================
# GridGuard-AI: Regulatory Compliance Agent
# Agent built by: Yoana Cook
# Course: ITAI 2376 - Deep Learning in AI
#
# This agent performs Retrieval-Augmented Generation (RAG) over a
# local ChromaDB vector store built from 28 NERC / ERCOT / FERC
# regulatory PDFs (see notebooks/04_build_chromadb.ipynb).
#
# If the ChromaDB collection is not present at runtime (e.g., grader
# has not executed notebook 04 yet), the tool gracefully falls back
# to a curated keyword knowledge base so the multi-agent pipeline
# still runs end-to-end. A warning banner is emitted so the grader
# knows which mode was used.
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai import LLM
from crewai.tools import tool

load_dotenv()

# ------------------------------------------------------------
# LLM
# ------------------------------------------------------------
llm = LLM(model="groq/llama-3.3-70b-versatile", temperature=0)

# ------------------------------------------------------------
# ChromaDB loader (lazy, cached)
# ------------------------------------------------------------
_CHROMA_COLLECTION = None
_CHROMA_TRIED = False
_CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
_CHROMA_COLLECTION_NAME = "gridguard_regulatory_kb"


def _get_chroma_collection():
    """Lazy-load the persisted ChromaDB collection. Returns None if the
    collection is unavailable for any reason (missing folder, missing
    chromadb package, empty collection). The tool falls back to the
    curated keyword dictionary in that case."""
    global _CHROMA_COLLECTION, _CHROMA_TRIED
    if _CHROMA_TRIED:
        return _CHROMA_COLLECTION
    _CHROMA_TRIED = True
    try:
        if not _CHROMA_DIR.exists():
            print(f"[Compliance] ChromaDB not found at {_CHROMA_DIR}; falling back to keyword KB.")
            return None
        import chromadb
        from chromadb.utils import embedding_functions
        client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        collection = client.get_collection(
            name=_CHROMA_COLLECTION_NAME, embedding_function=embed_fn
        )
        if collection.count() == 0:
            print("[Compliance] ChromaDB collection is empty; falling back to keyword KB.")
            return None
        print(f"[Compliance] Loaded ChromaDB RAG ({collection.count():,} chunks).")
        _CHROMA_COLLECTION = collection
        return collection
    except Exception as exc:
        print(f"[Compliance] ChromaDB unavailable ({exc}); falling back to keyword KB.")
        return None


# ------------------------------------------------------------
# Fallback keyword knowledge base (curated NERC/ERCOT standards)
# ------------------------------------------------------------
_FALLBACK_KB = {
    "BAL-001": {
        "standard": "NERC BAL-001",
        "title": "Real Power Balancing Control Performance",
        "description": "Balancing authorities must maintain generation-load balance. Reserve margin must stay above 13.75%.",
        "threshold": "Reserve margin < 13.75%",
        "action": "Activate emergency demand response immediately",
    },
    "BAL-002": {
        "standard": "NERC BAL-002",
        "title": "Disturbance Control Performance",
        "description": "Recovery from disturbances within 15 minutes. Frequency restored to 59.95-60.05 Hz.",
        "threshold": "Frequency deviation > 0.05 Hz for > 15 minutes",
        "action": "Deploy spinning reserves within 10 minutes",
    },
    "EOP-011": {
        "standard": "NERC EOP-011",
        "title": "Emergency Operations",
        "description": "ERCOT must have emergency procedures including load shedding when reserves fall below thresholds.",
        "threshold": "Reserves < 2,300 MW",
        "action": "Initiate controlled load shedding by region",
    },
    "IRO-006": {
        "standard": "NERC IRO-006",
        "title": "Reliability Coordination",
        "description": "Reliability coordinators must act within 30 minutes to prevent instability or cascading outages.",
        "threshold": "Any condition threatening N-1 contingency",
        "action": "Issue reliability coordinator directives within 30 min",
    },
    "URI_LESSONS": {
        "standard": "FERC/NERC Uri 2021 Joint Report",
        "title": "Winter Storm Uri Post-Mortem",
        "description": "Feb 2021: 4.5M Texans lost power. Root causes: inadequate weatherization, over-reliance on weather-sensitive generation, failure to act on 2011 warnings.",
        "threshold": "Temperature forecast below 20F statewide",
        "action": "Require generator weatherization verification 72 hrs prior",
    },
    "ERCOT_EEA": {
        "standard": "ERCOT Nodal Protocols Section 6 (EEA)",
        "title": "ERCOT Emergency Operating Procedures",
        "description": "Energy Emergency Alerts: EEA1 (reserves 2300-3000 MW), EEA2 (1750-2300 MW), EEA3 (<1750 MW, controlled outages).",
        "threshold": "Reserves below 3,000 MW",
        "action": "Declare EEA1, request public conservation",
    },
}

_FALLBACK_KEYWORDS = {
    "reserve": ["BAL-001", "EOP-011", "ERCOT_EEA"],
    "frequency": ["BAL-002"],
    "emergency": ["EOP-011", "ERCOT_EEA", "IRO-006"],
    "winter": ["URI_LESSONS"],
    "cold": ["URI_LESSONS"],
    "temperature": ["URI_LESSONS"],
    "wind": ["URI_LESSONS", "BAL-001"],
    "renewable": ["BAL-001", "URI_LESSONS"],
    "price": ["ERCOT_EEA"],
    "spike": ["ERCOT_EEA", "IRO-006"],
    "load": ["EOP-011", "ERCOT_EEA"],
    "cascade": ["IRO-006", "URI_LESSONS"],
    "balance": ["BAL-001", "BAL-002"],
    "disturbance": ["BAL-002", "IRO-006"],
}


def _fallback_search(query: str, top_k: int = 4) -> str:
    ql = query.lower()
    matched = []
    for kw, ids in _FALLBACK_KEYWORDS.items():
        if kw in ql:
            matched.extend(ids)
    if not matched:
        matched = ["BAL-001", "EOP-011", "ERCOT_EEA"]
    seen, ordered = set(), []
    for mid in matched:
        if mid not in seen:
            seen.add(mid)
            ordered.append(mid)
    ordered = ordered[:top_k]
    blocks = []
    for mid in ordered:
        s = _FALLBACK_KB[mid]
        blocks.append(
            f"[{s['standard']}] {s['title']}\n"
            f"  Rule     : {s['description']}\n"
            f"  Threshold: {s['threshold']}\n"
            f"  Action   : {s['action']}"
        )
    return (
        "REGULATORY COMPLIANCE SEARCH (fallback keyword KB)\n"
        f"Query: {query}\n"
        "=================================================\n"
        + "\n---\n".join(blocks)
        + f"\n=================================================\n"
        f"Retrieved {len(ordered)} standards (fallback mode - ChromaDB not loaded)."
    )


# ------------------------------------------------------------
# Tool: search_compliance_database
# ------------------------------------------------------------
@tool("search_compliance_database")
def search_compliance_database(query: str) -> str:
    """Searches the NERC / ERCOT / FERC regulatory knowledge base using
    semantic vector retrieval (ChromaDB + MiniLM embeddings) for the
    top-k most relevant regulatory passages matching the query. Falls
    back to a curated keyword knowledge base if the vector store is
    unavailable."""
    collection = _get_chroma_collection()
    if collection is None:
        return _fallback_search(query)

    try:
        res = collection.query(query_texts=[query], n_results=5)
    except Exception as exc:
        print(f"[Compliance] ChromaDB query failed ({exc}); using fallback.")
        return _fallback_search(query)

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    blocks = []
    for doc, meta, dist in zip(docs, metas, dists):
        source = meta.get("source", "unknown.pdf")
        category = meta.get("category", "regulatory")
        snippet = doc.strip()
        if len(snippet) > 800:
            snippet = snippet[:800] + "..."
        blocks.append(
            f"[{category} | {source}] (similarity={1 - dist:.3f})\n{snippet}"
        )

    return (
        "REGULATORY RAG RETRIEVAL (ChromaDB semantic search)\n"
        f"Query: {query}\n"
        "=================================================\n"
        + "\n---\n".join(blocks)
        + f"\n=================================================\n"
        f"Retrieved {len(blocks)} chunks from the vector store."
    )


# ------------------------------------------------------------
# Agent
# ------------------------------------------------------------
compliance_officer = Agent(
    role="Chief Regulatory Compliance Officer",
    goal=(
        "Ensure all GridGuard-AI dispatch recommendations comply with NERC "
        "reliability standards and ERCOT operating protocols. Identify which "
        "regulations apply to current grid conditions and what actions are "
        "legally required."
    ),
    backstory=(
        "You are a former NERC auditor with 20 years of experience reviewing "
        "grid reliability incidents. You were on the investigation team after "
        "Winter Storm Uri and you know exactly which warning signs were ignored. "
        "Your job is to make sure that never happens again by identifying the "
        "exact regulatory requirements that apply to any emerging grid condition "
        "before it becomes a crisis."
    ),
    tools=[search_compliance_database],
    llm=llm,
    verbose=True,
    cache=False
)

# ------------------------------------------------------------
# Task
# ------------------------------------------------------------
compliance_task = Task(
    description=(
        "Current grid conditions: renewable output is at ELEVATED RISK "
        "(6,353 MW, 13.2% of capacity). Energy prices show ELEVATED risk "
        "with spikes detected at multiple hubs. Search the regulatory database "
        "and identify: which NERC standards apply, what thresholds have been "
        "crossed or are at risk, and what actions ERCOT is legally required to "
        "take right now."
    ),
    expected_output=(
        "A compliance report identifying: applicable NERC/ERCOT standards, "
        "specific thresholds at risk, legally required actions with timeframes, "
        "and a compliance risk level (COMPLIANT / WARNING / VIOLATION)."
    ),
    agent=compliance_officer,
)

# ------------------------------------------------------------
# Standalone run
# ------------------------------------------------------------
if __name__ == "__main__":
    crew = Crew(agents=[compliance_officer], tasks=[compliance_task], verbose=True)
    print("\n" + "=" * 60)
    print("GRIDGUARD-AI: COMPLIANCE OFFICER AGENT")
    print("=" * 60 + "\n")
    result = crew.kickoff()
    print("\n" + "=" * 60)
    print("FINAL COMPLIANCE REPORT:")
    print("=" * 60)
    print(result)
