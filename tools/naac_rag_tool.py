"""
watsonx Orchestrate tool: NAAC Accreditation RAG retrieval + answer.
Registered as a native tool in the Orchestrate platform.
"""

import os
import sys

# Allow imports from the naac_rag directory whether running locally or in Orchestrate
_naac_dir = os.path.join(os.path.dirname(__file__), "..", "naac_rag")
if _naac_dir not in sys.path:
    sys.path.insert(0, _naac_dir)

from ibm_watsonx_orchestrate.agent_builder.tools import tool


@tool
def naac_rag_answer(question: str) -> str:
    """
    Answer a question about the NAAC Accreditation Manual for Autonomous Colleges.
    Retrieves the top relevant metric chunks and generates a grounded answer using
    IBM Granite via watsonx.ai, citing the exact Criterion, Key Indicator, and Metric.

    Args:
        question: The user's question about NAAC accreditation requirements,
                  criteria, key indicators, or metrics.

    Returns:
        A grounded answer with Criterion/Key Indicator/Metric citations,
        or "Not found in the manual" if no relevant content is found.
    """
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    from rag import answer
    result = answer(question)

    response_parts = [result["answer"]]
    if result["sources"]:
        response_parts.append("\n\nSources cited:")
        for s in result["sources"]:
            line = f"  • Criterion {s['criterion']} | KI {s['ki']} | Metric {s['metric']}"
            if s.get("type"):
                line += f" ({s['type']}"
                if s.get("weight"):
                    line += f", weight {s['weight']}"
                line += ")"
            response_parts.append(line)

    return "\n".join(response_parts)
