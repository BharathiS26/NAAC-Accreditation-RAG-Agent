"""
RAG pipeline: retrieves top-3 NAAC metric chunks and generates an answer
using IBM Granite via watsonx.ai text/generation endpoint.

Every answer is strictly grounded in retrieved chunks; if nothing relevant
is found the model is instructed to respond "Not found in the manual".
"""

import os
import requests
from typing import List
from dotenv import load_dotenv

from vector_store import retrieve
from parser import MetricChunk

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def _get_iam_token(api_key: str) -> str:
    resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if resp.status_code == 400:
        err = resp.json().get("errorMessage", resp.text)
        raise ValueError(f"IBM IAM token exchange failed: {err}")
    resp.raise_for_status()
    return resp.json()["access_token"]


def _build_prompt(query: str, chunks: List[MetricChunk]) -> str:
    context_blocks = []
    for c in chunks:
        context_blocks.append(f"{c.metadata_header()}\n{c.text}")
    context = "\n\n---\n\n".join(context_blocks)

    return (
        "You are a NAAC accreditation expert. Answer ONLY based on the "
        "provided NAAC manual excerpts below. "
        "Every answer MUST explicitly cite the exact Criterion number, "
        "Key Indicator number, and Metric number it draws from. "
        "If the answer is not present in the excerpts, respond exactly: "
        "\"Not found in the manual\".\n\n"
        "=== MANUAL EXCERPTS ===\n"
        f"{context}\n\n"
        "=== QUESTION ===\n"
        f"{query}\n\n"
        "=== ANSWER (cite Criterion / Key Indicator / Metric) ===\n"
    )


def answer(query: str) -> dict:
    """
    Returns a dict with keys:
        answer (str): the generated answer with citations
        sources (list[dict]): list of source chunk metadata
    """
    chunks = retrieve(query, top_k=3)
    if not chunks:
        return {"answer": "Not found in the manual", "sources": []}

    prompt = _build_prompt(query, chunks)

    api_key = os.environ.get("WATSONX_API_KEY", "")
    url = os.environ.get("WATSONX_URL", "")
    project_id = os.environ.get("WATSONX_PROJECT_ID", "")
    model_id = os.environ.get("WATSONX_MODEL_ID", "ibm/granite-4-h-small")

    token = _get_iam_token(api_key)

    payload = {
        "model_id": model_id,
        "project_id": project_id,
        "input": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.0,
            "repetition_penalty": 1.1,
        },
    }

    resp = requests.post(
        url,
        json=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    # text/generation response shape: results[0].generated_text
    try:
        generated = data["results"][0]["generated_text"]
    except (KeyError, IndexError):
        generated = str(data)

    sources = [
        {
            "criterion": f"{c.criterion_num} – {c.criterion_name}",
            "ki": f"{c.ki_num} – {c.ki_name}",
            "metric": c.metric_num,
            "type": c.metric_type,
            "weight": c.weight,
        }
        for c in chunks
    ]

    return {"answer": generated.strip(), "sources": sources}


if __name__ == "__main__":
    q = "What is required under Key Indicator 1.1?"
    result = answer(q)
    print("Q:", q)
    print("A:", result["answer"])
    print("Sources:", result["sources"])
