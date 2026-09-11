"""
CLI entry point for the NAAC Accreditation RAG Agent.
Usage:
    uv run orchestrate            – interactive Q&A loop
    uv run orchestrate --test     – run the 5 test queries
    uv run orchestrate --serve    – start the web server
    uv run orchestrate --build    – (re)build the FAISS index
"""

import os
import sys
import argparse

# Make sure naac_rag/ is importable regardless of CWD
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(_this_dir, "..", ".env"))


TEST_QUERIES = [
    "What is required under Key Indicator 1.1?",
    "What are the qualitative metrics for Curriculum Enrichment?",
    "What does Academic Flexibility measure?",
    "What is the weightage of Metric 1.1.2?",
    "What is required for Institutional Distinctiveness?",
]


def _ensure_index():
    from vector_store import build_index
    pdf = os.environ.get("PDF_PATH", "")
    if not pdf:
        parent = os.path.dirname(_this_dir)
        for f in os.listdir(parent):
            if f.lower().endswith(".pdf") and "naac" in f.lower():
                pdf = os.path.join(parent, f)
                os.environ["PDF_PATH"] = pdf
                break
    if not pdf or not os.path.exists(pdf):
        print(f"[ERROR] PDF not found. Set PDF_PATH in .env or pass the path.")
        sys.exit(1)
    build_index(pdf)


def cmd_build():
    print("[BUILD] Rebuilding FAISS index ...")
    from vector_store import build_index
    pdf = os.environ.get("PDF_PATH", "")
    if not pdf or not os.path.exists(pdf):
        print(f"[ERROR] PDF not found at PDF_PATH={pdf!r}")
        sys.exit(1)
    build_index(pdf, force=True)
    print("[BUILD] Done.")


def cmd_test():
    _ensure_index()
    from rag import answer
    print("\n" + "=" * 70)
    print("NAAC RAG Agent — Automated Test Queries")
    print("=" * 70)
    for i, q in enumerate(TEST_QUERIES, 1):
        print(f"\n[Query {i}] {q}")
        print("-" * 60)
        try:
            result = answer(q)
            print(f"Answer:\n{result['answer']}")
            if result["sources"]:
                print("\nSources:")
                for s in result["sources"]:
                    print(
                        f"  • Criterion {s['criterion']} | KI {s['ki']} "
                        f"| Metric {s['metric']} ({s['type']}"
                        + (f", weight {s['weight']}" if s.get("weight") else "")
                        + ")"
                    )
        except Exception as exc:
            print(f"[ERROR] {exc}")
    print("\n" + "=" * 70)
    print("Test run complete.")


def cmd_serve():
    import uvicorn
    _ensure_index()
    print("[SERVE] Starting web server at http://0.0.0.0:8000")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        app_dir=_this_dir,
    )


def cmd_interactive():
    _ensure_index()
    from rag import answer
    print("\n" + "=" * 70)
    print("NAAC Accreditation RAG Agent — Interactive Mode")
    print("Type your question and press Enter. Type 'quit' to exit.")
    print("=" * 70)
    while True:
        try:
            q = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        try:
            result = answer(q)
            print(f"\nAnswer:\n{result['answer']}")
            if result["sources"]:
                print("\nSources:")
                for s in result["sources"]:
                    print(
                        f"  • Criterion {s['criterion']} | KI {s['ki']} "
                        f"| Metric {s['metric']} ({s['type']}"
                        + (f", weight {s['weight']}" if s.get("weight") else "")
                        + ")"
                    )
        except Exception as exc:
            print(f"[ERROR] {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="NAAC Accreditation RAG Agent CLI"
    )
    parser.add_argument("--test", action="store_true", help="Run 5 test queries")
    parser.add_argument("--serve", action="store_true", help="Start the web server")
    parser.add_argument("--build", action="store_true", help="(Re)build the FAISS index")
    args = parser.parse_args()

    if args.build:
        cmd_build()
    elif args.test:
        cmd_test()
    elif args.serve:
        cmd_serve()
    else:
        cmd_interactive()


if __name__ == "__main__":
    main()
