"""
PDF parser for NAAC Revised Manual for Autonomous Colleges (2020).
Chunks the document by Metric, grouped under Key Indicator and Criterion.

The QIF section (page 50 onward) uses a layout like:
  Criterion I – Curricular Aspects (150)
  Key Indicator – 1.1 Curriculum Design and Development (50)
  1.1.1  QlM    <description>   20
  1.1.2  QnM    <description>   20
  Key Indicator – 1.2 ...

Each MetricChunk is tagged with criterion/ki/metric metadata.
"""

import re
import os
from dataclasses import dataclass, field
from typing import List, Optional
import pymupdf


@dataclass
class MetricChunk:
    criterion_num: str
    criterion_name: str
    ki_num: str
    ki_name: str
    metric_num: str
    metric_type: str   # "Qualitative" or "Quantitative"
    weight: Optional[str]
    text: str

    def metadata_header(self) -> str:
        return (
            f"[Criterion {self.criterion_num} – {self.criterion_name}] "
            f"[Key Indicator {self.ki_num} – {self.ki_name}] "
            f"[Metric {self.metric_num} ({self.metric_type}"
            + (f", weight {self.weight}" if self.weight else "")
            + ")]"
        )

    def full_text(self) -> str:
        return self.metadata_header() + "\n" + self.text


# ── Patterns ──────────────────────────────────────────────────────────────────
# "Criterion I – Curricular Aspects (150)"  or "Criterion II – ..."
_CRIT_RE = re.compile(
    r"^Criterion\s+([IVX]+\b[^–\-]*?)\s*[–\-]\s*(.+?)(?:\s*\(\d+\))?\s*$",
    re.IGNORECASE,
)
# "Key Indicator – 1.1 Curriculum Design and Development (50)"
_KI_RE = re.compile(
    r"Key\s+Indicator\s*[-–]\s*([\d]+\.[\d]+)\s+(.*?)(?:\s*\([\d]+\))?\s*$",
    re.IGNORECASE,
)
# Metric number alone on a "cell": "1.1.1" or "1.1.2" etc.
_METRIC_NUM_RE = re.compile(r"^([\d]+\.[\d]+\.[\d]+)\s*$")
# QlM / QnM indicator line
_QLM_RE = re.compile(r"\bQlM\b")
_QNM_RE = re.compile(r"\bQnM\b")
# Weight: a number alone on a line (between 5 and 100) — used as column
_WEIGHT_RE = re.compile(r"^\s*(\d{1,3})\s*$")
# Footer / header boilerplate to skip
_SKIP_RE = re.compile(
    r"(Manual for Autonomous Colleges|NAAC for Quality and Excellence|"
    r"Metric\s+No\.|Weightage|Weightages|^\s*$)",
    re.IGNORECASE,
)
# Page numbers: a lone integer >100 (page numbers are always > metric weights)
_PAGENUM_RE = re.compile(r"^\s*(\d+)\s*$")


def _is_boilerplate(line: str) -> bool:
    return bool(_SKIP_RE.search(line))


def _extract_pages(pdf_path: str) -> List[str]:
    """Return list of page texts (strings), one per page."""
    doc = pymupdf.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return pages


def parse_manual(pdf_path: str) -> List[MetricChunk]:
    pages = _extract_pages(pdf_path)

    # The QIF starts around page 50 (index 49). We'll scan the whole doc but
    # only start collecting once we've seen the first "Criterion I" in the QIF.
    full_lines: List[str] = []
    for text in pages:
        for line in text.split("\n"):
            full_lines.append(line.rstrip())

    chunks: List[MetricChunk] = []

    cur_crit_num = "?"
    cur_crit_name = "Unknown Criterion"
    cur_ki_num = "?"
    cur_ki_name = "Unknown KI"

    # Current metric buffer
    cur_metric_num: Optional[str] = None
    cur_metric_type = "Qualitative"
    cur_metric_weight: Optional[str] = None
    cur_metric_lines: List[str] = []

    in_qif = False  # only collect once we're inside the QIF section

    def flush():
        nonlocal cur_metric_num, cur_metric_lines, cur_metric_type, cur_metric_weight
        if cur_metric_num is None:
            return
        body = " ".join(l for l in cur_metric_lines if l.strip()).strip()
        if body:
            chunks.append(MetricChunk(
                criterion_num=cur_crit_num,
                criterion_name=cur_crit_name,
                ki_num=cur_ki_num,
                ki_name=cur_ki_name,
                metric_num=cur_metric_num,
                metric_type=cur_metric_type,
                weight=cur_metric_weight,
                text=body,
            ))
        cur_metric_num = None
        cur_metric_lines = []
        cur_metric_type = "Qualitative"
        cur_metric_weight = None

    i = 0
    while i < len(full_lines):
        raw = full_lines[i]
        line = raw.strip()
        i += 1

        # Detect QIF section start
        if not in_qif:
            if re.search(r"Quality Indicator Framework.*QIF", line, re.IGNORECASE):
                in_qif = True
            # Also trigger on the first "Criterion I – Curricular Aspects"
            if re.search(r"Criterion\s+I\s*[–\-]\s*Curricular", line, re.IGNORECASE):
                in_qif = True

        if not in_qif:
            continue

        # Skip boilerplate — but never skip inside a metric (weight lives here)
        if _is_boilerplate(raw) and not _CRIT_RE.match(line) and not _KI_RE.search(line):
            if cur_metric_num is None:
                continue
            # Inside a metric: only skip if it's genuinely boilerplate text, not a lone number
            pm = _PAGENUM_RE.match(line)
            if not pm:
                continue
            # It's a lone number — let it fall through to weight detection below

        # ── Criterion header ──────────────────────────────────────────────
        cm = _CRIT_RE.match(line)
        if cm and not _KI_RE.search(line) and not _METRIC_NUM_RE.match(line):
            flush()
            cur_crit_num = cm.group(1).strip()
            cur_crit_name = cm.group(2).strip()
            continue

        # ── Key Indicator header ──────────────────────────────────────────
        km = _KI_RE.search(line)
        if km:
            flush()
            cur_ki_num = km.group(1).strip()
            cur_ki_name = km.group(2).strip()
            continue

        # ── Metric number (standalone cell like "1.1.1") ──────────────────
        mm = _METRIC_NUM_RE.match(line)
        if mm:
            # Weight appears as the last standalone number before the next metric
            if cur_metric_num is not None and cur_metric_weight is None and cur_metric_lines:
                last = cur_metric_lines[-1].strip()
                wm2 = _PAGENUM_RE.match(last)
                if wm2:
                    val2 = int(wm2.group(1))
                    if 5 <= val2 <= 100:
                        cur_metric_weight = str(val2)
                        cur_metric_lines.pop()
            flush()
            cur_metric_num = mm.group(1)
            cur_metric_lines = []
            cur_metric_type = "Qualitative"
            cur_metric_weight = None
            continue

        # ── Once inside a metric, collect lines ───────────────────────────
        if cur_metric_num is not None:
            # Type detection
            if _QNM_RE.search(line):
                cur_metric_type = "Quantitative"
                # Don't add bare "QnM" token as text
                if line.strip() in ("QnM", "QlM"):
                    continue
            elif _QLM_RE.search(line):
                cur_metric_type = "Qualitative"
                if line.strip() in ("QlM", "QnM"):
                    continue

            # Weight: a standalone number at end of a metric section
            wm = _WEIGHT_RE.match(line)
            if wm:
                val = int(wm.group(1))
                if 5 <= val <= 100:
                    if cur_metric_weight is None:
                        cur_metric_weight = str(val)
                    # Don't add the bare weight number as text
                    continue

            if line and not _is_boilerplate(raw):
                cur_metric_lines.append(line)

    flush()
    return chunks


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    pdf = os.environ.get("PDF_PATH", "")
    if not pdf or not os.path.exists(pdf):
        # Try parent dir
        parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for f in os.listdir(parent):
            if f.lower().endswith(".pdf") and "naac" in f.lower():
                pdf = os.path.join(parent, f)
                break
    if not pdf or not os.path.exists(pdf):
        print("PDF not found. Set PDF_PATH.")
        sys.exit(1)

    results = parse_manual(pdf)
    print(f"Parsed {len(results)} metric chunks.")
    for r in results[:5]:
        print("---")
        print(r.metadata_header())
        print(r.text[:300])
        print()
