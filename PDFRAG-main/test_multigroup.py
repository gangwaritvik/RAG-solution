"""Focused multi_group / compound (segments) classifier test battery.
Run: $env:PYTHONIOENCODING="utf-8"; ..\\.venv\\Scripts\\python.exe test_multigroup.py
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from backend.memory.classifiers.llm_classifier import LLMClassifier  # noqa: E402

DOCS = ["electricity.pdf", "iso27001.pdf"]

# kinds:
#   "multi"   -> expect dependency_type multi_group with >= min_sub sub_queries (one op, many subjects)
#   "segment" -> expect >= min_seg segments (different ops in one query)
#   "single"  -> expect NEITHER (a single operation, even if over multiple subjects framed as one)
CASES = [
    # ---- multi_group: comparisons (one comparison op over N subjects) ----
    {"id": "cmp 2 subj", "q": "compare voltmeter and ammeter", "kind": "multi", "min_sub": 2},
    {"id": "cmp 3 subj", "q": "compare voltmeter, ammeter and potentiometer", "kind": "multi", "min_sub": 3},
    {"id": "cmp vs", "q": "ohm's law vs kirchhoff's law", "kind": "multi", "min_sub": 2},
    {"id": "cmp difference", "q": "what's the difference between AC and DC", "kind": "multi", "min_sub": 2},
    {"id": "cmp 4 subj", "q": "compare resistance, capacitance, inductance and impedance", "kind": "multi", "min_sub": 4},

    # ---- multi_group: plain conjunction (one op = give/list, many subjects) ----
    {"id": "give 2", "q": "give me ohm's law and kirchhoff's law", "kind": "multi", "min_sub": 2},
    {"id": "give 3", "q": "state ohm's law, faraday's law and lenz's law", "kind": "multi", "min_sub": 3},
    {"id": "explain 2", "q": "explain conductors and insulators", "kind": "multi", "min_sub": 2},
    {"id": "define 2", "q": "define emf and potential difference", "kind": "multi", "min_sub": 2},

    # ---- multi_group spanning two documents (one op per doc subject) ----
    {"id": "cross-doc compare", "q": "compare what electricity.pdf and iso27001.pdf cover", "kind": "multi", "min_sub": 2},

    # ---- segments: genuinely different operations ----
    {"id": "seg compare+summarize", "q": "compare voltmeter and ammeter, and summarize the iso27001 pdf", "kind": "segment", "min_seg": 2},
    {"id": "seg define+extract", "q": "define resistance and give me the controls table from iso27001", "kind": "segment", "min_seg": 2},
    {"id": "seg summarize+list", "q": "summarize electricity.pdf and list all the iso27001 controls", "kind": "segment", "min_seg": 2},
    {"id": "seg derive+compare", "q": "derive drift velocity and compare voltmeter with ammeter", "kind": "segment", "min_seg": 2},
    {"id": "seg 3 ops", "q": "define resistance, compare AC and DC, and summarize iso27001", "kind": "segment", "min_seg": 3},

    # ---- single operation: must NOT split into segments ----
    # NOTE: "difference between X and Y" is multi_group by design (each subject retrieved
    # separately), so it is NOT a 'single' case — it belongs with the comparisons above.
    {"id": "single summary", "q": "summarize the electricity document", "kind": "single"},
    {"id": "single factual", "q": "what is the SI unit of current", "kind": "single"},
    {"id": "single relate", "q": "how do voltage and current relate in a resistor", "kind": "single"},  # relationship = 1 answer
]


def run():
    clf = LLMClassifier()
    passes = 0
    total = 0
    rows = []
    for c in CASES:
        total += 1
        try:
            r = clf.classify_query(query=c["q"], available_documents=DOCS)
        except Exception as e:
            rows.append(("ERROR", c["id"], c["q"][:46], f"exc: {str(e)[:50]}"))
            continue
        dep = r.get("dependency_type", "")
        intent = r.get("retrieval_intent", "")
        sub = r.get("sub_queries", []) or []
        seg = r.get("segments", []) or []

        notes = []
        if c["kind"] == "multi":
            if dep != "multi_group":
                notes.append(f"dep={dep} (want multi_group)")
            if len(sub) < c.get("min_sub", 2):
                notes.append(f"sub={len(sub)} (want >={c.get('min_sub', 2)})")
            if seg:
                notes.append(f"seg={len(seg)} (want 0)")
        elif c["kind"] == "segment":
            if len(seg) < c.get("min_seg", 2):
                notes.append(f"seg={len(seg)} (want >={c.get('min_seg', 2)})")
        elif c["kind"] == "single":
            if seg:
                notes.append(f"seg={len(seg)} (want 0)")
            if dep == "multi_group":
                notes.append(f"dep=multi_group (want single)")

        ok = not notes
        if ok:
            passes += 1
        detail = f"dep={dep} intent={intent} sub={len(sub)} seg={len(seg)}"
        rows.append(("PASS" if ok else "FAIL", c["id"], c["q"][:46], "; ".join(notes) or detail))

    print("\n" + "=" * 120)
    print(f"{'STATUS':<7}{'ID':<22}{'QUERY':<48}{'DETAIL / MISMATCH'}")
    print("=" * 120)
    for status, cid, q, note in rows:
        print(f"{status:<7}{cid:<22}{q:<48}{note}")
    print("=" * 120)
    print(f"RESULT: {passes}/{total} matched expectations\n")


if __name__ == "__main__":
    run()
