"""Ad-hoc classifier test harness — exercises a wide variety of query types and
prints the classification so we can judge the LLM's reasoning. Not a unit test; run
manually:  $env:PYTHONIOENCODING="utf-8"; ..\\.venv\\Scripts\\python.exe test_classifier.py
"""
import io
import sys

# Force UTF-8 so any non-ASCII never crashes a redirected Windows console.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from backend.memory.classifiers.llm_classifier import LLMClassifier  # noqa: E402

DOCS = ["electricity.pdf", "iso27001.pdf"]

# Reusable active-context snippets
CTX_DRIFT = {
    "active_group_topic": "Drift velocity",
    "active_group_summary": (
        "User: derive drift velocity\n"
        "Assistant: Drift velocity v_d = eE*tau/m is derived from the force eE on an "
        "electron, acceleration a=eE/m, over relaxation time tau. It is measured in m/s."
    ),
}
CTX_FIELD = {
    "active_group_topic": "Electric field in a conductor",
    "active_group_summary": (
        "User: explain electric field in a conductor\n"
        "Assistant: The electric field E drives current; it relates to potential gradient."
    ),
}
CTX_ISO = {
    "active_group_topic": "ISO 27001 overview",
    "active_group_summary": (
        "User: what is iso27001 about\n"
        "Assistant: ISO/IEC 27001 is an information security management system (ISMS) "
        "standard covering risk assessment, controls, leadership, and continual improvement."
    ),
}

# Each case is a dict. Required: id, query. Optional expectations:
#   intent, dep, answer_source, min_sub, min_seg, files_any (>=1 present)
#   ctx (kwargs merged into classify_query)
CASES = [
    # factual
    {"id": "factual: SI unit", "query": "what is the SI unit of current", "intent": "factual", "dep": "independent"},
    {"id": "factual: define", "query": "define drift velocity", "intent": "factual", "dep": "independent"},
    {"id": "factual: when", "query": "when was ohm's law discovered", "intent": "factual"},
    {"id": "factual: value", "query": "what is the resistivity of copper", "intent": "factual"},

    # analysis
    {"id": "analysis: derive", "query": "derive drift velocity", "intent": "analysis", "dep": "independent"},
    {"id": "analysis: how", "query": "how does a potentiometer work", "intent": "analysis"},
    {"id": "analysis: why", "query": "why does resistance increase with temperature", "intent": "analysis"},
    {"id": "analysis: explain process", "query": "explain how a transformer steps up voltage", "intent": "analysis"},
    {"id": "analysis: prove", "query": "prove that power dissipated is I squared R", "intent": "analysis"},

    # comparison
    {"id": "comparison: difference", "query": "difference between voltmeter and ammeter", "intent": "comparison"},
    {"id": "comparison: vs", "query": "ammeter vs voltmeter", "intent": "comparison"},
    {"id": "comparison: which better", "query": "which is better for measuring emf, voltmeter or potentiometer", "intent": "comparison"},
    {"id": "comparison: 3 subj", "query": "compare voltmeter, ammeter and potentiometer", "intent": "comparison", "dep": "multi_group", "min_sub": 3},
    {"id": "comparison: contrast", "query": "contrast series and parallel circuits", "intent": "comparison"},

    # summary
    {"id": "global_summary: named doc", "query": "what is the iso27001 pdf about", "intent": "global_summary", "files_any": ["iso27001.pdf"]},
    {"id": "global_summary: summarize doc", "query": "summarize electricity.pdf", "intent": "global_summary", "files_any": ["electricity.pdf"]},
    {"id": "global_summary: tell about", "query": "tell me about the electricity document", "intent": "global_summary", "files_any": ["electricity.pdf"]},
    {"id": "targeted_summary: section", "query": "summarize the access control section of iso27001", "intent": "targeted_summary"},

    # extraction
    {"id": "global_extraction: list all", "query": "list all the physical quantities in the electricity document", "intent": "global_extraction"},
    {"id": "global_extraction: every", "query": "enumerate every control in iso27001", "intent": "global_extraction"},
    {"id": "targeted_extraction: table", "query": "give me the information security controls table", "intent": "targeted_extraction"},

    # plain conjunction (multi_group, NOT comparison)
    {"id": "conjunction: give X and Y", "query": "give me ohm's law and kirchhoff's law", "dep": "multi_group", "min_sub": 2},
    {"id": "conjunction: state X and Y", "query": "state ohm's law and faraday's law", "dep": "multi_group", "min_sub": 2},

    # compound segments (different operations)
    {"id": "compound: compare + summarize", "query": "compare voltmeter and ammeter, and summarize the iso27001 pdf", "min_seg": 2},
    {"id": "compound: define + extract", "query": "define resistance and give me the controls table from iso27001", "min_seg": 2},

    # ambiguous (no context)
    {"id": "ambiguous: bare pronoun", "query": "what is the unit", "intent": "ambiguous", "dep": "ambiguous"},
    {"id": "ambiguous: the value", "query": "what is the value", "dep": "ambiguous"},
    {"id": "ambiguous: generic doc multi", "query": "what is this document about", "dep": "ambiguous"},

    # typo tolerance
    {"id": "typo: bulbs", "query": "what is the combination of bublbs", "dep": "independent"},
    {"id": "typo: electricity", "query": "summarize the elecitrcity pdf", "intent": "global_summary", "files_any": ["electricity.pdf"]},
    {"id": "typo: misspelled subj", "query": "difference between voltmeter and ammeeter", "intent": "comparison"},

    # dependent / back-reference
    {"id": "dependent: explain above", "query": "explain the above information", "ctx": CTX_DRIFT, "dep": "dependent"},
    {"id": "dependent: its unit", "query": "what is its unit", "ctx": CTX_DRIFT, "dep": "dependent", "intent": "factual"},
    {"id": "dependent: implicit cont", "query": "now the magnetic version", "ctx": CTX_FIELD, "dep": "dependent"},
    {"id": "dependent: elaborate", "query": "can you elaborate on that", "ctx": CTX_ISO, "dep": "dependent"},
    {"id": "dependent: what about parallel", "query": "what about in parallel", "ctx": CTX_FIELD, "dep": "dependent"},

    # independent despite same domain
    {"id": "independent: own subject", "query": "how does temperature affect resistance", "ctx": CTX_DRIFT, "dep": "independent"},
    {"id": "independent: new topic", "query": "what are the iso27001 access controls", "ctx": CTX_DRIFT, "dep": "independent"},

    # named doc partial / descriptive
    {"id": "named: partial iso", "query": "tell me about the iso pdf", "intent": "global_summary", "files_any": ["iso27001.pdf"]},
    {"id": "named: descriptive", "query": "summarize the security standard", "intent": "global_summary", "files_any": ["iso27001.pdf"]},

    # answer_source: previous_answer
    {"id": "prev_ans: tabulate", "query": "put that in a table", "ctx": CTX_DRIFT, "answer_source": "previous_answer"},
    {"id": "prev_ans: shorten", "query": "make that shorter", "ctx": CTX_ISO, "answer_source": "previous_answer"},
    {"id": "prev_ans: translate", "query": "translate the above to simple terms", "ctx": CTX_DRIFT, "answer_source": "previous_answer"},

    # multi_group mixed intent
    {"id": "mixed: compare + whole-doc", "query": "difference between voltmeter and ammeter and what is iso27001 about", "min_sub": 2},
]


def run():
    clf = LLMClassifier()
    passes = 0
    total = 0
    rows = []
    for c in CASES:
        total += 1
        ctx = c.get("ctx", {})
        try:
            r = clf.classify_query(query=c["query"], available_documents=DOCS, **ctx)
        except Exception as e:
            rows.append(("ERROR", c["id"], c["query"][:40], "", "", f"exc: {str(e)[:50]}"))
            continue
        dep = r.get("dependency_type", "")
        intent = r.get("retrieval_intent", "")
        ans = r.get("answer_source", "")
        sub = r.get("sub_queries", []) or []
        seg = r.get("segments", []) or []
        files = r.get("source_files", []) or []

        notes = []
        if "intent" in c and intent != c["intent"]:
            notes.append(f"intent={intent} (want {c['intent']})")
        if "dep" in c and dep != c["dep"]:
            notes.append(f"dep={dep} (want {c['dep']})")
        if "answer_source" in c and ans != c["answer_source"]:
            notes.append(f"ans={ans} (want {c['answer_source']})")
        if "min_sub" in c and len(sub) < c["min_sub"]:
            notes.append(f"sub={len(sub)} (want >={c['min_sub']})")
        if "min_seg" in c and len(seg) < c["min_seg"]:
            notes.append(f"seg={len(seg)} (want >={c['min_seg']})")
        if "files_any" in c and not (set(files) & set(c["files_any"])):
            notes.append(f"files={files} (want any of {c['files_any']})")

        ok = not notes
        if ok:
            passes += 1
        extra = f"sub={len(sub)} seg={len(seg)} files={files} ans={ans}"
        rows.append(("PASS" if ok else "FAIL", c["id"], c["query"][:40],
                     dep, intent, "; ".join(notes) or extra))

    print("\n" + "=" * 130)
    print(f"{'STATUS':<7}{'ID':<30}{'QUERY':<42}{'DEP':<12}{'INTENT':<18}")
    print("=" * 130)
    for status, cid, query, dep, intent, note in rows:
        print(f"{status:<7}{cid:<30}{query:<42}{dep:<12}{intent:<18}")
        if status != "PASS":
            print(f"       -> {note}")
    print("=" * 130)
    print(f"RESULT: {passes}/{total} matched expectations\n")


if __name__ == "__main__":
    run()
