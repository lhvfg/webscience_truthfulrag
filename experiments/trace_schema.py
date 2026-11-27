"""
Trace JSONL schema check + loader, shared by the runner and the analysis scripts.

The schema is the contract described in
SEUCourseReportTexTemplate/plan/plan.md, section 4. Fields may be ADDED later,
but never renamed or removed.

Usage
-----
    python experiments/trace_schema.py results/trace_*.jsonl      # validate files

    from trace_schema import load_traces
    df = load_traces("results/trace_*.jsonl")                      # pandas DataFrame
"""
import glob
import json
import sys
from typing import Dict, List

METHODS = {"wo_rag", "w_rag", "TruthfulRAG"}
DATASETS = {"faitheval", "musique_neg", "musique_gold", "squad_neg", "squad_gold", "realtimeqa"}

# key -> (allowed types, may be null)
COMMON = {
    "id": (str, False),
    "dataset": (str, False),
    "method": (str, False),
    "question": (str, False),
    "choices": ((list, dict, str), True),
    "gold": ((str, list), False),
    "gold_true": ((str, list), False),
    "pred": (str, False),
    "pred_raw": (str, True),
    "correct": (bool, False),
    "em": (bool, False),
    "answer_in_context": (bool, True),
    "graph": (dict, True),
    "H_base": ((int, float), True),
    "candidates": (list, True),
    "kept": (list, True),
    "n_llm_calls": (int, False),
    "n_tokens": (int, False),
    "latency_s": ((int, float), False),
}
GRAPH_KEYS = {"n_entities": int, "n_relations": int, "answer_in_graph": bool}
CANDIDATE_KEYS = {"idx": int, "text": str, "H": (int, float), "dH": (int, float), "has_answer": bool}


def validate(rec: Dict) -> List[str]:
    """Return a list of problems (empty list = valid)."""
    errs = []
    for k, (types, nullable) in COMMON.items():
        if k not in rec:
            errs.append(f"missing key '{k}'")
            continue
        v = rec[k]
        if v is None:
            if not nullable:
                errs.append(f"'{k}' must not be null")
            continue
        if isinstance(v, bool) and bool not in (types if isinstance(types, tuple) else (types,)):
            errs.append(f"'{k}' has bool, expected {types}")
        elif not isinstance(v, types):
            errs.append(f"'{k}' has {type(v).__name__}, expected {types}")

    if rec.get("method") not in METHODS:
        errs.append(f"unknown method {rec.get('method')!r}")
    if rec.get("dataset") not in DATASETS:
        errs.append(f"unknown dataset {rec.get('dataset')!r}")

    is_cl = rec.get("method") == "TruthfulRAG"
    for k in ("graph", "H_base", "candidates", "kept"):
        if is_cl and rec.get(k) is None:
            errs.append(f"TruthfulRAG record needs '{k}'")
        if not is_cl and rec.get(k) is not None:
            errs.append(f"baseline record must have '{k}' = null")

    if isinstance(rec.get("graph"), dict):
        for k, t in GRAPH_KEYS.items():
            if not isinstance(rec["graph"].get(k), t):
                errs.append(f"graph.{k} missing or not {t}")
    cands = rec.get("candidates") or []
    for i, c in enumerate(cands):
        for k, t in CANDIDATE_KEYS.items():
            if k not in c or not isinstance(c[k], t):
                errs.append(f"candidates[{i}].{k} missing or not {t}")
    n = len(cands)
    for j in rec.get("kept") or []:
        if not isinstance(j, int) or not 0 <= j < n:
            errs.append(f"kept index {j!r} out of range (0..{n - 1})")
    return errs


def iter_records(pattern: str):
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if line.strip():
                    yield path, lineno, json.loads(line)


def load_traces(pattern: str = "results/trace_*.jsonl", strict: bool = True):
    """Load every matching trace file into one DataFrame (one row per sample x method)."""
    import pandas as pd

    rows = []
    for path, lineno, rec in iter_records(pattern):
        errs = validate(rec)
        if errs and strict:
            raise ValueError(f"{path}:{lineno} invalid trace: {errs}")
        rows.append(rec)
    if not rows:
        raise FileNotFoundError(f"no trace records match {pattern!r}")
    df = pd.DataFrame(rows)
    dup = df.duplicated(["dataset", "method", "id"])
    if dup.any():
        raise ValueError(f"duplicated (dataset, method, id) rows: {int(dup.sum())}")
    return df


if __name__ == "__main__":
    bad = 0
    total = 0
    for pattern in sys.argv[1:] or ["results/trace_*.jsonl"]:
        for path, lineno, rec in iter_records(pattern):
            total += 1
            errs = validate(rec)
            if errs:
                bad += 1
                print(f"{path}:{lineno} [{rec.get('method')}/{rec.get('id')}]")
                for e in errs:
                    print(f"    - {e}")
    print(f"{total} records checked, {bad} invalid")
    sys.exit(1 if bad else 0)
