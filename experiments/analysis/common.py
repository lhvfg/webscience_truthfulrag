"""Shared helpers for Sean's analysis scripts (E1 / E2 / E3)."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from trace_schema import load_traces  # noqa: E402

STYLE = os.path.join(ROOT, "SEUCourseReportTexTemplate", "plan", "cl_style.mplstyle")
FIG_DIR = os.path.join(ROOT, "SEUCourseReportTexTemplate", "figure", "cl_method")
TAB_DIR = os.path.join(ROOT, "experiments", "out", "cl_method")

# Fixed colours (same as cl_style.mplstyle header)
C_MAIN = "#E45756"   # the method / positive class
C_GREY = "#9AA0A6"   # baseline / negative class
C_BLUE = "#4C78A8"

DATASET_ORDER = ["faitheval", "musique_neg", "realtimeqa", "squad_neg", "musique_gold", "squad_gold"]
DATASET_LABEL = {
    "faitheval": "FaithEval", "musique_neg": "MuSiQue", "realtimeqa": "RealtimeQA",
    "squad_neg": "SQuAD", "musique_gold": "MuSiQue-gold", "squad_gold": "SQuAD-gold",
}


def parse_args(desc, extra=None):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--traces", default=os.path.join(ROOT, "results", "trace_*.jsonl"),
                    help="glob of trace JSONL files")
    ap.add_argument("--fig-dir", default=FIG_DIR)
    ap.add_argument("--tab-dir", default=TAB_DIR)
    ap.add_argument("--tag", default="", help="suffix for output names (e.g. _mock)")
    if extra:
        extra(ap)
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(args.tab_dir, exist_ok=True)
    return args


def method_rows(df, method_name=None):
    """Rows of the KG method (the only method with candidates)."""
    if method_name is None:
        has = df["candidates"].notna()
        return df[has].copy()
    return df[df["method"] == method_name].copy()


def ordered(datasets):
    present = list(dict.fromkeys(datasets))
    return [d for d in DATASET_ORDER if d in present] + [d for d in present if d not in DATASET_ORDER]


def use_style():
    import matplotlib.pyplot as plt
    plt.style.use(STYLE)
    return plt
