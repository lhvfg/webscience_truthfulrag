"""
E3  Stage-wise answer recall ("funnel") of the KG pipeline.

Per sample of the KG method:
  S1 answer in retrieved context        (answer_in_context)
  S2 answer in constructed graph        (graph.answer_in_graph)
  S3 answer in some candidate path      (any candidates[].has_answer)
  S4 answer in some KEPT path           (any candidates[kept].has_answer)
  S5 final answer correct               (correct)
Reference: accuracy of w_rag on the same ids.

Because the final prompt still contains the full original context (see
modules.predict_answer), S5 is NOT bounded by S4. The table therefore also
reports accuracy conditioned on whether the kept paths contain the answer,
which shows how much the answer comes from the paths vs. the raw context.

Outputs
  fig:  e3_funnel{tag}.pdf
  tab:  e3_funnel{tag}.csv / .tex

Run:  python experiments/analysis/e3_funnel.py [--traces "results/trace_*.jsonl"]
"""
import os

import numpy as np
import pandas as pd

from common import C_BLUE, C_MAIN, DATASET_LABEL, method_rows, ordered, parse_args, load_traces, use_style

STAGES = [("S1", "in context"), ("S2", "in graph"), ("S3", "in candidates"),
          ("S4", "in kept paths"), ("S5", "answered correctly")]


def stage_flags(cl):
    f = pd.DataFrame({"dataset": cl.dataset.values, "id": cl.id.values})
    f["S1"] = cl.answer_in_context.map(lambda v: bool(v)).values
    f["S2"] = cl.graph.map(lambda g: bool(g and g.get("answer_in_graph"))).values
    f["S3"] = cl.candidates.map(lambda c: any(x["has_answer"] for x in (c or []))).values
    f["S4"] = [any(c[k]["has_answer"] for k in (kept or []) if k < len(c))
               for c, kept in zip(cl.candidates, cl.kept)]
    f["S5"] = cl.correct.astype(bool).values
    f["n_kept"] = cl.kept.map(lambda k: len(k or [])).values
    f["n_cand"] = cl.candidates.map(lambda c: len(c or [])).values
    return f


def summarise(f, wrag):
    rows = []
    groups = [(d, f[f.dataset == d]) for d in ordered(f.dataset)] + [("ALL", f)]
    for name, g in groups:
        r = dict(dataset=DATASET_LABEL.get(name, name), n=len(g))
        for s, _ in STAGES:
            r[s] = g[s].mean()
        r["acc_if_S4"] = g[g.S4].S5.mean() if g.S4.any() else np.nan
        r["acc_if_not_S4"] = g[~g.S4].S5.mean() if (~g.S4).any() else np.nan
        r["share_correct_without_S4"] = (g.S5 & ~g.S4).sum() / max(g.S5.sum(), 1)
        r["avg_cand"] = g.n_cand.mean()
        r["avg_kept"] = g.n_kept.mean()
        ids = set(g.id)
        w = wrag[wrag.id.isin(ids) & ((wrag.dataset == name) | (name == "ALL"))]
        r["w_rag_acc"] = w.correct.mean() if len(w) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def plot(summ, path):
    plt = use_style()
    s = summ[summ.dataset != "ALL"]
    n = len(s)
    fig, axes = plt.subplots(1, n, figsize=(max(4.8, 2.3 * n), 3.0), sharey=True, squeeze=False)
    for ax, (_, r) in zip(axes[0], s.iterrows()):
        vals = [r[k] for k, _ in STAGES]
        cols = [C_BLUE] * 4 + [C_MAIN]
        ax.bar(range(5), vals, color=cols, width=0.7)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=7)
        if not np.isnan(r["w_rag_acc"]):
            ax.axhline(r["w_rag_acc"], color="#333333", ls="--", lw=0.9, label="w/ RAG accuracy")
        ax.set_xticks(range(5))
        ax.set_xticklabels([lab for _, lab in STAGES], rotation=35, ha="right")
        ax.set_title(r["dataset"])
        ax.set_ylim(0, 1.12)
    axes[0][0].set_ylabel("fraction of samples")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles[:1], labels[:1], loc="upper right", bbox_to_anchor=(1.0, 1.06))
    fig.savefig(path)
    plt.close(fig)


def main():
    args = parse_args(__doc__)
    df = load_traces(args.traces)
    cl = method_rows(df)
    wrag = df[df.method == "w_rag"]
    f = stage_flags(cl)
    summ = summarise(f, wrag)

    base = os.path.join(args.tab_dir, f"e3_funnel{args.tag}")
    summ.to_csv(base + ".csv", index=False)
    summ.to_latex(base + ".tex", index=False, float_format="%.3f", na_rep="--")
    plot(summ, os.path.join(args.fig_dir, f"e3_funnel{args.tag}.pdf"))
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(summ.round(3).to_string(index=False))
    print("written:", base + ".csv/.tex", "and figure in", args.fig_dir)


if __name__ == "__main__":
    main()
