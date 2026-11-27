"""
E1  Does an entropy increase (dH > 0) identify "corrective" paths?

For every candidate path of the KG method we know dH and whether the path text
contains the gold answer (has_answer). If dH were a good conflict/correction
signal, answer-bearing paths should have clearly larger dH.

Outputs
  fig:  e1_dh_by_label{tag}.pdf   -- dH distribution, answer vs no-answer paths, per dataset
  tab:  e1_entropy{tag}.csv / .tex -- per dataset: #paths, pos rate, mean dH, AUC, MWU p,
                                     Hit@1 of dH ranking vs random-ranking expectation

Run:  python experiments/analysis/e1_entropy.py [--traces "results/trace_*.jsonl"]
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from common import C_GREY, C_MAIN, DATASET_LABEL, method_rows, ordered, parse_args, load_traces, use_style


def explode_candidates(cl):
    rows = []
    for _, r in cl.iterrows():
        for c in r["candidates"]:
            rows.append(dict(dataset=r["dataset"], id=r["id"], idx=c["idx"], dH=c["dH"],
                             has_answer=bool(c["has_answer"]), kept=c["idx"] in (r["kept"] or [])))
    return pd.DataFrame(rows)


def hit_at_1(cl):
    """Among samples with >=1 answer path: is the top-dH path an answer path?"""
    hits, expect = [], []
    for _, r in cl.iterrows():
        cands = r["candidates"]
        pos = sum(c["has_answer"] for c in cands)
        if not cands or pos == 0:
            continue
        top = max(cands, key=lambda c: c["dH"])
        hits.append(top["has_answer"])
        expect.append(pos / len(cands))          # random ranking
    return (np.mean(hits) if hits else np.nan, np.mean(expect) if expect else np.nan, len(hits))


def summarise(cand, cl):
    out = []
    groups = [(d, cand[cand.dataset == d], cl[cl.dataset == d]) for d in ordered(cand.dataset)]
    groups.append(("ALL", cand, cl))
    for name, g, gcl in groups:
        pos, neg = g[g.has_answer], g[~g.has_answer]
        both = len(pos) > 0 and len(neg) > 0
        h1, h1_rand, n_h = hit_at_1(gcl)
        out.append(dict(
            dataset=DATASET_LABEL.get(name, name),
            n_samples=gcl["id"].nunique(),
            n_paths=len(g),
            pos_rate=len(pos) / max(len(g), 1),
            dH_pos=pos.dH.mean(), dH_neg=neg.dH.mean(),
            frac_pos_dH_gt0=(pos.dH > 0).mean() if len(pos) else np.nan,
            auc=roc_auc_score(g.has_answer, g.dH) if both else np.nan,
            mwu_p=mannwhitneyu(pos.dH, neg.dH, alternative="greater").pvalue if both else np.nan,
            hit1_dH=h1, hit1_random=h1_rand, n_hit_samples=n_h,
            kept_precision=g[g.kept].has_answer.mean() if g.kept.any() else np.nan,
        ))
    return pd.DataFrame(out)


def plot(cand, path):
    plt = use_style()
    ds = ordered(cand.dataset)
    fig, ax = plt.subplots(figsize=(max(4.8, 1.6 * len(ds)), 3.2))
    w = 0.35
    for k, (lab, col) in enumerate([(True, C_MAIN), (False, C_GREY)]):
        data = [cand[(cand.dataset == d) & (cand.has_answer == lab)].dH.values for d in ds]
        pos = [i + (k - 0.5) * w for i in range(len(ds))]
        keep = [(p, v) for p, v in zip(pos, data) if len(v) > 1]
        if keep:
            vp = ax.violinplot([v for _, v in keep], positions=[p for p, _ in keep], widths=w * 0.95,
                               showmedians=True, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(col); b.set_alpha(0.55); b.set_edgecolor("none")
            vp["cmedians"].set_color(col)
        ax.plot([], [], color=col, lw=6, alpha=0.55,
                label="path contains answer" if lab else "path without answer")
    ax.axhline(0, color="#333333", lw=0.8, ls="--")
    ax.set_xticks(range(len(ds)))
    ax.set_xticklabels([DATASET_LABEL.get(d, d) for d in ds])
    ax.set_ylabel(r"$\Delta H$ (entropy with path $-$ without)")
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def main():
    args = parse_args(__doc__)
    df = load_traces(args.traces)
    cl = method_rows(df)
    cand = explode_candidates(cl)
    summ = summarise(cand, cl)

    base = os.path.join(args.tab_dir, f"e1_entropy{args.tag}")
    summ.to_csv(base + ".csv", index=False)
    summ[["dataset", "n_paths", "pos_rate", "dH_pos", "dH_neg", "auc", "mwu_p", "hit1_dH", "hit1_random"]] \
        .to_latex(base + ".tex", index=False, float_format="%.3f", na_rep="--")
    plot(cand, os.path.join(args.fig_dir, f"e1_dh_by_label{args.tag}.pdf"))

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summ.round(3).to_string(index=False))
    print("written:", base + ".csv/.tex", "and figure in", args.fig_dir)


if __name__ == "__main__":
    main()
