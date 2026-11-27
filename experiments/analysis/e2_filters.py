"""
E2  Compare path-filtering strategies on the SAME candidate paths.

Candidates and their entropies come from the trace, so changing the filter
only requires re-running the final answer generation (1 LLM call / sample).

Strategies (all capped at top_k=10 like the original code unless noted)
  code        rule actually implemented in modules.entropy_filter (re-computed from H / H_base;
              sanity-checked against the trace's `kept`)
  paper       rule written in the paper: keep dH > tau, ordered by dH
  all         no filtering: every candidate (<=30), as in the code's own ablation (run_ablation)
  random      random subset, same size as `code`
  reverse     lowest-dH paths, same size as `code`
  none        no paths at all (question + original context, CoT prompt)

Step 1 (offline, free):   python experiments/analysis/e2_filters.py
    -> e2_offline.csv: #kept, empty rate, answer coverage / precision per strategy & tau
Step 2 (online, ~1 call per sample per strategy):
    python experiments/analysis/e2_filters.py --regenerate --strategies code paper none random reverse all
    -> results/e2/e2_{strategy}_tau{tau}_{dataset}.jsonl  (resumable)
    -> e2_accuracy.csv / .tex, fig e2_accuracy.pdf
Env for step 2: same as the main runs (OPENAI_API_KEY / OPENAI_BASE_URL, all requirements).
"""
import asyncio
import importlib
import json
import os
import random

import numpy as np
import pandas as pd

from common import (C_GREY, C_MAIN, DATASET_LABEL, ROOT, method_rows, ordered, parse_args,
                    load_traces, use_style)

TOP_K = 10
DATA_FILES = {
    "faitheval": "faitheval_data.json", "musique_neg": "musique_negative.json",
    "musique_gold": "musique_golden.json", "squad_neg": "squad_negative.json",
    "squad_gold": "squad_golden.json", "realtimeqa": "timeqa_2022_nota.json",
}


# ------------------------------------------------------------------ strategies
def keep_code(cands, h_base, tau):
    """Exact re-implementation of modules.entropy_filter's selection."""
    sel = []
    for c in cands:
        if c["dH"] >= 0.0:
            sel.append(c)
        elif c["H"] != 0.0 and h_base / c["H"] <= 10 ** tau:
            sel.append(c)
    if not sel and cands:
        m = max(c["H"] for c in cands)
        sel = [c for c in cands if c["H"] == m]
    # entropy_filter de-duplicates by text (dict keyed by element); emulate.
    seen, uniq = set(), []
    for c in sorted(sel, key=lambda c: -c["dH"]):
        if c["text"] not in seen:
            seen.add(c["text"]); uniq.append(c)
    return [c["idx"] for c in uniq[:TOP_K]]


def keep_paper(cands, h_base, tau):
    return [c["idx"] for c in sorted((c for c in cands if c["dH"] > tau), key=lambda c: -c["dH"])[:TOP_K]]


def select(strategy, rec, tau, rng):
    cands, h = rec["candidates"], rec["H_base"]
    if strategy == "code":
        return keep_code(cands, h, tau)
    if strategy == "paper":
        return keep_paper(cands, h, tau)
    if strategy == "all":
        return [c["idx"] for c in cands]
    if strategy == "none":
        return []
    n = len(keep_code(cands, h, tau))
    if strategy == "random":
        return rng.sample([c["idx"] for c in cands], min(n, len(cands)))
    if strategy == "reverse":
        return [c["idx"] for c in sorted(cands, key=lambda c: c["dH"])[:n]]
    raise ValueError(strategy)


def tau_grid(strategy, taus):
    return taus if strategy in ("code", "paper") else [None]


# --------------------------------------------------------------- offline part
def offline(cl, strategies, taus, seed):
    rows, mismatch = [], []
    for _, r in cl.iterrows():
        rec = r.to_dict()
        if r.get("candidates") is not None:
            mismatch.append(keep_code(rec["candidates"], rec["H_base"], 1.0) != list(rec["kept"]))
        for s in strategies:
            for tau in tau_grid(s, taus):
                rng = random.Random(f"{seed}-{rec['id']}")
                k = select(s, rec, 1.0 if tau is None else tau, rng)
                hits = [rec["candidates"][i]["has_answer"] for i in k]
                rows.append(dict(dataset=rec["dataset"], id=rec["id"], strategy=s, tau=tau,
                                 n_kept=len(k), empty=len(k) == 0,
                                 coverage=any(hits), precision=np.mean(hits) if hits else np.nan))
    return pd.DataFrame(rows), float(np.mean(mismatch)) if mismatch else float("nan")


# ---------------------------------------------------------------- online part
def load_samples():
    out = {}
    for ds, fn in DATA_FILES.items():
        p = os.path.join(ROOT, "datas", fn)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out[ds] = {str(x["id"]): x for x in json.load(f)}
    return out


async def regenerate(cl, strategies, taus, seed, model, backend, concurrency):
    import sys
    sys.path.insert(0, ROOT)
    pkg = next(d for d in os.listdir(ROOT) if os.path.isfile(os.path.join(ROOT, d, "pipeline.py")))
    modules = importlib.import_module(f"{pkg}.modules")
    ev = importlib.import_module(f"{pkg}.evaluate")
    fmt = importlib.import_module(f"{pkg}.util").FormatConverter

    samples = load_samples()
    out_dir = os.path.join(ROOT, "results", "e2")
    os.makedirs(out_dir, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)

    async def one(rec, s, tau, fh, lock):
        sample = samples[rec["dataset"]][str(rec["id"])]
        rng = random.Random(f"{seed}-{rec['id']}")
        k = select(s, rec, 1.0 if tau is None else tau, rng)
        elems = [rec["candidates"][i]["text"] for i in k]
        async with sem:
            preds = await modules.predict_answer([sample], [{"id": sample["id"], "element": elems}],
                                                 backend_type=backend, model_name=model, generation_type="cot")
        raw = preds[sample["id"]]
        pred = fmt.extract_answer(raw)
        if isinstance(pred, list):
            pred = pred[0] if pred else ""
        pred = "" if pred is None else str(pred)
        gold = sample["answer"]
        line = dict(id=rec["id"], dataset=rec["dataset"], strategy=s, tau=tau, kept=k, pred=pred, pred_raw=raw,
                    correct=bool(ev.metric_max_over_ground_truths(ev.acc_score, pred, gold)),
                    em=bool(ev.metric_max_over_ground_truths(ev.exact_match_score, pred, gold)))
        async with lock:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n"); fh.flush()

    for s in strategies:
        for tau in tau_grid(s, taus):
            for ds in ordered(cl.dataset):
                path = os.path.join(out_dir, f"e2_{s}_tau{tau}_{ds}.jsonl")
                done = set()
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as f:
                        done = {json.loads(l)["id"] for l in f if l.strip()}
                todo = [r.to_dict() for _, r in cl[cl.dataset == ds].iterrows() if r["id"] not in done]
                print(f"[E2] {s} tau={tau} {ds}: {len(done)} done, {len(todo)} to run")
                lock = asyncio.Lock()
                with open(path, "a", encoding="utf-8") as fh:
                    await asyncio.gather(*[one(r, s, tau, fh, lock) for r in todo])


def accuracy_table(cl, args):
    import glob
    files = glob.glob(os.path.join(ROOT, "results", "e2", "e2_*.jsonl"))
    if not files:
        return None
    rows = [json.loads(l) for f in files for l in open(f, encoding="utf-8") if l.strip()]
    e2 = pd.DataFrame(rows)
    e2 = e2[e2.id.isin(set(cl.id))]
    e2["tau"] = e2["tau"].astype(object).where(e2["tau"].notna(), None)
    e2["setting"] = e2.apply(lambda r: r.strategy if r.tau is None else f"{r.strategy} (tau={r.tau:g})", axis=1)
    tab = e2.pivot_table(index="setting", columns="dataset", values="correct", aggfunc="mean")
    tab = tab[[d for d in ordered(tab.columns)]]
    tab.columns = [DATASET_LABEL.get(c, c) for c in tab.columns]
    tab["Avg."] = tab.mean(axis=1)
    orig = cl.groupby("dataset").correct.mean()
    tab.loc["(trace) original run"] = [orig.get(d, np.nan) for d in ordered(cl.dataset)] + [orig.mean()]
    tab = (tab * 100).round(1).sort_values("Avg.", ascending=False)
    return tab


def plot_acc(tab, path):
    plt = use_style()
    t = tab.drop(index="(trace) original run", errors="ignore")
    fig, ax = plt.subplots(figsize=(4.8, 0.35 * len(t) + 1.0))
    cols = [C_MAIN if s.startswith("code") else C_GREY for s in t.index]
    ax.barh(range(len(t)), t["Avg."], color=cols)
    for i, v in enumerate(t["Avg."]):
        ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=7)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t.index)
    ax.invert_yaxis()
    ax.set_xlabel("average accuracy (%)")
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def plot_tau(off, path):
    plt = use_style()
    g = off[off.strategy == "paper"].groupby("tau").agg(coverage=("coverage", "mean"), empty_rate=("empty", "mean"),
                                                         n_kept=("n_kept", "mean")).reset_index()
    fig, ax = plt.subplots()
    ax.plot(g.tau, g.coverage, marker="o", color=C_MAIN, label="kept paths contain answer")
    ax.plot(g.tau, g.empty_rate, marker="s", color=C_GREY, label="nothing kept")
    ax.set_xlabel(r"threshold $\tau$ (paper rule $\Delta H>\tau$)")
    ax.set_ylabel("fraction of samples")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def main():
    def extra(ap):
        ap.add_argument("--strategies", nargs="+", default=["code", "paper", "all", "random", "reverse", "none"])
        ap.add_argument("--taus", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0, 3.0])
        ap.add_argument("--regen-taus", nargs="+", type=float, default=[1.0],
                        help="taus used when regenerating (keep small: each costs 1 call/sample)")
        ap.add_argument("--seed", type=int, default=42)
        ap.add_argument("--regenerate", action="store_true")
        ap.add_argument("--model", default="gpt-4o-mini")
        ap.add_argument("--backend", default="openai")
        ap.add_argument("--concurrency", type=int, default=8)
    args = parse_args(__doc__, extra)

    df = load_traces(args.traces)
    cl = method_rows(df)

    off, mism = offline(cl, args.strategies, args.taus, args.seed)
    agg = off.groupby(["strategy", "tau"], dropna=False).agg(
        n_kept=("n_kept", "mean"), empty_rate=("empty", "mean"),
        coverage=("coverage", "mean"), precision=("precision", "mean")).reset_index()
    base = os.path.join(args.tab_dir, f"e2_offline{args.tag}")
    agg.to_csv(base + ".csv", index=False)
    agg.to_latex(base + ".tex", index=False, float_format="%.3f", na_rep="--")
    print(agg.round(3).to_string(index=False))
    print(f"sanity: re-implemented code rule differs from trace `kept` in {mism:.1%} of samples "
          "(should be ~0%; non-zero means traces were produced with another tau/top_k)")
    if "paper" in args.strategies:
        plot_tau(off, os.path.join(args.fig_dir, f"e2_tau_coverage{args.tag}.pdf"))

    if args.regenerate:
        asyncio.run(regenerate(cl, args.strategies, args.regen_taus, args.seed,
                               args.model, args.backend, args.concurrency))

    tab = accuracy_table(cl, args)
    if tab is not None:
        b = os.path.join(args.tab_dir, f"e2_accuracy{args.tag}")
        tab.to_csv(b + ".csv")
        tab.to_latex(b + ".tex", float_format="%.1f", na_rep="--")
        plot_acc(tab, os.path.join(args.fig_dir, f"e2_accuracy{args.tag}.pdf"))
        print(tab.to_string())


if __name__ == "__main__":
    main()
