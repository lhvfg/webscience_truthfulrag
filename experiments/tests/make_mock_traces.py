"""
Generate SYNTHETIC trace files for developing the analysis scripts before real
runs exist. Numbers are random and mean nothing -- never use them in the report.

Output: experiments/_mock/trace_{dataset}_{method}.jsonl
"""
import json
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from trace_schema import METHODS, validate  # noqa: E402

KG = next(m for m in METHODS if m not in ("wo_rag", "w_rag"))
OUT = os.path.join(ROOT, "experiments", "_mock")
DATASETS = {"faitheval": 100, "musique_neg": 100, "realtimeqa": 113, "squad_neg": 100}


def main(seed=0):
    rng = random.Random(seed)
    os.makedirs(OUT, exist_ok=True)
    for ds, n in DATASETS.items():
        files = {m: open(os.path.join(OUT, f"trace_{ds}_{m}.jsonl"), "w", encoding="utf-8") for m in METHODS}
        for i in range(n):
            gold, true = f"ans{i}", f"true{i}"
            base = dict(id=f"{ds}_{i:04d}", dataset=ds, question=f"q{i}?", choices=None,
                        gold=gold, gold_true=true if ds.endswith("_neg") else gold)
            in_ctx = rng.random() < 0.95
            in_graph = in_ctx and rng.random() < 0.85
            n_c = rng.randint(3, 30)
            h_base = rng.uniform(0.05, 1.5)
            cands = []
            for j in range(n_c):
                has = in_graph and rng.random() < 0.25
                dh = rng.gauss(0.25 if has else 0.0, 0.4)
                cands.append(dict(idx=j, text=f"Path:\nE{j}->{'ANS' if has else 'X'}",
                                  H=max(h_base + dh, 0.0), dH=dh, has_answer=has))
            kept = [c["idx"] for c in sorted(cands, key=lambda c: -c["dH"])[:10]]
            hit = any(cands[k]["has_answer"] for k in kept)
            recs = {
                "wo_rag": dict(correct=rng.random() < 0.15, calls=1, tok=90, lat=0.5),
                "w_rag": dict(correct=in_ctx and rng.random() < 0.7, calls=1, tok=600, lat=0.8),
                KG: dict(correct=(hit and rng.random() < 0.9) or (in_ctx and rng.random() < 0.4),
                         calls=4 + n_c, tok=8000 + 300 * n_c, lat=rng.uniform(25, 60)),
            }
            for m, r in recs.items():
                pred = gold if r["correct"] else "wrong"
                rec = dict(base, method=m, pred=pred, pred_raw=pred, correct=r["correct"], em=r["correct"],
                           answer_in_context=None if m == "wo_rag" else in_ctx,
                           graph=None, H_base=None, candidates=None, kept=None,
                           n_llm_calls=r["calls"], n_tokens=r["tok"], latency_s=round(r["lat"], 2))
                if m == KG:
                    rec.update(graph=dict(n_entities=rng.randint(5, 40), n_relations=rng.randint(4, 50),
                                          answer_in_graph=in_graph),
                               H_base=h_base, candidates=cands, kept=kept)
                assert not validate(rec), validate(rec)
                files[m].write(json.dumps(rec) + "\n")
        for f in files.values():
            f.close()
    print("mock traces written to", OUT)


if __name__ == "__main__":
    main()
