"""
Fix the sample subsets shared by Sean and Hakimi (seed=42), so every run and every
analysis uses exactly the same question ids.

  faitheval            100 random ids of 1000
  musique_{neg,gold}   100 random ids, the SAME ids for both (files are paired by id)
  squad_{neg,gold}     100 random ids, the SAME ids for both
  realtimeqa           all 113

Output: experiments/samples/ids_{dataset}.txt  (one id per line; .txt because *.json is gitignored)

Run:  python experiments/make_samples.py [--n 100] [--seed 42]
"""
import argparse
import json
import os
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "experiments", "samples")


def load_ids(name):
    with open(os.path.join(ROOT, "datas", name), encoding="utf-8") as f:
        return [str(x["id"]) for x in json.load(f)]


def write(dataset, ids):
    with open(os.path.join(OUT, f"ids_{dataset}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")
    print(f"{dataset:14s} {len(ids)} ids")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(a.seed)

    ids = load_ids("faitheval_data.json")
    write("faitheval", sorted(rng.sample(ids, min(a.n, len(ids)))))

    for key, gold_f, neg_f in [("musique", "musique_golden.json", "musique_negative.json"),
                               ("squad", "squad_golden.json", "squad_negative.json")]:
        gold, neg = load_ids(gold_f), load_ids(neg_f)
        shared = sorted(set(gold) & set(neg))
        assert len(shared) == len(gold) == len(neg), f"{key}: golden/negative ids are not paired"
        pick = sorted(rng.sample(shared, min(a.n, len(shared))))
        write(f"{key}_neg", pick)
        write(f"{key}_gold", pick)

    write("realtimeqa", load_ids("timeqa_2022_nota.json"))


if __name__ == "__main__":
    main()
