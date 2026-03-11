#!/usr/bin/env python3
'''
Counting baseline 
'''

import argparse
import glob
import os
from itertools import combinations
import numpy as np
import pandas as pd


def load_digitized_for_run(run: int, data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, f"digitized_trace_*_run{run}.csv")))
    users = [os.path.basename(f).split("_run")[0].replace("digitized_trace_", "") for f in files]
    traces = {}
    for u, f in zip(users, files):
        mat = pd.read_csv(f)[["presence", "typing", "message_sent"]].to_numpy(dtype=int)
        traces[u] = mat
    return users, traces


def load_all_runs(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "digitized_trace_*_run*.csv")))
    user_runs = {}
    for f in files:
        base = os.path.basename(f)
        user = base.split("_run")[0].replace("digitized_trace_", "")
        run = int(base.split("_run")[-1].replace(".csv", ""))
        mat = pd.read_csv(f)[["presence", "typing", "message_sent"]].to_numpy(dtype=int)
        user_runs.setdefault(user, {})[run] = mat
    return user_runs


def counts_vec(trace: np.ndarray) -> np.ndarray:
    # [presence count, typing count, sent count]
    return np.array([trace[:,0].sum(), trace[:,1].sum(), trace[:,2].sum()], dtype=float)


def calc_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    args = ap.parse_args()

    user_runs = load_all_runs(args.data_dir)
    users = sorted(user_runs.keys())

    feats = {u: {r: counts_vec(user_runs[u][r]) for r in user_runs[u]} for u in users}

    ranks = []
    top1 = 0
    top3 = 0
    top5 = 0
    total = 0

    for u in users:
        for held_run in feats[u].keys():
            templates = {}
            for v in users:
                runs_v = list(feats[v].keys())
                if v == u:
                    runs_v = [r for r in runs_v if r != held_run]
                if not runs_v:
                    continue
                templates[v] = np.mean(np.stack([feats[v][r] for r in runs_v], axis=0), axis=0)

            obs = feats[u][held_run]
            # compare an observation to the compiled templates
            ordered = sorted(templates.keys(), key=lambda v: calc_distance(obs, templates[v]))
            rank = ordered.index(u) + 1
            ranks.append(rank)

            top1 += int(rank == 1)
            top3 += int(rank <= 3)
            top5 += int(rank <= 5)
            total += 1

    print("Top-1: {:.1f}%".format(100*top1/total))
    print("Top-3: {:.1f}%".format(100*top3/total))
    print("Top-5: {:.1f}%".format(100*top5/total))
    print("Mean rank: {:.2f}".format(np.mean(ranks)))
    print("Median rank: {:.1f}".format(np.median(ranks)))


if __name__ == "__main__":
    main()
