#!/usr/bin/env python3
"""
Plot pairwise indistinguishability game esults for the 4-profile subset

Each unordered pair (A,B) produces two plotted groups:
  - "A* vs B": observed trace came from A
  - "B* vs A": observed trace came from B
  - and so on
"""

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# plt sizes
plt.rcParams["figure.figsize"] = (3.5, 2.6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

KEEP_USERS = {"Alice_1", "Bob_1", "Charlie_1", "Diane_1"}
INPUT_GLOB = "data/qif_pairwise_results_run*.csv"
OUT_PDF = "out/pairwise_4users.pdf"

# Utils
def sort_key(r):
        u1, u2 = r["pair_key"]
        # order by unordered pair then observed user
        return (short(u1), short(u2), 0 if r["observed"] == u1 else 1)

def drop_min_max_by_loss(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 3: # need atleast 3
        return df
    return df.sort_values("entropy_loss_bits").iloc[1:-1].copy()


def short(u: str) -> str:
    # Alice_1 -> A, Charlie_1 -> C, etc
    return u.split("_")[0][0]

# main run
def main():
    files = sorted(glob.glob(INPUT_GLOB))

    # bucket by (unordered_pair, observed_user)
    buckets = defaultdict(list)

    for path in files:
        df = pd.read_csv(path)
        df = df[df["mode"] == "pairwise"].copy()
        df = df[df["user_a"].isin(KEEP_USERS) & df["user_b"].isin(KEEP_USERS)].copy()

        for col in ["posterior_a", "posterior_b", "bayes_vuln", "entropy_loss_bits"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["posterior_a", "posterior_b", "bayes_vuln", "entropy_loss_bits"])

        for _, r in df.iterrows():
            ua = r["user_a"]
            ub = r["user_b"]
            obs = r["observed_user"]
            if obs not in {ua, ub}:
                continue

            pair_key = tuple(sorted([ua, ub]))
            buckets[(pair_key, obs)].append({
                "run": int(r["run"]),
                "user_a": ua,
                "user_b": ub,
                "observed_user": obs,
                "posterior_a": float(r["posterior_a"]),
                "posterior_b": float(r["posterior_b"]),
                "bayes_vuln": float(r["bayes_vuln"]),
                "entropy_loss_bits": float(r["entropy_loss_bits"]),
            })

    # Build the plotted groups
    plotted = []
    for (pair_key, obs), rows in buckets.items():
        u1, u2 = pair_key
        df = pd.DataFrame(rows)
        df = drop_min_max_by_loss(df)
        if len(df) < 2:
            continue

        p_u1 = []
        p_u2 = []
        for _, r in df.iterrows():
            ua = r["user_a"]
            ub = r["user_b"]
            pa = r["posterior_a"]
            pb = r["posterior_b"]
            if ua == u1 and ub == u2:
                p_u1.append(pa)
                p_u2.append(pb)
            elif ua == u2 and ub == u1:
                # swap because other direction now
                p_u1.append(pb)
                p_u2.append(pa)
            else:
                # shouldn't happen for this bucket
                continue

        p_u1 = np.array(p_u1, dtype=float)
        p_u2 = np.array(p_u2, dtype=float)

        # Bayes vuln for this direction
        bv = np.maximum(p_u1, p_u2)

        # Correctness, does the max correspond to observed_user
        if obs == u1:
            correct = (p_u1 >= p_u2).astype(float)
        else:
            correct = (p_u2 > p_u1).astype(float)

        loss = df["entropy_loss_bits"].to_numpy(dtype=float)

        # the * marks the observed user
        if obs == u1:
            group_label = f"{short(u1)}* vs {short(u2)}"
        else:
            group_label = f"{short(u2)}* vs {short(u1)}"

        plotted.append({
            "pair_key": pair_key,
            "observed": obs,
            "group_label": group_label,
            "u1": u1,
            "u2": u2,
            "p_u1_mean": float(p_u1.mean()),
            "p_u1_std": float(p_u1.std(ddof=0)),
            "p_u2_mean": float(p_u2.mean()),
            "p_u2_std": float(p_u2.std(ddof=0)),
            "bv_mean": float(bv.mean()),
            "bv_std": float(bv.std(ddof=0)),
            "acc": float(correct.mean()),
            "loss_mean": float(loss.mean()),
            "loss_std": float(loss.std(ddof=0)),
        })

    
    

    plotted = sorted(plotted, key=sort_key)


    # CLI summary
    print("\n==== Indistinguishability Game Summary for 4 Users ====")
    print("\n=======================================================")
    for r in plotted:
        u1, u2 = r["pair_key"]
        print(f"{r['group_label']}  (pair {u1} vs {u2}, observed={r['observed']}):")
        print(f"  P({u1}|O): {r['p_u1_mean']:.4f} +- {r['p_u1_std']:.4f}")
        print(f"  P({u2}|O): {r['p_u2_mean']:.4f} +- {r['p_u2_std']:.4f}")
        print(f"  BV: {r['bv_mean']:.4f} +- {r['bv_std']:.4f}   correct={r['acc']*100:.0f}%")
        print(f"  dH: {r['loss_mean']:.4f} +- {r['loss_std']:.4f}\n")

    # plot
    x = np.arange(len(plotted))
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    p_left, s_left, p_right, s_right = [], [], [], []

    for r in plotted:
        left_letter = r["group_label"].split("*")[0]
        right_letter = r["group_label"].split("vs")[1].strip()

        u1, u2 = r["pair_key"]  # actual names like Alice_1, Bob_1
        u1_letter = u1.split("_")[0][0]
        u2_letter = u2.split("_")[0][0]

        # plot on direction of observation
        if left_letter == u1_letter and right_letter == u2_letter:
            p_left.append(r["p_u1_mean"])
            s_left.append(r["p_u1_std"])
            p_right.append(r["p_u2_mean"])
            s_right.append(r["p_u2_std"])
        elif left_letter == u2_letter and right_letter == u1_letter:
            p_left.append(r["p_u2_mean"])
            s_left.append(r["p_u2_std"])
            p_right.append(r["p_u1_mean"])
            s_right.append(r["p_u1_std"])

    p_left = np.array(p_left)
    s_left = np.array(s_left)
    p_right = np.array(p_right)
    s_right = np.array(s_right)

    ax.bar(x - bar_w/2, p_left, width=bar_w, label="Chosen")
    ax.bar(x + bar_w/2, p_right, width=bar_w, label="Other")
    ax.errorbar(x - bar_w/2, p_left, yerr=s_left, fmt="k_", capsize=3)
    ax.errorbar(x + bar_w/2, p_right, yerr=s_right, fmt="k_", capsize=3)

    # include delta-h as label
    for i, r in enumerate(plotted):
        y = max(p_left[i] + s_left[i], p_right[i] + s_right[i]) + 0.03
        ax.text(
            x[i], y,
            f"ΔH={r['loss_mean']:.2f}",
            ha="center", va="bottom", fontsize=8
        )

    ax.set_xticks(x)
    ax.set_xticklabels([r["group_label"] for r in plotted], fontsize=9, rotation=25, ha="center")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Posterior probability", fontsize=12)
    ax.set_title("Indistinguishability Game (4 Profiles)", fontsize=12)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)

    plt.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Created {OUT_PDF}")


if __name__ == "__main__":
    main()
