#!/usr/bin/env python3
"""
Plot pool
"""
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROFILE_ORDER = {"Alice": 0, "Bob": 1, "Charlie": 2, "Diane": 3}
PROFILE_TO_LETTER = {"Alice": "A", "Bob": "B", "Charlie": "C", "Diane": "D"}

PROFILE_COLORS = {
    "Alice": "steelblue",
    "Bob": "darkorange",
    "Charlie": "seagreen",
    "Diane": "mediumpurple",
}

PROFILE_LABELS = {
    "Alice": "Deliberative",
    "Bob": "Low Engagement",
    "Charlie": "High Frequency",
    "Diane": "Intermittent",
}


def trim_min_max_by_entropy(group: pd.DataFrame) -> pd.DataFrame:
    if len(group) <= 2:
        return group.copy()

    g = group.sort_values("entropy_loss_bits", kind="mergesort").reset_index(drop=True)
    # drop first and last
    return g.iloc[1:-1].copy()


   # X tick labels: A_i, B_i, C_i, D_i
def to_letter_subscript(t: str) -> str:
    p, idx = t.split("_")
    letter = PROFILE_TO_LETTER.get(p, p[0].upper())
    
    return f"${letter}_{{{idx}}}$"


def main():
    files = sorted(glob.glob("data/qif_pool_results_run*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    df["profile"] = df["target"].apply(lambda x: x.split("_")[0])
    df["index"] = df["target"].apply(lambda x: int(x.split("_")[1]))
    df["sort_key"] = df["profile"].map(PROFILE_ORDER).fillna(99) + df["index"] / 100.0

    # allow for various pool size
    N_pool = int(df["target"].nunique())
    vuln_baseline = 1.0 / N_pool

    # trim outliers
    trimmed = (df.groupby("target", group_keys=False).apply(trim_min_max_by_entropy).reset_index(drop=True))

    # per user
    agg = trimmed.groupby("target").agg(
        loss_mean=("entropy_loss_bits", "mean"),
        loss_std=("entropy_loss_bits", "std"),
        vuln_mean=("bayes_vuln", "mean"),
        vuln_std=("bayes_vuln", "std"),
        profile=("profile", "first"),
        index=("index", "first"),
        n_used=("entropy_loss_bits", "count"),
    ).reset_index()

    agg["sort_key"] = agg["profile"].map(PROFILE_ORDER).fillna(99) + agg["index"] / 100.0
    agg = agg.sort_values("sort_key").reset_index(drop=True)

    agg["color"] = agg["profile"].map(PROFILE_COLORS).fillna("gray")

 

    xtick_labels = agg["target"].apply(to_letter_subscript).tolist()
    x = np.arange(len(agg))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    bars = ax1.bar(
        x,
        agg["loss_mean"].values,
        yerr=agg["loss_std"].fillna(0.0).values,
        capsize=4,
        color=agg["color"].values,
        alpha=0.95,
        error_kw={"ecolor": "tab:blue", "elinewidth": 1.5},
    )
    ax1.set_ylabel("Entropy Loss ($\\Delta H$, bits)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(xtick_labels, rotation=25, ha="center")

    # baseline
    ax1.axhline(y=0.0, color="gray", linestyle="--", linewidth=1)

    # BV
    ax2 = ax1.twinx()
    ax2.errorbar(
        x,
        agg["vuln_mean"].values,
        yerr=agg["vuln_std"].fillna(0.0).values,
        fmt="o",
        capsize=4,
        color="black",
        ecolor="black",
        elinewidth=1.2,
        markersize=5,
    )
    ax2.set_ylabel("Bayes Vulnerability ($V$)")

    # baseline uniform prior
    ax2.axhline(y=vuln_baseline, color="black", linestyle=":", linewidth=1)

    ax1.set_title(f"Pool Evaluation (N={N_pool}): $\\Delta H$ and Bayes Vulnerability")

    # legend
    handles = []
    seen = set()
    for p in agg["profile"].unique():
        label = PROFILE_LABELS.get(p, p)
        if label not in seen:
            handles.append(plt.Line2D([], [], color=PROFILE_COLORS.get(p, "gray"), marker="s", linestyle="None", label=label))
            seen.add(label)

    handles.append(plt.Line2D([], [], color="black", marker="o", linestyle="None", label="Bayes Vulnerability"))
    handles.append(plt.Line2D([], [], color="gray", linestyle="--", label="$\\Delta H$ baseline (0)"))
    handles.append(plt.Line2D([], [], color="black", linestyle=":", label=f"$V$ baseline (1/{N_pool})"))

    ax1.legend(handles=handles, loc="upper left", fontsize=9)

    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = "out/pool_entropy_vulnerability.pdf"
    plt.savefig(out)
    plt.show()

    # Summary stats
    print("\n==== Pool Summary ====")
    for _, r in agg.iterrows():
        ls = 0.0 if pd.isna(r["loss_std"]) else r["loss_std"]
        vs = 0.0 if pd.isna(r["vuln_std"]) else r["vuln_std"]
        print(f"{r['target']}: n={int(r['n_used'])}  ΔH={r['loss_mean']:.4f}+-{ls:.4f}     V={r['vuln_mean']:.4f}+-{vs:.4f}")

    print(f"Created {out}")


if __name__ == "__main__":
    main()


