#!/usr/bin/env python3
"""
Generate a scatter plot for true rank
"""

import pandas as pd
import matplotlib.pyplot as plt


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

PROFILE_SORT = ["Alice", "Bob", "Charlie", "Diane"]


def trim_min_max(df: pd.DataFrame) -> pd.DataFrame:
    kept = []
    for user, g in df.groupby("held_out_user", sort=False):
        g = g.sort_values("posterior_true", kind="mergesort").reset_index(drop=True)
        if len(g) > 2:
            g = g.iloc[1:-1]  # drop min and max
        kept.append(g)
    return pd.concat(kept, ignore_index=True)


def main():
    path = "data/truerank.csv"
    df = pd.read_csv(path)

    # Extract profile from held_out_user (e.g., Alice_1 -> Alice)
    df["profile"] = df["held_out_user"].apply(lambda x: x.split("_")[0])

    # Trim min and max
    df_trim = trim_min_max(df)
  

    # plot
    plt.figure(figsize=(9, 5))

    # order profiles
    profiles_present = [p for p in PROFILE_SORT if p in set(df_trim["profile"])]
    profiles_present += [p for p in sorted(df_trim["profile"].unique()) if p not in PROFILE_SORT]

    for profile in profiles_present:
        subset = df_trim[df_trim["profile"] == profile]
        if subset.empty:
            continue
        plt.scatter(
            subset["true_rank"],
            subset["posterior_true"],
            color=PROFILE_COLORS.get(profile, "gray"),
            label=PROFILE_LABELS.get(profile, profile),
            s=50,
            alpha=0.8,
        )

    # calculate baseline based on number of users
    mode_N = int(df_trim["N_pool"].mode().iloc[0])
    baseline = 1.0 / mode_N
    plt.axhline(y=baseline,color="gray",linestyle="--",linewidth=1,label=f"Uniform prior (1/{mode_N})")

    plt.xlabel("True User Rank in Posterior")
    plt.ylabel("Posterior Probability for True Identity")
    plt.title("True Rank Pool Re-identification")
    plt.legend(fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out = "out/true_rank.pdf"
    plt.savefig(out)
    plt.show()

    # Summary stats (trimmed)
    total = len(df_trim)
    print("\n==== True Rank ====")
    for k in [1, 3, 5]:
        count = (df_trim["true_rank"] <= k).sum()
        print(f"Top-{k} Accuracy: {count} / {total} ({100.0 * count / total:.1f}%)")
    
    print(f"Mean Rank: {df_trim['true_rank'].mean():.2f}")
    print(f"Median Rank: {df_trim['true_rank'].median()}")
    print(f"Mean Posterior(True): {df_trim['posterior_true'].mean():.3f}")
    print(f"Min Posterior(True): {df_trim['posterior_true'].min():.3f}")
    print(f"Max Posterior(True): {df_trim['posterior_true'].max():.3f}")

    print(f"\nCreated {out}")


if __name__ == "__main__":
    main()