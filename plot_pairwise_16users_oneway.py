'''
Plot all 16 users, one direction (for size and readability constraints)
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob

# File paths
DATA_GLOB = "data/qif_pairwise_results_run*.csv"
OUTPUT_FILE = "out/entropy_pairwise_bar_chart_simplified.pdf"

# Generate shaded colors based on profile to show different users
def generate_shaded_colors(base_colors, max_shades=4, darken=False):
    def adjust_color(rgb, factor):
        if darken:
            return tuple(max(0, c * factor) for c in rgb)
        else:
            return tuple(min(1, 1 - (1 - c) * factor) for c in rgb)

    user_colors = {}
    for profile, base in base_colors.items():
        base_rgb = mcolors.to_rgb(base)
        for i in range(1, max_shades + 1):
            factor = 1 - (i - 1) * 0.15
            shaded_rgb = adjust_color(base_rgb, factor)
            user_colors[f"{profile}_{i}"] = mcolors.to_hex(shaded_rgb)

    return user_colors

base_profile_colors = {
    "Alice": "steelblue",
    "Bob": "darkorange",
    "Charlie": "seagreen",
    "Diane": "mediumpurple"
}
profile_shades = generate_shaded_colors(base_profile_colors, max_shades=4)

def drop_min_max(series):
    vals = sorted(series)
    return vals[1:-1] if len(vals) > 2 else vals

# Load all result CSVs
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(DATA_GLOB))], ignore_index=True)

# use just the first user
df = df[df["observed_user"] == df["user_a"]].copy()

# Manual aggregation with min/max dropped
grouped = []
pairwise_groups = df.groupby(["user_a", "user_b"])

for (user_a, user_b), group in pairwise_groups:
    if len(group) < 3:
        continue

    pa_vals = drop_min_max(group["posterior_a"].tolist())
    pb_vals = drop_min_max(group["posterior_b"].tolist())
    loss_vals = drop_min_max(group["entropy_loss_bits"].tolist())

    grouped.append({
        "user_a": user_a,
        "user_b": user_b,
        "posterior_a_mean": np.mean(pa_vals),
        "posterior_a_std": np.std(pa_vals),
        "posterior_b_mean": np.mean(pb_vals),
        "posterior_b_std": np.std(pb_vals),
        "loss_mean": np.mean(loss_vals),
        "loss_std": np.std(loss_vals)
    })

grouped = pd.DataFrame(grouped)


# Labels
def short_label(user):
    prefix = user.split("_")[0][0]
    suffix = user.split("_")[1]
    return f"{prefix}{suffix}"

grouped["pair"] = grouped.apply(lambda row: f"{short_label(row['user_a'])} vs {short_label(row['user_b'])}", axis=1)

# Create the plot
bar_width = 0.35
x = np.arange(len(grouped))
unique_users = sorted(set(grouped["user_a"]) | set(grouped["user_b"]))

fig, ax = plt.subplots(figsize=(12, 5))
for i, row in grouped.iterrows():
    a, b = row["user_a"], row["user_b"]
    pa, pb = row["posterior_a_mean"], row["posterior_b_mean"]
    ax.bar(x[i] - bar_width/2, pa, width=bar_width, color=profile_shades.get(a, "gray"))
    ax.bar(x[i] + bar_width/2, pb, width=bar_width, color=profile_shades.get(b, "gray"))

ax.set_xticks(x)
ax.set_xticklabels(grouped["pair"], rotation=90, ha='center')
ax.tick_params(axis='x', labelsize=7)
ax.set_ylim(0, 1.1)
ax.set_xlim(-0.55, len(grouped) - 0.55)
ax.set_ylabel("Posterior Probability")
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
ax.set_title(f"Indistinguishability Game (N={len(unique_users)}) - One Direction")
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Legend
short_names = {u: short_label(u) for u in unique_users}
handles = [
    plt.Line2D([], [], color=profile_shades.get(u, "gray"), marker='s', linestyle='None', label=short_names[u])
    for u in unique_users
]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(handles), fontsize=8, columnspacing=0.8)

plt.tight_layout()
plt.savefig(OUTPUT_FILE)
plt.show()





