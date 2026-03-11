'''
Plot top/mid/btm performers of full pairwise
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob

DATA_GLOB = "data/qif_pairwise_results_run*.csv"
OUTPUT_FILE = "out/pairwise_topmidbottom16_starred_by_ptrue.pdf"

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

# Utils
def drop_min_max(values):
    values = list(values)
    if len(values) < 3:
        return values
    values = sorted(values)
    return values[1:-1]

def subscript_abbrev(user):
    base, idx = user.split("_")
    return f"{base[0]}$_{{{idx}}}$"

def starred_label(observed_user, other_user):
    obs = subscript_abbrev(observed_user)
    oth = subscript_abbrev(other_user)
    return f"{obs}$^\\ast$ vs {oth}"


files = sorted(glob.glob(DATA_GLOB))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

star_rows = []
for (a, b, obs), g in df.groupby(["user_a", "user_b", "observed_user"]):
    pa = drop_min_max(g["posterior_a"].values)
    pb = drop_min_max(g["posterior_b"].values)
    dh = drop_min_max(g["entropy_loss_bits"].values)

    if len(dh) < 1:
        continue

    posterior_a_mean = float(np.mean(pa))
    posterior_b_mean = float(np.mean(pb))

    # True posterior
    if obs == a:
        p_true_mean = posterior_a_mean
    elif obs == b:
        p_true_mean = posterior_b_mean

    star_rows.append({
        "user_a": a,
        "user_b": b,
        "observed_user": obs,
        "posterior_a_mean": posterior_a_mean,
        "posterior_a_std": float(np.std(pa)),
        "posterior_b_mean": posterior_b_mean,
        "posterior_b_std": float(np.std(pb)),
        "p_true_mean": p_true_mean,
        "loss_mean": float(np.mean(dh)),
        "loss_std": float(np.std(dh)),
        "n_effective": len(dh),
    })

star = pd.DataFrame(star_rows)

pair_scores = []
for (a, b), g in star.groupby(["user_a", "user_b"]):
    pair_scores.append({
        "user_a": a,
        "user_b": b,
        "pair_ptrue_mean": float(g["p_true_mean"].mean()),
        "pair_ptrue_std": float(g["p_true_mean"].std()) if len(g) > 1 else 0.0,
        "num_dirs": int(len(g))
    })

pairs = pd.DataFrame(pair_scores).sort_values("pair_ptrue_mean", ascending=False).reset_index(drop=True)

# select top mid bottom
top_pairs = pairs.head(3)
bot_pairs = pairs.tail(3)
mid_start = max(0, len(pairs)//2 - 1)
mid_pairs = pairs.iloc[mid_start:mid_start + 3]
selected_pairs = pd.concat([top_pairs, mid_pairs, bot_pairs], ignore_index=True)

# ordering
ticks = []
for _, pr in selected_pairs.iterrows():
    a, b = pr["user_a"], pr["user_b"]
    g = star[(star["user_a"] == a) & (star["user_b"] == b)].copy()

    row_a = g[g["observed_user"] == a]
    row_b = g[g["observed_user"] == b]

    if len(row_a) == 1:
        r = row_a.iloc[0].to_dict()
        r["observed"] = a
        r["other"] = b
        r["tick_label"] = starred_label(a, b)
        ticks.append(r)

    if len(row_b) == 1:
        r = row_b.iloc[0].to_dict()
        r["observed"] = b
        r["other"] = a
        r["tick_label"] = starred_label(b, a)
        ticks.append(r)

ticks = pd.DataFrame(ticks)

print("\n==== Selected Unordered Pairs - Top/Mid/Bottom ====")
for _, pr in selected_pairs.iterrows():
    print(f"{pr['user_a']} vs {pr['user_b']}: mean P(true|O) = {pr['pair_ptrue_mean']:.4f} ")

## plot
bar_width = 0.34
x = np.arange(len(ticks))

fig, ax = plt.subplots(figsize=(9, 5))

def posterior_for(user, r):
    if user == r["user_a"]:
        return r["posterior_a_mean"], r["posterior_a_std"]
    elif user == r["user_b"]:
        return r["posterior_b_mean"], r["posterior_b_std"]

for i, r in ticks.iterrows():
    obs = r["observed"]
    oth = r["other"]

    p_obs, s_obs = posterior_for(obs, r)
    p_oth, s_oth = posterior_for(oth, r)

    # Left bar is always the observed/starred user
    ax.bar(x[i] - bar_width/2, p_obs, width=bar_width, color=profile_shades.get(obs, "gray"))
    ax.bar(x[i] + bar_width/2, p_oth, width=bar_width, color=profile_shades.get(oth, "gray"))

    ax.errorbar(x[i] - bar_width/2, p_obs, yerr=s_obs, fmt='k_', capsize=3, linewidth=1)
    ax.errorbar(x[i] + bar_width/2, p_oth, yerr=s_oth, fmt='k_', capsize=3, linewidth=1)

    label_y = min(1.15, max(p_obs + s_obs, p_oth + s_oth) + 0.05)

ax.set_xticks(x)
ax.set_xticklabels(ticks["tick_label"], rotation=25, ha='center', fontsize=8)
ax.tick_params(axis='x', pad=0)

ax.set_ylim(0, 1.15)
ax.set_ylabel("Posterior Probability", fontsize=12)
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)

ax.set_title("Indistinguishability Game (Top / Mid / Bottom, N=16)", fontsize=13)
ax.grid(axis='y', linestyle='--', alpha=0.6)

# legend
involved_users = sorted(set(ticks["user_a"]).union(set(ticks["user_b"])))
handles = [
    plt.Line2D([], [], color=profile_shades.get(u, "gray"), marker='s', linestyle='None',
               label=subscript_abbrev(u))
    for u in involved_users
]
ncol = min(6, len(handles))
ax.legend(handles=handles, loc="upper center", ncol=ncol+5, fontsize=10,
          columnspacing=0.3, handletextpad=0.3, frameon=True)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"Created {OUTPUT_FILE}")
plt.show()
