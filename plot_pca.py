'''
Create a PCA plot
'''
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

SESSION_LEN = 300
OUT_PDF = "out/feature_space_pca.pdf"

def parse_user_from_path(path: str) -> str:
    # data/digitized_trace_Alice_1_run0.csv -> Alice_1
    base = path.split("/")[-1]
    base = base.replace("digitized_trace_", "")
    base = base.split("_run")[0]
    return base

def safe_mean(x):
    return float(np.mean(x)) if len(x) else 0.0

def safe_std(x):
    return float(np.std(x)) if len(x) else 0.0

def safe_median(x):
    return float(np.median(x)) if len(x) else 0.0

def extract_features(df: pd.DataFrame, session_gap_slots: int) -> list:
    presence = df["presence"].to_numpy(dtype=int)
    typing = df["typing"].to_numpy(dtype=int)
    sent = df["message_sent"].to_numpy(dtype=int)

    # Basic counts
    total_presence = int(presence.sum())
    total_typing = int(typing.sum())
    total_sent = int(sent.sum())

    # Ratios that capture passive vs active participation
    presence_only = int(((presence == 1) & (typing == 0) & (sent == 0)).sum())
    presence_only_frac = presence_only / max(total_presence, 1)

    # Typing to sent
    typing_to_sent = total_typing / max(total_sent, 1)
    sent_to_typing = total_sent / max(total_typing, 1)

    # Activity mask
    active = (presence | typing | sent).astype(int)

    # gaps between message_sent events
    sent_idx = np.where(sent == 1)[0]
    sent_gaps = np.diff(sent_idx) if len(sent_idx) > 1 else np.array([])
    sent_gap_mean = safe_mean(sent_gaps)
    sent_gap_std = safe_std(sent_gaps)
    sent_gap_median = safe_median(sent_gaps)

    # Burst structure for typing
    typing_bursts = []
    burst = 0
    for t in typing:
        if t == 1:
            burst += 1
        else:
            if burst > 0:
                typing_bursts.append(burst)
                burst = 0
    if burst > 0:
        typing_bursts.append(burst)

    typing_burst_mean = safe_mean(typing_bursts)
    typing_burst_std = safe_std(typing_bursts)
    typing_burst_max = float(max(typing_bursts)) if len(typing_bursts) else 0.0

    # Sessionization based on gaps in "any activity"
    active_idx = np.where(active == 1)[0]
    if len(active_idx) == 0:
        session_count = 0
        session_lengths = []
        session_activity = []
    else:
        splits = [0]
        for i in range(1, len(active_idx)):
            if (active_idx[i] - active_idx[i - 1]) > session_gap_slots:
                splits.append(i)
        splits.append(len(active_idx))

        session_lengths = []
        session_activity = []
        for a, b in zip(splits[:-1], splits[1:]):
            seg = active_idx[a:b]
            length = int(seg[-1] - seg[0] + 1)
            session_lengths.append(length)
            density = len(seg) / max(length, 1)
            session_activity.append(density)

        session_count = len(session_lengths)

    session_len_mean = safe_mean(session_lengths)
    session_len_std = safe_std(session_lengths)
    session_len_max = float(max(session_lengths)) if len(session_lengths) else 0.0
    session_density_mean = safe_mean(session_activity)
    session_density_std = safe_std(session_activity)

    # Presence frequency
    presence_rate = total_presence / max(len(df), 1)

    # Sent rate across the day
    sent_rate = total_sent / max(len(df), 1)

    return [
        total_presence,
        total_typing,
        total_sent,
        presence_only_frac,
        typing_to_sent,
        sent_to_typing,
        sent_gap_mean,
        sent_gap_std,
        sent_gap_median,
        typing_burst_mean,
        typing_burst_std,
        typing_burst_max,
        session_count,
        session_len_mean,
        session_len_std,
        session_len_max,
        session_density_mean,
        session_density_std,
        presence_rate,
        sent_rate,
    ]

def label_to_tex(user: str) -> str:
    # Alice_1 -> $Alice_{1}$ (and same for others)
    if "_" not in user:
        return user
    base, idx = user.rsplit("_", 1)
    return f"${base}_{{{idx}}}$"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=0, help="run number used in filenames (default: 0)")
    parser.add_argument("--data_dir", type=str, default="data", help="directory containing digitized traces")
    args = parser.parse_args()

    # load files
    pattern = f"{args.data_dir}/digitized_trace_*_run{args.run}.csv"
    digitized_files = sorted(glob.glob(pattern))

    users = [parse_user_from_path(f) for f in digitized_files]

    feature_matrix = []
    labels = []
    colors = []
    profile_colors = {
        "Alice": "steelblue",
        "Bob": "darkorange",
        "Charlie": "seagreen",
        "Diane": "mediumpurple"
    }

    for f, u in zip(digitized_files, users):
        df = pd.read_csv(f)

        feats = extract_features(df, session_gap_slots=SESSION_LEN)
        feature_matrix.append(feats)
        labels.append(u)

        assigned = False
        for profile in profile_colors:
            if profile in u:
                colors.append(profile_colors[profile])
                assigned = True
                break
        if not assigned:
            colors.append("gray")

    feature_matrix = np.asarray(feature_matrix, dtype=float)

    # z-score normalize features across users
    mu = feature_matrix.mean(axis=0)
    sigma = feature_matrix.std(axis=0) + 1e-8
    norm = (feature_matrix - mu) / sigma

    pca = PCA(n_components=2)
    coords = pca.fit_transform(norm)
    explained = pca.explained_variance_ratio_

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=55)
    for i, u in enumerate(labels):
        plt.text(coords[i, 0], coords[i, 1], label_to_tex(u), fontsize=9, ha="center", va="top")
    plt.title("PCA of Behavioral Feature Vectors")
    plt.xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_PDF)
    plt.show()

    # Summary stats
    print("\n==== PCA Summary ====")
    print(f"Explained variance by PC1: {explained[0]:.4f}")
    print(f"Explained variance by PC2: {explained[1]:.4f}")
    for u, (x, y) in zip(labels, coords):
        print(f"{u}: PC1 = {x:.3f}, PC2 = {y:.3f}")

if __name__ == "__main__":
    main()