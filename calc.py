#!/usr/bin/env python3
import argparse
import glob
import os
from itertools import combinations
from math import log2

import numpy as np
import pandas as pd

OUT_DIR = 'data'

# Utils
def compute_entropy(probs: dict) -> float:
    return -sum(p * log2(p) for p in probs.values() if p > 0)


def safe_softmax(log_scores: dict) -> dict:
    keys = list(log_scores.keys())
    arr = np.array([log_scores[k] for k in keys], dtype=float)
    m = float(np.max(arr))
    ex = np.exp(arr - m)
    Z = float(np.sum(ex))
    if Z <= 0:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: float(v) for k, v in zip(keys, ex / Z)}


def euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# Feature extractor
def extract_features(trace: np.ndarray) -> np.ndarray:
    """
    Extract the feature vectors from the data set.
    """
    presence = trace[:, 0]
    typing = trace[:, 1]
    sent = trace[:, 2]

    total_presence = float(presence.sum())
    total_typing = float(typing.sum())
    total_sent = float(sent.sum())

    presence_only = float(((presence == 1) & (typing == 0) & (sent == 0)).sum())
    presence_only_frac = presence_only / max(total_presence, 1.0)

    # Typing burst lengths
    burst_lengths = []
    burst = 0
    for t in typing:
        if t == 1:
            burst += 1
        elif burst > 0:
            burst_lengths.append(burst)
            burst = 0
    if burst > 0:
        burst_lengths.append(burst)

    avg_burst = float(np.mean(burst_lengths)) if burst_lengths else 0.0
    max_burst = float(np.max(burst_lengths)) if burst_lengths else 0.0

    # Any-activity slots (presence OR typing OR sent)
    active_slots = np.where(trace.any(axis=1))[0]
    if len(active_slots) == 0:
        session_count = 0.0
        median_gap_any = 0.0
        frac_long_gaps_any = 0.0
        active_minutes = 0.0
    else:
        gaps_any = np.diff(active_slots)
        session_count = float(1 + np.sum(gaps_any > 300))
        median_gap_any = float(np.median(gaps_any)) if len(gaps_any) else 0.0
        frac_long_gaps_any = float(np.mean(gaps_any > 300)) if len(gaps_any) else 0.0
        active_minutes = float(len(np.unique(active_slots // 60)))

    # Timing between message_sent events
    sent_slots = np.where(sent == 1)[0]
    median_gap_sent = float(np.median(np.diff(sent_slots))) if len(sent_slots) >= 2 else 0.0

    # Timing between typing events
    typing_slots = np.where(typing == 1)[0]
    median_gap_typing = float(np.median(np.diff(typing_slots))) if len(typing_slots) >= 2 else 0.0

    # Burst starts and spacing
    burst_starts = []
    in_burst = False
    for i, t in enumerate(typing):
        if t == 1 and not in_burst:
            burst_starts.append(i)
            in_burst = True
        elif t == 0 and in_burst:
            in_burst = False
    median_gap_burst_starts = float(np.median(np.diff(burst_starts))) if len(burst_starts) >= 2 else 0.0

    return np.array([
        total_presence,
        total_typing,
        total_sent,
        presence_only_frac,
        avg_burst,
        max_burst,
        session_count,
        median_gap_any,
        median_gap_sent,
        median_gap_typing,
        frac_long_gaps_any,
        active_minutes,
        median_gap_burst_starts,
    ], dtype=float)


def load_digitized_for_run(data_dir: str, run: int) -> tuple[list[str], dict]:
    '''
    Load digitized traces for a specific run
    '''
    files = sorted(glob.glob(os.path.join(data_dir, f"digitized_trace_*_run{run}.csv")))
    if not files:
        raise FileNotFoundError(f"No files found: {data_dir}/digitized_trace_*_run{run}.csv")

    users = [os.path.basename(f).split("_run")[0].replace("digitized_trace_", "") for f in files]
    traces = {}
    for u, f in zip(users, files):
        df = pd.read_csv(f)
        traces[u] = df[["presence", "typing", "message_sent"]].to_numpy(dtype=int)

    lengths = [traces[u].shape[0] for u in users]
    if len(set(lengths)) != 1:
        min_T = min(lengths)
        for u in users:
            traces[u] = traces[u][:min_T, :]

    return users, traces


def load_all_runs(data_dir: str) -> dict:
    '''
    Laod all runs.
    '''
    files = sorted(glob.glob(os.path.join(data_dir, "digitized_trace_*_run*.csv")))
    if not files:
        raise FileNotFoundError(f"No digitized_trace_*_run*.csv files found in {data_dir}/")

    user_runs = {}
    for f in files:
        base = os.path.basename(f)
        user = base.split("_run")[0].replace("digitized_trace_", "")
        run = int(base.split("_run")[-1].replace(".csv", ""))
        mat = pd.read_csv(f)[["presence", "typing", "message_sent"]].to_numpy(dtype=int)
        user_runs.setdefault(user, {})[run] = mat

    return user_runs



def create_ref_other_runs(users: list[str], data_dir: str, held_out_run: int) -> dict:
    ref_files = sorted(glob.glob(os.path.join(data_dir, "digitized_trace_*_run*.csv")))
    ref_traces_by_user = {u: [] for u in users}

    for f in ref_files:
        base = os.path.basename(f)
        if f"_run{held_out_run}.csv" in base:
            continue
        user = base.split("_run")[0].replace("digitized_trace_", "")
        if user in ref_traces_by_user:
            mat = pd.read_csv(f)[["presence", "typing", "message_sent"]].to_numpy(dtype=int)
            ref_traces_by_user[user].append(mat)

    # sanity check we're not missing some files if we're generating different numbers
    missing = [u for u, traces in ref_traces_by_user.items() if len(traces) == 0]
    if missing:
        raise RuntimeError(f"Missing reference runs for users: {missing}.")

    theta = {}
    for u, traces in ref_traces_by_user.items():
        feats = np.stack([extract_features(tr) for tr in traces], axis=0)
        theta[u] = feats.mean(axis=0)

    return theta


def zscore_from_reference(theta: dict) -> tuple[np.ndarray, np.ndarray]:
    U = list(theta.keys())
    M = np.stack([theta[u] for u in U], axis=0)
    mean = M.mean(axis=0)
    std = M.std(axis=0) + 1e-8
    return mean, std


def zscore_apply(vec: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (vec - mean) / std


def posterior_from_distance(observed_z: np.ndarray, theta_z: dict, tau: float, priors: dict) -> dict:
    dists = {u: euclid(observed_z, f_u) for u, f_u in theta_z.items()}
    tau_eff = float(tau)

    log_scores = {}
    for u, d in dists.items():
        log_scores[u] = (-d / tau_eff) + np.log(priors[u])

    return safe_softmax(log_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pairwise", "pool", "loo"], required=True)
    parser.add_argument("--run", type=int, help="Run number (required for pairwise/pool)")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    if args.mode in {"pairwise", "pool"}:
        if args.run is None:
            raise ValueError("--run is required for --mode pairwise or --mode pool")

        users, obs_traces = load_digitized_for_run(args.data_dir, args.run)
        theta = create_ref_other_runs(users, args.data_dir, args.run)

        mean_ref, std_ref = zscore_from_reference(theta)

        theta_z = {u: zscore_apply(theta[u], mean_ref, std_ref) for u in users}
        obs_z = {u: zscore_apply(extract_features(obs_traces[u]), mean_ref, std_ref) for u in users}

        rows = []

        if args.mode == "pairwise":
            for a, b in combinations(users, 2):
                prior = {a: 0.5, b: 0.5}
                H_prior = compute_entropy(prior)
                candidates = {a: theta_z[a], b: theta_z[b]}

                # observed = a
                post_a = posterior_from_distance(obs_z[a], candidates, args.tau, prior)
                dH_a = H_prior - compute_entropy(post_a)
                vuln_a = max(post_a.values())

                rows.append({
                    "run": args.run,
                    "mode": "pairwise",
                    "pair": f"{a} vs {b}",
                    "observed_user": a,
                    "user_a": a,
                    "user_b": b,
                    "posterior_a": round(post_a[a], 6),
                    "posterior_b": round(post_a[b], 6),
                    "entropy_loss_bits": round(dH_a, 6),
                    "bayes_vuln": round(vuln_a, 6),
                })

                # observed = b
                post_b = posterior_from_distance(obs_z[b], candidates, args.tau, prior)
                dH_b = H_prior - compute_entropy(post_b)
                vuln_b = max(post_b.values())

                rows.append({
                    "run": args.run,
                    "mode": "pairwise",
                    "pair": f"{a} vs {b}",
                    "observed_user": b,
                    "user_a": a,
                    "user_b": b,
                    "posterior_a": round(post_b[a], 6),
                    "posterior_b": round(post_b[b], 6),
                    "entropy_loss_bits": round(dH_b, 6),
                    "bayes_vuln": round(vuln_b, 6),
                })

            out_csv = os.path.join(OUT_DIR, f"qif_pairwise_results_run{args.run}.csv")
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"Created {out_csv}")
            return

        # pool
        N = len(users)
        prior = {u: 1.0 / N for u in users}
        H_prior = compute_entropy(prior)

        for target in users:
            post = posterior_from_distance(obs_z[target], theta_z, args.tau, prior)
            dH = H_prior - compute_entropy(post)
            vuln = max(post.values())

            sorted_users = sorted(post.keys(), key=lambda u: post[u], reverse=True)
            true_rank = sorted_users.index(target) + 1

            row = {
                "run": args.run,
                "mode": "pool",
                "target": target,
                "H_prior": round(H_prior, 6),
                "H_post": round(compute_entropy(post), 6),
                "entropy_loss_bits": round(dH, 6),
                "bayes_vuln": round(vuln, 6),
                "true_rank": true_rank,
                "top1": int(true_rank == 1),
                f"top{args.topk}_hit": int(true_rank <= args.topk),
            }
            for u in users:
                row[f"posterior_{u}"] = round(post[u], 6)
            rows.append(row)

        out_csv = os.path.join(OUT_DIR, f"qif_pool_results_run{args.run}.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Created {out_csv}")

        return

    # loo / true rank
    user_runs = load_all_runs(args.data_dir)
    users = sorted(user_runs.keys())

    feats = {u: {r: extract_features(user_runs[u][r]) for r in user_runs[u]} for u in users}

    rows = []

    for held_user in users:
        for held_run in sorted(feats[held_user].keys()):
            theta = {}
            for u in users:
                run_list = sorted(feats[u].keys())
                if u == held_user:
                    run_list = [r for r in run_list if r != held_run]
                if not run_list:
                    continue
                theta[u] = np.mean(np.stack([feats[u][r] for r in run_list], axis=0), axis=0)

            mean_ref, std_ref = zscore_from_reference(theta)

            theta_z = {u: zscore_apply(theta[u], mean_ref, std_ref) for u in theta.keys()}
            obs_z = zscore_apply(feats[held_user][held_run], mean_ref, std_ref)

            U = list(theta_z.keys())
            N = len(U)
            prior = {u: 1.0 / N for u in U}

            post = posterior_from_distance(obs_z, theta_z, args.tau, prior)

            H_prior = compute_entropy(prior)
            H_post = compute_entropy(post)
            dH = H_prior - H_post
            vuln = max(post.values())

            sorted_users = sorted(post.keys(), key=lambda u: post[u], reverse=True)
            true_rank = sorted_users.index(held_user) + 1 if held_user in post else None
            posterior_true = float(post.get(held_user, 0.0))

            rows.append({
                "held_out_user": held_user,
                "held_out_run": held_run,
                "N_pool": N,
                "H_prior": round(H_prior, 6),
                "H_post": round(H_post, 6),
                "entropy_loss_bits": round(dH, 6),
                "bayes_vuln": round(vuln, 6),
                "posterior_true": round(posterior_true, 6),
                "true_rank": true_rank,
                "top1": int(true_rank == 1) if true_rank is not None else 0,
                f"top{args.topk}_hit": int(true_rank is not None and true_rank <= args.topk),
            })

    out_csv = os.path.join(OUT_DIR, "truerank.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"Created {out_csv}")
    return


if __name__ == "__main__":
    main()


