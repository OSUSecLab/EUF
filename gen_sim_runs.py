import argparse
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os
import json
import hashlib
from typing import Dict, Any, Tuple

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, default=0, help="Run number for reproducibility")
parser.add_argument("--deliberative", type=int, default=4, help="Number of deliberative users")
parser.add_argument("--low_engagement", type=int, default=4, help="Number of low-engagement users")
parser.add_argument("--high_frequency", type=int, default=4, help="Number of high-frequency users")
parser.add_argument("--intermittent", type=int, default=4, help="Number of intermittent users")
args = parser.parse_args()

# Constants
base_date = datetime(year=2025, month=4, day=2)
AWAY_TIMEOUT = timedelta(minutes=5)

os.makedirs("data", exist_ok=True)

# Loadable profile paths
PROFILE_BASELINES_FILE = "data/profile_baselines.json"
USER_DEVIATIONS_FILE = "data/user_deviations.json"

# User Profiles
USER_PROFILES = {
    "deliberative":   {"target_sends": 90, "abandon_prob": 0.05, "presence_multiplier": 0.50},
    "low_engagement": {"target_sends": 90, "abandon_prob": 0.10, "presence_multiplier": 1.00},
    "high_frequency": {"target_sends": 90, "abandon_prob": 0.05, "presence_multiplier": 0.20},
    "intermittent":   {"target_sends": 90, "abandon_prob": 0.00, "presence_multiplier": 0.00},
}

# Workday times
WORK_START = datetime.strptime("09:00:00", "%H:%M:%S")
WORK_END   = datetime.strptime("19:00:00", "%H:%M:%S")

# Helpers
def stable_int_from_str(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def minutes_to_time_str(base: datetime, minutes_offset: int) -> str:
    t = base + timedelta(minutes=minutes_offset)
    return t.strftime("%H:%M:%S")

def parse_hms(hms: str) -> datetime:
    return datetime.strptime(hms, "%H:%M:%S").replace(
        year=base_date.year, month=base_date.month, day=base_date.day
    )


def default_profile_baselines() -> Dict[str, Any]:
    """
    Baseline definitions for each profile.
    """
    return {
        "deliberative": {
            "start_mean_min": 30,     
            "start_sd_min": 12,
            "dur_mean_hr": 8.0,
            "dur_sd_hr": 0.6,
            "away_timeout_min": 5.0,
            "away_timeout_sd_min": 0.8,
            "burstiness_mean": 1.35,
            "burstiness_sd": 0.25,
            "session_window_mean_sec": 8 * 60,
            "session_window_sd_sec": 3 * 60,
            "sessions_mean": 6,
            "sessions_sd": 2,
        },
        "low_engagement": {
            "start_mean_min": 60,
            "start_sd_min": 15,
            "dur_mean_hr": 7.5,
            "dur_sd_hr": 0.7,
            "away_timeout_min": 5.5,
            "away_timeout_sd_min": 1.0,
            "burstiness_mean": 1.15,
            "burstiness_sd": 0.20,
            "session_window_mean_sec": 10 * 60,
            "session_window_sd_sec": 4 * 60,
            "sessions_mean": 4,
            "sessions_sd": 2,
        },
        "high_frequency": {
            "start_mean_min": 15,
            "start_sd_min": 10,
            "dur_mean_hr": 8.5,
            "dur_sd_hr": 0.6,
            "away_timeout_min": 4.5,
            "away_timeout_sd_min": 0.8,
            "burstiness_mean": 1.70,
            "burstiness_sd": 0.30,
            "session_window_mean_sec": 6 * 60,
            "session_window_sd_sec": 2 * 60,
            "sessions_mean": 8,
            "sessions_sd": 2,
        },
        "intermittent": {
            "start_mean_min": 45,
            "start_sd_min": 15,
            "dur_mean_hr": 8.0,
            "dur_sd_hr": 0.8,
            "away_timeout_min": 5.0,
            "away_timeout_sd_min": 0.9,
            "num_bursts_mean": 3,
            "num_bursts_sd": 1,
            "burst_window_mean_sec": 6 * 60,
            "burst_window_sd_sec": 3 * 60,
            "burst_spacing_mean_sec": 75 * 60,
            "burst_spacing_sd_sec": 25 * 60,
        },
    }

def load_or_create_profile_baselines() -> Dict[str, Any]:
    '''
    Load or create a profile baseline.
    '''

    # If we have baseline files already...
    if os.path.exists(PROFILE_BASELINES_FILE):
        with open(PROFILE_BASELINES_FILE, "r") as f:
            return json.load(f)
    
    # or generate them.
    baselines = default_profile_baselines()

    with open(PROFILE_BASELINES_FILE, "w") as f:
        json.dump(baselines, f, indent=2, sort_keys=True)
    return baselines


def load_or_create_user_deviations(users: Dict[str, Any], baselines: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load or create per-user deviations around profile baselines.
    """
    if os.path.exists(USER_DEVIATIONS_FILE):
        with open(USER_DEVIATIONS_FILE, "r") as f:
            dev = json.load(f)
    else:
        dev = {}

    changed = False
    for user, cfg in users.items():
        if user in dev:
            continue

        profile = cfg["profile"]
        b = baselines[profile]
        r = random.Random(stable_int_from_str(f"user_dev::{user}"))

        # schedule deviations
        start_offset_min = int(round(r.gauss(0, b.get("start_sd_min", 12))))
        dur_offset_hr = float(r.gauss(0, b.get("dur_sd_hr", 0.6)))

        # timing / session structure deviations
        away_timeout_offset_min = float(r.gauss(0, b.get("away_timeout_sd_min", 0.9)))

        entry = {
            "start_offset_min": start_offset_min,
            "dur_offset_hr": dur_offset_hr,
            "away_timeout_offset_min": away_timeout_offset_min,
        }

        # non-intermittent additional deviations
        if profile != "intermittent":
            entry.update({
                "burstiness_offset": float(r.gauss(0, b.get("burstiness_sd", 0.25))),
                "sessions_offset": int(round(r.gauss(0, b.get("sessions_sd", 2)))),
                "session_window_offset_sec": int(round(r.gauss(0, b.get("session_window_sd_sec", 180)))),
            })
        else:
            entry.update({
                "num_bursts_offset": int(round(r.gauss(0, b.get("num_bursts_sd", 1)))),
                "burst_window_offset_sec": int(round(r.gauss(0, b.get("burst_window_sd_sec", 180)))),
                "burst_spacing_offset_sec": int(round(r.gauss(0, b.get("burst_spacing_sd_sec", 1500)))),
            })

        dev[user] = entry
        changed = True

    if changed:
        with open(USER_DEVIATIONS_FILE, "w") as f:
            json.dump(dev, f, indent=2, sort_keys=True)

    return dev


def generate_user_configs(baselines: Dict[str, Any], run_seed: int) -> Dict[str, Any]:
    """
    Generate users with start/duration derived from profile baselines + deviations.
    """
    users: Dict[str, Any] = {}
    profile_groups = [
        ("Alice", "deliberative", args.deliberative),
        ("Bob", "low_engagement", args.low_engagement),
        ("Charlie", "high_frequency", args.high_frequency),
        ("Diane", "intermittent", args.intermittent),
    ]

    for prefix, profile, count in profile_groups:
        for i in range(count):
            name = f"{prefix}_{i+1}"
            users[name] = {"profile": profile}

    return users


def compute_user_schedule(user: str, profile: str, baselines: Dict[str, Any], deviations: Dict[str, Any], run_rng: random.Random) -> Tuple[str, int]:
    """
    Determine the users schedule.
    """
    b = baselines[profile]
    d = deviations[user]

    # profile baseline
    start_mean_min = int(b.get("start_mean_min", 30))
    dur_mean_hr = float(b.get("dur_mean_hr", 8.0))

    # stable deviations
    start_min = start_mean_min + int(d.get("start_offset_min", 0))
    dur_hr = dur_mean_hr + float(d.get("dur_offset_hr", 0.0))

    # small per-run jitter
    start_min += int(round(run_rng.gauss(0, 3)))   # +/- a few minutes
    dur_hr += float(run_rng.gauss(0, 0.15))        # +/- ~10 minutes

    # clamp to reasonable bounds
    start_min = int(clamp(start_min, 0, 120))

    # working duration between 6 and 10 hours
    dur_hr = clamp(dur_hr, 6.0, 10.0)

    start_str = minutes_to_time_str(WORK_START, start_min)
    duration_hours = int(round(dur_hr))
    duration_hours = int(clamp(duration_hours, 6, 10))

    return start_str, duration_hours


def sample_session_windows(total_seconds: int, sessions: int, run_rng: random.Random) -> list:
    """
    Pick session start times spread across day.
    """
    if sessions <= 0:
        return []
    # Keep sessions well spread by sampling from coarse buckets
    bucket = max(1, total_seconds // max(1, sessions))
    starts = []
    for si in range(sessions):
        lo = si * bucket
        hi = min(total_seconds - 1, (si + 1) * bucket - 1)
        if hi <= lo:
            starts.append(lo)
        else:
            starts.append(run_rng.randint(lo, hi))
    starts.sort()
    return starts


def sample_nonintermittent_times(total_seconds: int, n_events: int, sessions: int, session_window_sec: int, burstiness: float, run_rng: random.Random) -> list:
    """
    Sample nonintermittent profile times
    """
    n_events = max(0, int(n_events))
    if total_seconds <= 1 or n_events <= 0:
        return []

    # If burstiness low, mix in more uniform samples
    uniform_frac = clamp(1.2 - burstiness, 0.0, 0.6)  # burstiness 1.7 => ~0, burstiness 1.0 => ~0.2
    n_uniform = int(round(n_events * uniform_frac))
    n_session = n_events - n_uniform

    times = set()

    # Uniform portion
    if n_uniform > 0:
        n_uniform = min(n_uniform, total_seconds)
        times.update(run_rng.sample(range(total_seconds), n_uniform))

    # Sessionized portion
    sessions = max(1, int(sessions))
    session_window_sec = int(clamp(session_window_sec, 60, 20 * 60))

    starts = sample_session_windows(total_seconds, sessions, run_rng)
    if not starts:
        starts = [0]

    remaining = n_session
    for si, s in enumerate(starts):
        left = len(starts) - si
        if remaining <= 0:
            break

        # base allocation
        mean_here = remaining / left
        alloc = int(round(clamp(run_rng.gauss(mean_here, max(1.0, mean_here * 0.25)), 1, remaining)))
        remaining -= alloc

        # sample within window
        w = min(session_window_sec, max(1, total_seconds - s))
        
        # allow some drift beyond the window based on burstiness (higher => tighter)
        tight = clamp(burstiness / 2.0, 0.3, 1.0)
        for _ in range(alloc):
            if run_rng.random() < tight:
                off = run_rng.randint(0, max(1, w) - 1)
            else:
                # looser: spill outside window a bit
                spill = int(round(w * 0.75))
                off = int(clamp(run_rng.gauss(w / 2, spill), 0, max(1, w) - 1))
            t = s + off
            if 0 <= t < total_seconds:
                times.add(t)

    # Maks sure we have enough events
    if len(times) < n_events:
        remaining = n_events - len(times)
        candidates = [t for t in range(total_seconds) if t not in times]
        if candidates:
            times.update(run_rng.sample(candidates, min(remaining, len(candidates))))

    return sorted(times)

def sample_intermittent_bursts(total_seconds: int, sends_target: int, num_bursts: int, burst_window_sec: int, burst_spacing_sec: int, run_rng: random.Random) -> list:
    """
    Intermittent profile only.
    """
    sends_target = max(0, int(sends_target))
    if total_seconds <= 1 or sends_target <= 0:
        return []

    num_bursts = int(clamp(num_bursts, 2, 6))
    burst_window_sec = int(clamp(burst_window_sec, 60, 20 * 60))
    burst_spacing_sec = int(clamp(burst_spacing_sec, 20 * 60, 3 * 60 * 60))

    safety_margin = 900
    latest_start = max(0, total_seconds - safety_margin)

    burst_starts = []
    for _ in range(num_bursts):
        for _try in range(2000):
            s = run_rng.randint(0, latest_start) if latest_start > 0 else 0
            if all(abs(s - prev) >= burst_spacing_sec for prev in burst_starts):
                burst_starts.append(s)
                break
    burst_starts.sort()
    if not burst_starts:
        burst_starts = [0]

    base_times = []
    remaining = sends_target

    for bi, s in enumerate(burst_starts):
        left = len(burst_starts) - bi
        if remaining <= 0:
            break

        mean_here = remaining / left

        # intermittent bursts
        alloc = int(round(clamp(run_rng.gauss(mean_here, max(2.0, mean_here * 0.35)), 1, remaining)))
        remaining -= alloc

        alloc = min(alloc, burst_window_sec)
        alloc = max(1, alloc)

        offsets = run_rng.sample(range(burst_window_sec), alloc) if burst_window_sec > 1 else [0] * alloc
        offsets.sort()
        for off in offsets:
            t = s + off
            if 0 <= t < total_seconds:
                base_times.append(t)

    base_times.sort()
    return base_times

# Simulation 
def simulate_user(user: str, profile_name: str, start_str: str, duration_hours: int, baselines: Dict[str, Any], deviations: Dict[str, Any], run_rng: random.Random):
    profile = USER_PROFILES[profile_name]
    b = baselines[profile_name]
    d = deviations[user]

    # Apply baseline + deviation + small run jitter to away timeout
    away_timeout_min = float(b.get("away_timeout_min", 5.0)) + float(d.get("away_timeout_offset_min", 0.0))
    away_timeout_min += float(run_rng.gauss(0, 0.25))  # small run jitter
    away_timeout_min = clamp(away_timeout_min, 2.5, 10.0)
    away_timeout = timedelta(minutes=away_timeout_min)

    # add noise to event occurances
    sends_target = int(profile["target_sends"] + round(run_rng.gauss(0, 2)))
    sends_target = int(clamp(sends_target, 75, 105))

    abandon_prob = profile["abandon_prob"] + float(run_rng.gauss(0, 0.01))
    abandon_prob = clamp(abandon_prob, 0.0, 0.35)

    presence_multiplier = profile["presence_multiplier"] + float(run_rng.gauss(0, 0.05))
    presence_multiplier = max(0.0, presence_multiplier)

    start_time = parse_hms(start_str)
    end_time = start_time + timedelta(hours=duration_hours)
    total_seconds = int((end_time - start_time).total_seconds())
    total_seconds = max(1, total_seconds)

    # Build base times
    if profile_name == "intermittent":
        num_bursts = int(b.get("num_bursts_mean", 3)) + int(d.get("num_bursts_offset", 0)) + int(round(run_rng.gauss(0, 0.5)))
        burst_window_sec = int(b.get("burst_window_mean_sec", 360)) + int(d.get("burst_window_offset_sec", 0)) + int(round(run_rng.gauss(0, 30)))
        burst_spacing_sec = int(b.get("burst_spacing_mean_sec", 4500)) + int(d.get("burst_spacing_offset_sec", 0)) + int(round(run_rng.gauss(0, 120)))

        base_times = sample_intermittent_bursts(total_seconds=total_seconds, sends_target=sends_target, num_bursts=num_bursts, burst_window_sec=burst_window_sec, burst_spacing_sec=burst_spacing_sec, run_rng=run_rng)
    
    # other profiles
    else:
        burstiness = float(b.get("burstiness_mean", 1.3)) + float(d.get("burstiness_offset", 0.0)) + float(run_rng.gauss(0, 0.08))
        burstiness = clamp(burstiness, 0.85, 2.25)

        sessions = int(b.get("sessions_mean", 6)) + int(d.get("sessions_offset", 0)) + int(round(run_rng.gauss(0, 0.5)))
        sessions = int(clamp(sessions, 2, 12))

        session_window_sec = int(b.get("session_window_mean_sec", 480)) + int(d.get("session_window_offset_sec", 0)) + int(round(run_rng.gauss(0, 30)))
        session_window_sec = int(clamp(session_window_sec, 60, 20 * 60))

        base_times = sample_nonintermittent_times(total_seconds=total_seconds,n_events=sends_target,sessions=sessions,session_window_sec=session_window_sec,burstiness=burstiness,run_rng=run_rng)

    # Make events
    log = []
    occupied_times = []

    for t_sec in base_times:
        t_typing = start_time + timedelta(seconds=int(t_sec))

        need_presence = (not occupied_times) or ((t_typing - occupied_times[-1]) > away_timeout)
        if need_presence:
            log.append((t_typing, user, "presence"))
            occupied_times.append(t_typing)

        # typing: single contiguous typing phase, every send must start with typing
        typing_len = int(clamp(round(run_rng.gauss(3, 1)), 1, 8))
        for k in range(typing_len):
            log.append((t_typing + timedelta(seconds=1 + k), user, "typing"))
        occupied_times.append(t_typing + timedelta(seconds=1 + typing_len))

        if run_rng.random() > abandon_prob:
            t_sent = t_typing + timedelta(seconds=1 + typing_len + run_rng.randint(1, 4))
            log.append((t_sent, user, "message_sent"))
            occupied_times.append(t_sent)

    # Passive presence-only
    num_extra = int(round(sends_target * presence_multiplier))
    num_extra = max(0, num_extra)

    if num_extra > 0:
        extra_times = sorted(run_rng.sample(range(total_seconds), min(num_extra, total_seconds)))
        for t_sec in extra_times:
            t_presence = start_time + timedelta(seconds=int(t_sec))
            if any(abs((t_presence - t).total_seconds()) < 300 for t in occupied_times):
                continue
            log.append((t_presence, user, "presence"))
            occupied_times.append(t_presence)

    return log

# Let's go!!

# Run RNG (only for run-level jitter and event placement)
run_rng = random.Random(args.run)

# Load baselines and generate users
baselines = load_or_create_profile_baselines()
users = generate_user_configs(baselines, run_seed=args.run)

# Load stable per-user deviations
deviations = load_or_create_user_deviations(users, baselines)

# Compute schedules (baseline + deviation + small run jitter)
for user, cfg in users.items():
    profile = cfg["profile"]
    start_str, duration_hours = compute_user_schedule(
        user=user,
        profile=profile,
        baselines=baselines,
        deviations=deviations,
        run_rng=run_rng
    )
    cfg["start"] = start_str
    cfg["duration"] = duration_hours

# Simulate
all_logs = []
for name, cfg in users.items():
    all_logs.extend(simulate_user(user=name,profile_name=cfg["profile"],start_str=cfg["start"],duration_hours=cfg["duration"],baselines=baselines,deviations=deviations,run_rng=run_rng))

df = pd.DataFrame(all_logs, columns=["timestamp", "user", "event_type"])
df = df.sort_values("timestamp").reset_index(drop=True)

# and digitize! 
df["timestamp"] = pd.to_datetime(df["timestamp"])
activities = ["presence", "typing", "message_sent"]
start_time = df["timestamp"].min().floor("min")
end_time = df["timestamp"].max().ceil("min")
T = int((end_time - start_time).total_seconds()) + 1
T = max(1, T)

for user in df["user"].unique():
    mat = np.zeros((T, len(activities)), dtype=int)
    user_df = df[df["user"] == user]
    for _, row in user_df.iterrows():
        slot = int((row["timestamp"] - start_time).total_seconds())
        if 0 <= slot < T:
            mat[slot, activities.index(row["event_type"])] = 1
    df_out = pd.DataFrame(mat, columns=activities)
    df_out["time_slot"] = np.arange(T)
    df_out.to_csv(f"data/digitized_trace_{user}_run{args.run}.csv", index=False)

print(f"Created per-user digitized traces: data/digitized_trace_*_run{args.run}.csv")
print(f"Finished!")