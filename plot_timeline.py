import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def subscript_label(name):
    base, idx = name.split("_")
    return f"${base}_{{{idx}}}$"

# Load digitized traces
digitized_files = sorted(glob.glob("data/digitized_trace_*_run0.csv"))
users = [f.split("_run")[0].replace("data/digitized_trace_", "") for f in digitized_files]

profile_order = ["Alice", "Bob", "Charlie", "Diane"]
users_sorted = sorted(users, key=lambda u: (profile_order.index(u.split("_")[0]), int(u.split("_")[1])))
users_sorted = users_sorted[::-1]


# Sort digitized_files to match users_sorted
user_to_file = {u: f for u, f in zip(users, digitized_files)}
digitized_files_sorted = [user_to_file[u] for u in users_sorted]

# Set baseline start
trace_start = datetime.strptime("09:00:00", "%H:%M:%S")
trace_seconds = []

# Get max trace length
for f in digitized_files_sorted:
    df = pd.read_csv(f)
    trace_seconds.append(len(df))

max_seconds = max(trace_seconds)
trace_end = trace_start + timedelta(seconds=max_seconds)

# Add padding and round
start_display = trace_start - timedelta(hours=1)
end_display = trace_end + timedelta(hours=1)
start_display = start_display.replace(minute=0, second=0)
end_display = (end_display + timedelta(minutes=59)).replace(minute=0, second=0)

# Event config
event_offsets = {
    "presence": -0.2,
    "typing": 0.0,
    "message_sent": 0.2
}
event_styles = {
    "presence": {"color": "#1f77b4", "marker": "o", "label": "Presence"},
    "typing": {"color": "#d62728", "marker": "^", "label": "Typing"},
    "message_sent": {"color": "#2ca02c", "marker": "s", "label": "Message"}
}

# Plot setup
fig, ax = plt.subplots(figsize=(12, 0.6 * len(users_sorted)))
spacing = 1.0

for i, (file, user) in enumerate(zip(digitized_files_sorted, users_sorted)):
    df = pd.read_csv(file)[["presence", "typing", "message_sent"]]
    y_base = i * spacing

    ax.annotate("",
                xy=(mdates.date2num(end_display), y_base),
                xytext=(mdates.date2num(start_display), y_base),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    for event, offset in event_offsets.items():
        t_indices = np.where(df[event] == 1)[0]
        if len(t_indices) > 0:
            times = [trace_start + timedelta(seconds=int(t)) for t in t_indices]
            ax.scatter(mdates.date2num(times),
                       [y_base + offset] * len(times),
                       color=event_styles[event]["color"],
                       marker=event_styles[event]["marker"],
                       s=20,
                       label=event_styles[event]["label"] if i == 0 else None)

# Y-axis
ax.set_yticks([i * spacing for i in range(len(users_sorted))])
ax.set_yticklabels([subscript_label(u) for u in users_sorted], fontsize=8)


# X-axis formatting
ax.set_xlim(mdates.date2num(start_display), mdates.date2num(end_display))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45)
ax.set_xlabel("Time of Day")

ax.set_title("User Activity Timeline (Presence, Typing, Message Sent)", fontsize=12)
ax.grid(True, axis='x', linestyle='--', alpha=0.3)
ax.tick_params(axis='y', length=0)
for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_visible(False)

# Legend
handles = [
    plt.Line2D([], [], color=event_styles[event]["color"],
        marker=event_styles[event]["marker"],
        linestyle='None',
        label=event_styles[event]["label"])
    for event in event_styles
]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=8, columnspacing=1.5)

plt.tight_layout()
plt.savefig("out/simulated_timelines16.pdf")
plt.show()


