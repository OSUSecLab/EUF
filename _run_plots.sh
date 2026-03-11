#!/bin/bash

echo "Generating plots one by one. They will be saved in ./out/. Close each window to proceed to the next."

# Generate timeline
echo "[1] Generating timeline plot"
python3 plot_timeline.py

# Generate pairwise - 4 users
echo "[2] Generating pairwise plot with 4 users"
python3 plot_pairwise_4users.py

# Generate top/mid/btm
echo "[3] Generating top-mid-btm for 16 users"
python3 plot_pairwise_topmidbtm.py

# Generate 16 user
echo "[4] Generating 16-user pairwise"
python3 plot_pairwise_16users_oneway.py

# Generate true rank
echo "[5] Generating true rank plot"
python3 plot_truerank.py

# Run naive counting baseline
echo "[6] Running counting baseline for comparison"
python3 run_baseline_counting.py

# Generate PCA
echo "[7] Generating PCA"
python3 plot_pca.py

# Generate pool
echo "[8] Generating pool"
python3 plot_pool.py

# Done!
echo "[!] Finished! All plots are saved to ./out"
