#!/bin/bash

# Number of runs
N=10

# create folders
mkdir out
mkdir data

# Generate simulation data
for ((i=0; i<N; i++))
do
    echo "[1] Generating simulation run $i"
    python3 gen_sim_runs.py --run "$i"
done

# Calculate pairwise
for ((i=0; i<N; i++))
do
    echo "[2] Calculating pairwise data for run $i"
    python3 calc.py --mode pairwise --run "$i"
done

# Calculate pool
for ((i=0; i<N; i++))
do
    echo "[3] Calculating pool data for run $i"
    python3 calc.py --mode pool --run "$i"
done

# Calculate ranking
echo "[4] Calculating ranking"
python3 calc.py --mode loo
