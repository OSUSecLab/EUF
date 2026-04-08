# Exclusive-Use Formalization: Simulation and Analysis

This repo contains numerous scripts to generate simulated user interactions and perform quantification analysis. This approach is motivated to study numerous user behavior profiles that are empirically and functionally grounded in Microsoft Teams (or similar chat programs) interactions.

# Environment

A requirements.txt file is included for virtual environment setup of Python dependencies, using Python 3.10.12 running on Ubuntu 22.04.1. Note, all CLI commands and output assume this root as the current working directory. 

First, ensure you have the virtualenv package installed using:
```
sudo apt install python3-venv
```

Then, creating and initializing a Python virtual environment is typically:

```
virtualenv ./.venv
source ./.venv/bin/activate
pip install -r requirements.txt
```

# Execution Pipeline

There is a specific ordering of execution required. Each file has numerous options, but often with defaults, and can be explored in the source.

## Run
For ease of use, quick run scripts are provided which generate and perform analysis over numerous (10) runs. First:

```
source _run_sim_calc.sh
```

will generate user data, and perform pool, pairwise, and leave-one-out calculations. Then:

```
source _run_plots.sh
```
will generate the corresponding plots using the data.




