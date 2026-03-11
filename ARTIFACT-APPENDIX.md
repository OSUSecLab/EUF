# Artifact Appendix

Paper title: Access Granted, Privacy Lost: Formalizing & Quantifying the Hidden Anonymity Risks of Exclusive-Use Systems

Requested Badge(s):
  - [x] **Available**
  - [x] **Functional**
  - [x] **Reproduced**


## Description

Paper title: Access Granted, Privacy Lost: Formalizing & Quantifying the Hidden Anonymity Risks of Exclusive-Use Systems  
Authors: Christopher Ellis, Zhiqiang Lin  
Publication: PETS'26  

Artifact Description: This repository contains files to simulate web chat/collaboration service interactions and quantify anonymity loss over various metrics to serve as a case study evaluation in our paper.

### Security/Privacy Issues and Ethical Concerns

There are no known security/privacy issues or ethical concerns with this code base. It is primarily a simulation and quantification of anonymity loss, not requiring an ethical review or IRB process. 

## Basic Requirements

### Hardware Requirements

No specialized hardware is required and performance/evaluation is not dependent on machine specifications. Evaluations were performed on a commodity Intel i7 laptop.

### Software Requirements

Evaluations were performed using Ubunutu 22.04.1 and Python 3.10.12. A requirements.txt file is included for ease of virtualenv setup and required Python packages. No other containerization is required.

### Estimated Time and Storage Consumption

The full pipeline can be executed in under one minute and requires trivial storage space.

## Environment

### Accessibility

Our artifact repository is publicly available at: https://github.com/OSUSecLab/EUF

View README.md for additional setup and execution steps.


### Set up the environment

One may clone the repository using the following:

```bash
git clone https://github.com/OSUSecLab/EUF
```

Then, refer to the repo's README.md for setup details. Summary statistics are shown in the CLI and plots are shown one-by-one, and saved in the "out" directory.

### Testing the Environment

The environment only requires only requires Python 3.10.x(12) and the required packages installed from the requirements.txt. The absence of runtime errors ensures the environment is running correctly. 

## Artifact Evaluation

### Main Results and Claims

Our evaluations produce numerous metrics and are discussed in the case study section of our paper. These metrics are summarized in the CLI output and were used to help interpret the plots.

==== Indistinguishability Game Summary for 4 Users ====  
A* vs B  (pair Alice_1 vs Bob_1, observed=Alice_1):  
  P(Alice_1|O): 0.8501 +- 0.0790  
  P(Bob_1|O): 0.1499 +- 0.0790  
  BV: 0.8501 +- 0.0790   correct=100%  
  dH: 0.4196 +- 0.1427

B* vs A  (pair Alice_1 vs Bob_1, observed=Bob_1):  
  P(Alice_1|O): 0.1315 +- 0.0478  
  P(Bob_1|O): 0.8685 +- 0.0478  
  BV: 0.8685 +- 0.0478   correct=100%  
  dH: 0.4531 +- 0.1309  

A* vs C  (pair Alice_1 vs Charlie_1, observed=Alice_1):  
  P(Alice_1|O): 0.7122 +- 0.0865  
  P(Charlie_1|O): 0.2878 +- 0.0865  
  BV: 0.7122 +- 0.0865   correct=100%  
  dH: 0.1596 +- 0.0931  

C* vs A  (pair Alice_1 vs Charlie_1, observed=Charlie_1):  
  P(Alice_1|O): 0.2883 +- 0.1232  
  P(Charlie_1|O): 0.7117 +- 0.1232  
  BV: 0.7211 +- 0.1053   correct=88%  
  dH: 0.1865 +- 0.1456  


==== Selected Unordered Pairs - Top/Mid/Bottom ====  
Bob_3 vs Diane_3: mean P(true|O) = 0.9977  
Bob_3 vs Diane_1: mean P(true|O) = 0.9968  
Bob_3 vs Diane_4: mean P(true|O) = 0.9966  
Bob_2 vs Charlie_1: mean P(true|O) = 0.8865   
Bob_1 vs Bob_4: mean P(true|O) = 0.8851  
Bob_2 vs Charlie_2: mean P(true|O) = 0.8832   
Charlie_2 vs Charlie_4: mean P(true|O) = 0.5015   
Charlie_1 vs Charlie_3: mean P(true|O) = 0.4946  
Charlie_1 vs Charlie_2: mean P(true|O) = 0.4893  

==== True Rank ====  
Top-1 Accuracy: 70 / 128 (54.7%)  
Top-3 Accuracy: 114 / 128 (89.1%)  
Top-5 Accuracy: 119 / 128 (93.0%)  
Mean Rank: 2.23  
Median Rank: 1.0  

vs. Counting Baseline:  
Top-1: 44.4%  
Top-3: 71.9%  
Top-5: 86.9%  
Mean rank: 2.79  
Median rank: 2.0  

==== Pool Summary ====  
Alice_1: n=8  ΔH=0.8516+-0.1081     V=0.2572+-0.0272  
Alice_2: n=8  ΔH=0.6760+-0.0662     V=0.1936+-0.0209  
Alice_3: n=8  ΔH=1.0033+-0.1430     V=0.2614+-0.0506  
Alice_4: n=8  ΔH=0.8342+-0.1169     V=0.2823+-0.0663  
Bob_1: n=8  ΔH=1.1864+-0.2146     V=0.3076+-0.0751  
Bob_2: n=8  ΔH=1.0913+-0.2295     V=0.3147+-0.0575  
Bob_3: n=8  ΔH=1.6076+-0.2269     V=0.4549+-0.0847  
Bob_4: n=8  ΔH=1.0807+-0.2916     V=0.2959+-0.0679  
Charlie_1: n=8  ΔH=0.8671+-0.1742     V=0.2278+-0.0397  
Charlie_2: n=8  ΔH=0.9249+-0.2220     V=0.2405+-0.0480  
Charlie_3: n=8  ΔH=0.7810+-0.1687     V=0.2268+-0.0469  
Charlie_4: n=8  ΔH=0.8579+-0.1797     V=0.2510+-0.0425  
Diane_1: n=8  ΔH=1.7489+-0.2747     V=0.5038+-0.1387  
Diane_2: n=8  ΔH=2.4368+-0.4877     V=0.7072+-0.1015  
Diane_3: n=8  ΔH=1.7307+-0.2729     V=0.5013+-0.0758  
Diane_4: n=8  ΔH=1.7219+-0.1959     V=0.4784+-0.1205  


### Experiments

All experiments can be ran using the provided scripts:
```
source _run_sim_calc.sh
```
will generate user data, and perform pool, pairwise, and leave-one-out calculations. Then:

```
source _run_plots.sh
```
will generate the corresponding plots using the data and output statistical summaries used in the paper.

They should all execute in roughly one minute and take up trivial storage space.


## Limitations

All experiments are reproducible.

## Notes on Reusability

This repository contains scripts that creates user profiles for 3 distinct semantic events that occur generally in web chat/collaboration programs and thus allows tunable parameters to create distinct profiles. Additional profiles can be created beyond the four provided. 
