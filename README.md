# LEAP - Atmospheric Physics using AI (ClimSim) - 7th Place Solution (Ryota Part)
![certificate](./appendix/certificate.png)
This repository contains the code for the 7th place solution (Ryota Part) in the LEAP - Atmospheric Physics using AI (ClimSim) competition hosted on Kaggle. In this competition, participants were tasked with developing machine learning models that accurately emulate subgrid-scale atmospheric physics in an operational climate model, an important step in improving climate projections and reducing uncertainty surrounding future climate trends.

## Solution Summary
In my solution, I primarily combined LSTM, Transformer, and Conv1D models for training. The model inputs included not only the original data but also data generated through feature engineering that considered domain knowledge. Additionally, to address the issue that the best optimization differs among multiple target columns, I first trained using MAE and then conducted additional training using MSE.

|  | Public LB | Private LB |
| --- | --- | --- |
| LSTM based | **0.78682** | **0.78120** |
| Transformer based | 0.78567 | 0.78058 |
| Conv1D based | 0.78301 | 0.77506 |


## Preparation
You can set up the environment and download the required data by running the following commands.
```sh
. ./bin/setup.sh
. ./bin/download.sh
```

## Reproducing the Solution

[Provide step-by-step instructions on how to run the code to reproduce the results, including any necessary configuration settings.]

## Links
- Competition website : [link](https://www.kaggle.com/c/leap-atmospheric-physics-ai-climsim)
- 7th place solution summary : [link](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/524111)
