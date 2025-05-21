# DeepRFM

## Description
This repository contains implementations of various hit-and-run Random Feature Maps as described in https://arxiv.org/pdf/2501.06661 .


## Table of contents
- [Installation](#installation)
- [Usage](#usage)
    - [Navigation](#navigation)
    - [Modules](#modules)
        - [Classes for implementing Random Feature Maps](#classes-for-random-feature-maps)
        - [One-shot hit-and-run sampling of non-trainable parameters](#one-shot-hit-and-run-sampling-of-non-trainable-parameters)
    - [How to read the data files](#data)
- [Tutorials](#tutorials)
    - [Quick Start](#quick-start)
    - [Notebooks](#notebooks)
- [License](#license)


## Installation
The code was tested on Python 3.11.5 (macOS with torch-metal) and 3.11.12 (Colab).
```sh
# Clone the repository
git clone https://github.com/pinakm9/DeepRFM.git
cd repository

# Install dependencies
pip install -r requirements.txt 
```


## Usage
### Navigation
This repository is structured as follows:
```plaintext 
DeepRFM/
│── data/                  
│   ├── KS-200           # Data for Kuramoto-Sivashinsky (512D),
|   |   ├── config_1     # Data for Table in Appendix
|   |   ├── config_local # Data for experiments to determine the optimal localization scheme
|   |   ...
│   ├── L63              # Data for Lorenz-63 (3D) 
|   |   ├── config_0_s   # Data for dt=0.02
|   |   ├── config_1_s   # Data for dt=0.01
|   |   ├── config_3_s   # Data for dt=0.1
|   |   ...           
|   |── L96              # Data for Lorenz-96 (40D)
|   |   ├── config_1_s   # Data for Tables in Appendix
|   |   ├── config_local # Data for experiments to determine the optimal localization scheme
|   |   ...
|   ...
|── modules/
|   |──rfm.py            # Base Python classes for all variants
|   |──skipRFM.py        # Implementation of SkipRFM
|   |──deepRFM.py        # Implementation of DeepRFM
|   |──deepSkip.py       # Implementation of DeepSkip
|   |──localRFM.py       # Implementation of LocalRFM
|   |──localSkip.py      # Implementation of LocalSkip
|   |──localDeepRFM.py   # Implementation of LocalDeepRFM
|   |──localDeepSkip.py  # Implementation of LocalDeepSkip
|   |──localRFMN.py      # Implementation of LocalRFM that adds noise to the training data
|   |──localSkipN.py     # Implementation of LocalSkip that adds noise to the training data
|   |──localDeepRFMN.py  # Implementation of LocalDeepRFM that adds noise to the training data
|   |──localDeepSkipN.py # Implementation of LocalDeepSkip that adds noise to the training data
|   |──oneshot.py        # Implementation of hit-and-run sampling of the non-trainable parameters
|   |──wasserstein.py    # Implementation of Sinkhorn divergence in torch
|   |──config.py         # Configuration dicts for various experiments as a function of the dyanmical system
|   ... 
│── notebooks/           # Jupyter notebooks for testing code and generating plots
│── tutorials/
    │── colab/           # contains example Colab notebook for running experiments in bulk
    │── jupyter/         # contains example Jupyer notebook for optimizing beta
│── .gitignore           # Git ignore file
│── LICENSE              # License file
│── README.md            # Project readme file
│── requirements.txt     # Dependencies list
```
### Modules
#### Classes for implementing Random Feature Maps
Each Random Feature Map variant has an associated [RFM name].py file in the modules/ directory. This file typically contains 3 main classes named: [RFM name], DeepRF and BatchDeepRF. The first class codes the associated architecture. The second class (DeepRF) allows one to contruct a model using the said architecture and learn a dynamical system from data with the help of the "learn" function. 
``` plaintext
DeepRF(rfm.DeepRF)
    |── learn(train, seed)              # performs necessary regressions to learn a dynamical system from training data, 
    |                                   # seed controls the random weights
    |── forecast(u)                     # given u_n, returns u_(n+1) 
    |── multistep_forecast(u, n_steps)  # generates a trajectory of length n by recursively propagating u
    |── compute_tau_f                   # calculates VPT for a multistep prediction
    |── save                            # saves the entire model including all the random and trainable parameters
    |── load                            # loads a saved model
    ...
```
The third class (BatchDeepRF) allows one to execute bulk experiments using the associated RFM variant. 
``` plaintext
DeepRF(rfm.BatchDeepRF)
    |── run              # performs a batch of learning experiments where each one differs in random weights,
    |                    # training and testing data, and saves the results in a batch_data.csv file 
    |── search_beta      # looks for the optimal regularization hyperparameter with a grid search 
    |── get_data         # reads the results of a batch of experiments from batch_data.csv
    ...
```
Another important class that appears in rfm.py is BetaTester. This class allows one to optimize $\beta$ in bulk for different model and experiment configurations.
``` plaintext
BetaTester 
    |── search_beta      # looks for the optimal regularization hyperparameter with a grid search for a collection of
    |                    # model and experiment configurations
    ...                     
```
>**Note**: The functions for computing VPT takes an error_threshold argument which equals $\varepsilon^2$ for $\varepsilon$ in the paper.

#### One-shot hit-and-run sampling of non-trainable parameters
modules/oneshot.py contains the class GoodRowSampler whose "sample_vec" method is a parallelized (tensorized) version of the one-shot hit-and-run sampling algorithm in the paper. This method is called by DeepRF.learn to initialize the random non-trainable parameters.

>**Tip**: If you are unsure about what a function does, check its docstring with Python's help command.

### How to read the data files

The experimental data is mainly contained in the batch_data.csv files which have the following columns:
``` plaintext
batch_data.csv
    |── l             # experiment index
    |── model seed    # seed for generating the non-trainable parameters of the model 
    |                 # note that on two different machines the same seed may result in different parameters
    |── train_index   # index specifying training data
    |── train_index   # index specifying test data
    |── tau_f_nmse    # VPT as defined in the paper
    |── tau_f_se      # validity time as defined in https://arxiv.org/abs/2107.06658v3
    |                 # (note that before the updates in Feb 2025, this quantity was defined as in https://arxiv.org/abs/2108.03561)
    |── train_time    # time spent on training a model, includes the time spent on the random initialization which is negligible
    ...
```
Results for the grid-search for $\beta$ are contained in folders named beta which contain beta.csv and beta_s.csv files. beta_s.csv contains smoothed data from beta.csv using a moving average. beta.csv contains the following columns:
``` plaintext
beta.csv
    |── D_r             # model width
    |── B               # model depth
    |── beta            # optimal beta
    |── tau_f_nmse_mean # estimate of E[VPT] 
    |── tau_f_nmse_std  # estimate of standard deviation of VPT
    |── tau_f_se_mean   # estimate of E[validity time] defined in https://arxiv.org/abs/2107.06658v3
    |                   # (note that before the updates in Feb 2025, this quantity was defined as in https://arxiv.org/abs/2108.03561)
    |── tau_f_se_std    # estimate of standard deviation of tau_f
    ...
```
# Tutorials

## Quick Start
```sh
# import necessary modules
import os, sys
sys.path.insert(0, "path to DeepRFM/modules") #<--- add modules folder to Python's seach path
import l96
import localDeepSkip #<--- import the version of random feature model you'd like to train

# load data
train, test = l96.gen_data()
device = "cuda" if torch.cuda.is_available() else "cpu"
train, test = torch.tensor(train, device=device), torch.tensor(test, device=device)

# set up architecture and hit-and-run
G, I = 2, 2 #<--- localization scheme
model = localDeepSkip.DeepRF(D_r=2048, B=2, L0=0.4, L1=3.5, Uo=train, beta=9.64e-9,\
                            name='surrogate_model', save_folder='.', normalize=False, G=2, I=2)
# assign the non-trainable parameters and train the rest
model.learn(train[:, :int(1e5)], seed=42) 
# evaluate the model on a test trajectory
vpt = model.compute_tau_f(test[99:100], error_threshold=0.25, dt=0.01, Lyapunov_time=1/2.27)[0]
print(f"VPT on the test path = {vpt[0].item():.2f}")

```

## Notebooks
Tutorials can be found in the tutorials folder in the form of notebooks. The colab folder contains a tutorial on how to run bulk training and VPT computations on Colab. The jupyter folder contains a tutorial on how to bulk optimize $\beta$ with BetaTester for different model and experiment configurations on a local machine. The tutorials are set up in this manner to enable users to run code both on cloud and locally. Every module in this repository can be run on both.

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. See LICENSE for details.