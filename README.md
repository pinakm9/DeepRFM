# DeepRFM

## Description
This repository contains implementations of various hit-and-run Random Feature Maps as described in https://arxiv.org/pdf/2501.06661 .


## Table of contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)


## Installation

```sh
# Clone the repository
git clone https://github.com/pinakm9/DeepRFM.git
cd repository

# Install dependencies
pip install -r requirements.txt 
```


## Usage
### Directory strcuture
This repository is structured as follows:
```plaintext 
DeepRFM/
│── data/                  
│   ├── KS-200           # Data for Kuramoto-Sivashinsky (512D),
|   |   ├── config_1     # Data for Table in Appendix
|   |   ├── config_local # Data for experiments to determine the optimal localization scheme
│   ├── L63              # Data for Lorenz-63 (3D) 
|   |   ├── config_0_s   # Data for dt=0.02
|   |   ├── config_1_s   # Data for dt=0.01
|   |   ├── config_3_s   # Data for dt=0.1           
|   |── L96              # Data for Lorenz-96 (40D)
|   |   ├── config_1_s   # Data for Tables in Appendix
|   |   ├── config_local # Data for experiments to determine the optimal localization scheme
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
│── notebooks/           # Jupyter notebooks for testing code and generating plots
│── .gitignore           # Git ignore file
│── LICENSE              # License file
│── README.md            # Project readme file
│── requirements.txt     # Dependencies list
```
### Modules
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
### Data

The experimental data is mainly contained in the batch_data.csv files which have the following columns:
``` plaintext
batch_data.csv
    |──
```

1. Column "tau_f_nmse" contains the VPT data in batch_data.csv files. 
8. error_threshold = epsilon^2
9. config.py contains different configurations used for running batches of experiments on Colab.

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. See LICENSE for details.