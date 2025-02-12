# Personalized Federated Learning of Probabilistic Models: A PAC-Bayesian Approach
This repository provides source code corresponding the paper Personalized Federated Learning of Probabilistic Models: A PAC-Bayesian Approach.

A demo using the synthetic toy dataset with our method, PAC-PFL, and all baseline methods is provided in **experiments/Toy_dataset**.

The **server** package contains the code ran by the server for different methods.

The **baselines** directory contains the implementations of MAML, MTL, and pooled baselines for all versions of the PV dataset.
PAC-PFL, Vanilla, and pFedGP have a similar implementation, and are evaluated in **experiments/PV/train_full_gp**.
It is possible to first train the prior mean, using **experiments/PV/train_full_gp**, and then train a zero-mean GP on the residuals, by running **experiments/PV/train_only_kernel**.

New clients are simulated and evaluated by running **experiments/PV/new_clients**.

Structure search for all NNs and hyper-parameter tuning is performed using the functions provided in the **utils** folder.

## Installation
To install full dependencies, please run in the main directory of this repository
```bash
pip install -r requirements.txt
```

## Usage
The GPU is disabled by default. To enable the GPU, set disable_gpu = False in **config.py**.

The **demo.ipynb** notebook can be ran from **experiments/Toy_dataset** without downloading further files.


To conduct the PV experiments, please follow these steps:

* 1. Download the dataset files from the **PV_dataset** folder in the provided [*Google Drive link*](https://drive.google.com/drive/u/3/my-drive).
* 2. Unzip the downloaded files and place them in the **third_party/Synthetic_PV_Profiles/saved_results** folder. Ensure that there are four files in the folder, each ending with _env.
* 3. Download the pre-trained models for the PV-EW (150) and PV-EW (610) datasets from the **pre_trained** folder in the provided Google Drive link.
* 4. Unzip the downloaded models and place them in the **experiments/PV/saved_results** folder. The folders **PV_BiModal** and **PV_BiModal_NewClients** should appear directly in the **experiments/PV/saved_results** folder.
* 5. Run the **experiments/PV/paper_figs.ipynb** file. Please note that executing this file may take some time.
By following these instructions, you should be able to reproduce the boxplots shown in Figure 2 of the paper.

