import utility as ut 
import l96
import pandas as pd
import numpy as np
import os
import json
import wasserstein
import torch
import time
import localDeepSkip
import count_params as cp



@ut.timer
def run_single(drf_kwargs, data_gen_kwargs, train_kwargs, eval_kwargs, device):
    if not os.path.exists(train_kwargs["save_folder"]):
        os.makedirs(train_kwargs["save_folder"])
    results = {}

    # prepare data
    data, _ = l96.gen_data(**data_gen_kwargs)
    N = int(data.shape[1]/2)
    np.save("{}/data.npy".format(train_kwargs["save_folder"]), data) 
    std = data.std(axis=1)

    # train model
    start = time.time()
    model = localDeepSkip.DeepRF(drf_kwargs["D_r"], drf_kwargs["B"], drf_kwargs["L0"], drf_kwargs["L1"], torch.tensor(data[:, :N], device=device), drf_kwargs["beta"],\
                           'drf_model', train_kwargs["save_folder"], False, drf_kwargs["G"], drf_kwargs["I"])
    model.learn(torch.tensor(data[:, :N], device=device), seed=train_kwargs["model_seed"])
    train_time = time.time() - start
    print(f"Model trained for {train_time:.2f} seconds")

    # save model
    # model.save(None)

    # generate data for VPT
    x = data[:, N]
    Y = model.multistep_forecast(torch.tensor(x, device=device), eval_kwargs["vpt_steps"]).detach().cpu().numpy().T
    np.save("{}/vpt_trajectory.npy".format(train_kwargs["save_folder"]), Y)

    # calculate VPT
    Y0 = data[:, N:N+eval_kwargs["vpt_steps"]]
    l = np.argmax((((Y - Y0) / std[:, None])**2).sum(axis=0) > eval_kwargs["vpt_epsilon"]**2)
    results["VPT"] = float(l * data_gen_kwargs["dt"] / eval_kwargs["Lyapunov_time"])

    # generate data for RMSE
    x = data[:, N:N+eval_kwargs["n_RMSE"]]
    t = np.arange(N, N+eval_kwargs["n_RMSE"]) * data_gen_kwargs["dt"]
    Y = model.forecast(torch.tensor(x, device=device)).detach().cpu().numpy().T
    np.save("{}/rmse_trajectory.npy".format(train_kwargs["save_folder"]), Y)

    # calculate RMSE and MAE
    Y0 = data[:, N:N+eval_kwargs["n_RMSE"]]
    results["RMSE"] = float(np.sqrt(np.mean(((Y - Y0)**2).sum(axis=0))))
    results["MAE"] = float(np.mean(np.abs(Y - Y0).sum(axis=0)))
    np.save("{}/rmse_true_trajectory.npy".format(train_kwargs["save_folder"]), Y0)

    # generate data for Wasserstein
    x = data[:, N]
    t = N * data_gen_kwargs["dt"]
    Y = model.multistep_forecast(torch.tensor(x, device=device), eval_kwargs["n_sample_w2"]).detach().cpu().numpy().T
    # Y = np.squeeze(Y, axis=-1).T
    np.save("{}/w2_trajectory.npy".format(train_kwargs["save_folder"]), Y)

    # calculate Wasserstein distance
    Y0 = data[:, N:]
    A = torch.tensor(Y.T[:eval_kwargs["n_sample_w2"]], dtype=torch.float32, device=device)
    B = torch.tensor(Y0.T[:eval_kwargs["n_sample_w2"]], dtype=torch.float32, device=device)
    results["W2"] = float(wasserstein.sinkhorn_div(A, B).item())

    # add model size and training time
    results["training_time"] = float(train_time)
    results["model_size"] = int(cp.LocalDeepSkip(40, drf_kwargs["D_r"], drf_kwargs["B"], drf_kwargs["G"], drf_kwargs["I"]))
    results["experiment_seed"] = int(data_gen_kwargs["train_seed"])
    results["model_seed"] = int(train_kwargs["model_seed"])

    # save results
    # print(results)
    with open(f"{train_kwargs['save_folder']}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    config = {}
    config.update(drf_kwargs)
    config.update(data_gen_kwargs)
    config.update(train_kwargs)
    config.update(eval_kwargs)  

    with open(f"{train_kwargs['save_folder']}/config.json", "w") as f:  
        json.dump(config, f, indent=2)
