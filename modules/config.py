def config_1(dynamical_system):
    if dynamical_system == 'L63':
        prediction_time_config = {"error_threshold": 0.09, "dt": 0.01, "Lyapunov_time": 1/0.91}
        train_test_config = {"training_points": int(5e4), "n_repeats": 500}
        data_gen_config = {"dt": prediction_time_config["dt"], "train_seed": 22, "train_size": 2*train_test_config["training_points"],\
                           "test_seed": 43, "test_num": train_test_config["n_repeats"], "test_size": 2500, "save_folder": None}
        beta_config = {"negative_log10_range": [6, 11], "resolution":25, "n_repeats": 5,\
                       "training_points": train_test_config["training_points"]}
        beta_arch_config = {"RFM": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                            "SkipRFM": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                            "DeepSkip":[[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]]}
        arch_configs = [{"RFM": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                         "SkipRFM": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                         "DeepSkip":[[1024, 1], [1024, 2], [1024, 4], [1024, 8], [1024, 16]]}, 
                         {"DeepSkip":[[4096, 1], [4096, 2], [4096, 4], [4096, 8]]}] 
        return {"prediction_time": prediction_time_config, "train_test": train_test_config, "data_gen": data_gen_config,\
                 "beta": beta_config, "beta_arch": beta_arch_config, "arch": arch_configs}
    
    elif dynamical_system == 'L96':
        prediction_time_config = {"error_threshold": 0.25, "dt": 0.01, "Lyapunov_time": 1/2.27}
        train_test_config = {"training_points": int(1e5), "n_repeats": 500}
        data_gen_config = {"dt": prediction_time_config["dt"], "train_seed": 22, "train_size": 2*train_test_config["training_points"],\
                           "test_seed": 43, "test_num": train_test_config["n_repeats"], "test_size": 1000, "save_folder": None}
        beta_config = {"negative_log10_range": [6, 11], "resolution":25, "n_repeats": 5,\
                       "training_points": train_test_config["training_points"]}
        beta_arch_config = {"SkipRFM": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1]],\
                            "DeepSkip":[[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1]],\
                            "LocalSkip_2_2": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                            "LocalDeepSkip_2_2": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]]}
        arch_configs = [{"SkipRFM": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1]],\
                         "DeepSkip":[[1024, 1], [1024, 2], [1024, 4], [1024, 8], [1024, 16]],\
                         "LocalSkip_2_2": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                         "LocalDeepSkip_2_2":[[1024, 1], [1024, 2], [1024, 4], [1024, 8], [1024, 16], [4096, 2], [4096, 4], [4096, 8], [8192, 2], [8192, 4], [16384, 2]]}]
        return {"prediction_time": prediction_time_config, "train_test": train_test_config, "data_gen": data_gen_config,\
                 "beta": beta_config, "beta_arch": beta_arch_config, "arch": arch_configs}
    
    
    elif dynamical_system == 'KS':
        prediction_time_config = {"error_threshold": 0.25, "dt": 0.25, "Lyapunov_time": 1/0.094}
        train_test_config = {"training_points": int(1e5), "n_repeats": 500}
        data_gen_config = {"dt": prediction_time_config["dt"], "train_seed": 22, "train_size": 2*train_test_config["training_points"],\
                           "test_seed": 43, "test_num": train_test_config["n_repeats"], "test_size": 1000, "save_folder": None}
        beta_config = {"negative_log10_range": [6, 11], "resolution":25, "n_repeats": 5,\
                       "training_points": train_test_config["training_points"]}
        beta_arch_config = {"LocalSkip_8_1": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                            "LocalDeepSkip_8_1": [[512, 1], [1024, 1], [2048, 1], [4096, 1],[8192, 1], [16384, 1]]}
        arch_configs = [{"LocalSkip_8_1": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                         "LocalDeepSkip_8_1":[[1024, 1], [1024, 2], [1024, 4], [1024, 8], [1024, 16]]}]
        return {"prediction_time": prediction_time_config, "train_test": train_test_config, "data_gen": data_gen_config,\
                 "beta": beta_config, "beta_arch": beta_arch_config, "arch": arch_configs}
    







def config_0(dynamical_system):
    if dynamical_system == 'L63':
        prediction_time_config = {"error_threshold": 0.05, "dt": 0.02, "Lyapunov_time": 1/0.91}
        train_test_config = {"training_points": int(2e4), "n_repeats": 500}
        data_gen_config = {"dt": prediction_time_config["dt"], "train_seed": 22, "train_size": 2*train_test_config["training_points"],\
                           "test_seed": 43, "test_num": train_test_config["n_repeats"], "test_size": 1000, "save_folder": None}
        beta_config = {"negative_log10_range": [6, 11], "resolution":50, "n_repeats": 25,\
                       "training_points": train_test_config["training_points"]}
        beta_arch_config = {"RFM": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                            "SkipRFM": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                            "DeepSkip":[[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]]}
        arch_configs = [{"RFM": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                       "SkipRFM": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                       "DeepSkip":[[1024, 1], [1024, 2], [1024, 4], [1024, 8], [1024, 16]]}]
        return {"prediction_time": prediction_time_config, "train_test": train_test_config, "data_gen": data_gen_config,\
                 "beta": beta_config, "beta_arch": beta_arch_config, "arch": arch_configs}
    
    elif dynamical_system == 'L96':
        prediction_time_config = {"error_threshold": 0.25, "dt": 0.01, "Lyapunov_time": 1/2.27}
        train_test_config = {"training_points": int(1e5), "n_repeats": 500}
        data_gen_config = {"dt": prediction_time_config["dt"], "train_seed": 22, "train_size": 2*train_test_config["training_points"],\
                           "test_seed": 43, "test_num": train_test_config["n_repeats"], "test_size": 1000, "save_folder": None}
        beta_config = {"negative_log10_range": [6, 11], "resolution":50, "n_repeats": 25,\
                       "training_points": train_test_config["training_points"]}
        beta_arch_config = {"SkipRFM": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1]],\
                            "DeepSkip":[[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1]],\
                            "LocalSkip_2_2": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                            "LocalDeepSkip_2_2": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]]}
        arch_configs = [{"SkipRFM": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1]],\
                         "DeepSkip":[[1024, 1], [1024, 2], [1024, 4], [1024, 8], [1024, 16]],\
                         "LocalSkip_2_2": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                         "LocalDeepSkip_2_2":[[1024, 1], [1024, 2], [1024, 4], [1024, 8], [1024, 16]]}]
        return {"prediction_time": prediction_time_config, "train_test": train_test_config, "data_gen": data_gen_config,\
                 "beta": beta_config, "beta_arch": beta_arch_config, "arch": arch_configs}
    
    

    







def config_test(dynamical_system):
    if dynamical_system == 'L63':
        prediction_time_config = {"error_threshold": 0.09, "dt": 0.01, "Lyapunov_time": 1/0.91}
        train_test_config = {"training_points": int(5e4), "n_repeats": 500}
        data_gen_config = {"dt": prediction_time_config["dt"], "train_seed": 22, "train_size": 2*train_test_config["training_points"],\
                           "test_seed": 43, "test_num": train_test_config["n_repeats"], "test_size": 2500, "save_folder": None}
        beta_config = {"negative_log10_range": [6, 11], "resolution":25, "n_repeats": 5,\
                       "training_points": train_test_config["training_points"]}
        beta_arch_config = {"RFM": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                            "SkipRFM": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                            "DeepSkip":[[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]]}
        arch_configs = [{"RFM": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                         "SkipRFM": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                         "DeepSkip":[[1024, 1], [1024, 2], [1024, 4], [1024, 8], [1024, 16]]}, 
                         {"DeepSkip":[[4096, 1], [4096, 2], [4096, 4], [4096, 8]]}] 
        return {"prediction_time": prediction_time_config, "train_test": train_test_config, "data_gen": data_gen_config,\
                 "beta": beta_config, "beta_arch": beta_arch_config, "arch": arch_configs}
    
    elif dynamical_system == 'L96':
        prediction_time_config = {"error_threshold": 0.25, "dt": 0.01, "Lyapunov_time": 1/2.27}
        train_test_config = {"training_points": int(1e5), "n_repeats": 500}
        data_gen_config = {"dt": prediction_time_config["dt"], "train_seed": 22, "train_size": 2*train_test_config["training_points"],\
                           "test_seed": 43, "test_num": train_test_config["n_repeats"], "test_size": 1000, "save_folder": None}
        beta_config = {"negative_log10_range": [6, 11], "resolution":25, "n_repeats": 5,\
                       "training_points": train_test_config["training_points"]}
        beta_arch_config = {"SkipRFM": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1]],\
                            "DeepSkip":[[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1]],\
                            "LocalSkip_2_2": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                            "LocalDeepSkip_2_2": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]]}
        arch_configs = [{"SkipRFM": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1]],\
                         "DeepSkip":[[1024, 1], [1024, 2], [1024, 4], [1024, 8], [1024, 16]],\
                         "LocalSkip_2_2": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                         "LocalDeepSkip_2_2":[[1024, 1], [1024, 2], [1024, 4], [1024, 8], [1024, 16], [4096, 2], [4096, 4], [4096, 8], [8192, 2], [8192, 4], [16384, 2]]}]
        return {"prediction_time": prediction_time_config, "train_test": train_test_config, "data_gen": data_gen_config,\
                 "beta": beta_config, "beta_arch": beta_arch_config, "arch": arch_configs}
    
    
    elif dynamical_system == 'KS':
        prediction_time_config = {"error_threshold": 0.25, "dt": 0.25, "Lyapunov_time": 1/0.094}
        train_test_config = {"training_points": int(1e5), "n_repeats": 100}
        data_gen_config = {"dt": prediction_time_config["dt"], "train_seed": 22, "train_size": 2*train_test_config["training_points"],\
                           "test_seed": 43, "test_num": train_test_config["n_repeats"], "test_size": 1000, "save_folder": None}
        beta_config = {"negative_log10_range": [-1, 8], "resolution":25, "n_repeats": 2,\
                       "training_points": train_test_config["training_points"]}
        beta_arch_config = {"LocalSkip_8_1": [[2048, 1]]}
        arch_configs = [{"LocalSkip_8_1": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                         "LocalDeepSkip_8_1":[[1024, 1], [1024, 2], [1024, 4], [1024, 8], [1024, 16]]}]
        return {"prediction_time": prediction_time_config, "train_test": train_test_config, "data_gen": data_gen_config,\
                 "beta": beta_config, "beta_arch": beta_arch_config, "arch": arch_configs}
    


def config_test_12(dynamical_system):
    if dynamical_system == 'KS':
        prediction_time_config = {"error_threshold": 0.25, "dt": 0.25, "Lyapunov_time": 1/0.003}
        train_test_config = {"training_points": int(1e5), "n_repeats": 100}
        data_gen_config = {"dt": prediction_time_config["dt"], "train_seed": 22, "train_size": train_test_config["training_points"],\
                           "test_seed": 43, "test_num": 1, "test_size": int(1e5), "save_folder": None}
        beta_config = {"negative_log10_range": [2, 8], "resolution":5, "n_repeats": 2,\
                       "training_points": train_test_config["training_points"]}
        beta_arch_config = {"SkipRFM": [[512, 1]],\
                            "LocalSkip_12_1": [[512, 1]]}
        arch_configs = [{"LocalSkip_12_1": [[512, 1], [1024, 1], [2048, 1], [4096, 1]],\
                         "LocalDeepSkip_12_1":[[1024, 1], [1024, 2], [1024, 4], [1024, 8], [1024, 16]]}]
        return {"prediction_time": prediction_time_config, "train_test": train_test_config, "data_gen": data_gen_config,\
                 "beta": beta_config, "beta_arch": beta_arch_config, "arch": arch_configs}
    

def config_test_22(dynamical_system):
    if dynamical_system == 'KS':
        prediction_time_config = {"error_threshold": 0.25, "dt": 0.25, "Lyapunov_time": 1/0.043}
        train_test_config = {"training_points": int(1e5), "n_repeats": 100}
        data_gen_config = {"dt": prediction_time_config["dt"], "train_seed": 22, "train_size": train_test_config["training_points"],\
                           "test_seed": 43, "test_num": train_test_config["n_repeats"], "test_size": 1000, "save_folder": None}
        beta_config = {"negative_log10_range": [-1, 8], "resolution":50, "n_repeats": 2,\
                       "training_points": train_test_config["training_points"]}
        beta_arch_config = {"LocalSkip_16_1": [[1024, 8]]}
        arch_configs = [{"LocalSkip_8_1": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                         "LocalDeepSkip_8_1":[[1024, 1], [1024, 2], [1024, 4], [1024, 8], [1024, 16]]}]
        return {"prediction_time": prediction_time_config, "train_test": train_test_config, "data_gen": data_gen_config,\
                 "beta": beta_config, "beta_arch": beta_arch_config, "arch": arch_configs}
    
    
def config_test_200(dynamical_system):
    if dynamical_system == 'KS':
        prediction_time_config = {"error_threshold": 0.25, "dt": 0.25, "Lyapunov_time": 1/0.094}
        train_test_config = {"training_points": int(1e5), "n_repeats": 100}
        data_gen_config = {"dt": prediction_time_config["dt"], "train_seed": 22, "train_size": train_test_config["training_points"],\
                           "test_seed": 43, "test_num": train_test_config["n_repeats"], "test_size": 1000, "save_folder": None}
        beta_config = {"negative_log10_range": [-1, 8], "resolution":25, "n_repeats": 4,\
                       "training_points": train_test_config["training_points"]}
        beta_arch_config = {"LocalSkip_8_1": [[2048, 1]]}
        arch_configs = [{"LocalSkip_8_1": [[512, 1], [1024, 1], [2048, 1], [4096, 1], [8192, 1], [16384, 1]],\
                         "LocalDeepSkip_8_1":[[1024, 1], [1024, 2], [1024, 4], [1024, 8], [1024, 16]]}]
        return {"prediction_time": prediction_time_config, "train_test": train_test_config, "data_gen": data_gen_config,\
                 "beta": beta_config, "beta_arch": beta_arch_config, "arch": arch_configs}