import json
import os
import pandas as pd

class Logger:
    def __init__(self, save_folder) -> None:
        self.folder = save_folder
        self.config = {}
        self.train_log = {}
        self.config_file = '{}/config.json'.format(save_folder)
        self.train_file = '{}/train_log.csv'.format(save_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
    def update(self, start, kwargs):
        self.config.update(kwargs)
        if start:
            with open(self.config_file, 'w') as file:
                json.dump(kwargs, file, indent=2)
        else:
            with open(self.config_file, 'a') as file:
                json.dump(kwargs, file, indent=2)


    def log(self, start=False, **kwargs):
        df = pd.DataFrame.from_dict(kwargs)
        if start:
            df.to_csv(self.train_file, index=False)
        else:
            df.to_csv(self.train_file, mode='a', index=False, header=False)

    def print(self, **kwargs):
        line = ''
        for key, value in kwargs.items():
            line += '{}={:.3f}\t'.format(key, value[0])
        print(line)

    def get_config(self):
        return json.loads(self.save_folder + '/config.json')

