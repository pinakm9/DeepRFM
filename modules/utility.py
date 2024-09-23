# A helper module for various sub-tasks
from time import time
import pandas as pd
import glob, os
import numpy as np

def timer(func):
	"""
	Timing wrapper for a generic function.
	Prints the time spent inside the function to the output.
	"""
	def new_func(*args, **kwargs):
		start = time()
		val = func(*args,**kwargs)
		end = time()
		print(f'Time taken by {func.__name__} is {end-start:.4f} seconds')
		return val
	return new_func



class ExperimentLogger:
    def __init__(self, folder, description):
        self.folder = folder
        self.description = description
        self.objects = []
        self.logFile = folder + '/experiment_log.txt'
        with open(self.logFile, 'w') as file:
            file.write("=======================================================================\n")
            file.write("This is a short description of this experiment\n")
            file.write("=======================================================================\n")
            descriptionWithNewLine= description.replace('. ', '.\n')
            file.write(f"{descriptionWithNewLine}\n\n\n")
        
    def addSingle(self, object):
        self.objects.append(object)
        with open(self.logFile, 'a') as file:
            file.write("=======================================================================\n")
            file.write(f"Object: {object.name}\n")
            file.write("=======================================================================\n")
            for key, value in object.__dict__.items():
                file.write(f"{key} = {value}\n")
                file.write("-----------------------------------------------------------------------\n")
            file.write("\n\n")
    
    def add(self, *objects):
        for object in objects:
            self.addSingle(object)


def collect_beta():
    for dynamical_system in ['L63', 'L96', 'KS']:
        folder = f'../data/{dynamical_system}'
        folders = glob.glob(folder + '/*/*/beta')
        for folder in folders:
            print(f"Working on {folder} ...")
            agg = []
            for file in glob.glob(folder + '/*.csv'):
                filename = os.path.basename(file)
                if filename != 'beta.csv':
                    D_r, B = filename[9:].split('_')
                    D_r, B = int(D_r), int(B.split('.')[0][2:])
                    print(file, D_r, B)
                    data = pd.read_csv(file)
                    idx = np.argmax(data['tau_f_nmse_mean'])
                    agg.append([D_r, B] + data.iloc[idx].to_list())
            pd.DataFrame(sorted(agg), columns=['D_r', 'B'] + list(data.columns))\
                        .to_csv(folder + '/beta.csv', index=False, mode='w')
            



def get_data(dynamical_system, config_id, architecture, D_r, B):
    return pd.read_csv(f'../data/{dynamical_system}/config_{config_id}/{architecture}/D_r-{D_r}_B-{B}/batch_data.csv')