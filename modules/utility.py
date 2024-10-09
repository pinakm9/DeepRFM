# A helper module for various sub-tasks
from time import time
import pandas as pd
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import count_params as cp

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


def collect_beta(smoothing_window=10):
    for dynamical_system in ['L63', 'L96', 'KS-12', 'KS-22', 'KS-200']:
        folder = f'../data/{dynamical_system}'
        folders = glob.glob(folder + '/*/*/beta')
        for folder in folders:
            print(f"Working on {folder} ...")
            agg = []
            agg_s = []
            for file in glob.glob(folder + '/*.csv'):
                filename = os.path.basename(file)
                if filename != 'beta.csv' and filename != 'beta_s.csv':
                    D_r, B = filename[9:].split('_')
                    D_r, B = int(D_r), int(B.split('.')[0][2:])
                    print(file, D_r, B)
                    data = pd.read_csv(file)
                    idx = np.argmax(data['tau_f_nmse_mean'])
                    agg.append([D_r, B] + data.iloc[idx].to_list())
                    pd.DataFrame(sorted(agg), columns=['D_r', 'B'] + list(data.columns))\
                        .to_csv(folder + '/beta.csv', index=False, mode='w')
                    for column in list(data.columns):
                        if column != 'beta':
                            data[column] = smooth(data[column], smoothing_window)
                    idx = np.argmax(data['tau_f_nmse_mean'])
                    agg_s.append([D_r, B] + data.iloc[idx].to_list())
                pd.DataFrame(sorted(agg_s), columns=['D_r', 'B'] + list(data.columns))\
                        .to_csv(folder + '/beta_s.csv', index=False, mode='w')
            
            

def gather_beta(dynamical_system):
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


def waterfall(data, filename=None, drf=None, cmap='magma', levels=15, width=10, **tau_f_kwargs):
    n, m = data.shape
    x, y = np.linspace(1, n, num=n), np.linspace(1, m, num=m) * tau_f_kwargs['dt'] / tau_f_kwargs['Lyapunov_time']
    if drf is None:
        fig = plt.figure(figsize=(width, 5))
        ax = fig.add_subplot(111)
        origin = 'lower'
        im = ax.contourf(x, y, data.T, levels=levels, cmap=cmap, origin=origin)
        # ax.contour(x, y, data.T, levels=levels, linewidths=0.5, colors='k')
        # ax.contour(x, y, data.T, origin=origin, linewidths=0.15)
        fig.colorbar(im, ax=[ax])
    else:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)
        im = ax.contourf(x, y, data.T, levels=levels, cmap=cmap)
        z = drf.multistep_forecast(data[:, 0], m).cpu().numpy()
        print(z.shape)
        ax1.contourf(x, y, z.T, levels=levels, cmap=cmap)
        # ax1.set_ylabel(r'$t/T^{\Lambda_1}$')
        ax1.set_title(r'Forecast')
        fig.colorbar(im, ax=[ax, ax1])
        ax1.set_yticks([])
    ax.set_ylabel(r'$t/T^{\Lambda_1}$')
    ax.set_title(r'Truth')
    if filename is not None:
        plt.savefig(f'../data/plots/{filename}.png', bbox_inches='tight', dpi=300)
    plt.show()



def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]


def smooth(y, box_pts=10):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def get_arch(folder):
    try:
        if 'L63' in folder:
            D = 3
        elif 'L96' in folder:
            D = 40
        elif 'KS-12' in folder:
            D = 48
        elif 'KS-22' in folder:
            D = 64
        elif 'KS-200' in folder:
            D = 512
        architecture, folder = folder.split('/')[-2:]
        split = folder.split('-')
        D_r = int(split[1].split('_')[0])
        B   = int(split[2])
        if 'Local' in architecture:
            G, I = architecture.split('_')[1:]
            G, I = int(G), int(I)
            args = [D, D_r, B, G, I]
        else:
            args = [D, D_r, B]
        return [architecture] + args + [getattr(cp, architecture.split('_')[0])(*args)]
    except:
        return []
    

def summary(dynamical_system, column='tau_f_nmse', verbose=False):
    root = f'../data/{dynamical_system}'
    vpts = {}
    for folder in glob.glob(f'{root}/*'):
        if 'config' in folder:
            config = folder.split('/')[-1]
            vpts[config] = {}
            for folder_arc in glob.glob(folder + '/*'):
                arch = folder_arc.split('/')[-1]
                vpts[config][arch] = {}
                for subfolder in glob.glob(f'{folder_arc}/*'):
                    if not 'beta' in subfolder:
                        args = get_arch(subfolder)
                        try:
                            if 'Local' in args[0]:
                                vpts[config][arch][tuple(args[2:6])] = pd.read_csv(f'{subfolder}/batch_data.csv')[column].mean()
                            else:
                                vpts[config][arch][tuple(args[2:4])] = pd.read_csv(f'{subfolder}/batch_data.csv')[column].mean()
                        except:
                            pass
                vpts[config][arch] = {k: v for k, v in sorted(vpts[config][arch].items(), key=lambda item: item[1])}
    if verbose:
        for config in vpts:
            for arch in vpts[config]:
                print(f"Looking at data for {dynamical_system}-{config}-{arch}:")
                for k, v in vpts[config][arch].items():
                    print(f"{k}: {v}")
    return vpts