# A helper module for various sub-tasks
from time import time
import pandas as pd
import glob, os, json
import numpy as np
import matplotlib.pyplot as plt
import count_params as cp
from scipy.stats import gaussian_kde

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
        """
        Constructor for ExperimentLogger.
        
        Parameters
        ----------
        folder : str
            The folder where log files are saved.
        description : str
            A description of the experiment.
        
        Attributes
        ----------
        folder : str
            The folder where log files are saved.
        description : str
            A description of the experiment.
        objects : list
            A list of objects which are to be logged.
        logFile : str
            The path to the log file.
        """
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
        """
        Logs a single object's attributes to the log file.
        
        Parameters
        ----------
        object : object
            The object to be logged.
        """
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
        """
        Logs multiple objects' attributes to the log file.
        
        Parameters
        ----------
        objects : object
            The objects to be logged.
        """
        for object in objects:
            self.addSingle(object)


def collect_beta(smoothing_window=10, data_folder="../data"):
    """
    Collects and processes beta values for various dynamical systems.

    This function iterates over a predefined list of dynamical systems,
    collects beta values from CSV files in specific directories, processes
    them, and saves the aggregated results. 

    Parameters
    ----------
    smoothing_window : int, optional
        The window size used for smoothing data columns, by default 10.
    """

    for dynamical_system in ['L63', 'L96', 'KS-12', 'KS-22', 'KS-200']:
        folder = f'{data_folder}/{dynamical_system}'
        folders = glob.glob(folder + '/*/*/beta')
        for folder in folders:
            print(f"Working on {folder} ...")
            agg = []
            agg_s = []
            for file in glob.glob(folder + '/*.csv'):
                filename = os.path.basename(file)
                if filename in ['beta.csv', 'beta_s.csv']:
                    continue    
                D_r, B = filename[9:].split('_')
                D_r, B = int(D_r), int(B.split('.')[0][2:])
                print(file, D_r, B)
                data = pd.read_csv(file)
                # print(data)
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
    
    for file in glob.glob('../data/*/*/*/*/config.json'):
        with open(file, 'r') as json_file:
            text = json_file.read().split('}')[0]+'}'
        with open(file, 'w') as json_file:
            json_file.write(text)
            
            

def gather_beta(dynamical_system):
    """
    Gathers and processes beta values for a given dynamical system.

    This function iterates over the subdirectories of the given dynamical
    system, collects beta values from CSV files, processes them, and saves
    the aggregated results.

    Parameters
    ----------
    dynamical_system : str
        The name of the dynamical system.
    """
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
    """
    Retrieves data from a CSV file located in a specific directory.

    Parameters
    ----------
    dynamical_system : str
        The name of the dynamical system.
    config_id : str
        The configuration ID.
    architecture : str
        The architecture name.
    D_r : int
        The number of reservoir nodes.
    B : int
        The batch size.

    Returns
    -------
    DataFrame
        A pandas DataFrame containing the data.
    """
    return pd.read_csv(f'../data/{dynamical_system}/config_{config_id}/{architecture}/D_r-{D_r}_B-{B}/batch_data.csv')


def waterfall(data, filename=None, drf=None, cmap='magma', levels=15, width=10, **tau_f_kwargs):
    """
    Creates a waterfall plot for the given data, which is expected to be a 2D numpy array.

    Parameters
    ----------
    data : array_like
        The data to plot.
    filename : str, optional
        If provided, the plot is saved to a file with the given name.
    drf : DeepRF, optional
        If provided, the forecast is computed using the given DeepRF object and the result is plotted alongside the truth.
    cmap : str, optional
        The colormap to use, defaults to 'magma'.
    levels : int, optional
        The number of contour levels to use, defaults to 15.
    width : int, optional
        The width of the plot in inches, defaults to 10.
    **tau_f_kwargs : dict
        Additional keyword arguments for tau_f computation, such as error_threshold, dt, and Lyapunov_time.

    Returns
    -------
    None
    """
    n, m = data.shape
    x, y = np.linspace(1, n, num=n), np.linspace(1, m, num=m) * tau_f_kwargs['dt'] / tau_f_kwargs['Lyapunov_time']
    if drf is None:
        fig = plt.figure(figsize=(width, 5))
        ax = fig.add_subplot(111)
        origin = 'lower'
        im = ax.contourf(x, y, data.T, levels=levels, cmap=cmap, origin=origin)
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
    ax.set_ylabel(r'$t/T_{\Lambda}$')
    ax.set_title(r'Truth')
    if filename is not None:
        plt.savefig(f'../data/plots/{filename}.png', bbox_inches='tight', dpi=300)
    plt.show()



def autocorr(x, lags):
    """
    Computes the autocorrelation of a given time series for specified lags.

    Parameters
    ----------
    x : array_like
        The input time series data.
    lags : array_like
        An array of lag values at which to compute the autocorrelation.

    Returns
    -------
    np.ndarray
        An array containing the autocorrelation values for each specified lag.
    """

    corr = [1. if l==0 else np.corrcoef(x[l:], x[:-l])[0][1] for l in lags]
    return np.array(corr)


def smooth(y, box_pts=10):
    """
    Smooth a time series using a boxcar average.

    Parameters
    ----------
    y : array_like
        The time series to be smoothed.
    box_pts : int, optional
        The window size for the boxcar average. Defaults to 10.

    Returns
    -------
    y_smooth : array_like
        The smoothed time series.
    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def get_arch(folder):
    """
    Extracts the architecture parameters from a given folder name.

    Parameters
    ----------
    folder : str
        The name of the folder to extract the parameters from.

    Returns
    -------
    list
        A list containing the architecture name and its parameters.
    """
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
    

def summary(dynamical_system, column='tau_f_nmse', verbose=False, root='../data'):
    """
    Summarizes statistical information from batch data files for a given dynamical system.

    This function processes directories and subdirectories related to the specified 
    dynamical system, extracts statistical values from CSV files, associates them with 
    the corresponding architecture and configuration, and returns a nested dictionary 
    structure containing this information.

    Parameters
    ----------
    dynamical_system : str
        The name of the dynamical system to analyze.
    column : str, optional
        The column name in the CSV files to use for statistical calculations, by default 'tau_f_nmse'.
    verbose : bool, optional
        If True, prints detailed information about the processed data, by default False.
    root : str, optional
        The root directory containing data, by default '../data'.

    Returns
    -------
    dict
        A nested dictionary where keys are configuration names and values are dictionaries of 
        architectures, each containing tuples of architecture-specific parameters mapped to 
        statistical summaries.
    """

    vpts = {}
    for folder in glob.glob(f'{root}/{dynamical_system}/*'):
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
                            data = pd.read_csv(f'{subfolder}/batch_data.csv')
                            v = data[column]
                            with open(f'{subfolder}/config.json', 'r') as file:
                                beta = json.load(file)['beta']
                            stat = [v.mean(), v.std(), v.median(), v.min(), v.max(), beta, data['train_time'].mean(), args[-1]]
                            if 'Local' in args[0]:
                                vpts[config][arch][tuple(args[2:6])] = stat
                            else:
                                vpts[config][arch][tuple(args[2:4])] = stat
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


def latexify(vpts):
    """
    Generate LaTeX code for a table summarizing results.

    Args:
    vpts: A nested dictionary where keys are configuration names and values are dictionaries of 
          architectures, each containing tuples of architecture-specific parameters mapped to 
          statistical summaries.

    Returns:
    A string containing LaTeX code for a table summarizing results.
    """
    arch_list = ["RFM", "SkipRFM", "DeepRFM", "DeepSkip", "LocalRFM", "LocalSkip", "LocalDeepRFM", "LocalDeepSkip",\
                  "LocalRFMN",  "LocalSkipN", "LocalDeepRFMN", "LocalDeepSkipN"]
    exist_arch_list = []
    exist_arch_labels = []

    for architecture in arch_list:
        for x in list(vpts.keys()):
            if x.split('_')[0] == architecture:
                if x.startswith('Local'):
                    y = f"{x[:-4]}$_{{{x[-3:].replace('_', ',')}}}$"
                else:
                    y = x + ''
                exist_arch_list.append(x)
                exist_arch_labels.append(y)
                
    print(exist_arch_list)
    table = "\n\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|} \\hline\n"
    table += "\\multicolumn{4}{|c|}{Model} &\\multicolumn{5}{c|}{VPT} & \multicolumn{2}{c|}{}" + "\\\\ \\hline\n"
    table += "architecture & $D_r$ & $B$ & model size & mean & std & median & min & max &" + r"$\beta$ & $\mathbb{E}[t_{\rm train}]$(s)" + "\\\\ \\hline\\hline\n"
    
    for i, architecture in enumerate(exist_arch_list):
        rows, sizes, means = [], [], []
        entry = f"\\multirow{{{len(vpts[architecture])}}}{{*}}{{{exist_arch_labels[i]}}}"
        try:
            for j, (k, v) in enumerate(vpts[architecture].items()):
                    try:
                        beta = f"{v[-3]:.2e}"#.replace('e-0', 'e-')
                        rows.append(f" & {k[0]} & {k[1]} & {v[-1]} & {v[0]:.1f} & {v[1]:.1f} & {v[2]:.1f} & {v[3]:.1f} & {v[4]:.1f} & {beta} & {v[-2]:.1e}\\\\ \\cline{{2-11}}\n")
                        sizes.append((v[-1], k[0]))
                        means.append(v[0])
                    except:
                        pass
            rows = [r for _, r in sorted(zip(sizes, rows))]
            means = [m for _, m in sorted(zip(sizes, means))]
            l = np.argmax(means)
            rows[l] = rows[l].replace("&", "& \\cellcolor{pink}") # color best row
            rows[0] = entry + rows[0] # add architecture on first row
            table += ''.join(rows) 
            if i < len(vpts) - 1:
                table += "\\hline\\hline\n"
            else:
                table += "\\cline{1-2}\n"
        except:
            pass 
    table += "\\end{tabular}\n"
    print(table)



def get_best_models(dynamical_system, config, root='../data'):
    """
    Get the best models from the batch data files for a given dynamical system and configuration.

    Parameters
    ----------
    dynamical_system : str
        The name of the dynamical system to analyze.
    config : str
        The name of the configuration for which to get the best models.
    root : str, optional
        The root directory containing the data, by default '../data'.

    Returns
    -------
    list
        A list of lists, where each sublist contains the architecture name and the best parameters
        for that architecture in the following order: mean, std, median, min, max, beta, train_time.
    """
    results = []
    gist = summary(dynamical_system, root=root)[config]
    for architecture in gist:
        l = sorted(zip(gist[architecture].values(), gist[architecture].keys()))
        try:
            results.append([architecture] + list(l[-1][::-1]))
        except:
            pass
    return results



def l2_kde(data1, data2, n_eval=500):
    """
    Compute the L2 difference between two distributions given by the data1 and data2 arrays.
    
    Parameters
    ----------
    data1 : array_like
        The first data set.
    data2 : array_like
        The second data set.
    n_eval : int, optional
        The number of evaluation points in the range [data1.min(), data1.max()].
    
    Returns
    -------
    r : float
        The L2 difference between the two distributions.
    """
    pdf1 = gaussian_kde(data1)
    pdf2 = gaussian_kde(data2) 
    x = np.linspace(data1.min(), data1.max(), num=n_eval, endpoint=False)
    delx = (data1.max() - data1.min()) / n_eval
    y1 = pdf1.evaluate(x)
    y2 = pdf2.evaluate(x)
    r = np.sqrt((1./n_eval)*((y1-y2)**2).sum()*delx)
    return r



    
  