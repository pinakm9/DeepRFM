import json
import os
import pandas as pd

class Logger:
    def __init__(self, save_folder) -> None:
        """
        Constructor for Logger class.

        Parameters
        ----------
        save_folder : str
            The directory where log files are saved.

        Attributes
        ----------
        folder : str
            The directory where log files are saved.
        config : dict
            A dictionary containing hyperparameters and other configuration.
        train_log : dict
            A dictionary containing training and validation metrics.
        config_file : str
            The path to the json file containing the configuration.
        train_file : str
            The path to the csv file containing the training log.

        Returns
        -------
        None
        """
        self.folder = save_folder
        self.config = {}
        self.train_log = {}
        self.config_file = '{}/config.json'.format(save_folder)
        self.train_file = '{}/train_log.csv'.format(save_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
    def update(self, start, kwargs):
        """
        Updates the configuration dictionary and writes it to a json file.

        Parameters
        ----------
        start : bool
            If True, overwrite the existing config file. If False, append to it.
        kwargs : dict
            A dictionary of key-value pairs containing hyperparameters and other configuration.

        Returns
        -------
        None
        """
        
        self.config.update(kwargs)
        if start:
            with open(self.config_file, 'w') as file:
                json.dump(kwargs, file, indent=2)
        else:
            with open(self.config_file, 'a') as file:
                json.dump(kwargs, file, indent=2)


    def log(self, start=False, **kwargs):
        """
        Logs training data to a CSV file.

        Parameters
        ----------
        start : bool, optional
            If True, overwrite the existing training log file. If False, append to it.
        kwargs : dict
            A dictionary of key-value pairs representing the training data to log.

        Returns
        -------
        None
        """
        df = pd.DataFrame.from_dict(kwargs)
        if start:
            df.to_csv(self.train_file, index=False)
        else:
            df.to_csv(self.train_file, mode='a', index=False, header=False)

    def print(self, **kwargs):
        """
        Prints a line of training data to the console.

        Parameters
        ----------
        kwargs : dict
            A dictionary of key-value pairs representing the training data to print.

        Returns
        -------
        None
        """
        line = ''
        for key, value in kwargs.items():
            line += '{}={:.3f}\t'.format(key, value[0])
        print(line)

 

