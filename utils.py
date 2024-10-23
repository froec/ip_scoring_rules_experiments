import numpy as np
from typing import Dict, Any
import os


import logging


class SimpleLogger():
    def __init__(self, name, log_file=None, level=logging.INFO, console=True):
        """
        Creates a logger with specified name, log file, and verbosity level.
        Written by gemini

        Parameters
        ----------
        name : str
            The name of the logger.
        log_file : str
            The path to the log file (optional).
        level : one of {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}

        Returns:
        -------
            A configured logger object.
        """

        # remove log if it already exists
        if os.path.exists(log_file):
            os.remove(log_file)

        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Create a formatter for the log messages
        

        if console:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # Add a file handler if a log file is specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        self.logger = logger 


    def debug(self, message):
        """ 
        Logs a debug message

        Parameters
        ----------
        message : str
            obviously, the message to be logged
        """
        self.logger.debug(str(message))


    def info(self, message):
        """ 
        Logs an info message

        Parameters
        ----------
        message : str
            obviously, the message to be logged
        """
        self.logger.info(str(message))


    def debug_dict_4f(self, dictionary: Dict[Any, float]) -> None:
        """
        Logs the content of a dictionary where the values are floats, formatted to 4 decimal places.

        Parameters
        ----------
            dictionary (dict): A dictionary with numeric values.

        Returns:
        -------
            None. Prints the formatted dictionary to the console.
        """

        for key, value in dictionary.items():
            self.logger.debug(f"{key}: {value:.4f}")



"""
An extremely stupid logger who just forgets everything
"""
class DummyLogger():
    def __init__(self):
        pass
    def info(self, message):
        pass 

    def debug(self, message):
        pass
    def debug_dict_4f(self, dict):
        pass



def printdict_4f(dictionary: Dict[Any, float]) -> None:
    """
    Prints a dictionary where the values are floats, formatted to 4 decimal places.

    Parameters
    ----------
        dictionary (dict): A dictionary with numeric values.

    Returns:
    -------
        None. Prints the formatted dictionary to the console.
    """

    for key, value in dictionary.items():
        print(f"{key}: {value:.4f}")


        

def computeAccuracy(y_preds: np.ndarray, y_test: np.ndarray) -> float:
    """
    Computes the accuracy for binary classification.
    Ts function assumes a task of binary classification with labels encoded as 0. and 1.

    Parameters
    ----------
        y_preds: np.ndarray 
            Predicted labels, 1-D array of shape (n_samples,).
        y_test: np.ndarray
            True labels, 1-D array of shape (n_samples,).

    Returns:
    -------
        float: The accuracy score, a value between 0.0 and 1.0.

    Raises:
        AssertionError: If inputs are not 1-D arrays of the same length.

    
    """
    y_preds = np.array(y_preds)
    y_test = np.array(y_test)
    assert y_preds.ndim == 1, "y_preds must be a 1-D array"
    assert y_test.ndim == 1, "y_test must be a 1-D array"
    assert len(y_preds) == len(y_test), "y_preds and y_test must have the same length"
    return (y_preds == y_test).sum() / len(y_preds)




# Example doc string
"""
Parameters
----------
first : array_like
    the 1st param name `first`
second :
    the 2nd param
third : {'value', 'other'}, optional
    the 3rd param, by default 'value'

Returns
-------
string
    a value in a string

Raises
------
KeyError
    when a key error
OtherError
    when an other error
"""