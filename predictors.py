from abc import ABC, abstractmethod
import numpy as np
import typing 
from typing import List
from loss_functions import LossFunction
from trainers import Trainer

class Predictor(ABC):
    """
    An abstract class for a predictor  

    Attributes
    -------
    name : str
        a human readable name for this predictor

    Methods
    -------
    predict(X_test)
        Get the predictor's prediction for the given test data.

    getRecommendedActions(X_test, loss_function)
        Get the action recommendation of the predictor for the given test data and loss function.
    """

    @abstractmethod
    def predict(self, X_test : np.ndarray) -> np.ndarray:
        pass  

    @abstractmethod
    def getRecommendedActions(self, X_test: np.ndarray, loss_function : LossFunction) -> np.ndarray:
        pass




class PrecisePredictor(Predictor):
    """ 
    This class is a trivial wrapper around a trainer, resulting in a precise probabilistic predictor
    or more generally it takes some object which is expected to implement a predict(X_test) function
    which returns the precise probabilities

    Note: this calls the trainer's transform method when predicting! So it expects untransformed data

    Attributes
    -------
    trainer : Trainer
        the underlying trainer object, to be used for prediction
    name : str
        a human readable name for this predictor

    Methods
    -------
    predict(X_test)
        Get the predictor's _precise probabilistic_ prediction for the given test data.

    getRecommendedActions(X_test, loss_function)
        Get the action recommendation of the predictor for the given test data and loss function.
        For a precise predictor this is the Bayes optimal action

    """


    def __init__(self, trainer : Trainer, name : str):
        """
        Initializes a `PrecisePredictor` instance.

        Parameters
        ----------
        trainer : Trainer
            The underlying trainer or object used for prediction.
            This object is expected to have a `predict(X_test)` method.
        name (str, optional): A human-readable name for this predictor.
        """
        self.trainer = trainer
        self.name = name

    def predict(self, X_test : np.ndarray) -> np.ndarray:        
        """
        This returns the precise probabilistic predictions

        Parameters:
        -----------
        X_test : numpy.ndarray
            A 2D array of shape (N, number_of_features), where N is the number 
            of samples and number_of_features is the number of features.

        Returns:
        -----------
        np.ndarray
            A one-dimensional array of shape (N,)
        """
        return self.trainer.predict(self.trainer.transform(X_test))


    def getRecommendedActions(self, X_test, loss_function):
        """
        Get the recommended actions for an array of test points of shape X_test 
        and the specified loss function
        Note: this crucially depends on the loss function!

        Parameters:
        -----------
        X_test : numpy.ndarray
            A 2D array of shape (N, number_of_features), where N is the number 
            of samples and number_of_features is the number of features.

        Returns:
        -----------
        np.ndarray
            A one-dimensional array of shape (N,), containing the action recommendations
            which are Bayes optimal actions under the predicted probabilities
        """
        preds = self.predict(X_test)
        return loss_function.getBayesAction(preds)






class MinMaxEnsemble(Predictor):
    def __init__(self, ensemble_members : List[Predictor], name : str):
        """
        A MinMax Ensemble of a set of precise predictors
        Each precise predictor should take care of transforming the data correctly..

        Attributes:
        -----------
        ensemble_members : List[Predictor]
            An array of trainers, each implementing a predict method

        name : str
            A human-readable identifier for the predictor

        Methods
        -------
        predict(X_test)
            Get the predictor's _precise probabilistic_ prediction for the given test data.

        getRecommendedActions(X_test, loss_function)
            Get the action recommendation of the predictor for the given test data and loss function.
            For a precise predictor this is the Bayes optimal action


        """
        self.ensemble_members = ensemble_members
        self.name = name


    def predict(self, X_test):        
        """
        This returns an imprecise probabilistic predictions based on the ensemble

        Parameters:
        -----------
        X_test : numpy.ndarray
            A 2D array of shape (N, number_of_features), where N is the number 
            of samples and number_of_features is the number of features.

        Returns:
        -----------
        np.ndarray
            A two-dimensional array of shape (k,N)
            where k is the number of ensemble members
        """
        return np.array([mem.predict(X_test) for mem in self.ensemble_members])


    def getRecommendedActions(self, X_test, loss_function):
        """
        Get the recommended actions for an array of test points of shape X_test 
        and the specified loss function
        Note: this crucially depends on the loss function!

        Parameters:
        -----------
        X_test : numpy.ndarray
            A 2D array of shape (N, number_of_features), where N is the number 
            of samples and number_of_features is the number of features.

        Returns:
        -----------
        np.ndarray
            A one-dimensional array of shape (N,), containing the action recommendations
            which are Bayes optimal actions under the predicted probabilities
        """
        y_preds = np.array([mem.predict(X_test) for mem in self.ensemble_members])

        # Compute the min and max along the first axis (over ensemble members)
        y_min = np.min(y_preds, axis=0)
        y_max = np.max(y_preds, axis=0)

        # Use list comprehension to compute a_star for each X in X_test
        a_stars = [loss_function.getMinMaxAction(min_val, max_val) for min_val, max_val in zip(y_min, y_max)]

        return np.array(a_stars)

