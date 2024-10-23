from abc import ABC, abstractmethod

class DataDictionaryLoader(ABC):
    """
    An abstract class for a class which loads a data dictionary
    i.e. a dataset

    A data dictionary is a dictionary of the following structure:

    data_dictionary[state]["X_train"] : np.ndarray
         array of shape (N,k), containing the training data, where N is the number of samples and k the number of features
    data_dictionary[state]["y_train"] : np.ndarray
        array of shape (N,) containing the training labels as floats
    data_dictionary[state]["X_test"] : np.ndarray
         array of shape (M,k), containing the test data, where M is the number of samples and k the number of features
    data_dictionary[state]["y_test"] : np.ndarray
        array of shape (M,) containing the test labels as floats
    data_dictionary[state]["N_train"] = len(X_train)
    data_dictionary[state]["N_test"] = len(X_test)


    Attributes
    -------
    data_dict : dict
        the data dictionary
    states: List[str]
        list of the group labels
    feature_names : List[str]
        human-readable names of the features

    Methods
    -------
    init
    """
    def __init__(self, datadir, logger, subsample=None, seed=42, test_size=0.2):
        """
        Loads the data dictionary

        Parameters
        ----------
        datadir : str
            The directory where the data is to be found
        logger : .utils.SimpleLogger or utils.DummyLogger
            used to log messages when loading the data
        subsample : one of {int, float between 0 and 1, np.ndarray}
            Semantics of the *group-wise* subsampling: 
                if subsample=None, use the full data for each group
                if subsample=N for some int, then for each of the subgroups we will take a subsample of size N
                if subsample=p, 0<p<1, then we will take a fraction p of the data _for each group_
                if subsample : np.ndarray is an array of ints of shape (G,), where G is the number of groups, then we will
                    take the respective number of samples for each group
        seed : int
            the random seed
        test_size : float
            fraction to be used for testing
        """

        pass