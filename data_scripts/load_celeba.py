from .data_dictionary_loader import DataDictionaryLoader
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA

class CelebALoader(DataDictionaryLoader):
    """
    CelebALoader

    """
    def __init__(self, datadir, logger, seed=42):

        # todo: try https://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html to prevent overfitting
        N = 9999999
        X_train_male = np.load(datadir + 'embeddings_train_male.npy')[:N]
        y_train_male = np.load(datadir + 'labels_train_male.npy')[:N]

        logger.debug("X_train_male:")
        logger.debug(X_train_male.shape)

        X_test_male = np.load(datadir + 'embeddings_test_male.npy')[:N]
        y_test_male = np.load(datadir + 'labels_test_male.npy')[:N]
        logger.debug("X_test_male:")
        logger.debug(X_test_male.shape)
        X_train_nonmale = np.load(datadir + 'embeddings_train_nonmale.npy')[:N]
        y_train_nonmale = np.load(datadir + 'labels_train_nonmale.npy')[:N]

        X_test_nonmale = np.load(datadir + 'embeddings_test_nonmale.npy')[:N]
        y_test_nonmale = np.load(datadir + 'labels_test_nonmale.npy')[:N]


        
        
        X_train_combined = np.vstack([X_train_male,X_train_nonmale])
        y_train_combined = np.hstack([y_train_male,y_train_nonmale])

        X_test_combined = np.vstack([X_test_male,X_test_nonmale])
        y_test_combined = np.hstack([y_test_male,y_test_nonmale])


        
        n_components = 100
        #ipca = IncrementalPCA(n_components=n_components, batch_size=128)
        ipca = PCA(n_components=n_components)
        X_train_combined = ipca.fit_transform(X_train_combined)
        X_train_male = ipca.transform(X_train_male)
        X_train_nonmale = ipca.transform(X_train_nonmale)
        X_test_male = ipca.transform(X_test_male)
        X_test_nonmale = ipca.transform(X_test_nonmale)
        print("explained variance in PCA:")
        print(ipca.explained_variance_ratio_.cumsum())
        print("data shape:")
        print(X_train_combined.shape)
        


        data_dictionary = {}

        # male
        data_dictionary["male"] = {}
        data_dictionary["male"]["X_train"] = X_train_male.astype(float)
        data_dictionary["male"]["y_train"] = y_train_male.astype(float)

        data_dictionary["male"]["X_test"] = X_test_male.astype(float)
        data_dictionary["male"]["y_test"] = y_test_male.astype(float)
        data_dictionary["male"]["N_train"] = len(X_train_male)
        data_dictionary["male"]["N_test"] = len(X_test_male)

        # nonmale
        data_dictionary["nonmale"] = {}
        data_dictionary["nonmale"]["X_train"] = X_train_nonmale.astype(float)
        data_dictionary["nonmale"]["y_train"] = y_train_nonmale.astype(float)

        data_dictionary["nonmale"]["X_test"] = X_test_nonmale.astype(float)
        data_dictionary["nonmale"]["y_test"] = y_test_nonmale.astype(float)
        data_dictionary["nonmale"]["N_train"] = len(X_train_nonmale)
        data_dictionary["nonmale"]["N_test"] = len(X_test_nonmale)


        states = ['male','nonmale']
        self.states = states
        self.data_dictionary = data_dictionary
        # dummy feature names
        self.feature_names = ['lin_feat']*X_train_male.shape[0]