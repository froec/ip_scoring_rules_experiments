from .data_dictionary_loader import DataDictionaryLoader


import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import dill




class FraminghamLoader(DataDictionaryLoader):
    """
    FraminghamLoader

    """
    def __init__(self, datadir, logger, subsample=None, seed=42, test_size=0.2):
        assert subsample is None
        # for framingham we don't allow subsampling



        # we follow approximately the preprocessing by https://www.kaggle.com/code/kapiluniyara/framingham-dataset
        df = pd.read_csv(datadir + 'framingham.csv')

        y = df['TenYearCHD'].to_numpy().flatten()
        df = df.drop(columns=['TenYearCHD'])

        groupcol = 'age'

        group_data = df[groupcol]
        logger.debug("group data:")
        logger.debug(group_data)
        states = ['young','old']
        #group_masks = [(df[groupcol]==s).to_numpy() for s in states]
        group_masks = [(df[groupcol]<=60.).to_numpy(),(df[groupcol]>60.).to_numpy()]
        logger.debug("group sizes:")
        logger.debug([m.sum() for m in group_masks])
        # we don't drop the group-inducing column here. If you want to do that:
        #df = df.drop(columns=[groupcol])


        # Imputer for mean
        mean_imputer = SimpleImputer(strategy='mean')

        # Columns to impute with mean (normally distributed/symmetric)
        mean_cols = ['heartRate']

        # Impute the columns with mean
        df[mean_cols] = mean_imputer.fit_transform(df[mean_cols])

        # Imputer for median
        median_imputer = SimpleImputer(strategy='median')

        # Columns to impute with median (skewed distributions or categorical)
        median_cols = ['education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'glucose']

        # Impute the columns with median
        df[median_cols] = median_imputer.fit_transform(df[median_cols])

        logger.debug(df.isnull().sum())

        logger.debug(df)

        logger.debug("how many have heart disease?")
        logger.debug(y.sum())

        logger.debug(df.describe())



        """
        We found that outlier removal, like in the above kaggle ressource, doesn't have substantial effects, so we simply don't do it
        (since it would require some justification anyway)
        """


        X = df.to_numpy()


        # add interaction terms!
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)


        # Fit and transform X
        X = poly.fit_transform(X)

        poly_feature_names = poly.get_feature_names_out(df.columns)


        data_dictionary = {}
        base_rates = []
        for i,state in enumerate(states):
            logger.debug(state)
            data_dictionary[state] = {}
            X_s = X[group_masks[i]]
            y_s = y[group_masks[i]]

            ros = RandomOverSampler(random_state=seed)
            X_s, y_s = ros.fit_resample(X_s, y_s)

            base_rate = y_s.sum()/len(y_s)
            base_rates.append(base_rate)

            X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, random_state=seed, shuffle=True, test_size=test_size)


            data_dictionary[state]["X_train"] = X_train.astype(float)
            data_dictionary[state]["y_train"] = y_train.astype(float)

            data_dictionary[state]["X_test"] = X_test.astype(float)
            data_dictionary[state]["y_test"] = y_test.astype(float)
            data_dictionary[state]["N_train"] = len(X_train)
            data_dictionary[state]["N_test"] = len(X_test)



        base_rates_dict = dict(zip(states,base_rates))
        base_rates_dict_sorted = dict(sorted(base_rates_dict.items(), key=lambda item: item[1]))
        logger.debug("base rates of y=1:")
        logger.debug_dict_4f(base_rates_dict_sorted)

        ### dump data to pickle file
        dill.dump(data_dictionary, open(datadir + "data_dictionary_framingham.pkl","wb"))

        self.data_dictionary = data_dictionary
        self.states = states
        self.feature_names = poly_feature_names