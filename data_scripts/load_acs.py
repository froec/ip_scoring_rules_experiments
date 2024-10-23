from .data_dictionary_loader import DataDictionaryLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import dill

class ACSLoader(DataDictionaryLoader):

    def __init__(self, datadir, logger, subsample=None, seed=42, test_size=0.2, states=None):
        # if states is None, we will use all states

        state_abbrevs = pd.read_csv(datadir + "US_states_ansi_codes.csv")
        all_states = list(state_abbrevs["state abbreviation"])
        logger.debug("list of states:")
        logger.debug(all_states)

        data_dictionary = {}


        feature_columns = ["AGEP", # age
                           "SCHL", # educational attainment
                           "MAR", # marital status
                           "SEX", # sex
                           "ESP", # employment status of parents
                           "MIG", # mobility (lived here 1 year ago?)
                           "GCL", # is grandparents with children?
                           "HICOV", # health insurance
                           "SCIENGP", # has science degree

                           # disability related
                           "DIS", # has disability?
                           "DEAR", # hearing difficulty
                           "DEYE", # vision difficulty
                           "DREM", # cognitive difficulty

                           # origin/ethnicity related
                           "CIT", # citizen status
                           "NATIVITY", # nativity
                           "ANC", # ancestry
                           "RAC1P", # race recoded
                           "LANX", # speaks another language than english at home?
                           'isHisp', # of hispanic origin
                           'isWhiteOnly' # white and non-hispanic
                          ]


        # the states for which we collect the data
        # try ['DE','WV'] quite different baselines and different coefficients
        if states is None:
            states = all_states 

        #['AK','CA'] #['AK','HI'] #['CA','MS']
        # if we don't balance ID-MS but take frac=0.2, interesting entropy curve
        # ['AK','CA'] # this seems to be crazy 2 states - they get all the DRO weight of the first 20 states
        #all_states[:20] # all_states
        # ['ID','MS'] good one to showcase max ent difference with log loss vs brier vs asymmetric loss 
        # nice one ['AK','SC'], max ent is at lamb=0.65 roughly
        # try ['CA', 'UT'] # max ent is inbetween
        #['NM','IA'] max ent=NM
         #['ID','MS'] # nice example for where max ent is in the middle
        # MN is a very easy state
        # ['OK','MN'] is a good couple, DRO will put all weight on OK since MN is so easy.
        # Maximum entropy dist is clearly at the boundary=OK then

        logger.debug("number of states: %s" % len(states))


        logger.debug("load the data..")
        base_rates = []
        data_dictionary = {}
        for i,state in enumerate(states):
            logger.debug("State: %s" % state)
            data = pd.read_csv(datadir + ("processed_data_%s" % state) + ".csv")
            logger.debug(len(data))


            if subsample is None:
                data_subsampled = data
            elif isinstance(subsample, float) and subsample<=1.:
                data_subsampled = data.sample(frac=subsample, replace=False, random_state=seed)
            elif isinstance(subsample, list):
                assert len(subsample) == len(states)
                data_subsampled = data.sample(n=subsample[i], replace=False, random_state=seed)
            elif isinstance(subsample, int):
                data_subsampled = data.sample(n=subsample, replace=False, random_state=seed)
            

            logger.debug("length of subsampled data:")
            logger.debug(len(data_subsampled))

            X = data_subsampled[feature_columns].values.astype(float)
            Y = data_subsampled['isEmployed'].values.astype(float)

            base_rate = Y.sum()/len(Y)
            base_rates.append(base_rate)


            ### polynomial features

            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            # Fit and transform X
            X = poly.fit_transform(X)

            poly_feature_names = poly.get_feature_names_out(feature_columns)

            
            if test_size > 0.:
                X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=seed, shuffle=True, test_size=test_size) 
            else:
                X_train = X 
                y_train = Y 
                X_test = []
                y_test = []

            data_dictionary[state] = {}
            data_dictionary[state]["X_train"] = X_train
            data_dictionary[state]["y_train"] = y_train
            data_dictionary[state]["X_test"] = X_test
            data_dictionary[state]["y_test"] = y_test
            data_dictionary[state]["N_train"] = len(X_train)
            data_dictionary[state]["N_test"] = len(X_test)

            dill.dump(data_dictionary, open(datadir + "data_dictionary_acs.pkl","wb"))

            self.data_dictionary = data_dictionary
            self.states = states
            self.feature_names = poly_feature_names


        # all data has been loaded


