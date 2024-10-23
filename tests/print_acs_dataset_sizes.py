import numpy as np 
import pandas as pd

datadir = "../data/ACS_unemployment_data/"

state_abbrevs = pd.read_csv(datadir + "US_states_ansi_codes.csv")
all_states = list(state_abbrevs["state abbreviation"])
print("list of states:")
print(all_states)

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



states = all_states

print("number of states: %s" % len(states))


print("load the data..")
base_rates = []
dataset_sizes = {}
for i,state in enumerate(states):
    print("State: %s" % state)
    data = pd.read_csv(datadir + ("processed_data_%s" % state) + ".csv")
    print(len(data))
    dataset_sizes[state] = len(data)


# for info: order states by number of N_train
print("States by number of training samples:")
sorted_lens = dict(sorted(dataset_sizes.items(), key=lambda item: item[1], reverse=True))
for key, value in sorted_lens.items():
        print(f"{key}: {value:.4f}")

print("total:")
print(sum(sorted_lens.values()))