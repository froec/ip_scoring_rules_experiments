"""
Assume the data for all states have been downloaded already.
This script takes care of preprocessing.
The raw data is in csv format (one file for each state): we load it into a pandas dataframe (for each state),
and only keep some of the columns as features, doing some feature encoding.

We apply an age filter: 16 < age < 90
"""

import numpy as np
import pandas as pd

datadir = "../data/ACS_unemployment_data/"

# read the states info from the csv file and turn into dictionary
US_states_info = pd.read_csv(datadir + "US_states_ansi_codes.csv",sep=",")
US_states_info.columns=["state abbreviation","state code"]
US_states_info = US_states_info.set_index('state abbreviation')

print(US_states_info)
print(US_states_info.loc["AL"])

year = 2018 # let's use the newest available data
print("year: %s" % year)

# for a chosen U.S. state (by abbreviation) and year, get the corresponding CSV file name for the raw data file
def get_file_name(state, year):
    state_code = US_states_info.loc[state]["state code"]
    state_code = str(state_code).zfill(2)
    file_name = f'psam_p{state_code}.csv'
    return file_name


############ first: which columns do we keep from the raw data?
# compared to folktables:
# dropped MIL ("is in military" feature)

# added: PWGTP (person's weight), HICOV (has health insurance)
columns_to_keep = ["AGEP", # age
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
                   "HISP", # hispanic origin
                   "LANX", # speaks another language than english at home?

                   # general info:
                   "PWGTP", # person's weight

                   # the target
                   "ESR" # employment status recoded
                  ]

# give readable names to these columns
columns_names_dict = {'AGEP' : 'age', 'SCHL' : 'educational attainment', 'MAR' : 'marital status', 'SEX' : 'sex', 'ESP' : 'employment of parents', \
                     'MIG' : '1-year-non-mover', 'GCL' : 'grandparent with grandchildren', 'HICOV' : 'health insured', 'SCIENGP' : 'science/eng degree',\
                     'DIS' : 'disability', 'DEAR' : 'hearing difficulty', 'DEYE' : 'vision difficulty', 'DREM' : 'cognitive difficulty',\
                     'CIT' : 'citizen status', 'NATIVITY' : 'nativity status', 'ANC' : 'ancestry status', 'RAC1P' : 'Race', 'HISP' : 'Hispanic (detailled)', \
                     'LANX' : 'non-english lang at home', 'PWGTP' : 'person weight', 'ESR' : 'employment status',\
                     # new fields
                     'isEmployed' : 'employed', 'ThreeClassesEmploymentStatus' : '3-class employment status',\
                      'isHisp' : 'hispanic origin', 'isWhiteOnly' : 'white (non-hispanic)'\
                     }
column_names = pd.DataFrame.from_dict(columns_names_dict, orient='index')
column_names.to_csv(datadir + "column_names.csv",index=True)

############     
    
    
############  MAIN LOOP   
# for each US state
for state, row in US_states_info.iterrows():
    # state is an abbreviation of the from AZ, CA, etc.
    print("State: (abbreviation,code)")
    statecode = row["state code"]
    print("statecode:")
    print(statecode)
    print(state,row)
    file_name = get_file_name(state,year)
    print("Raw data file: %s" % file_name)
    state_data = pd.read_csv(datadir + file_name)
    
    # there shouldn't be any persons with weight less than 1 in the data
    # note that person's weight does not refer to their body, but to the statistical sampling weight
    assert (state_data["PWGTP"]<1.).sum() == 0
    assert (not state_data["PWGTP"].isna().any())
    
    # apply age filter: we employ the same age filter as folktables
    # AGEP (Age) must be strictly greater than 16 and strictly less than 90.
    print("number of person records in raw ACS data: %s" % len(state_data))
    state_data_filtered = state_data[(state_data["AGEP"] > 16.) & (state_data["AGEP"] < 90.)]
    
    # now that we applied the age filter there shouldn't be any NaNs left in the unemployment feature
    # since here NaN means less than 16 years old
    assert (not state_data_filtered["ESR"].isna().any())
    
    print("number of person records after filtering: %s" % len(state_data_filtered))
    
    
    
    # only keep these columns
    state_data_relevant = state_data_filtered[columns_to_keep].astype(float)
    
    # Replace all NaN values with 0.0
    # this is fine since no variable has meaningful 0 values
    state_data_relevant = state_data_relevant.fillna(0.0)
    
    
    print(state_data_relevant.head(10))
    
    ############## feature engineering
    """
    augment the dataframe with some new columns based on encodings
    
    the target variable: we want to predict whether a person is employed (Y=1) or unemployed (Y=0)
    a threfold distinction for ESR feature:
    1. employed as civilian or in military (even if potentially not currently working due to sickness or similar)
    2. unemployed (i.e. potentially job-seeking)
    3. not in labor force (i.e. not seeking jobs)
    
    One possibility: predict who is employed
    Construct the label Y such that Y==1 iff ESR==1, and thus Y==0 iff (ESR == 2 or ESR == 3)
    
    Other possibility: predict who is unemployed
    Construct the label Y such that Y==1 iff ESR==2, and thus Y==0 iff (ESR == 1 or ESR == 3)
    # seems questionable
    
    Other possibility: construe as multi-classification problem using the three categories.
    """
    
    # is a person employed?
    employed_column = state_data_relevant["ESR"].isin([1.,2.,4.,5.])
    state_data_relevant["isEmployed"] = employed_column
    
    # three-class multi class column
    multi_class_column = np.zeros_like(state_data_relevant["ESR"])
    multi_class_column[state_data_relevant["ESR"] == 3] = 0. # unemployed
    multi_class_column[state_data_relevant["ESR"].isin([1.,2.,4.,5.])] = 1. # employed
    multi_class_column[state_data_relevant["ESR"] == 6] = 2. # out of labor force
    state_data_relevant["ThreeClassesEmploymentStatus"] = multi_class_column
    
    
    # encode binary hispanic feature
    hispanic_column = state_data_relevant["HISP"] != 1.
    state_data_relevant["isHisp"] = hispanic_column
    
    # white and not hispanic
    white_only = (state_data_relevant["RAC1P"] == 1) & (state_data_relevant["isHisp"] == False)
    state_data_relevant["isWhiteOnly"] = white_only
    
    ##############
    
    
    # save the preprocessed data to CSV file
    state_data_relevant.to_csv(datadir + "processed_data_%s.csv" % state, index=False)
    
                       
                       
print("finished preprocessing for all states.")