"""
This script downloads ACS PUMS data for all 50 U.S. states, making use of some utilies in the folktables library
For each state, the downloaded data will be a single csv file
Example: for California, which has ANSI state code 06, the csv file will be saved to datadir/psam_p06.csv
"""



""" 
This script uses and adapts some utility functions from the folktables (https://github.com/socialfoundations/folktables/tree/main/folktables) library
folktables license:

MIT License

Copyright (c) 2021 ZYKLS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import requests
import zipfile
import configparser
import pandas as pd

year = 2018

# important flags for downloading
datadir = "../data/ACS_unemployment_data/"


if __name__ == "__main__":
    print("The year defined in the config file: %s" % year)
    print("ACS PUMS data for this year will be downloaded")
    print("")


# read the states info from the csv file and turn into dictionary
US_states_info = pd.read_csv(datadir + "US_states_ansi_codes.csv",sep=",")
US_states_ansi_codes = US_states_info.set_index('state abbreviation')['state code'].to_dict()
US_states_ansi_codes = {key: ("%.2d" % value) for key, value in US_states_ansi_codes.items()}
US_states_abbrev = US_states_ansi_codes.keys()
#print(US_states_ansi_codes)


# the U.S. has 50 states
# folktable also includes puerto rico, but it is not a state
# on the other hand, ACS also has more data available, for example for Washington D.C., which is also not a proper state
if __name__ == "__main__":
    print("number of U.S. states for which we download data:")
    print(len(US_states_abbrev))
    print(len(US_states_ansi_codes))
    print("")


# for a chosen U.S. state (by abbreviation) and year, get the corresponding CSV file name for the raw data file
def get_file_name(state, year):
    state_code = US_states_ansi_codes[state]
    file_name = f'psam_p{state_code}.csv'
    return file_name

# adapted from folktables
def download_and_extract(url, datadir, remote_fname, file_name, delete_download=False):
    """Helper function to download and unzip files."""
    download_path = os.path.join(datadir, remote_fname)
    response = requests.get(url)
    with open(download_path, 'wb') as handle:
        handle.write(response.content)
    
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extract(file_name, path=datadir)
    
    if delete_download and download_path != os.path.join(datadir, file_name):
        os.remove(download_path)

# adapted from folktables
def initialize_and_download(datadir, state, year):
    """Download the dataset (if required)."""
    assert int(year) >= 2014
    assert state in US_states_abbrev

    state_code = US_states_ansi_codes[state]
    file_name = get_file_name(state, year)
    
    # Assume is the path exists and is a file, then it has been downloaded
    # correctly
    file_path = os.path.join(datadir, file_name)
    if os.path.isfile(file_path):
        return file_path
      
    print(f'Downloading data for {year} person survey for {state}...')
    # Download and extract file
    base_url= f'https://www2.census.gov/programs-surveys/acs/data/pums/{year}/1-Year'
    remote_fname = f'csv_p{state.lower()}.zip'
    url = f'{base_url}/{remote_fname}'
    try:
        download_and_extract(url, datadir, remote_fname, file_name, delete_download=True)
    except Exception as e:
        print(f'\n{os.path.join(datadir, remote_fname)} may be corrupted. Please try deleting it and rerunning this command.\n')
        print(f'Exception: ', e)

    return file_path


if __name__ == '__main__':
    # create the main directory for all data
    os.makedirs(datadir, exist_ok=True)

    # MAIN LOOP    
    # now download data for each state
    for state in US_states_abbrev:
        print("downloading data for state %s" % state)
        initialize_and_download(datadir, state, year)


    print("all downloads finished.")