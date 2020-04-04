# This script builds a function to clean the data! 
# SKD Feb 13

######################################################
### Builds a function to clean, OHE data           ###
### Uses: justifications_long_training.csv         ###
### Creates: justifications_clean_text_ohe.csv     ###
### Script needs to be cleaned                     ###
######################################################

import os
from pathlib import Path
import re
import pandas as pd
import numpy as np

this_file_path = os.path.abspath(__file__)
folder_root = os.path.split(this_file_path)[0]
train_path = os.path.join(folder_root) + '/'

repo_root = os.path.split(folder_root)[0]
repo_path = os.path.join(repo_root)

df_just = pd.read_csv(os.path.join(train_path, "justifications_complete.csv"))
df_just.head

# Sample text to test function
#df_test = df_just.iloc[0:10, 2:3]
#df_test = df_just[0:10]
#df_test = df_just.iloc[[22, 30, 85, 130, 131, 157, 159, 177, 194, 239, 240, 241], :] # combo of messed up, all caps, and normal text
#df_test.columns

# Function to clean text and return it as either a dataframe or a text list
def clean_func(column, df, analysis = False):
    new_col = column.replace(r"(Page.\d.:)(.*)", np.nan, regex=True)
    new_col = new_col.str.lower()
    new_col = new_col.replace(r"[^0-9a-z #+_]", "", regex=True)
    if analysis is False: 
        df['clean_text'] = new_col
        return(df)
    else:
        list_clean_col = list(new_col)
        return(list_clean_col)

clean_func(column = df_just['text'], df = df_just) # Default analysis = false
#clean_func(column = df_test['text'], df = df_test, analysis = True)

# One hot encoding after running this function
cat_columns = ['justification']
df_just_ohe = pd.get_dummies(df_just, columns=cat_columns)

# Add justification column back in (for looping purposes later)
df_just_ohe['justification_cat'] = df_just['justification']

# Save it as a csv
df_just_ohe.to_csv(os.path.join(repo_root, 'justifications_clean_text_ohe.csv'))