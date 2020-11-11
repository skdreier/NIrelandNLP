###############################################
## NI NLP Project
##  Adopt/augment justification code to randomly select internment docs for ICR
###############################################

import os
from pathlib import Path
import re
import pandas as pd
import numpy as np

this_file_path = os.path.abspath(__file__)
folder_root = os.path.split(this_file_path)[0]
folder_path = os.path.join(folder_root) + '/'
repo_root = os.path.split(folder_root)[0]
j_path = os.path.join(repo_root, 'orig_text_data/int_1002') + '/'

sys.path # make sure the repo is in the path 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocess as pr

################################################
##### 1) COMPILE Internment docs #####
################################################

## Load txt document
raw = pr.text_preprocess(j_path)
raw.files

cat_text = raw.nvivo_clumps()

for key, value in cat_text.items() :
    print (key)
# Create df this will create a df with lists where each row is the justification
# category (12) followed by a list of the individual "files" within the category
temp_df = pd.DataFrame.from_dict(cat_text, orient='index')

# What we want is a column for the category repeated by each entry of the list 
# which will becore rows 
df_long = temp_df.stack().reset_index(level=1, drop=True).to_frame(name='raw_text')

# The previous created a long df with the category as the index
# we create a column using that index
df_long.reset_index(level=0, inplace=True)

# rename the column 
df_long = df_long.rename(columns = {'index':'classification'})

## Extract file ID as a unique column
df_long["img_file_orig"] = df_long["raw_text"].str.extract(r"(?<=\d{2}\\\\)(.*)(?=.-)")

# make sure this worked (should be ~1400 unique documents)
int_docs2 = df_long['img_file_orig'].unique()
int_docs2.shape

random = pd.DataFrame(np.random.choice(int_docs2, 20, replace=False))

random.to_csv(os.path.join(repo_root, 'misc_tasks/random_sample_justification_ICR_among_all_internment_docs.csv'))

# pull two more docs that aren't in DEFE 24 1214
random = pd.DataFrame(np.random.choice(int_docs2, 2, replace=False))
print(random)

# New docs:
# 0   IMG_6303_PREM_15_484
# 1  IMG_7165_PREM_15_1005

random.to_csv(os.path.join(repo_root, 'misc_tasks/random_sample_justification_ICR_among_all_internment_docs_2.csv'))


