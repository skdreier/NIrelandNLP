###############################################
## NI NLP Project
## Append each of 48 date code .txt files
## Cleans file/image names as we did with justification codes
###############################################

import os
from pathlib import Path
import re
import pandas as pd

this_file_path = os.path.abspath(__file__)
subfolder_root = os.path.split(this_file_path)[0]
folder_root = os.path.split(subfolder_root)[0]
repo_root = os.path.split(folder_root)[0]

date_path = os.path.join(repo_root, 'orig_text_data/dates_0123') + '/'

## Load txt document
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(date_path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))
            

files[1]
name = Path(files[1]).name 
# two different ways  
name.replace('.txt', '')
re.sub('.txt', '', name)

# loop through the files exclude Nodes.txt <-- nVivo relic 
# create lists 
p_text = []
cat = []

# create file paths list 
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(date_path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))

# create method to extract relevant text and appropriate categories from file name
for f in files:
 #   if "Nodes" not in f:
        print(f)
        # Extract text and parse into df
        docs = open(f, "r")
        text = docs.read()
        docs.close()
        text = re.split(r'.*(?=Files)', text)
        # get the file name 
        cat_code = Path(f).name 
        cat.append(re.sub('.txt', '', cat_code))
        p_text.append(list(filter(None, text)))
# create dict that we can use to create a df


cat_text = dict(zip(cat, p_text))

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
df_long = df_long.rename(columns = {'index':'date'})

df_long.head


# we can now proceed with the other regex magic procedures
# JH note: maybe create a function of this...

## Extracts image ID as a unique column
df_long["image_id"] = df_long["raw_text"].str.extract(r"(IMG_\d{4})")

## Extracts file ID as a unique column
df_long["file_id_orig"] = df_long["raw_text"].str.extract(r"(DEFE\w+|PREM\w+|CJ\w+)")

## Fixing File/Image number issue (PREM 15 478, 1010, 1689). 
# Note: 1010/1689 and 487 have overlapping image numbers (1010/1689: IMGs 001-205; 487: IMGs 001-258)
# This will be a problem later if we use IMG as a unique identifier
df_long["image_id_issue"] = df_long["file_id_orig"].str.extract(r"(PREM_15_.*_\S*)") 
df_long["image_id_issue"] = r"IMG_0" + df_long["image_id_issue"].str.extract(r"(\d{3}$)")
df_long["image_id"] = df_long["image_id"].fillna(df_long["image_id_issue"])

df_long["file_id"] = df_long["file_id_orig"].str.extract(r"(PREM_15_.*_\S*)") 
df_long["file_id"] = df_long["file_id"].str.extract(r"(PREM_15_\d*)")
df_long["file_id"] = df_long["file_id"].fillna(df_long["file_id_orig"])

df_long["image_id"].head

# Merge image and file names to output original doc name (for merging)
df_long["img_file"] = df_long.image_id.map(str) + "_" + df_long.file_id

df_long = df_long[['date', 'image_id', 'file_id', 'img_file']]

df_long.head

## Write out as a csv
df_long.to_csv(os.path.join(subfolder_root, 'dates_long_parsed.csv'))

