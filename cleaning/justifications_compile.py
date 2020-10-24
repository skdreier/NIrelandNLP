###############################################
## NI NLP Project
##  1) Compile justifications from Nvivo
##      - Append each of 12 justification.txt files
##      - Parse text into components and fix image naming issue
##      - Pull "screenshot" codes from above to be hand-transcribed
##  2) Merge text justifications with hand-transcribed screenshot justifications
##      (after hand-transcription is completed)
##  3) Merge with date files
##  4) Save justification output file ('justifications_long_training.csv')
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
j_path = os.path.join(repo_root, 'orig_text_data/just_0404') + '/'
#j_path = os.path.join(repo_root, 'orig_text_data/just_icr') + '/' #For ICR task

sys.path # make sure the repo is in the path 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocess as pr

################################################
##### 1) COMPILE JUSTIFICATIONS FROM NVIVO #####
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
df_long = df_long.rename(columns = {'index':'justification'})

# we can now proceed with the other regex magic procedures
# JH note: maybe create a function of this...

## Extracts image ID as a unique column
df_long["image_id"] = df_long["raw_text"].str.extract(r"(IMG_\d{4})")
df_long.head

## Extracts file ID as a unique column
df_long["file_id_orig"] = df_long["raw_text"].str.extract(r"(DEFE\w+|PREM\w+|CJ\w+)")
df_long.head

## Fixing File/Image number issue (PREM 15 478, 1010, 1689). 
# Note: 1010/1689 and 487 have overlapping image numbers (1010/1689: IMGs 001-205; 487: IMGs 001-258)
# This will be a problem later if we use IMG as a unique identifier
df_long["image_id_issue"] = df_long["file_id_orig"].str.extract(r"(PREM_15_.*_\S*)") 
df_long["image_id_issue"] = r"IMG_0" + df_long["image_id_issue"].str.extract(r"(\d{3}$)")
df_long["image_id"] = df_long["image_id"].fillna(df_long["image_id_issue"])

df_long["file_id"] = df_long["file_id_orig"].str.extract(r"(PREM_15_.*_\S*)") 
df_long["file_id"] = df_long["file_id"].str.extract(r"(PREM_15_\d*)")
df_long["file_id"] = df_long["file_id"].fillna(df_long["file_id_orig"])

# Merge Image and File names to output the original file name for merging purposes
df_long["img_file"] = df_long.image_id.map(str) + "_" + df_long.file_id

# Maintain original document ID number to merge with .txt from full corpus
df_long["img_file_orig"] = df_long["raw_text"].str.extract(r"(?<=\d{2}\\\\)(.*)(?=.-)")

## Extracts justification text as its own raw-text column (Removes “Reference”)
df_long["just_plain_text_all_ref"] = df_long["raw_text"].str.replace(r"(?<!Files)(.*)(?<=Coverage)", "").str.strip()
df_long["just_plain_text_all_ref"].head
df_long["just_plain_text_all_ref"] = df_long["just_plain_text_all_ref"].str.replace(r"\W+", " ") # This removes all non-letter characters, including \n spaces and punctuation.

## Extracts justification text as its own raw-text column (Retains “Reference” markers which can be used to split each unique code).
df_long["just_text_all_ref"] = df_long["raw_text"].str.replace(r"(?<!Files)(.*)(?<=Coverage])", "").str.strip()
df_long["just_text_all_ref"].head

## Extract the number of unique codes in this justification category a given document received
df_long["ref_count"] = df_long["raw_text"].str.extract(r"(\d\sreference)")
df_long["ref_count"] = df_long["ref_count"].str.extract(r"(\d)")

# split multiple reference text entries into new columns
test = df_long["just_text_all_ref"].str.split(r'.*(?=Reference)',expand = True) 
# 
new_test = pd.concat([df_long[['image_id', 'file_id', 'img_file_orig', 'justification']], test], axis=1).drop([0], axis=1)
new_test.info()

idx = ['image_id', 'file_id', 'img_file_orig', 'justification']

multi_df = new_test.set_index(idx)
multi_df.head

stacked_df = multi_df.stack(dropna=False)
long_df = stacked_df.reset_index()
long_df.columns
long_df.shape
long_df = long_df.rename(columns = {0: 'text'})
long_df.head(10)

# clean up "None" missing text HERE
long_df.iloc[[3]]
long_df.dropna(subset=['text'], inplace=True)
long_df.shape

# clean up "Refernce and weird characters here"
long_df['text'] = long_df['text'].replace({"(Reference.\d)(.*)(Coverage\\n)": ""}, regex=True)
long_df.head(20)
long_df

# save the data 
#long_df.to_csv(os.path.join(folder_root, 'justifications_long_training.csv'))

# For ICR files:
#long_df['justification_cat'] = long_df['justification']
#long_df_icr = long_df[['file_id', 'image_id', 'justification_cat', 'text']]
#long_df_icr.head
#long_df_icr.to_csv(os.path.join(folder_root, 'justifications_icr.csv'))

# Pull text that needs to be transcribed by hand
# Creating a List within a column (proved to be a headache due to the non characters)
# Create a list within a single column 
# df_long["coded_refs"] = df_long["just_text_all_ref"].str.split(r'.*(?=Reference)',expand = False) 

## Extracts docs that have page capture codes (rather than text codes) and outputs it to a different doc

df_pg_ref = long_df[long_df['text'].str.contains(r"(Page.\d.:)")]
df_pg_ref.shape
df_pg_ref.columns
df_pg_ref.head

#len(df_pg_ref['level_4'])
#df_pg_ref.level_4.unique()

df_pg_ref = df_pg_ref[['image_id', 'file_id', 'img_file_orig', 'justification', 'level_4']]
df_pg_ref.to_csv(os.path.join(folder_root, 'page_ref.csv'))

long_df_icr.head

########################################################################################
##### 2) MERGE TEXT JUSTIFICATIONS WITH HAND-TRANSCRIBED SCREENSHOT JUSTIFICATIONS #####
########################################################################################

df_just = long_df
df_trans = pd.read_csv(os.path.join(folder_path, "page_ref_transcribed_04-2020.csv"))

df_trans.columns
df_just.columns

df_just['occurances'] = df_just['level_4']
df_trans['text'] = df_trans['transcribed_text']

df_just_merge = df_just[['image_id', 'file_id', 'img_file_orig', 'justification', 'occurances', 'text']]
df_trans_merge = df_trans[['image_id', 'file_id', 'img_file_orig', 'justification', 'occurances', 'text']]

all_just = pd.concat([df_just_merge, df_trans_merge])

#### Cross-reference number of justification codes w Nvivo record:
all_just.head
all_just.shape #2277 rows
df_just.shape #2063 rows 
df_trans.shape #214 rows 
176+77+327+36+48+23+101+338+377+148+351+60 #Nvivo number of codes / justification: 2062 unique codes

### NOTES:
### Resulting DF has 2277 lines. This includes:
# 1849 text-coded justifications
# 214 place holders for the screenshot refs to be transcribed (these will be omitted later) 
# 214 the transcribed references
### Result: 2063 lines, 2062 unique coded justifications; this aligns w Nvivo record (2062)
### CONCLUSION: THIS WORKED; NOTHING APPEARS TO HAVE BEEN DROPPED OR DOUBLE-COUNTED

# Save it as a csv
#all_just.to_csv(os.path.join(folder_root, 'justifications_long_training.csv'))

####################################
##### 3) MERGE WITH DATE FILES #####
####################################

###############################################
## NI NLP Project
## Merge date with justification codes
## NOTE: This is a temporary output. Once we get the whole data file as a .txt, we can: 
## 1) run a loop that assigns every date to the previous date, then
## 2) merge this to the justification file output.
###############################################

## Date range (based on archive file records)
df_just = all_just

# Upload dates coded file -- all documents that received a Round 1 Date code 
# (e.g., first page of multi-page docs)
df_dates = pd.read_csv(os.path.join(folder_root, "date_files/dates_long_parsed.csv"))
df_dates['date_coded'] = df_dates['date']
df_dates['img_file_orig'] = df_dates['img_file']
df_dates = df_dates[['date_coded', 'img_file', 'img_file_orig']]

# Upload date range (based on file archive record) file
df_date_range = pd.read_csv(os.path.join(folder_root, "date_files/date_range.csv"), dtype=str)
df_date_range.columns
df_date_range["file_id"] = df_date_range["file1"] + "_" + df_date_range["file2"] + "_" + df_date_range["file3"]
df_date_range = df_date_range[['file_start_year', 'file_start_month', 'file_end_year', 'file_end_month', 'file_id']]

df_date_range.shape
df_dates.shape
df_just.shape

# Merge justification and date ranges
df_merge = pd.DataFrame.merge(df_just, df_date_range, 'left', on='file_id')

# Merge justification/date ranges w date codes
df_merge_final = pd.DataFrame.merge(df_merge, df_dates, 'left', on="img_file_orig")

##################################################################################
##### 4) Save justification output file ('justifications_long_training.csv') #####
##################################################################################

df_just_dates = df_merge_final[['image_id', 'file_id', 'img_file_orig',
      'file_start_year', 'file_start_month', 'file_end_year', 'file_end_month', 'date_coded', 
      'justification', 'occurances', 'text',]]

# output csv
df_just_dates.to_csv(os.path.join(folder_root, 'justifications_long_training.csv'))
