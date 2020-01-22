###############################################
## NI NLP Project
## 1) Append each of 12 justification.txt files
## 2) Parse text into components
## 3) Fix image naming issue
###############################################

import os
from pathlib import Path
import re
import pandas as pd

## Load txt document
#test_file = '/Users/josehernandez/Documents/eScience/projects/NIreland_NLP/justifications_01-06/justifications_txt/J_Denial.txt'
test_file = "/Users/sarahdreier/OneDrive/Incubator/NIreland_NLP/just_0106/J_Denial.txt"

f = open(test_file, "r")
text = f.read()
f.close()

#path = '/Users/josehernandez/Documents/eScience/projects/NIreland_NLP/'
path = '/Users/sarahdreier/OneDrive/Incubator/NIreland_NLP/'


files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
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
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))

# create method to extract relevant text and appropriate categories from file name
for f in files:
    if "Nodes" not in f:
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
test
# will need to join to overall df 

# better to create a list within a single column 
df_long["coded_refs"] = df_long["just_text_all_ref"].str.split(r'.*(?=Reference)',expand = False) 
df_long["coded_refs"] # need to remove empty list items 

## Extract pieces coded as photo block (rather than as text)
# This is more detailed than necessary in order to avoid false positives
# ISSUE: THIS DOESN'T PULL ANY PAGE CAPTURE BEYOND THE FIRST REFERENCE!!!! WHATEVER SHALL WE DO? #
df_long["page_captures"] = df_long["coded_refs"].str.extract(r"(Reference.\d.*\nPage.\d.:.*)")
df_long["page_captures"] = df_long["page_captures"].str.replace(r"(?<=Reference.\d.)(.*\n)", "").str.strip()

## Write out as a csv
df_long.to_csv(os.path.join(path, 'justifications_long_parsed.csv'))