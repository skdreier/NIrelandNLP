################################################################
#### This is code for using regular expressions to clean /  ####
####    parse text from Nvivo .txt into a workable format.  ####
#### Thurs Jan 09					    ####
#### Jose and Sarah					    ####
#### environment: ni_nlp 				    ####
################################################################

## Activate environment
Condo activate ni_nlp

## cd into project
cd OneDrive/Incubator/NIreland_NLP/

## Launch: python OR jupyter notebook


## Import python programs ##
import pandas as pd
import re 
import os


## Load txt document
test_file = "/Users/sarahdreier/OneDrive/Incubator/NIreland_NLP/just_0106/J_Denial.txt"
f = open(test_file, "r")
text = f.read()
f.close()

## View txt file
text

## Split into unique string for each document (r signals regular expression)
test = re.split(r'.*(?=Files)', text)

## Examine output
test[2]
len(test) #120 lines, line 0 is blank

## Filter out blank lines
test2 = list(filter(None,test))
len(test2) #118 documents have the “denial” code - this is the same as in Nvivo

## Puts the list into a dataframe and name the raw text as its only column
prelim_df = pd.DataFrame(test2,columns=["raw_text"])
prelim_df.head

## Extracts image ID as a unique column
prelim_df["image_id"] = prelim_df["raw_text"].str.extract(r"(IMG_\d{4})")
prelim_df.head

## Extracts file ID as a unique column
prelim_df["file_id"] = prelim_df["raw_text"].str.extract(r"(DEFE\w+|PREM\w+|CJ\w+)")
prelim_df.head

## Fixing File/Image number issue (PREM 15 478, 1010, 1689). 
# Note: 1010/1689 and 487 have overlapping image numbers (1010/1689: IMGs 001-205; 487: IMGs 001-258)
# This will be a problem later if we use IMG as a unique identifier
prelim_df["image_id2"] = prelim_df["file_id"].str.extract(r"(PREM_15_.*_\S*)") 
prelim_df["image_id2"] = r"IMG_0" + prelim_df["image_id2"].str.extract(r"(\d{3}$)")
prelim_df["image_id"] = prelim_df["image_id"].fillna(prelim_df["image_id2"])

## Extracts justification text as its own raw-text column (Removes “Reference”)
prelim_df["just_text_lump"] = prelim_df["raw_text"].str.replace(r"(?<!Files)(.*)(?<=Coverage)", "").str.strip()
prelim_df["just_text_lump"].head
prelim_df["just_text_lump2"] = prelim_df["just_text_lump"].str.replace(r"\W+", " ") # This removes all non-letter characters, including \n spaces and punctuation.

## Extracts justification text as its own raw-text column (Retains “Reference” markers which can be used to split each unique code).
prelim_df["just_text"] = prelim_df["raw_text"].str.replace(r"(?<!Files)(.*)(?<=Coverage])", "").str.strip()
prelim_df["just_text"].head

## Extract the number of unique codes in this justification category a given document received
prelim_df["reference_count"] = prelim_df["raw_text"].str.extract(r"(\d\sreference)")

## Write out as a csv
prelim_df.to_csv("prelim_denial.csv")

## Text work to create a new variable for each unique reference
#re.compile(r"^>(\w+)$$([.$]+)^$", re.MULTILINE) #
#(Reference.\d)(.*)(?<=Reference.[2-5])
#prelim_df["test"] = prelim_df["raw_text"].str.extract(r"(Reference)(.*)(Files)", re.MULTILINE)

##################


## To set different rules for output
#pd.set_option('display.max_rows', 10)
#pd.set_option('display.max_rows', None)

## See number of references / codes per document — THIS DOESN’T WORK
#prelim_df["code_quantity"] = prelim_df["raw_text"].str.extract(r”(\d{2}references)")

