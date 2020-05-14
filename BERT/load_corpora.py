######################################################
### Retrieves Text Corpora for Sophia ###
### Uses: NI_docs, preprocessing.py                ###
######################################################

import os, sys
import pandas as pd
import numpy as np

# Define file / project paths
this_file_path = os.path.abspath(__file__)
project_root = os.path.split(os.path.split(this_file_path)[0])[0]
sys.path.append(project_root)

# Import scripts to concatenate archive text 
import preprocess as pr

##################################
###### Retrieve Full Corpus ######
##################################

# Define path for full data corpus
path_corpus = os.path.join(project_root, "orig_text_data/NI_docs/") 

## Load txt document file names
ocr_text = pr.text_preprocess(path_corpus)
ocr_text.files #6946 documents

# run through function to gather all text (as dictionary)
ocr_text_corpus = ocr_text.nvivo_ocr()

# Convert to Dataframe
ocr_corpus = pd.DataFrame(ocr_text_corpus.items())
ocr_corpus.columns = ['img_file', 'raw_text']

###############################################################
###### Subset Corpus: Pages that contain a justification ######
###### Sofia: I don't think you'll need this for now ##########
###############################################################

# Subset to pages that contain a justification
df = pd.read_csv(os.path.join(project_root, 'justifications_clean_text_ohe.csv'))
just_imgs = np.ndarray.tolist(df['img_file_orig'].unique())
ocr_corpus_subset = ocr_corpus.loc[ocr_corpus['img_file'].isin(just_imgs)]

df['clean_text']
# Define whether you want to use whole corpus or subset to text w justifications
corpus = ocr_corpus
#corpus = ocr_corpus_subset

# Function to clean text
def clean_func(column, df):
    new_col = column.str.lower()
    new_col = new_col.replace(r"\n", " ", regex=True) # Remove line spaces
    new_col = new_col.replace(r"[^0-9a-z #+_]", "", regex=True) # Remove non-alpha-numeric digits
    #new_col = new_col.replace(r"[^a-z #+_]", " ", regex=True) # (Removes non-alpha digits)
    new_col = new_col.replace(r"#", " ", regex=True) # Removes hashtag (didn't get droped from code above)
    #new_col = new_col.replace(r'\b\w{1}\b', '', regex=True) # remove one-letter words (loses "I" and "a")
    #new_col = new_col.replace(remove_word_set, '', regex=True) # remove identified words
    df['clean_text'] = new_col
    return(df)

corpus_clean = clean_func(corpus['raw_text'], corpus) 
