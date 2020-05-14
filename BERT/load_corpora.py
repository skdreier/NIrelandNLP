######################################################
##### Builds Text Corpora for BERT analysis      #####
##### corpus_full.csv: contains all archive text #####
##### corpus_just.csv: contains all sentences coded as justification #####
##### corpus_subset_pages_w_just.csv: contains all text in pages that contain a justification #####
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

####################################
###### Function to clean text ######
####################################

# Function to clean text
def clean_func(column, df):
    new_col = column.str.lower()
    new_col = new_col.replace(r"\n", " ", regex=True) # Remove line spaces
    new_col = new_col.replace(r"[^0-9a-z #+_]", "", regex=True) # Remove non-alpha-numeric digits
    #new_col = new_col.replace(r"[^a-z #+_]", " ", regex=True) # (Removes non-alpha digits)
    new_col = new_col.replace(r"#", " ", regex=True) # Removes hashtag (didn't get droped from code above)
    #new_col = new_col.replace(r'\b\w{1}\b', '', regex=True) # remove one-letter words (loses "I" and "a")
    #new_col = new_col.replace(remove_word_set, '', regex=True) # remove identified words
    new_col = new_col.replace(r"(Page.\d.:)(.*)", np.nan, regex=True) # remove reference to page captures (placeholder for transcribed text)
    df['clean_text'] = new_col
    return(df)

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

# Clean text (using function from above)
full_corpus = clean_func(ocr_corpus['raw_text'], ocr_corpus) 

# Save corpus as csv
full_corpus.to_csv('corpus_full.csv')

###########################################
###### Retrieve Justification Corpus ######
###########################################

# Column data wrangling
just_corpus = pd.read_csv(os.path.join(project_root, 'justifications_clean_text_ohe.csv'))
just_corpus['img_file'] = just_corpus['img_file_orig']
just_corpus['raw_text'] = just_corpus['text']
just_corpus['clean_text0'] = just_corpus['clean_text']
just_corpus['justification_category'] = just_corpus['justification_cat']

# Clean text (using function from above)
just_corpus = clean_func(just_corpus['raw_text'], just_corpus) 

# Subset to columns we need
just_corpus = just_corpus[['img_file', 'raw_text', 'clean_text', 'transcription_data',
       'justification_category', 'justification_J_Denial',
       'justification_J_Development-Unity', 'justification_J_Emergency-Policy',
       'justification_J_Intelligence',
       'justification_J_Intl-Domestic_Precedent',
       'justification_J_Last-resort', 'justification_J_Law-and-order',
       'justification_J_Legal_Procedure', 'justification_J_Misc',
       'justification_J_Political-Strategic', 'justification_J_Terrorism',
       'justification_J_Utilitarian-Deterrence',
]]

# Save corpus as csv
just_corpus.to_csv('corpus_just.csv')

####################################################################################
###### Retrieve Subset Corpus: All text in pages that contain a justification ######
###### Sofia: I don't think you'll need this for now                          ######
####################################################################################

# Subset to pages that contain a justification
df = pd.read_csv(os.path.join(project_root, 'justifications_clean_text_ohe.csv'))
just_imgs = np.ndarray.tolist(df['img_file_orig'].unique())
ocr_corpus_subset = ocr_corpus.loc[ocr_corpus['img_file'].isin(just_imgs)]

ocr_corpus_subset.to_csv('corpus_subset_pages_w_just.csv')

# Load all three corpora
full_corpus_df = pd.read_csv('corpus_full.csv')
just_corpus_df = pd.read_csv('corpus_just.csv')
subset_corpus_df = pd.read_csv('corpus_subset_pages_w_just.csv')

