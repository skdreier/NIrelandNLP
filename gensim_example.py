## Gensim how-to and test with subset 
import pandas as pd
import os
import numpy as np
from gensim.models import Word2Vec

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]

sys.path.append(project_root)
import preprocess as pr

path_corpus = os.path.join(project_root, "data/") 


## Load txt document file names
ocr_text = pr.text_preprocess(path_corpus)
ocr_text.files

# run through function to gather all text (as dictionary)
ocr_text_corpus = ocr_text.nvivo_ocr()

# Convert to Dataframe
ocr_corpus = pd.DataFrame(ocr_text_corpus.items())
ocr_corpus.columns = ['img_file', 'raw_text']

# Function to clean text
def clean_func(column, df):
    new_col = column.str.lower()
    new_col = new_col.replace(r"\n", " ", regex=True)
    #new_col = new_col.replace(r"[^0-9a-z #+_]", "", regex=True)
    new_col = new_col.replace(r"[^a-z #+_]", "", regex=True)
    new_col = new_col.replace(r"#", " ", regex=True)
    new_col = new_col.replace(r'\b\w{1,3}\b', '', regex=True)
    df['clean_text'] = new_col
    return(df)

test = clean_func(ocr_corpus['raw_text'], ocr_corpus) 

text = list(test['clean_text'])

tokenized_docs = [word_tokenize(i) for i in text]
for i in tokenized_docs:
    print(i)

model = Word2Vec(tokenized_docs, min_count=1) # this should train the model

words = list(model.wv.vocab) # this should show the words in the model, but it is showing characters rather than words
print(sorted(words))
len(words)

print(model['terrorism']) # will show embeddings 

print(model)



