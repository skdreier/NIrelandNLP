## Neural network boiler plate 
## using pretrained word2vec

# You will need to download the models first
# Glove is smaller and might be esier to work with initially 
# download here: [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
    # SKD downloaded glove.840B.300d (Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download)
# Google world news vectors:
# download here: [GoogleNews-vectors-negative300.bin.gz - Google Drive](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

# Load models 

## Glove	

import os
from pathlib import Path
import gensim
import numpy as np
import pandas as pd

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]
path = os.path.join(project_root, 'wordvec') + '/'

from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = os.path.join(path, 'glove.6B/glove.6B.100d.txt')
word2vec_output_file = os.path.join(path, 'glove.6B.100d.txt.word2vec')
glove2word2vec(glove_input_file, word2vec_output_file)

from gensim.models import KeyedVectors
# load the Stanford GloVe model
filename = word2vec_output_file
glove_model = KeyedVectors.load_word2vec_format(filename, binary=False)

# calculate: (king - man) + woman = ?
result = glove_model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
result = glove_model.most_similar(positive=['london', 'germany'], negative=['berlin'], topn=1)

print(result)

# import numpy as np

# with open('LOCATION/glove.6B.50d.txt', 'rb') as lines:
#    w2v_glove = {line.split()[0]: np.array(map(float, line.split()[1:]))
#            for line in lines}

## google

# w2v_google = gensim.models.Word2Vec.load_word2vec_format(‘WHERE YOU DOWNLOADED THE MODEL/GoogleNews-vectors-negative300.bin', binary=True)  
from gensim.models import KeyedVectors

filename = os.path.join(path, 'GoogleNews-vectors-negative300.bin')

google_model = KeyedVectors.load_word2vec_format(filename, binary=True)

## Then you can inspect the models 
result = google_model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
result = google_model.most_similar(positive=['sushi', 'Italian'], negative=['pizza'], topn=1)

print(result)

###### Our archive corpus

import preprocess as pr

path_corpus = '/Users/sarahdreier/OneDrive/Incubator/NI_docs/'
path_corpus = '/Users/josehernandez/Documents/eScience/projects/NIreland_NLP/data/'
## Load txt document file names
ocr_text = pr.text_preprocess(path_corpus)
ocr_text.files

# run through function to gather all text (as dictionary)
ocr_text.nvivo_ocr(img_id=['IMG_1247_DEFE_24_876'])

# Convert to Dataframe to clean text
ocr_corpus = pd.DataFrame(ocr_text_corpus.items())
ocr_corpus.columns = ['img_file', 'raw_text']

# Function to clean text
def clean_func(column, df):
    new_col = column.str.lower()
    new_col = new_col.replace(r"\n", " ", regex=True)
    new_col = new_col.replace(r"[^0-9a-z #+_]", "", regex=True)
    new_col = new_col.replace(r"#", " ", regex=True)
    df['clean_text'] = new_col
    return(df)

clean_func(ocr_corpus['raw_text'], ocr_corpus) 

# Ready to be developed as a training model. 

# load the list of Image IDs that have a justification and pass to ocr function: not working but not necessary
#df = pd.read_csv(os.path.join(j_path, 'justifications_clean_text_ohe.csv'))
#just_imgs = np.ndarray.tolist(df['img_file_orig'].unique())
#ocr_justifications_docs_corpus = ocr_text.nvivo_ocr(img_id = just_imgs) # This step isn't working
