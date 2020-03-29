######### JOSE: Line 115 begins to train the word2vec model with our own corpus -- this is what goes wrong. ##########

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
from gensim.models import Word2Vec
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
result = google_model.most_similar(positive=['sushi', 'italian'], negative=['pizza'], topn=1)

print(result)

##################################################################################################################
###### FOR CODE TO TRAIN BASED ON OUR OWN ARCHIVE CORPUS, SEE: "gensim_example.py" ###############################
##################################################################################################################
##################################################################################################################
##################################################################################################################

###################################################################
###### OLD TEXT: Training data based on our archive corpus ########
###################################################################

import preprocess as pr
import pandas as pd
import os
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer

path_corpus = '/Users/sarahdreier/OneDrive/Incubator/NI_docs/'

## Load txt document file names
ocr_text = pr.text_preprocess(path_corpus)
ocr_text.files

# run through function to gather all text (as dictionary)
ocr_text_corpus = ocr_text.nvivo_ocr()
len(ocr_text_corpus)

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

clean_func(ocr_corpus['raw_text'], ocr_corpus) 

# Subset to pages that contain a justification
j_path = '/Users/sarahdreier/OneDrive/Incubator/NIreland_NLP'
df = pd.read_csv(os.path.join(j_path, 'justifications_clean_text_ohe.csv'))
just_imgs = np.ndarray.tolist(df['img_file_orig'].unique())
ocr_corpus_subset = ocr_corpus.loc[ocr_corpus['img_file'].isin(just_imgs)]
#ocr_text_corpus_just = ocr_text.nvivo_ocr(img_id=just_imgs)

# This is the text corpus to train our new model
sentences = ocr_corpus_subset['clean_text'] 
len(sentences) #605 unique documents / sentences
type(sentences)

## Example of model training
sentences_ex = [["cat", "say", "meow"], ["dog", "say", "woof"]]
type(sentences_ex)
model = Word2Vec(sentences_ex, min_count=1)
model.wv.vocab

## Prepare our text corpus
model = Word2Vec(sentences, min_count=1)
model.wv.vocab # word2vec at the CHARACTER level. Try to tokenize first

## Tokenize text corpus
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
#sequences = tokenizer.texts_to_sequences(sentences)
sentences2 = list(sentences)
type(sentences2)
print(sentences2)
len(sentences2)

def extractDigits(lst): 
    return [[el] for el in lst] 
                  
sentences3 = extractDigits(sentences2)
type(sentences3)

model = Word2Vec(sentences3, min_count=1)
model.wv.vocab
words = list(model.wv.vocab) # this should show the words in the model, but it is showing characters rather than words
print(sorted(words))
len(words)

model = Word2Vec(sentences2, min_count=1)
model.wv.vocab # word2vec at the CHARACTER level, even after tokenizing and transforming to list

# This was our original code from last Friday -- transforms corpus to a dictionary
word_index = tokenizer.word_index # word and their token # ordered by most frequent
print('Found %s unique tokens.' % len(word_index)) #13,197 unique tokens
type(word_index) ### I tried to transform from a dict to a string but that didn't fix the issue
model = Word2Vec(word_index, min_count=1) # this should train the model
words = list(model.wv.vocab) # this should show the words in the model, but it is showing characters rather than words
print(sorted(words))
len(words)

# This is our new code as of today (Mon Mar 09)
from gensim.models import Word2Vec
sentences
type(sentences)
sentences = list(sentences)
model = Word2Vec(sentences, min_count=1)
model.wv.vocab


### PCA model output ###
### Gensim model output ###

# One problem that might happen: 
# what words are more likely