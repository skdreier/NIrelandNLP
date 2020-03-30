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
folder_root = os.path.split(this_file_path)[0]
path = os.path.join(folder_root, 'wordvec') + '/'

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