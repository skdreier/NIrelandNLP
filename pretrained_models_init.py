## Neural network boiler plate 
## using pretrained word2vec

# You will need to download the models first
# Glove is smaller and might be esier to work with initially 
# download here: [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
# Google world news vectors:
# download here: [GoogleNews-vectors-negative300.bin.gz - Google Drive](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

# Load models 

## Glove	
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

from gensim.models import KeyedVectors
# load the Stanford GloVe model
filename = 'glove.6B.100d.txt.word2vec'
glove_model = KeyedVectors.load_word2vec_format(filename, binary=False)

# calculate: (king - man) + woman = ?
result = glove_model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)

# import numpy as np

# with open('LOCATION/glove.6B.50d.txt', 'rb') as lines:
#    w2v_glove = {line.split()[0]: np.array(map(float, line.split()[1:]))
#            for line in lines}

## google
import gensim

# w2v_google = gensim.models.Word2Vec.load_word2vec_format(‘WHERE YOU DOWNLOADED THE MODEL/GoogleNews-vectors-negative300.bin', binary=True)  
from gensim.models import KeyedVectors

filename = 'LOCATION/GoogleNews-vectors-negative300.bin'
google_model = KeyedVectors.load_word2vec_format(filename, binary=True)

## Then you can inspect the models 
result = google_model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)



