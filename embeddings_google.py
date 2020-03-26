import numpy as np
import pandas as pd
import nltk
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras import utils

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import os
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors



## Set the file pathway and download corpus
this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]
j_path = os.path.join(project_root) 

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]
PATH_TO_GV = os.path.join(project_root, 'wordvec') + '/'

df = pd.read_csv(os.path.join(j_path, 'justifications_clean_text_ohe.csv'))

# Collapse justification categories from 12 to 6 -- approach #2
df['just_category_6'] = df['justification_cat']
df['just_category_6'] = df['just_category_6'].replace(['J_Emergency-Policy', 'J_Intelligence', 'J_Last-resort', 'J_Utilitarian-Deterrence', 'J_Law-and-order'], 'J_Security')
df['just_category_6'] = df['just_category_6'].replace(['J_Legal_Procedure'], 'J_Legal')
df['just_category_6'] = df['just_category_6'].replace(['J_Political-Strategic'], 'J_Political')
df['just_category_6'] = df['just_category_6'].replace(['J_Denial', 'J_Intl-Domestic_Precedent'], 'J_DenyHRVio') #
df['just_category_6'] = df['just_category_6'].replace(['J_Development-Unity'], 'J_Misc')
df['just_categories'] = df['just_category_6']

# Create a unique number id for each justification category
col = ['just_categories', 'clean_text'] 
df = df[col]
df = df[pd.notnull(df['clean_text'])]
df.columns = ['just_categories', 'clean_text']
df['category_id'] = df['just_categories'].factorize()[0]
category_id_df = df[['just_categories', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'just_categories']].values)
df.head()

######################################
### Stem sentences outside of grid ###
######################################

ps = PorterStemmer()

def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

df['stem_text'] = df['clean_text'].apply(stem_sentences)

#############################################
### Divide into training and testing data ###
#############################################

#sentences = df['stem_text'].values # include stopwords, stemmed
sentences = df['clean_text'] # include stopwords, unstemmed
y = df['just_categories']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

word_index = tokenizer.word_index # word and their token # ordered by most frequent
print('Found %s unique tokens.' % len(word_index))

max_words = 5153 # total words of vocabulary we will consider

num_words = [len(words.split()) for words in sentences]
max_seq_len = max(num_words) + 1

from keras.preprocessing.sequence import pad_sequences

text_tok = pad_sequences(sequences, maxlen=max_seq_len+1)
text_tok.shape
np.mean(text_tok > 0)

from keras.utils import to_categorical

encoder = LabelEncoder()
encoder.fit(y)
labels = encoder.transform(y)

num_classes = np.max(labels) + 1
labels = utils.to_categorical(labels, num_classes)

print('Shape of data tensor:', text_tok.shape)
print('Shape of label tensor:', labels.shape)

# split training data into test, validation
x_train, x_test, y_train, y_test = train_test_split(text_tok, labels, test_size=0.2, random_state = 42)

# Prepare embedding matrix
word_vector_dim=100
vocabulary_size= max_words+1
embedding_matrix = np.zeros((vocabulary_size, word_vector_dim))

nb_filters = 64
filter_size_a = 2
drop_rate = 0.5
my_optimizer = 'adam'

from keras.layers import Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Concatenate, MaxPooling1D, Flatten
from keras.models import Model, load_model
from keras.layers import SpatialDropout1D

my_input = Input(shape=(None,))
embedding = Embedding(input_dim=embedding_matrix.shape[0], input_length=max_seq_len,
    output_dim=word_vector_dim, trainable=True,)(my_input)
        
x = Conv1D(filters = nb_filters, kernel_size = filter_size_a,
    activation = 'relu',)(embedding)
x = SpatialDropout1D(drop_rate)(x)
x = MaxPooling1D(pool_size=5)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
prob = Dense(6, activation = 'softmax',)(x)
model = Model(my_input, prob)
    
model.compile(loss='categorical_crossentropy', optimizer = my_optimizer,
    metrics = ['accuracy']) 

model.fit(x_train, y_train, # Target vector
    epochs=20, # Three epochs
    verbose=1, # No output
    batch_size=100, # Number of observations per batch
    validation_data=(x_test, y_test))

# add the google embeddings 

# Prepare embedding matrix
word_vectors = KeyedVectors.load_word2vec_format(PATH_TO_GV + 'GoogleNews-vectors-negative300.bin', binary=True)

word_vector_dim=300
vocabulary_size= max_words + 1
embedding_matrix = np.zeros((vocabulary_size, word_vector_dim))

for word, i in word_index.items():
    if i>=max_words:
        continue
    try:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),word_vector_dim)


len(embedding_matrix)
embedding_matrix.shape
type(embedding_matrix)

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
nonzero_elements / max_words

# Setting parameters for the NN
nb_filters = 128
filter_size_a = 3
drop_rate = 0.5
my_optimizer = 'adam'

from keras.layers import Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Concatenate, MaxPooling1D, Flatten
from keras.models import Model, load_model

## Build the neural network

my_input = Input(shape=(max_seq_len+1,))

embedding = Embedding(input_dim=embedding_matrix.shape[0], # vocab size, including the 0-th word used for padding
                        output_dim=word_vector_dim,
                        weights=[embedding_matrix], # we pass our pre-trained embeddings
                        input_length=max_seq_len+1,
                        trainable=True
                        )(my_input)

# Kernel size is how big your window is. Putting x number of words into the NN together at a time from each sentence.
x = Conv1D(filters = nb_filters, kernel_size = filter_size_a,
    activation = 'relu',)(embedding)

x = MaxPooling1D(pool_size=5)(x)

x = Flatten()(x)

x = Dense(128, activation='relu')(x)

prob = Dense(6, activation = 'softmax',)(x)
model = Model(my_input, prob)
    
model.compile(loss='categorical_crossentropy', optimizer = my_optimizer,
    metrics = ['accuracy']) 

x = model.fit(x_train, y_train, # Target vector
    epochs=20, # Three epochs
    verbose=1, # No output
    batch_size=100, # Number of observations per batch
    validation_data=(x_test, y_test))