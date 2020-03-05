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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import os
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split


## Set the file pathway and download corpus
from IPython.display import display

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]
j_path = os.path.join(project_root) 

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
sentences = df['clean_text'].values # include stopwords, unstemmed

tokenizer = Tokenizer(num_words=max_words)
word_index = tokenizer.word_index # dictionary of word and token number
print('Found %s unique tokens.' % len(word_index)) # 5153 unique tokens (not total tokens)

y = df['just_categories'].values
labels = df['just_categories'].factorize()
label_names = list(labels[1])
label_values = list(range(0,6))
labels_index = dict(zip(label_names, label_values))
labels = list(labels[0])

max_words = 5000 #not all vocab is valuable
# Set random seed
np.random.seed(0)

# Set the number of features we want (how many features do we need?)
# look at the number of words

# Load data from scikit learn pipeline

# Convert feature data to a one-hot encoded feature matrix
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

x = tokenizer.sequences_to_matrix(sentences, mode='binary')
# One-hot encode target vector to create a target matrix
labels

x_train, x_test, y_train, y_test = train_test_split(sequences, labels, test_size=.2, random_state = 40)

# Start neural network
network = models.Sequential()

# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=10, activation='relu', input_shape=(max_words,)))

# Add fully connected layer with a softmax activation function for the output
network.add(layers.Dense(units=6, activation='softmax')) # ensures 0-1 probabilities for each class

# Compile neural network
network.compile(loss='categorical_crossentropy', # Cross-entropy
                optimizer='adam', # Root Mean Square Propagation
                metrics=['accuracy']) # Accuracy performance metric

# Train neural network
history = network.fit(x_train, # Features
                      y_train, # Target vector
                      epochs=3, # Three epochs
                      verbose=0, # No output
                      batch_size=100, # Number of observations per batch
                      validation_data=(x_test, y_test))




num_words = [len(words.split()) for words in sentences]
max_sen_leng = max(num_words)