import numpy as np
import pandas
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

# Set random seed
np.random.seed(0)

# Set the number of features we want (how many features do we need?)
# look at the number of words

number_of_features = # we need to look at the size of our vocabulary 

# Load data from scikit learn pipeline

# Convert feature data to a one-hot encoded feature matrix
tokenizer = Tokenizer(num_words=number_of_features)

train_features = tokenizer.sequences_to_matrix(train_data, mode='binary')
test_features = tokenizer.sequences_to_matrix(test_data, mode='binary')

# One-hot encode target vector to create a target matrix
train_target = to_categorical(train_target_vector)
test_target = to_categorical(test_target_vector)

# Start neural network
network = models.Sequential()

# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=10, activation='relu', input_shape=(number_of_features,)))

# Add fully connected layer with a softmax activation function for the output
network.add(layers.Dense(units=6, activation='softmax')) # ensures 0-1 probabilities for each class

# Compile neural network
network.compile(loss='categorical_crossentropy', # Cross-entropy
                optimizer='adam', # Root Mean Square Propagation
                metrics=['accuracy']) # Accuracy performance metric

# Train neural network
history = network.fit(train_features, # Features
                      train_target, # Target vector
                      epochs=3, # Three epochs
                      verbose=0, # No output
                      batch_size=100, # Number of observations per batch
                      validation_data=(test_features, test_target))
