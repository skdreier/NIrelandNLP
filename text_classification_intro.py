###############################################
## NI NLP Project
## start analysis
###############################################

import os
from pathlib import Path
#import re
import pandas as pd
import numpy as np
import preprocess as pr

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]
j_path = os.path.join(project_root) 

df_just = pd.read_csv(os.path.join(j_path, 'justifications_clean_text_ohe.csv'))
df_just.shape

df_just.dropna(subset=['clean_text'], inplace=True)
df_just.shape

sentences = df_just['clean_text']
#sentences = str(df_just['clean_text'])


from sklearn.feature_extraction.text import CountVectorizer

# takes the words of each sentence and creates a vocabulary of all the unique words in the sentences.
vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences)
vectorizer.vocabulary_

# take each sentence and get the word occurrences of the words based on the previous vocabulary. The vocabulary consists of all five words in our sentences, each representing one word in the vocabulary. When you take the previous two sentences and transform them with the CountVectorizer you will get a vector representing the count of each word of the sentence:
# vector line for each of 1734 sentences 
# bag of words model
vectorizer.transform(sentences).toarray() 

###########################
## Defining a base model ##
###########################

# Split into training and testing dataset

from sklearn.model_selection import train_test_split

sentences = df_just['clean_text'].values
y = df_just['justification_J_Terrorism'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

# Use BOW model to vectorize the sentences. Create vocabulary using only training data.
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

# Logit classifier
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print("Accuracy:", score)
## Accuracy: 0.76 for terrorism

## Loop through all justifictions

# First, rename justification categories so they match the OHE dummy variable names
df_just['justification_cat2'] = 'justification_' + df_just['justification_cat'].astype(str)
df_just['justification_cat2'].unique()

for source in df_just['justification_cat2'].unique():
    sentences = df_just['clean_text'].values
    y = df_just[source].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)

    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    X_test  = vectorizer.transform(sentences_test)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print('Accuracy for {} code: {:.4f}'.format(source, score))


# Deep Neural Networks -- I got pretty lost here, but everything 
# seemed to work until the function

from keras.models import Sequential
from keras import layers

input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
    epochs=100,
    verbose=False,
    validation_data=(X_test, y_test),
    batch_size=10)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

plot_history(history)
