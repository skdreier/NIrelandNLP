###############################################
## NI NLP Project
## start analysis
###############################################

import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
import preprocess as pr
import matplotlib.pyplot as plt


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]
j_path = os.path.join(project_root) 

df_just = pd.read_csv(os.path.join(j_path, 'justifications_clean_text_ohe.csv'))
df_just.shape

df_just.dropna(subset=['clean_text'], inplace=True)
df_just.shape

sentences = df_just['clean_text']

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
sentences = df_just['clean_text'].values
y = df_just['justification_J_Terrorism'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

# Use BOW model to vectorize the sentences. Create vocabulary using only training data.
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

# Logit binarify classifier: Terrorism category
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print("Accuracy:", score)
## Accuracy: 0.76 for terrorism

# Loop through all justifictions
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

    #classifier = LogisticRegression()
    #classifier = GaussianNB() # this one is better for overlap
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print('Accuracy for {} code: {:.4f}'.format(source, score))


### What word vectors are driving these classifications?

# Trying for one model (terrorism):
sentences = df_just['clean_text'].values
y = df_just['justification_J_Terrorism'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)


logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('log', LogisticRegression(n_jobs=1, C=1e5)),
               ])

logreg.fit(sentences_train, y_train)

y_pred = logreg.predict(sentences_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
###
log = logreg.named_steps['log']
cv = logreg.named_steps['vect']
log.classes_
log.coef_.ravel()
len(cv.vocabulary_)
feature_names = cv.get_feature_names()
coef = log.coef_.ravel()
top_positive_coefficients = np.argsort(coef)[-20:]
top_negative_coefficients = np.argsort(coef)[:20]
top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
 # create plot
plt.figure(figsize=(15, 5))
colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
plt.bar(np.arange(2 * 20), coef[top_coefficients], color=colors)
feature_names = np.array(feature_names)
plt.xticks(np.arange(1, 1 + 2 * 20), feature_names[top_coefficients], rotation=60, ha='right')
plt.show()
plt.savefig('biclass_logreg/terrorism' + '.png')


# Plot for all category models:

accuracy_score_output={}
#classification_report_output={}
#confusion_matrix_output={}

for cat in df_just['justification_cat2'].unique():
    sentences = df_just['clean_text'].values
    y = df_just[cat].values
    cat_title = cat.replace(r'justification_J_', "").strip()

    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)

    logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('log', LogisticRegression(n_jobs=1, C=1e5)),
               ])

    logreg.fit(sentences_train, y_train)

    y_pred = logreg.predict(sentences_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    accuracy_score_output.update({cat_title:accuracy_score(y_pred, y_test)})

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

###
    log = logreg.named_steps['log']
    cv = logreg.named_steps['vect']
    log.classes_
    log.coef_.ravel()
    len(cv.vocabulary_)
    feature_names = cv.get_feature_names()
    coef = log.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-20:]
    top_negative_coefficients = np.argsort(coef)[:20]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    plt.title(cat_title)
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * 20), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * 20), feature_names[top_coefficients], rotation=60, ha='right')
    plt.savefig('biclass_logreg/' + cat_title + '.png')
    plt.close()









##################################################################
##################################################################
# Deep Neural Networks -- I got pretty lost here, but everything #
# seemed to work until the function
##################################################################
##################################################################

from keras.models import Sequential
from keras import layers

input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu')) #relu is a uniform distribution
model.add(layers.Dense(1, activation='sigmoid')) #binary classification

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

##### Word embeddings

### One hot encoding for each category (we already did this same thing with simpler code):

# List justification categories into categorical integer values
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
justification_labels = encoder.fit_transform(df_just['justification_cat'])
justification_labels

# Reshape array
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
justification_labels = justification_labels.reshape((len(justification_labels), 1))
encoder.fit_transform(justification_labels)

# Vectorize a text corpus into a list of integers
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

print(sentences_train[2])
print(X_train[2])

# choose some words and look at where they show up
for word in ['the', 'internment', 'hmg', 'faulkner', 'belfast']:
    print('{}: {}'.format(word, tokenizer.word_index[word]))

# pad sequence
from keras.preprocessing.sequence import pad_sequences
maxlen = 100 # max sentence length
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

 print(X_train[0, :]) # look at first sentence in corpus

# Keras embedding layer
from keras.models import Sequential
from keras import layers

embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#vocab_size*embedding_dim

# Train model

history = model.fit(X_train, y_train,
    epochs=20,
    verbose=False,
    validation_data=(X_test, y_test),
    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history) # plot still not working