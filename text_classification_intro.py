###############################################
## NI NLP Project
## start analysis
## Code runs binary classifiers for each category
## Plots the words that have the most weight in classifying categories
## Can use code for all words or just stopwords
## this code can be cleaned/streamlined w a few functions (for a later date)
## Updated: Sarah 23/02/2020
###############################################

import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
import preprocess as pr
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder 


##### Load data #####
this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]
j_path = os.path.join(project_root) 

df_just = pd.read_csv(os.path.join(j_path, 'justifications_clean_text_ohe.csv'))

# Drops observations based on page captures (not highlighted text) -- these will be transcribed later
df_just.dropna(subset=['clean_text'], inplace=True)

# Object with sentences-based text for analysis (type: pd.Series)
clean_sentences = df_just['clean_text']

###### Function to remove stopwords (optional) ######
def rmv_stopwords(sent):
        STOPWORDS = set(stopwords.words("english"))
        sent = [' '.join(word for word in x.split() if word not in STOPWORDS) for x in sent.tolist()]
        return sent

sentences_nosw = rmv_stopwords(clean_sentences)

###### Prep for binary classification

# Create a vocabulary of all the unique words in the sentences
vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences_nosw)
vectorizer.vocabulary_
vectorizer.transform(sentences_nosw).toarray() 

#######################################
## Defining a base model: BOW Logreg ##
#######################################

###########################################################
###### Logreg Binary Classifier: Terrorism category #######
###########################################################

# Split into training and testing dataset
# DECIDE WHETHER TO EXCLUDE STOPWORDS 
# sentences = df_just['clean_text'].values # For all words
sentences = pd.Series(sentences_nosw).values # exclude stopwords 
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

###### Determine which word vectors drive this terrorism classification ######
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

#######################################################
###### Logreg Binary Classifier: All categories #######
#######################################################

# Loop through all justifictions
# First, rename justification categories so they match the OHE dummy variable names
df_just['justification_cat2'] = 'justification_' + df_just['justification_cat'].astype(str)
df_just['justification_cat2'].unique()

for source in df_just['justification_cat2'].unique():
    #sentences = df_just['clean_text'].values
    sentences = pd.Series(sentences_nosw).values # exclude stopwords 
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

###### Plot for all category models:

accuracy_score_output={}
#classification_report_output={}
#confusion_matrix_output={}

for cat in df_just['justification_cat2'].unique():
    # sentences = df_just['clean_text'].values
    sentences = pd.Series(sentences_nosw).values # exclude stopwords 
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
    plt.figure(figsize=(15, 8))
    plt.title(cat_title)
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * 20), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * 20), feature_names[top_coefficients], rotation=60, ha='right')
    plt.savefig('biclass_logreg/' + cat_title + '.png')
    plt.close()

###################################################
############### End Script ########################
###################################################

# Attempt at function that includes stopwords and tokenizes
# Tokenization had issues

    def rmv_stopwords(sent, tokenize = False):
        STOPWORDS = set(stopwords.words("english"))
        sent = [' '.join(word for word in x.split() if word not in STOPWORDS) for x in sent.tolist()]
        #sent = sent.apply(lambda x: [word for word in x if word not in STOPWORDS])
        return sent
       # if tokenize is True:
       #     results = []
       #     for sentence in sent:
       #         tokenized_sentences = []
       #         for s in sentence:
       #             tokenized_sentences.append(nltk.word_tokenize(sentence))
       #         results.append(tokenized_sentences)
       #     return sent
       #     return results