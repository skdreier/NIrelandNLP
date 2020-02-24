### For now: clean text with no stopwords: sentences_nosw
### 7 justification categories: justification_cat

import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
import nltk

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from io import StringIO

from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]
j_path = os.path.join(project_root) 

df = pd.read_csv(os.path.join(j_path, 'justifications_clean_text_ohe.csv'))

# Collapse justification categories from 12 to 7
df['just_category_12'] = df['justification_cat']
df['just_category_7'] = df['justification_cat']
df['just_category_7'] = df['just_category_7'].replace('J_Intl-Domestic_Precedent', 'J_Denial')
df['just_category_7'] = df['just_category_7'].replace(['J_Utilitarian-Deterrence', 'J_Intelligence', 'J_Law-and-order', 'J_Development-Unity'], 'J_Outcome')
df['just_category_7'] = df['just_category_7'].replace('J_Last-resort', 'J_Emergency-Policy')
df['just_category_7'].unique()

# Define which category size to use in analysis (12, 7, etc)
df['just_categories'] = df['just_category_12']
#df['just_categories'] = df['just_category_7']

# Create a unique number id for each justification category
col = ['just_categories', 'clean_text'] 
df = df[col]
df = df[pd.notnull(df['clean_text'])]
df['category_num'] = df['just_categories'].factorize()[0]
category_num_df = df[['just_categories', 'category_num']].drop_duplicates().sort_values('category_num')
category_to_num = dict(category_num_df.values)
num_to_category = dict(category_num_df[['category_num', 'just_categories']].values)

###### Function to remove stopwords (optional) ######
def rmv_stopwords(sent):
        STOPWORDS = set(stopwords.words("english"))
        sent = [' '.join(word for word in x.split() if word not in STOPWORDS) for x in sent.tolist()]
        return sent

sentences_nosw = rmv_stopwords(df['clean_text'])

sentences = pd.Series(sentences_nosw).values # exclude stopwords 
# sentences = df['clean_text'].values # include stopwords
y = df['just_categories'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

## Pipeline
multinom = Pipeline([('vect', CountVectorizer()), # ngram_range=(1, 1), tokenizer = LemmaTokenizer
                ('tfidf', TfidfTransformer()),     # use_idf=True
                ('multiclass', MultinomialNB()),    # alpha=1.0 # smoothing parameter
               ])

# 4 hyperparameters: ngrams, tfidf, alpha, lemma tokenizer

from sklearn.model_selection import GridSearchCV
# create dictionary {} of parameters
parameters = {
    'vect__ngram_range': [(1,1),(1,4),(1,10)],
    'vect__tokenizer': (LemmaTokenizer, None),
    'tfidf__use_idf': (True, False),
    'multiclass__alpha': (0.5, 0.2, 2.0)
}

grid_output = GridSearchCV(multinom, parameters)

grid_output.fit(sentences_train, y_train)

grid_output.best_score_

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, grid_output.best_params_[param_name]))


### Another approach to stemming -- doesn't seem to be workig

import nltk
import string

def tokenize(text):
    stem = nltk.stem.SnowballStemmer('english')
    text = text.lower()

    for token in nltk.word_tokenize(text):
        if token in string.punctuation: continue
        yield stem.stem(token)


stem = nltk.stem.SnowballStemmer('english')
x = stem.stem(str(sentences_nosw))