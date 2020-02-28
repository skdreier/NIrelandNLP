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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from io import StringIO

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')


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
#df['just_categories'] = df['just_category_12']
df['just_categories'] = df['just_category_7']

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

fig = plt.figure(figsize=(8,6))
df.groupby('just_categories').clean_text.count().plot.bar(ylim=0)
plt.show()


######################################
### Stem sentences outside of grid ###
######################################

from nltk.stem import PorterStemmer
ps = PorterStemmer()

def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

df['stem_text'] = df['clean_text'].apply(stem_sentences)

####### Divide into training and testing data
sentences = df['stem_text'].values # include stopwords, stemmed
#sentences = df['clean_text'].values # include stopwords, unstemmed

y = df['just_categories'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

###### Pipeline
multinom = Pipeline([
                #('vect', CountVectorizer(tokenizer=LemmaTokenizer())), # ngram_range=(1, 1), tokenizer = LemmaTokenizer
                ('vect2', TfidfVectorizer(sublinear_tf=True, norm='l2', encoding='latin-1')),
                #('tfidf', TfidfTransformer()),     # use_idf=True
                ('multiclass', MultinomialNB()),    # alpha=1.0 # smoothing parameter
               ])

###### Look at outputs without parameters
multinom.fit(sentences_train, y_train)
y_pred = multinom.predict(sentences_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


####### Add hyperparameters: min_df, ngrams,  stop words, tfidf, alpha, lemma tokenizer, stemming

# create dictionary {} of parameters
parameters = {
    #'vect__ngram_range': [(1,1),(1,4),(1,10)],
    'vect2__min_df': (0,3,5,10),
    'vect2__ngram_range': [(1,1), (2,2), (1,2)], 
    'vect2__stop_words': ('None', 'english'),
    'vect2__use_idf': (True, False),
    'multiclass__alpha': (0.5, 0.2, 2.0)
}

grid_output = GridSearchCV(multinom, parameters)

grid_output.fit(sentences_train, y_train)

grid_output.best_score_

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, grid_output.best_params_[param_name]))

##### Best score and parameters for stemmed sentences:
# score: 0.41923
# alpha: 0.5
# Min_df: 10
# Ngram range: (1, 2) -- includes unigrams and bigrams
# Stopwords: english
# use_idf: False

##### Best score and parameters for unstemmed sentences:
# score: 0.42846
# alpha: 0.5
# Min_df: 5
# Ngram range: (1, 2) -- includes unigrams and bigrams
# Stopwords: english
# use_idf: False


#### Look at different machine learning models ####

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.clean_text).toarray()
labels = df.just_categories
features.shape

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []

for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

cv_df.groupby('model_name').accuracy.mean()

##################################################################
# Multinomial NB, LogReg, and Linear all perform well
# Now look at what is producing misclassifications among Multinomial NB
##################################################################

model = MultinomialNB()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.just_categories, yticklabels=category_id_df.just_categories)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

from IPython.display import display

# attempted correction
for predicted in df.category_id:
    for actual in df.category_id:
        if predicted != actual and conf_mat[actual, predicted] >= 10:
            print("'{}' predicted as '{}' : {} examples.".format(category_[actual], id_to_category[predicted], conf_mat[actual, predicted]))
            display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['just_categories', 'stem_text']])
            print('')

# original code
for predicted in category_id_df.category_id:
    for actual in category_id_df.category_id:
        if predicted != actual and conf_mat[actual, predicted] >= 10:
            print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
            display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['just_categories', 'stem_text']])
            print('')
            

# THIS WEEK:
# Do grid search on stemming with ngrams (ADD STEMMING OUTSIDE THE GRID)
# Not sure if this is necessary: Work on getting rid of more categories; think about categorization substantively. Especially since Legal Procedure captures all of them
# Try to encorporate word2vec based on article I was reading through; take notes on where things break.
# Feature selection within a Pipeline
# https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
## Friday: Simple multi-class neuro network implementation as a baseline
### Broad goal: get over 50 percent (better than flipping a coin)



### NOTES


# Look at outputs: CountVectorizer v. TfidfVectorizer
x = CountVectorizer(tokenizer=LemmaTokenizer()) 
y = x.fit_transform(sentences_nosw)
y = pd.DataFrame(y.toarray(), columns=x.get_feature_names())
print(y.columns.tolist())
len(y.columns.tolist())

x = TfidfVectorizer(sublinear_tf=True, norm='l2', encoding='latin-1', min_df=5, ngram_range=(1, 2), stop_words='english')
y = x.fit_transform(df['clean_text'])
y = pd.DataFrame(y.toarray(), columns=x.get_feature_names())
print(y.columns.tolist())
len(y.columns.tolist())
# Here, ngram_range=(1,2) looks at one-unit and two-unit matches
# min_df=5 really dramatically drops out words.

###### Function to remove stopwords ######
def rmv_stopwords(sent):
    STOPWORDS = set(stopwords.words("english"))
    sent = [' '.join(word for word in x.split() if word not in STOPWORDS) for x in sent.tolist()]
    return sent

sentences_nosw = rmv_stopwords(df['clean_text'])
sentences_nosw = pd.Series(sentences_nosw).values # exclude stopwords 


## Add lemmonizer option for Pipeline (more sophisticated than stem words)
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

TfidfVectorizer(tokenizer=LemmaTokenizer)

## Add stemming option for Pipeline

stemmer = nltk.stem.SnowballStemmer('english')

class StemmedVectorizer:
    def build_analyzer(self):
        analyzer = super(StemmedVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

TfidfVectorizer(analyzer=StemmedVectorizer)

parameters = {
    #'vect__ngram_range': [(1,1),(1,4),(1,10)],
    'vect2__min_df': (0,3,5,10),
    'vect2__ngram_range': [(1,1), (2,2), (1,2)], 
    'vect2__stop_words': ('None', 'english'),
    'vect2__use_idf': (True, False),
    'multiclass__alpha': (0.5, 0.2, 2.0),
    'vect2__tokenizer': (None, LemmaTokenizer),
    'vect2__analyzer': (None, stemmed_words)
}