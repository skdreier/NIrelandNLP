####################################################################################################################
#### Code examines various multi-class model and parameters for predicting classifications.
#### Basic code adapted from: https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
#### Added code: text stemming, pipelines to add hyperparmeters
#### 7 justification categories
#### Feb 28 2020
####################################################################################################################

import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
import nltk
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

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
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk import word_tokenize          
nltk.download('wordnet')

from IPython.display import display

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]
j_path = os.path.join(project_root) 

df = pd.read_csv(os.path.join(j_path, 'justifications_clean_text_ohe.csv'))

###################################################
### Reclassify justifications into 7 categories ###
###################################################

# Collapse justification categories from 12 to 7 -- approach #1
df['just_category_12'] = df['justification_cat']
df['just_category_7'] = df['justification_cat']
df['just_category_7'] = df['just_category_7'].replace('J_Intl-Domestic_Precedent', 'J_Denial')
df['just_category_7'] = df['just_category_7'].replace(['J_Utilitarian-Deterrence', 'J_Intelligence', 'J_Law-and-order', 'J_Development-Unity'], 'J_Outcome')
df['just_category_7'] = df['just_category_7'].replace('J_Last-resort', 'J_Emergency-Policy')
df['just_category_7'].unique()

# Collapse justification categories from 12 to 6 -- approach #2
df['just_category_6'] = df['justification_cat']
df['just_category_6'] = df['just_category_6'].replace(['J_Emergency-Policy', 'J_Intelligence', 'J_Last-resort', 'J_Utilitarian-Deterrence', 'J_Law-and-order'], 'J_Security')
df['just_category_6'] = df['just_category_6'].replace(['J_Legal_Procedure'], 'J_Legal')
df['just_category_6'] = df['just_category_6'].replace(['J_Political-Strategic'], 'J_Political')
df['just_category_6'] = df['just_category_6'].replace(['J_Denial', 'J_Intl-Domestic_Precedent'], 'J_HR_not_violated')
df['just_category_6'] = df['just_category_6'].replace(['J_Development-Unity'], 'J_Misc')
# Keep terrorism and Misc as-is
df['just_category_6'].unique()

# Collapse justification categories from 12 to 5 -- approach #3
df['just_category_5'] = df['justification_cat']
df['just_category_5'] = df['just_category_5'].replace(['J_Emergency-Policy', 'J_Intelligence', 'J_Last-resort', 'J_Utilitarian-Deterrence', 'J_Law-and-order', 'J_Terrorism'], 'J_Security')
df['just_category_5'] = df['just_category_5'].replace(['J_Legal_Procedure'], 'J_Legal')
df['just_category_5'] = df['just_category_5'].replace(['J_Political-Strategic'], 'J_Political')
df['just_category_5'] = df['just_category_5'].replace(['J_Denial', 'J_Intl-Domestic_Precedent'], 'J_HR_not_violated')
df['just_category_5'] = df['just_category_5'].replace(['J_Development-Unity'], 'J_Misc')
# Keep terrorism and Misc as-is
df['just_category_5'].unique()

## Security: Emergency-Policy, Intelligence, Last-resort, Utilitarian-Deterrence, Law-and-order,
## Legal: Legal_Procedure
## Political: Political-Strategic
## Terrorism: Terrorism
## HR_maintained: Denial, Intl-Domestic_Precedent
## Misc: Misc, Development-Unity

# Define which category size to use in analysis (12, 7, etc)
#df['just_categories'] = df['just_category_12']
#df['just_categories'] = df['just_category_7']
df['just_categories'] = df['just_category_6']
#df['just_categories'] = df['just_category_5']


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
plt.savefig('multiclass_LR/just_6' + '.png')
plt.show()

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

sentences = df['stem_text'].values # include stopwords, stemmed
#sentences = df['clean_text'].values # include stopwords, unstemmed

y = df['just_categories'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

type(sentences_train)
type(y_train)

from matplotlib import pyplot as plt 
import numpy as np  

a = df['just_categories']
plt.hist(sorted(a), bins = 11, normed=True) 
plt.title("hist_whole") 
plt.show()

a = y_train
plt.hist(sorted(a), bins = 11, normed=True) 
plt.title("hist_train") 
plt.show()

a = y_test
plt.hist(sorted(a), bins = 11, normed=True) 
plt.title("hist_test") 
plt.show()

############################################################
### Determine which machine learning models perform best ###
############################################################

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

# Plot accuracy outputs
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.savefig('multiclass_LR/models6' + '.png')
plt.show()

cv_df.groupby('model_name').accuracy.mean()

# Here, Logistic Regression performs the best (for both J=7 and J=6)

############################################
### Develop Pipeline for hyperparameters ###
############################################

###### Pipeline
multinom = Pipeline([
                #('vect', CountVectorizer(tokenizer=LemmaTokenizer())), # ngram_range=(1, 1), tokenizer = LemmaTokenizer
                ('vect2', TfidfVectorizer(sublinear_tf=True, norm='l2', encoding='latin-1')),
                #('tfidf', TfidfTransformer()),     # use_idf=True
                #('multiclass', MultinomialNB()),    # alpha=1.0 # smoothing parameter
                ('multiclass', LogisticRegression(random_state=0) )
               ])
help(LogisticRegression)
###### Look at outputs without parameters
multinom.fit(sentences_train, y_train)
y_pred = multinom.predict(sentences_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Accuracy for J=6: 0.449

####### Add hyperparameters: min_df, ngrams,  stop words, tfidf, alpha, lemma tokenizer, stemming, multi_class

# create dictionary {} of parameters
parameters = {
    #'vect__ngram_range': [(1,1),(1,4),(1,10)],
    'vect2__min_df': (0,3,5,10),
    'vect2__ngram_range': [(1,1), (2,2), (1,2)], 
    'vect2__stop_words': ('None', 'english'),
    'vect2__use_idf': (True, False),
#    'multiclass__alpha': (0.5, 0.2, 2.0)
    'multiclass__multi_class': ('ovr', 'multinomial') # ovr: one v. rest
}

grid_output = GridSearchCV(multinom, parameters) #5-fold cross-validation

grid_output.fit(sentences_train, y_train)

grid_output.best_score_

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, grid_output.best_params_[param_name]))

##### Best NB score and parameters for stemmed words:
# score: 0.41923
# alpha: 0.5
# Min_df: 10
# Ngram range: (1, 2) -- includes unigrams and bigrams
# Stopwords: english
# use_idf: False

##### Best NB score and parameters for unstemmed sentences:
# score: 0.42846
# alpha: 0.5
# Min_df: 5
# Ngram range: (1, 2) -- includes unigrams and bigrams
# Stopwords: english
# use_idf: False

##### Best LogisticRegression score and parameters for stemmed sentences:
# score: 0.4538
# alpha: NA
# Min_df: 3
# Ngram range: (1, 2) -- includes unigrams and bigrams
# Stopwords: english
# use_idf: False
# multi_class: ovr (one v. rest)

##### Best LogisticRegression score and parameters for stemmed sentences, J=6:
# score: 0.4808
# alpha: NA
# Min_df: 10
# Ngram range: (1, 2) -- includes unigrams and bigrams
# Stopwords: english
# use_idf: False
# multi_class: ovr (one v. rest)

##### Best LogisticRegression score and parameters for stemmed sentences, J=5:
# score: 0.5638
# alpha: NA
# Min_df: 10
# Ngram range: (1, 1)
# Stopwords: english
# use_idf: True
# multi_class: ovr (one v. rest)

#####################################################################################
#### Look at what is producing misclassifications among LogReg w best parameters ####
#####################################################################################

multinom_best = Pipeline([
                ('vect2', TfidfVectorizer(sublinear_tf=True, norm='l2', encoding='latin-1', min_df=3, ngram_range=(1,2), stop_words='english', use_idf=False)),
                ('multiclass', LogisticRegression(random_state=0, multi_class='ovr')) 
               ])

multinom_best.fit(sentences_train, y_train)
y_pred = multinom_best.predict(sentences_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(4,4))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.just_categories, yticklabels=category_id_df.just_categories)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

###### This code should pull the sentences that are getting most misclasified; not working #######
###### One issue may be that it works w NB and not LogReg but this isn't the only issue. #########
for predicted in category_id_df.category_id:
    for actual in category_id_df.category_id:
        if predicted != actual and conf_mat[actual, predicted] >= 10:
            print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
            display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['just_categories', 'stem_text']])
            print('')
            


################################################################################
#### Use chi-squared test to find terms most correlated with each category: ####
################################################################################

# make a write-up of the interative / inductive / mixed-methods approach to machine learning that we can use based on these outputs:

model = LogisticRegression()
model.fit(features, labels)

N = 5
for just, category_id in sorted(category_to_id.items()):
  indices = np.argsort(model.coef_[category_id])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  print("# '{}':".format(just))
  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))

from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names=df['just_categories'].unique()))


# THIS WEEKEND:
# UPDATE categories based on today's outputs Especially since Legal Procedure captures all of them
# Try to encorporate word2vec based on article I was reading through; take notes on where things break.
# Feature selection (currently each word in dict is OHE; feature selection trims the features input into the model, but maybe not necessary for us -- pipeline is fast)
## Friday: Simple multi-class neuro network implementation as a baseline
### Broad goal: get over 50 percent (better than flipping a coin)


####################################################################################################################
### NOTES / Code that didn't work
####################################################################################################################

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