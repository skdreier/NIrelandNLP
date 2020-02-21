import os
from pathlib import Path
import re
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]
j_path = os.path.join(project_root) 

df_just = pd.read_csv(os.path.join(j_path, 'justifications_clean_text_ohe.csv'))

df_just.columns

from io import StringIO
col = ['justification_cat', 'clean_text'] 
df = df_just[col]
df = df[pd.notnull(df['clean_text'])]
df.columns = ['justification_cat', 'clean_text']
df['category_id'] = df['justification_cat'].factorize()[0]
category_id_df = df[['justification_cat', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'justification_cat']].values)
df.head

sentences = df['clean_text'].values
y = df['justification_cat'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

multinom = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('multiclass', MultinomialNB()),
               ])

multinom.fit(sentences_train, y_train)

y_pred = multinom.predict(sentences_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

## Visualize confusion matrix