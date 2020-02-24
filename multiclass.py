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
from sklearn.metrics import plot_confusion_matrix

from nltk.tokenize import word_tokenize
nltk.download('stopwords')

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]
j_path = os.path.join(project_root) 

df_just = pd.read_csv(os.path.join(j_path, 'justifications_clean_text_ohe.csv'))

# Create a unique number id for each justification category
from io import StringIO
col = ['justification_cat', 'clean_text'] 
df = df_just[col]
df = df[pd.notnull(df['clean_text'])]
#df.columns = ['justification_cat', 'clean_text'] # this line not necessary
df['category_id'] = df['justification_cat'].factorize()[0]
category_id_df = df[['justification_cat', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'justification_cat']].values)
df.head

# Collapse justification categories from 12 to 7
df['category_id2'] = df['justification_cat']

df['category_id2'] = df['category_id2'].replace('J_Intl-Domestic_Precedent', 'J_Denial')
df['category_id2'] = df['category_id2'].replace(['J_Utilitarian-Deterrence', 'J_Intelligence', 'J_Law-and-order', 'J_Development-Unity'], 'J_Outcome')
df['category_id2'] = df['category_id2'].replace('J_Last-resort', 'J_Emergency-Policy')

df['category_id2'].unique()

### If you want to look at aggregate categories rather than original, rename as follows:
df['justification_cat'] = df['category_id2']

df['category_id'] = df['justification_cat'].factorize()[0]
category_id_df = df[['justification_cat', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'justification_cat']].values)
df.head

###### Function to remove stopwords (optional) ######
def rmv_stopwords(sent):
        STOPWORDS = set(stopwords.words("english"))
        sent = [' '.join(word for word in x.split() if word not in STOPWORDS) for x in sent.tolist()]
        return sent

sentences_nosw = rmv_stopwords(clean_sentences)

sentences = pd.Series(sentences_nosw).values # exclude stopwords 
# sentences = df['clean_text'].values # include stopwords
y = df['justification_cat'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

multinom = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('multiclass', MultinomialNB()),
               ])

classifier = multinom.fit(sentences_train, y_train)

y_pred = multinom.predict(sentences_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

## Visualize confusion matrix
np.set_printoptions(precision=2)

# Plot non-normalized and normalized confusion matrix

titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, sentences_test, y_test,
                                 #display_labels=id_to_category,
                                 display_labels=category_to_id,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    plt.xticks(np.arange(0, len(category_to_id)), category_to_id, rotation=60, ha='right')

    print(title)
    print(disp.confusion_matrix)

    #plt.savefig('multiclass_NB/confusion_matrix12_' + title + '.png')
    plt.savefig('multiclass_NB/confusion_matrix7_' + title + '.png')
    plt.close()