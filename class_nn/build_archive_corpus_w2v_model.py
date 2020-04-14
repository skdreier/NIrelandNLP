######################################################
### Builds Word vector model based on archive data ###
### Inc: "cleaning" tactics (probably unnecessary) ###
### Examines / visualizes model output             ###
### Uses: NI_docs, preprocessing.py                ###
### Creates: archive_corpus_w2v_model.bin          ###
######################################################

## Gensim how-to and test with subset 
import pandas as pd
import os, sys
import numpy as np
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

this_file_path = os.path.abspath(__file__)
folder_root = os.path.split(this_file_path)[0]
repo_root = os.path.split(folder_root)[0]

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocess as pr

path_corpus = os.path.join(repo_root, "orig_text_data/NI_docs/") 

## Load txt document file names
ocr_text = pr.text_preprocess(path_corpus)
ocr_text.files #6946

# run through function to gather all text (as dictionary)
ocr_text_corpus = ocr_text.nvivo_ocr()

# Convert to Dataframe
ocr_corpus = pd.DataFrame(ocr_text_corpus.items())
ocr_corpus.columns = ['img_file', 'raw_text']

# Subset to pages that contain a justification
df = pd.read_csv(os.path.join(repo_root, 'justifications_clean_text_ohe.csv'))
just_imgs = np.ndarray.tolist(df['img_file_orig'].unique())
ocr_corpus_subset = ocr_corpus.loc[ocr_corpus['img_file'].isin(just_imgs)]

# Define whether you want to use whole corpus or subset to text w justifications
corpus = ocr_corpus
#corpus = ocr_corpus_subset

# Function to clean text
def clean_func(column, df):
    new_col = column.str.lower()
    new_col = new_col.replace(r"\n", " ", regex=True)
    #new_col = new_col.replace(r"[^0-9a-z #+_]", "", regex=True)
    new_col = new_col.replace(r"[^a-z #+_]", " ", regex=True)
    new_col = new_col.replace(r"#", " ", regex=True)
    new_col = new_col.replace(r'\b\w{1}\b', '', regex=True) # remove one-letter words (loses "I" and "a")
    #new_col = new_col.replace(remove_word_set, '', regex=True) # remove identified words
    df['clean_text'] = new_col
    return(df)

corpus_clean = clean_func(corpus['raw_text'], corpus) 

corpus_text = list(corpus_clean['clean_text'])

tokenized_docs = [word_tokenize(i) for i in corpus_text]
for i in tokenized_docs:
    print(i)

# save model w/out any cleaning
model= Word2Vec(tokenized_docs, min_count=1) 
len(list(model.wv.vocab))
model_name = 'archive_corpus_embedding_w2v_big.txt'
model.wv.save_word2vec_format(model_name, binary=False)
# 

### Cleaning step 1: Remove words from the model that appear only once or twice in the corpus (Original yield: 15567 words)
model= Word2Vec(tokenized_docs, min_count=3) 
len(list(model.wv.vocab))

### Cleaning step 2: Look at the most frequently occurring 1000 words and remove non-word "noise"; rerun model 
model.wv.index2entity[:1000]
    # non-word "words" to keep (abbreviations, acronyms, incomplete but informative, etc.): th, int, pr, sep, prot, pd, vsi, col, ni, cgs, tho, fco, gsw, goc, rc
    # non-word "words" to remove:
remove_word_set = ['bn', 'er', 'pte', 'si', 'ar', 'rt', 'il', 'al', 'ir', 'mod', 'regt', 'nd', 'mil', 'asd', 'br', 'ad', 'ra', 'sqn', 'bde', 'baor', 'li', 'gs', 'lt', 'ma', 'iii', 'ti', 'te', 'ds', 'rd', 'ii']

tokenized_docs2 = [[word for word in sub if word not in remove_word_set] for sub in tokenized_docs] 

model= Word2Vec(tokenized_docs2, min_count=3) 

words = list(model.wv.vocab) 
print(sorted(words))
len(words) #15537

### Cleaning step 3: Look at output among words most similar to the substantively most relevant words

# Function to look at most similar words
def most_sim(keyword, num):
    return(model.most_similar(positive=[keyword], topn=num))
most_sim('internment', 5)

# Loop to print most similar words for a list
keyword_list = ['internment', 'intern', 'interrogation', 'interrogate', 'detention', 'detain', 'security', 
'emergency', 'special', 'powers', 'spa', 'parliament', 'policy', 'political', 'terrorism', 'terrorists', 'ira',
'hmg', 'faulkner', 'protestant', 'catholic', 'ulster', 'westminster', 'stormont', 'britain', 'london',
'belfast', 'ireland', 'england', 'northern', 'republican', 'loyalist']

def most_sim_list(keyword, num):
    new_list = []
    for word in keyword_list:
        new_list.append(model.most_similar(positive=word, topn=num))
    return(new_list)
    
most_sim_list(keyword_list, 25)

import re
tokenized_docs3 = [[re.sub('(irel|irland|reland|iroland|ieland|irelan|irelnd|roland|irelandnd|tireland|irelandan|irelandanid)','ireland', x) for x in i] for i in tokenized_docs2]
tokenized_docs3 = [[re.sub('(orthern|norther|ofnorthern|nothern|ncrthern|innorthern|rthern|thenorthern|northrn|nortern|northen|northernn|northorn|northernnn|xnorthern)','northern', x) for x in i] for i in tokenized_docs3]

model= Word2Vec(tokenized_docs3, min_count=3) 

words = list(model.wv.vocab) 
print(sorted(words))
len(words) #15522

# Save archive word2vec model:
model.wv.save_word2vec_format('archive_corpus_w2v_model.bin')
#model.wv.save_word2vec_format('archive_corpus_w2v_model.txt', binary=false) # Saved as ASCII format to view contents
# Then to use later:
# model = Word2Vec.load('archive_corpus_w2v_model.bin')
print(model)
words = list(model.wv.vocab)
print(sorted(words))
len(words)

print(model['terrorism']) # will show embeddings 
model.wv.most_similar('terrorism')

# save model
model_name = 'archive_corpus_embedding_w2v.txt'
model.wv.save_word2vec_format(model_name, binary=False)
# 


most_sim('faulkner', 5)
most_sim('interrogation', 5)
most_sim('internment', 5)
most_sim('terrorism', 5)

print(model['terrorism'])

# other words to remove: thesecurity, ander, securi, rmoved, millen, hewas, rees, nio, pss, fco, irelandoffice
# irelandgovernment, ofth, northernireland, aoc, ofth, cesa, toal, theruc
# "words" to keep: sf, eec, jsc

##################################################
### Visualize word embeddings
##################################################

from sklearn.decomposition import PCA
from matplotlib import pyplot

X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
pyplot.scatter(result[:, 0], result[:, 1])

# Plot scatterplot for words most similar to Internment

# Build dataframe w 2-dim PCA and words
words = list(model.wv.vocab)
df = pd.DataFrame(
    {'pca1': result[:, 0],
    'pca2': result[:, 1],
    'word': words
    })

# Pull words most associated w Internment
token = 'faulkner'
token_lst = pd.DataFrame(model.most_similar(positive=token, topn=25))[0]
token_lst = token_lst.append(pd.Series([token]))
df_token = df.loc[df['word'].isin(token_lst)]

ax = df_token.plot(kind='scatter', x='pca1', y='pca2')
df_token[['pca1','pca2','word']].apply(lambda row: ax.text(*row),axis=1);

