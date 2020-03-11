## Gensim how-to and test with subset 
import pandas as pd
import os
import numpy as np
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]
data_root = os.path.split(project_root)[0]

sys.path.append(project_root)
import preprocess as pr

path_corpus = os.path.join(data_root, "NI_docs/") 

## Load txt document file names
ocr_text = pr.text_preprocess(path_corpus)
ocr_text.files #6946

# run through function to gather all text (as dictionary)
ocr_text_corpus = ocr_text.nvivo_ocr()

# Convert to Dataframe
ocr_corpus = pd.DataFrame(ocr_text_corpus.items())
ocr_corpus.columns = ['img_file', 'raw_text']

# Subset to pages that contain a justification
df = pd.read_csv(os.path.join(project_root, 'justifications_clean_text_ohe.csv'))
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
    #new_col = new_col.replace(r'\b\w{1,2}\b', '', regex=True)
    df['clean_text'] = new_col
    return(df)

corpus_clean = clean_func(corpus['raw_text'], corpus) 

corpus_text = list(corpus_clean['clean_text'])

tokenized_docs = [word_tokenize(i) for i in corpus_text]
for i in tokenized_docs:
    print(i)

model= Word2Vec(tokenized_docs, min_count=2) # this should train the model

words = list(model.wv.vocab) # this should show the words in the model, but it is showing characters rather than words
print(sorted(words))
print(model['terrorism'])
len(words)
model.wv.save_word2vec_format('archive_corpus_w2v_model.bin')
#model.wv.save_word2vec_format('archive_corpus_w2v_model.txt', binary=false) # Saved as ASCII format to view contents
# Then to use later:
# model = Word2Vec.load('archive_corpus_w2v_model.bin')
print(model)

model.most_similar(positive=['terrorism'], topn=5)
model.most_similar(positive=['ira'], topn=5)
model.most_similar(positive=['hmg'], topn=5)
model.most_similar(positive=['internment'], topn=5)
model.most_similar(positive=['faulkner'], topn=5)

model.most_similar(positive=['ulster'], topn=5)
model.most_similar(positive=['westminster'], topn=5)
model.most_similar(positive=['stormont'], topn=5)

model.most_similar(positive=['protestant'], topn=5)
model.most_similar(positive=['catholic'], topn=5)

### Visualize word embeddings
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
token = 'internment'
token_lst = pd.DataFrame(model.most_similar(positive=token, topn=25))[0]
token_lst = token_lst.append(pd.Series([token]))
df_token = df.loc[df['word'].isin(token_lst)]

ax = df_token.plot(kind='scatter', x='pca1', y='pca2')
df_token[['pca1','pca2','word']].apply(lambda row: ax.text(*row),axis=1);




###########################################################################################
###########################################################################################
###### FULL CORPUS does better than the subsetted corpus for the words tested below #######
###########################################################################################


model_full = model
words = list(model_full.wv.vocab) # this should show the words in the model, but it is showing characters rather than words
print(sorted(words))
len(words)

model_subset = model
words = list(model_subset.wv.vocab) # this should show the words in the model, but it is showing characters rather than words
print(sorted(words))
len(words)

print(model['terrorism']) # will show embeddings 
print(model)

model.most_similar(positive=['terrorism'], topn=5)
model.most_similar(positive=['ira'], topn=5)
model.most_similar(positive=['hmg'], topn=5)
model.most_similar(positive=['internment'], topn=5)
model.most_similar(positive=['faulkner'], negative=['england'], topn=1)



model_full.most_similar(positive=['terrorism'], topn=5)
model_subset.most_similar(positive=['terrorism'], topn=5)

model_full.most_similar(positive=['ira'], topn=5)
model_subset.most_similar(positive=['ira'], topn=5)

model_full.most_similar(positive=['hmg'], topn=5)
model_subset.most_similar(positive=['hmg'], topn=5)

model_full.most_similar(positive=['internment'], topn=5)
model_subset.most_similar(positive=['internment'], topn=5)

model_full.most_similar(positive=['faulkner'], topn=5)
model_subset.most_similar(positive=['faulkner'], topn=5)

model_full.most_similar(positive=['ulster'], topn=5)
model_full.most_similar(positive=['westminster'], topn=5)
model_full.most_similar(positive=['stormont'], topn=5)
