# T-sne
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from sklearn.manifold import TSNE
import numpy as np

from gensim.models import KeyedVectors

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(os.path.split(this_file_path)[0])[0]

w_vec_model = KeyedVectors.load_word2vec_format(os.path.join(project_root, "class_nn/archive_corpus_embedding_w2v.txt"), binary=False)

w_vec_model.vocab
len(w_vec_model.vocab)

keys = ['terrorism', 'faulkner', 'internment', 'arrest', 'catholic', 'protest']

embedding_clusters = []
word_clusters = []
for word in keys:
    embeddings = []
    words = []
    for similar_word, _ in w_vec_model.most_similar(word, topn=10):
        words.append(similar_word)
        embeddings.append(w_vec_model[similar_word])
    embedding_clusters.append(embeddings)
    word_clusters.append(words)

embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape

tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)


def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename):
    plt.figure(figsize=(16, 9))
    colors = cm.Dark2(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=12)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()


tsne_plot_similar_words('Archive Corpus Word Embeddings', keys, embeddings_en_2d, word_clusters, 0.7,
                        'similar_words.png')
