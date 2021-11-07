import pandas as pd
import re
from string import whitespace, punctuation
from util import make_directories_as_necessary
import os
import numpy as np
import datetime
from tqdm import tqdm
from gensim.models import Word2Vec


lowercase_all_text = False
source_fname = 'data/binary_mindsduplicates_withcontext_linesupwithmultiway_withfilename_BIGGERDATA_train.csv'

output_dir = 'data/biggerdata_'
if 'linesupwithmultiway' in source_fname:
    output_dir += 'linesupwithmultiway_'
if lowercase_all_text:
    output_dir += 'lowercase_'
else:
    output_dir += 'fullcase_'
output_dir += 'pretrainedword2vec/'
make_directories_as_necessary(output_dir + 'something.txt')
train_df = pd.read_csv(source_fname)


def train_and_save_word_embeddings(training_data_df, directory_for_filename,
                                   inds_to_words, total_num_sentences, unk_ind, words_to_inds, word_embedding_dimension,
                                   iterations_for_training_word_embeddings, unk_token_during_embedding_training,
                                   lowercase_all_text, regular_exp, run_check=False):
    print("Starting to put together word embeddings at " + str(datetime.datetime.now()))
    sentence_iterator = SentenceIterator(training_data_df, words_to_inds, unk_token_during_embedding_training,
                                         lowercase_all_text, regular_exp=regular_exp)
    trained_model = Word2Vec(None, iter=iterations_for_training_word_embeddings,
                             min_count=0, size=word_embedding_dimension, workers=4)
    print("Starting to build vocabulary in gensim Word2Vec model at " + str(datetime.datetime.now()))
    trained_model.build_vocab(sentence_iterator)
    print("Starting to train Word2Vec embeddings at " + str(datetime.datetime.now()))
    trained_model.train(sentence_iterator, total_examples=total_num_sentences,
                        epochs=iterations_for_training_word_embeddings)

    if run_check:
        assert len(trained_model.wv.vocab) == len(inds_to_words) + 1  # since inds_to_words doesn't include unk
        part_to_query = trained_model.wv
        print('Most similar to Faulkner:')
        try:
            print(part_to_query.most_similar(positive=['faulkner' if lowercase_all_text else 'Faulkner'],
                                             negative=[])[:10])
        except KeyError:
            print(('faulkner' if lowercase_all_text else 'Faulkner') + ' not in vocabulary')
        print('Most similar to arrest:')
        try:
            print(part_to_query.most_similar(positive=['arrest'], negative=[])[:10])
        except KeyError:
            print('arrest not in vocabulary')
        print('Most similar to provisional:')
        try:
            print(part_to_query.most_similar(positive=['provisional'], negative=[])[:10])
        except KeyError:
            print('provisional not in vocabulary')


    temp_filename = os.path.join(directory_for_filename, "_tempgensim")
    trained_model.save(temp_filename)

    print("Starting to move trained embeddings into numpy matrix at " + str(datetime.datetime.now()))
    num_vocab_words = len(inds_to_words)
    embedding_matrix = np.zeros((num_vocab_words + 2, word_embedding_dimension))
    for word_ind in tqdm(inds_to_words.keys(), total=len(inds_to_words)):
        assert word_ind != 0, 'We later assume the index 0 corresponds to padding'
        embedding_matrix[word_ind] = trained_model[inds_to_words[word_ind]]
    embedding_matrix[unk_ind] = trained_model[unk_token_during_embedding_training]
    norm_of_embeddings = np.linalg.norm(embedding_matrix, axis=1)
    norm_of_embeddings[norm_of_embeddings == 0] = 1e-13
    embedding_matrix = embedding_matrix / norm_of_embeddings[:, None]
    np.save(os.path.join(directory_for_filename, 'word2vec_embeddings.npy'), embedding_matrix)
    print('Saved trained embeddings to ' + os.path.join(directory_for_filename, 'word2vec_embeddings.npy'))

    print("Removing temporary gensim model files at " + str(datetime.datetime.now()))
    # remove gensim model files, now that embedding matrix has been saved
    if os.path.isfile(temp_filename):
        os.remove(temp_filename)
    if os.path.isfile(temp_filename + ".syn1neg.npy"):
        os.remove(temp_filename + ".syn1neg.npy")
    if os.path.isfile(temp_filename + ".wv.syn0.npy"):
        os.remove(temp_filename + ".wv.syn0.npy")


def tokenize_sentence(sentence_as_string, regular_exp=None):
    if regular_exp is None:
        delimiters = [char for char in whitespace]
        regular_exp = '|'.join(map(re.escape, delimiters))
    sentence_pieces = re.split(regular_exp, sentence_as_string.strip())
    tokens_to_return = []
    for piece in sentence_pieces:
        if len(piece) == 0:
            continue
        # if it's punctuation on the end of a token, then split it off and make it its own token
        cur_ind = 0
        while cur_ind < len(piece) and piece[cur_ind] in punctuation:
            cur_ind += 1

        if cur_ind == len(piece):
            if len(piece.strip()) > 0:
                tokens_to_return.append(piece)  # the whole thing is punctuation
            continue
        else:
            piece_to_append = piece[:cur_ind].strip()
            if len(piece_to_append) > 0:
                tokens_to_return.append(piece_to_append)

        first_ind_of_non_punctuation = cur_ind

        cur_ind = len(piece) - 1
        while cur_ind > first_ind_of_non_punctuation and piece[cur_ind] in punctuation:
            cur_ind -= 1

        piece_to_append = piece[first_ind_of_non_punctuation: cur_ind + 1].strip()
        if len(piece_to_append) > 0:
            tokens_to_return.append(piece_to_append)
        if cur_ind < len(piece) - 1:
            piece_to_append = piece[cur_ind + 1:].strip()
            if len(piece_to_append) > 0:
                tokens_to_return.append(piece_to_append)

    return tokens_to_return


def develop_vocabulary(dataframe_with_data, lowercase_all_text, regular_exp=None, cutoff_for_being_in_vocab = 4):
    vocab_in_progress = {}
    total_num_sentences = 0
    for i, row in dataframe_with_data.iterrows():
        text_in_sentence = row['text']
        tokens_in_sent = tokenize_sentence(text_in_sentence, regular_exp=regular_exp)
        for token in tokens_in_sent:
            if lowercase_all_text:
                token = token.lower()
            if token not in vocab_in_progress:
                vocab_in_progress[token] = 1
            else:
                vocab_in_progress[token] += 1
        if len(tokens_in_sent) > 0:
            total_num_sentences += 1
    words_to_inds = {}
    inds_to_words = {}
    unk_token = '<unk>'
    next_available_ind = 1  # we don't start with 0 because that's reserved for padding
    for tokentype, count in vocab_in_progress.items():
        if count >= cutoff_for_being_in_vocab:
            words_to_inds[tokentype] = next_available_ind
            inds_to_words[next_available_ind] = tokentype
            next_available_ind += 1
    while unk_token in words_to_inds:
        unk_token = '<' + unk_token + '>'
    return words_to_inds, inds_to_words, unk_token, total_num_sentences


class SentenceIterator:
    def __init__(self, dataframe_with_data, words_to_inds, unk_token_during_embedding_training, lowercase,
                 regular_exp):
        self.df = dataframe_with_data
        self.words_to_inds = words_to_inds
        self.unk_token_string = unk_token_during_embedding_training
        self.lowercase = lowercase
        self.regular_exp = regular_exp

    def __iter__(self):
        for i, row in self.df.iterrows():
            text_in_sentence = row['text']
            tokens_in_sent = tokenize_sentence(text_in_sentence, regular_exp=self.regular_exp)
            inds_to_unk = []
            if self.lowercase:
                tokens_in_sent = [token.lower() for token in tokens_in_sent]
            for i, token in enumerate(tokens_in_sent):
                if token not in self.words_to_inds:
                    inds_to_unk.append(i)
            for ind in inds_to_unk:
                tokens_in_sent[ind] = self.unk_token_string
            if len(tokens_in_sent) > 0:
                yield tokens_in_sent

    def __call__(self, *args, **kwargs):
        return iter(self)


def save_vocab_index(inds_to_words, directory, unk_token, unk_ind):
    fname = os.path.join(directory, 'vocab_in_index_order.txt')
    make_directories_as_necessary(fname)
    assert unk_ind == len(inds_to_words) + 1
    with open(fname, 'w') as f:
        for i in range(1, len(inds_to_words) + 1):
            f.write(inds_to_words[i] + '\n')
        f.write(unk_token + '\n')
    print('Saved vocab index to ' + fname)


delimiters = [char for char in whitespace]
regular_exp = '|'.join(map(re.escape, delimiters))
words_to_inds, inds_to_words, unk_token, total_num_sentences_in_train = \
    develop_vocabulary(train_df, lowercase_all_text, cutoff_for_being_in_vocab=4, regular_exp=regular_exp)
unk_ind = len(words_to_inds) + 1
save_vocab_index(inds_to_words, output_dir, unk_token, unk_ind)
train_and_save_word_embeddings(train_df, output_dir, inds_to_words, total_num_sentences_in_train,
                               unk_ind, words_to_inds, 100,
                               50, unk_token, lowercase_all_text,
                               regular_exp=regular_exp, run_check=True)
