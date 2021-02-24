import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from prep_data import extract_and_tag_next_document


def get_doc2doc_similarity_scores(doctext_fname_tuples):
    # get full vocabulary over all documents
    cv = CountVectorizer()
    matrix = cv.fit_transform([tup[0] for tup in doctext_fname_tuples])

    # normalize each vector to have a Euclidean norm of 1
    if matrix.shape[0] == len(doctext_fname_tuples):
        matrix = normalize(matrix, norm='l2', axis=1)
    elif matrix.shape[1] == len(doctext_fname_tuples):
        matrix = normalize(matrix, norm='l2', axis=0)
    else:
        assert False

    # multiply that matrix by its transpose and return the resulting matrix
    return np.multiply(matrix, np.transpose(matrix))


def make_file_of_tagpairs_to_scores(similarity_score_matrix, corresponding_tag_list, filename):
    print('Starting to make ' + filename)
    assert similarity_score_matrix.shape[0] == len(corresponding_tag_list) and \
           similarity_score_matrix.shape[1] == len(corresponding_tag_list)

    ordered_list = []
    for i in range(len(corresponding_tag_list)):
        for j in range(len(corresponding_tag_list)):
            if j >= i:
                break
            ordered_list.append((float(similarity_score_matrix[i, j]), (i, j)))
    ordered_list = sorted(ordered_list, key=lambda x: x[0], reverse=True)
    with open(filename, 'w') as f:
        for score, indextup in ordered_list:
            f.write(str(score) + ': ' + corresponding_tag_list[indextup[0]] + ' to ' +
                    corresponding_tag_list[indextup[1]] + '\n')


def main():
    full_doc_fname = '../orig_text_data/internment.txt'
    output_filename = 'document_bow_cosinesimilarity.txt'

    previously_extracted_header = None
    tags_to_documents = {}
    with open(full_doc_fname, 'r', encoding='utf-8-sig') as f:
        keep_going = True
        while keep_going:
            document, tag, previously_extracted_header = \
                extract_and_tag_next_document(f, previously_extracted_header=previously_extracted_header)
            if document is None:
                keep_going = False
            else:
                tags_to_documents[tag] = document

    document_text_filename_tuples = [(doc, tag[0] + '/' + tag[1]) for tag, doc in tags_to_documents.items()]
    make_file_of_tagpairs_to_scores(get_doc2doc_similarity_scores(document_text_filename_tuples),
                                    [tup[1] for tup in document_text_filename_tuples],
                                    output_filename)


if __name__ == '__main__':
    main()
