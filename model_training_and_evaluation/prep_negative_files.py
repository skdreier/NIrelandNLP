import pandas as pd
import numpy as np
from copy import deepcopy
from group_duplicates_for_split import get_doc2doc_similarity_scores
from prep_data import extract_file_image_tag_from_relevant_part_of_header_string, get_sentence_split_inds, \
    make_classification_split, save_splits_as_csv, fix_df_format


dataframe = pd.read_csv('../justifications_clean_text_ohe.csv')
all_fnames_currently_in_data = set(dataframe['img_file_orig'])
with open('../../OCRdata/NI_docs/negative_filenames_also_in_current_data.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line != '':
            all_fnames_currently_in_data.add(line)
for fname in all_fnames_currently_in_data:
    assert '/' not in fname

complete_list_of_all_filenames = []
fname_to_slashformfname = {}
slashformfname_to_fname = {}
with open('../../OCRdata/NI_docs/all_txt_filenames.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line.endswith('.txt'):
            line = line[:line.rfind('.txt')]
        if line != '':
            complete_list_of_all_filenames.append(line)
            assert '/' not in line
            slashform = extract_file_image_tag_from_relevant_part_of_header_string(line)
            slashform = slashform[0] + '/' + slashform[1]
            fname_to_slashformfname[line] = slashform
            slashformfname_to_fname[slashform] = line


def get_list_of_all_filenames_in_data_file(fname):
    df = pd.read_csv(fname)
    list_in_slashformat = list(set(list(df['filename'])))
    return [slashformfname_to_fname[item] for item in list_in_slashformat]


cur_train_fnames = get_list_of_all_filenames_in_data_file('data/binary_mindsduplicates_withcontext_train.csv')
cur_dev_fnames = get_list_of_all_filenames_in_data_file('data/binary_mindsduplicates_withcontext_dev.csv')
cur_test_fnames = get_list_of_all_filenames_in_data_file('data/binary_mindsduplicates_withcontext_test.csv')


complete_list_of_all_filenames_reordered = []
filenames_found_to_not_be_in_data = 0
print('Length of complete_list_of_all_filenames: ' + str(len(complete_list_of_all_filenames)))
for fname in complete_list_of_all_filenames:
    if fname not in all_fnames_currently_in_data:
        filenames_found_to_not_be_in_data += 1
        complete_list_of_all_filenames_reordered.append(fname)
print('Number of those filenames not occurring in the current data: ' + str(filenames_found_to_not_be_in_data))
all_fnames_not_currently_accounted_for_in_data = deepcopy(complete_list_of_all_filenames_reordered)
set_of_all_fnames_not_currently_accounted_for_in_data_slashform = \
    set([extract_file_image_tag_from_relevant_part_of_header_string(fname)[0] + '/' +
         extract_file_image_tag_from_relevant_part_of_header_string(fname)[1]
         for fname in all_fnames_not_currently_accounted_for_in_data])
new_fname_endindplus1 = len(complete_list_of_all_filenames_reordered)
complete_list_of_all_filenames_reordered += cur_train_fnames
train_fname_endindplus1 = len(complete_list_of_all_filenames_reordered)
complete_list_of_all_filenames_reordered += cur_dev_fnames
dev_fname_endindplus1 = len(complete_list_of_all_filenames_reordered)
complete_list_of_all_filenames_reordered += cur_test_fnames
test_fname_endindplus1 = len(complete_list_of_all_filenames_reordered)
assert len(complete_list_of_all_filenames_reordered) == len(complete_list_of_all_filenames)
complete_list_of_all_filenames = complete_list_of_all_filenames_reordered


def get_text_for_file(tag_part_of_fname):
    expanded_fname = '../../OCRdata/NI_docs/NI_docs_all/' + tag_part_of_fname + '.txt'
    text = ''
    with open(expanded_fname, 'r') as f:
        for line in f:
            text += line
    while text.endswith('\n\n'):
        text = text[:-1]
    return text


doctext_fname_tuples = [(get_text_for_file(fname), fname) for fname in complete_list_of_all_filenames]
matrix_of_simscores = get_doc2doc_similarity_scores(doctext_fname_tuples).todense()
print('Computed score matrix of shape ' + str(matrix_of_simscores.shape))
print((new_fname_endindplus1, train_fname_endindplus1, dev_fname_endindplus1, test_fname_endindplus1))


# check that any cur-excluded docs that match >= a threshold of .9 with a cur-included doc don't also match with any
# cur-included docs in a different data partition
def check_that_existing_data_split_isnt_linked_to_any_others_now(ind_range_for_slice, str_desc):
    # first, confirm that this data split isn't already linked to any others
    submatrix_to_check = matrix_of_simscores[new_fname_endindplus1:, new_fname_endindplus1:]
    adjusted_ind_range = [ind_range_for_slice[0] - new_fname_endindplus1,
                          ind_range_for_slice[1] - new_fname_endindplus1]
    pieces = []
    if 0 < adjusted_ind_range[0]:
        pieces.append(submatrix_to_check[adjusted_ind_range[0]: adjusted_ind_range[1], :adjusted_ind_range[0]])
    if adjusted_ind_range[1] < submatrix_to_check.shape[0]:
        pieces.append(submatrix_to_check[adjusted_ind_range[0]: adjusted_ind_range[1], adjusted_ind_range[1]:])
    encountered_a_problem = False
    for i in range(len(pieces)):
        encountered_a_problem = encountered_a_problem or (pieces[i] >= .9).any()
    assert not encountered_a_problem

    # now figure out which new documents this data split is linked to
    matrix_to_check = matrix_of_simscores[:new_fname_endindplus1, ind_range_for_slice[0]: ind_range_for_slice[1]]
    meets_threshold = matrix_to_check >= .9
    bool_indices = np.squeeze(np.array(np.any(meets_threshold, axis=1)))
    assert len(bool_indices.shape) == 1 and bool_indices.shape[0] == new_fname_endindplus1, bool_indices.shape
    print('The ' + str_desc + ' split of the old data pings matches for ' + str(np.sum(bool_indices)) +
          ' of the new documents to be added.')

    # finally, figure out whether those new documents also meet the threshold for any old documents in other splits
    # of the data (they hopefully shouldn't)
    def check_minimat(minimat):
        return np.any(minimat[bool_indices, :] >= .9)

    if str_desc != 'test':
        assert not check_minimat(np.array(matrix_of_simscores[:new_fname_endindplus1,
                                          ind_range_for_slice[1]:]))
        print(str_desc + ' doesn\'t conflict with what comes after it')
    if str_desc != 'train':
        assert not check_minimat(np.array(matrix_of_simscores[:new_fname_endindplus1,
                                          new_fname_endindplus1: ind_range_for_slice[0]]))
        print(str_desc + ' doesn\'t conflict with what comes before it')

    indices_to_return = []
    for i in range(len(bool_indices)):
        if bool_indices[i]:
            indices_to_return.append(i)
    return set(indices_to_return)


new_inds_that_go_to_dev = \
    check_that_existing_data_split_isnt_linked_to_any_others_now([train_fname_endindplus1, dev_fname_endindplus1],
                                                                 'dev')
new_inds_that_go_to_test = \
    check_that_existing_data_split_isnt_linked_to_any_others_now([dev_fname_endindplus1, test_fname_endindplus1],
                                                                 'test')
new_inds_that_go_to_train = \
    check_that_existing_data_split_isnt_linked_to_any_others_now([new_fname_endindplus1, train_fname_endindplus1],
                                                                 'train')
print()


def get_list_of_sents_in_text(text):
    sent_split_inds = get_sentence_split_inds(text)
    list_of_sents = []
    start_ind = 0
    for end_ind in sent_split_inds:
        sent_to_add = text[start_ind: end_ind].strip()
        if sent_to_add != '':
            list_of_sents.append(sent_to_add)
        start_ind = end_ind
    return list_of_sents


def get_all_sent_tuples_for_file(text, fname, classification_split_formatting=False):
    sents = get_list_of_sents_in_text(text)
    list_to_return = []
    slashformat = fname_to_slashformfname[fname]
    for sent in sents:
        # positive_sentence, tag, is_problem_filler, label
        if classification_split_formatting:
            slash_pieces = slashformat.split('/')
            list_to_return.append((sent, (slash_pieces[0], slash_pieces[1]), False, 'Negative'))
        else:
            list_to_return.append((sent, 'Negative', slashformat))
    if len(list_to_return) == 0:
        print('Found a file with no sentences in it: ' + fname + ', with text "' + text.strip() + '"')
    return list_to_return


def get_all_sent_tuples_for_split(inds, classification_split_formatting=False):
    list_to_return = []
    for ind in inds:
        list_to_return += get_all_sent_tuples_for_file(doctext_fname_tuples[ind][0], doctext_fname_tuples[ind][1],
                                                       classification_split_formatting=classification_split_formatting)
    return list_to_return


# text, strlabel, labels, source_handcoded_sent, filename (in slash format)
new_train_sents = get_all_sent_tuples_for_split(new_inds_that_go_to_train)
new_dev_sents = get_all_sent_tuples_for_split(new_inds_that_go_to_dev)
new_test_sents = get_all_sent_tuples_for_split(new_inds_that_go_to_test)


set_of_filenames_already_represented = set()
existing_num_train_sents = fix_df_format(pd.read_csv('data/binary_mindsduplicates_withcontext_train.csv'))
print('There are ' + str(sum(list(existing_num_train_sents['labels']))) + ' positive binary training sentences in ' +
      'the preexisting data split out of ' + str(existing_num_train_sents.shape[0]) + ' preexisting training sents.')
num_positive_train_sents = sum(list(existing_num_train_sents['labels']))
for fname in list(existing_num_train_sents['filename']):
    set_of_filenames_already_represented.add(fname)
existing_num_train_sents = existing_num_train_sents.shape[0]
existing_num_dev_sents = fix_df_format(pd.read_csv('data/binary_mindsduplicates_withcontext_dev.csv'))
print('There are ' + str(sum(list(existing_num_dev_sents['labels']))) + ' positive binary dev sentences in ' +
      'the preexisting data split out of ' + str(existing_num_dev_sents.shape[0]) + ' preexisting dev sents.')
num_positive_dev_sents = sum(list(existing_num_dev_sents['labels']))
for fname in list(existing_num_dev_sents['filename']):
    set_of_filenames_already_represented.add(fname)
existing_num_dev_sents = existing_num_dev_sents.shape[0]
existing_num_test_sents = fix_df_format(pd.read_csv('data/binary_mindsduplicates_withcontext_test.csv'))
print('There are ' + str(sum(list(existing_num_test_sents['labels']))) + ' positive binary test sentences in ' +
      'the preexisting data split out of ' + str(existing_num_test_sents.shape[0]) + ' preexisting test sents.')
num_positive_test_sents = sum(list(existing_num_test_sents['labels']))
for fname in list(existing_num_test_sents['filename']):
    set_of_filenames_already_represented.add(fname)
existing_num_test_sents = existing_num_test_sents.shape[0]
print('Sample filename logged from old data, to ensure formatting is as expected: ' +
      list(set_of_filenames_already_represented)[0])
print()


new_inds_that_still_need_to_be_determined = []
doctext_fname_constrained = []
for i, tup in enumerate(doctext_fname_tuples):
    if i < new_fname_endindplus1 and i not in new_inds_that_go_to_dev and i not in new_inds_that_go_to_test \
            and i not in new_inds_that_go_to_train:
        new_inds_that_still_need_to_be_determined.append(i)
        doctext_fname_constrained.append(tup)
print('We still need to decide where to place ' + str(len(new_inds_that_still_need_to_be_determined)) + ' documents.')
up_for_grabs_sents = get_all_sent_tuples_for_split(new_inds_that_still_need_to_be_determined,
                                                   classification_split_formatting=True)
matrix_of_simscores = get_doc2doc_similarity_scores(doctext_fname_constrained).todense()
print('Shape of matrix_of_simscores based on constrained: ' + str(matrix_of_simscores.shape))


list_to_sort = []
for i in range(matrix_of_simscores.shape[0]):
    for j in range(i):
        list_to_sort.append((float(matrix_of_simscores[i, j]), (fname_to_slashformfname[doctext_fname_constrained[i][1]],
                                                                fname_to_slashformfname[doctext_fname_constrained[j][1]])))
list_to_sort = sorted(list_to_sort, key=lambda x: x[0], reverse=True)
scorelist = [x[0] for x in list_to_sort]
corr_tagpairs = [x[1] for x in list_to_sort]


total_num_sents = existing_num_train_sents + existing_num_dev_sents + existing_num_test_sents + \
    len(new_train_sents) + len(new_dev_sents) + len(new_test_sents) + len(up_for_grabs_sents)
desired_num_test_sents = .1 * total_num_sents
num_test_sents_left_to_grab = desired_num_test_sents - existing_num_test_sents - len(new_test_sents)
test_sent_percent_to_grab = num_test_sents_left_to_grab / len(up_for_grabs_sents)
desired_num_dev_sents = .1 * total_num_sents
num_dev_sents_left_to_grab = desired_num_dev_sents - existing_num_dev_sents - len(new_dev_sents)
dev_sent_percent_to_grab = num_dev_sents_left_to_grab / len(up_for_grabs_sents)
print('Adjusted fraction to grab of dev: ' + str(dev_sent_percent_to_grab))
print('Adjusted fraction to grab of test: ' + str(test_sent_percent_to_grab))


for slashformfname in set_of_all_fnames_not_currently_accounted_for_in_data_slashform:
    assert slashformfname not in set_of_filenames_already_represented
all_tags_in_up_for_grabs_sents = set()
for sent_tup in up_for_grabs_sents:
    assert sent_tup[1][0] + '/' + sent_tup[1][1] in set_of_all_fnames_not_currently_accounted_for_in_data_slashform
    all_tags_in_up_for_grabs_sents.add(sent_tup[1][0] + '/' + sent_tup[1][1])
for tagpair in corr_tagpairs:
    assert tagpair[0] in set_of_all_fnames_not_currently_accounted_for_in_data_slashform
    assert tagpair[1] in set_of_all_fnames_not_currently_accounted_for_in_data_slashform
    assert tagpair[0] in all_tags_in_up_for_grabs_sents
    assert tagpair[1] in all_tags_in_up_for_grabs_sents
assert 'PREM_15_1013/IMG_8028' in set_of_all_fnames_not_currently_accounted_for_in_data_slashform


train_sents, dev_sents, test_sents = make_classification_split(up_for_grabs_sents, include_filename_as_field=True,
                                                               allowed_to_not_have_matches_for_all_docs=False,
                                                               ideal_percent_dev=dev_sent_percent_to_grab,
                                                               ideal_percent_test=test_sent_percent_to_grab,
                                                               scorelist_for_clustering=scorelist,
                                                               corr_tagpair_list=corr_tagpairs)

train_sents = new_train_sents + train_sents
for i, tup in enumerate(train_sents):
    assert '/' in tup[2], str(i) + ' (length of new_train_sents is ' + str(len(new_train_sents)) + ')'
    assert tup[2] not in set_of_filenames_already_represented
dev_sents = new_dev_sents + dev_sents
for tup in dev_sents:
    assert '/' in tup[2]
    assert tup[2] not in set_of_filenames_already_represented
test_sents = new_test_sents + test_sents
for tup in test_sents:
    assert '/' in tup[2]
    assert tup[2] not in set_of_filenames_already_represented
print('Assembled new additions to data and ensured that they don\'t contain any old filenames.')
print(str(len(train_sents)) + ' new train sents --> there are now ' + str(num_positive_train_sents) +
      ' positive training sents out of ' + str(len(train_sents) + existing_num_train_sents))
print(str(len(dev_sents)) + ' new dev sents --> there are now ' + str(num_positive_dev_sents) +
      ' positive dev sents out of ' + str(len(dev_sents) + existing_num_dev_sents))
print(str(len(test_sents)) + ' new test sents --> there are now ' + str(num_positive_test_sents) +
      ' positive test sents out of ' + str(len(test_sents) + existing_num_test_sents))


save_splits_as_csv(train_sents, dev_sents, test_sents, 'new_train_sents_withoutcontext.csv',
                   'new_dev_sents_withoutcontext.csv', 'new_test_sents_withoutcontext.csv',
                   'new_label_file_containing_only_negative.txt',
                   include_filename_as_field=True)
print('Successfully completed partitioning script.')
