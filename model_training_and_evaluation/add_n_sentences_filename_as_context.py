from prep_data import read_in_presplit_data, get_sentence_split_inds, fix_df_format, extract_and_tag_next_document
import pandas as pd

full_doc_fname = '../orig_text_data/internment.txt'

source_train_filename = 'data/binary_mindsduplicates_train.csv'
source_dev_filename = 'data/binary_mindsduplicates_dev.csv'
source_test_filename = 'data/binary_mindsduplicates_test.csv'

new_train_filename = 'data/binary_mindsduplicates_withcontext_train.csv'
new_dev_filename = 'data/binary_mindsduplicates_withcontext_dev.csv'
new_test_filename = 'data/binary_mindsduplicates_withcontext_test.csv'

num_preceding_sents_to_use_as_context = 1

# read train, dev, test files into different dictionaries
# they should go to a list of (tuple of two things: number index in file, (all other associated tags as tuple))
def read_in_existing_csv_files(train_fname, dev_fname, test_fname):
    train_df, dev_df, test_df, _ = read_in_presplit_data(train_fname, dev_fname, test_fname, None, shuffle_data=False)
    """print(train_df.columns)
    for i, row in train_df.iterrows():
        print(str(row['contextbefore']) + '\t' + str(row['text']))
        print('\n')
        if i == 20:
            quit()"""
    train_dict = {}
    dev_dict = {}
    test_dict = {}
    non_text_columns = []
    for i, col_name in enumerate(train_df.columns):
        col_name = str(col_name)
        if col_name != 'text' and col_name != 'filename' and col_name != 'contextbefore':
            non_text_columns.append(col_name)
        else:
            assert i == 0  # we expect the text column to come first
    for i, col_name in enumerate(dev_df.columns):
        col_name = str(col_name)
        assert (col_name == 'text' or col_name == 'filename' or col_name == 'contextbefore') or \
               col_name == non_text_columns[i - 1]
    for i, col_name in enumerate(test_df.columns):
        col_name = str(col_name)
        assert (col_name == 'text' or col_name == 'filename' or col_name == 'contextbefore') or \
               col_name == non_text_columns[i - 1]

    def populate_dict_with_data(df, dict_to_populate, is_train_for_debugging=False):
        if is_train_for_debugging:
            print('\n\n')

        for i, row in df.iterrows():
            text_from_row = row['text']
            all_other_parts_of_row = tuple([row[colname] for colname in non_text_columns])
            if text_from_row in dict_to_populate:
                dict_to_populate[text_from_row].append([i, all_other_parts_of_row])
            else:
                dict_to_populate[text_from_row] = [[i, all_other_parts_of_row]]

    populate_dict_with_data(train_df, train_dict, is_train_for_debugging=False)
    populate_dict_with_data(dev_df, dev_dict)
    populate_dict_with_data(test_df, test_dict)

    return train_dict, dev_dict, test_dict, non_text_columns


def have_found_context_for_sentence(val_from_valuelist_for_sentence, num_nontext_cols_not_including_new_ones):
    if len(val_from_valuelist_for_sentence) == 2:
        return False
    elif len(val_from_valuelist_for_sentence) == 4:
        # we should add filename and contextbefore, in that order
        return True
    assert False, str(num_nontext_cols_not_including_new_ones) + ', ' + str(val_from_valuelist_for_sentence[1])


def get_sentence_ind_in_file(val_from_valuelist_for_sentence):
    return val_from_valuelist_for_sentence[0]


def get_start_end_inds_in_dict_containing_all_sentences(list_of_sentences, candidate_dict,
                                                        num_nontext_cols_not_including_new_ones):
    #print('\n\nStarting document')
    set_of_match_inds_in_df = {}
    set_of_sentences_processed_so_far = set()
    for i, sent in enumerate(list_of_sentences):
        for val in candidate_dict[sent]:
            if not have_found_context_for_sentence(val, num_nontext_cols_not_including_new_ones):
                assert sent in set_of_sentences_processed_so_far or val[0] not in set_of_match_inds_in_df
                set_of_match_inds_in_df[val[0]] = sent
        set_of_sentences_processed_so_far.add(sent)
    """print("Length of set_of_match_inds: " + str(len(set_of_match_inds_in_df)) +
          '\tNum sents in doc: ' + str(len(list_of_sentences)))"""

    # select a rough group of potential contiguous-document-chunk starts based on whether or not there are
    # len(list_of_sentences) consecutive indices in set_of_match_inds_in_df starting at a hypothesized start
    match_inds_in_df = sorted(list(set_of_match_inds_in_df.keys()))
    might_work_as_start = []
    if len(list_of_sentences) > 1:
        for potential_start in match_inds_in_df[:(-1 * (len(list_of_sentences) - 1))]:
            might_work = True
            for val_that_also_has_to_be_there_for_this_to_work in range(potential_start + len(list_of_sentences) - 1,
                                                                        potential_start,
                                                                        -1):
                if val_that_also_has_to_be_there_for_this_to_work not in set_of_match_inds_in_df:
                    might_work = False
                    break
            if might_work:
                might_work_as_start.append(potential_start)
                # yes, it's really inefficient not to break here, but this is a check worth doing
    else:
        might_work_as_start = match_inds_in_df
    #print(str(len(might_work_as_start)) + ": " + str(might_work_as_start))

    vetted_potential_starts = []
    sent_to_docindex = {}
    for i, sent in enumerate(list_of_sentences):
        if sent not in sent_to_docindex:
            sent_to_docindex[sent] = [[i], 0]
        else:
            sent_to_docindex[sent][0].append(i)
    for potential_start in might_work_as_start:
        checklist = [False] * len(list_of_sentences)
        for list_to_reset in sent_to_docindex.values():
            list_to_reset[1] = 0
        for ind in range(potential_start, potential_start + len(list_of_sentences)):
            pandas_sent_corresponding_to_ind_line = set_of_match_inds_in_df[ind]
            indices_of_that_sent_in_document = sent_to_docindex[pandas_sent_corresponding_to_ind_line][0]
            if len(indices_of_that_sent_in_document) > 1:
                ind_of_that_sent_in_document = sent_to_docindex[pandas_sent_corresponding_to_ind_line][0][
                    sent_to_docindex[pandas_sent_corresponding_to_ind_line][1]]
                sent_to_docindex[pandas_sent_corresponding_to_ind_line][1] = \
                    1 + sent_to_docindex[pandas_sent_corresponding_to_ind_line][1]
            else:
                ind_of_that_sent_in_document = indices_of_that_sent_in_document[0]
            checklist[ind_of_that_sent_in_document] = True
        if all(checklist):
            vetted_potential_starts.append(potential_start)
        """print(sum(checklist), end=', ')
        for i in range(len(checklist)):
            if not checklist[i]:
                print(list_of_sentences[i], end=':')
                for val in candidate_dict[list_of_sentences[i]]:
                    if not have_found_context_for_sentence(val, num_nontext_cols_not_including_new_ones):
                        print(val[0], end=', ')
                print()"""

    #print(len(vetted_potential_starts))

    if len(vetted_potential_starts) > 1:
        print('Found a duplicated document')
    elif len(vetted_potential_starts) == 0:
        return None, None
    start_ind = vetted_potential_starts[-1]
    end_ind = start_ind + len(list_of_sentences) - 1
    return start_ind, end_ind


def find_which_split_the_document_got_sorted_into(list_of_sentences, candidate_dicts, num_non_text_columns):
    winning_dict = None
    ind_of_start_sent_in_doc = None
    ind_of_end_sent_in_doc = None
    for candidate_dict in candidate_dicts:
        compatible_with_this_dict = True
        for i, sentence in enumerate(list_of_sentences):
            if sentence not in candidate_dict:
                compatible_with_this_dict = False
                break
        if compatible_with_this_dict:
            if ind_of_start_sent_in_doc is None:
                ind_of_start_sent_in_doc, ind_of_end_sent_in_doc = \
                    get_start_end_inds_in_dict_containing_all_sentences(list_of_sentences, candidate_dict,
                                                                        num_non_text_columns)
                if ind_of_start_sent_in_doc is None:
                    continue  # not actually a winning dict
            assert winning_dict is None
            winning_dict = candidate_dict

            # now just verify again that all the sentences correspond to the inds we think they do
            for sent in list_of_sentences:
                list_of_vals_to_check = []
                for corr_val in winning_dict[sent]:
                    if not have_found_context_for_sentence(corr_val, num_non_text_columns):
                        list_of_vals_to_check.append(corr_val[0])
                # at least one of the values in here needs to fall in the start-to-end range we defined
                at_least_one_val_meets_criteria = False
                for val in list_of_vals_to_check:
                    if ind_of_start_sent_in_doc <= val <= ind_of_end_sent_in_doc:
                        at_least_one_val_meets_criteria = True
                        break
                assert at_least_one_val_meets_criteria, str(list_of_vals_to_check) + ', ' + \
                                                        str(ind_of_start_sent_in_doc) + '-' + str(ind_of_end_sent_in_doc)
            """for j in range(ind_of_start_sent_in_doc, ind_of_end_sent_in_doc + 1):
                assert j in [val[0] for val in winning_dict[list_of_sentences[j - ind_of_start_sent_in_doc]]], \
                    '\n'.join([str(k) + ': ' +
                               str([val[0] for val in winning_dict[list_of_sentences[k - ind_of_start_sent_in_doc]]])
                               for k in range(ind_of_start_sent_in_doc, ind_of_end_sent_in_doc + 1)]) + '\n' + \
                    get_sentences_surrounding_first_mismatch(list_of_sentences, winning_dict, ind_of_start_sent_in_doc,
                                                             ind_of_end_sent_in_doc)"""

    assert winning_dict is not None, str(list_of_sentences) + '\n' + \
                                     str([list_of_sentences[0] in candidate_dict for candidate_dict in candidate_dicts])
    assert ind_of_start_sent_in_doc is not None and ind_of_end_sent_in_doc is not None, str(list_of_sentences)
    return winning_dict, ind_of_start_sent_in_doc, ind_of_end_sent_in_doc


def get_sentences_surrounding_first_mismatch(list_of_sentences, candidate_dict, ind_of_start_sent_in_doc,
                                             ind_of_end_sent_in_doc):
    first_mismatch_ind = None
    for k in range(ind_of_start_sent_in_doc, ind_of_end_sent_in_doc + 1):
        if k not in [val[0] for val in candidate_dict[list_of_sentences[k - ind_of_start_sent_in_doc]]]:
            first_mismatch_ind = k
            break
    string_to_return = "In list of sentences:\n"
    if first_mismatch_ind > ind_of_start_sent_in_doc:
        string_to_return += '"' + list_of_sentences[first_mismatch_ind - 1 - ind_of_start_sent_in_doc] + '"\n'
    string_to_return += '"' + list_of_sentences[first_mismatch_ind - ind_of_start_sent_in_doc] + '"\n'
    if first_mismatch_ind < ind_of_end_sent_in_doc:
        string_to_return += '"' + list_of_sentences[first_mismatch_ind + 1 - ind_of_start_sent_in_doc] + '"\n'

    string_to_return += 'In candidate dict:\n'
    if first_mismatch_ind > ind_of_start_sent_in_doc:
        string_to_return += '"' + list_of_sentences[first_mismatch_ind - 1 - ind_of_start_sent_in_doc] + '"\n'
    # now find which sentence in the candidate dict contains the missing index we're looking for

    def get_corr_key_from_candidate_dict(ind_to_find):
        for key, list_of_vals in candidate_dict.items():
            for val in list_of_vals:
                if ind_to_find == val[0]:
                    return key

    string_to_return += '"' + get_corr_key_from_candidate_dict(first_mismatch_ind) + '"\n'
    if first_mismatch_ind < ind_of_end_sent_in_doc:
        string_to_return += '"' + get_corr_key_from_candidate_dict(first_mismatch_ind + 1) + '"\n'
    return string_to_return


def add_contexts_for_document(list_of_sentences, corresponding_dict, document_filename,
                              ind_of_start_sent_in_original_splitfile, ind_of_end_sent_in_original_splitfile,
                              num_nontext_cols):
    sent_to_numtimesoccurredindocsofar = {}
    for sentence_ind_for_end, sentence in enumerate(list_of_sentences):
        if sentence in sent_to_numtimesoccurredindocsofar:
            sent_to_numtimesoccurredindocsofar[sentence] += 1
        else:
            sent_to_numtimesoccurredindocsofar[sentence] = 0

        all_possible_vals = corresponding_dict[sentence]
        num_matching_vals_passed_so_far = 0
        matching_val = None
        """if sentence == 'i.':
            print('OCCURRENCE OF i.')"""
        for i, val in enumerate(all_possible_vals):
            val_ind = get_sentence_ind_in_file(val)
            if ind_of_start_sent_in_original_splitfile <= val_ind <= ind_of_end_sent_in_original_splitfile \
                    and not have_found_context_for_sentence(val, num_nontext_cols):
                # assert matching_val is None (this breaks things if we have a duplicated sentence in the doc)
                """if sentence == 'i.':
                    print('\tCONSIDERING ADDING CONTEXT: ' + str(num_matching_vals_passed_so_far) + ', ' + 
                          str(sent_to_numtimesoccurredindocsofar[sentence]))"""
                """if num_matching_vals_passed_so_far == sent_to_numtimesoccurredindocsofar[sentence]:
                    if sentence == 'i.':
                        print('\t\tACTUALLY ADDING CONTEXT')
                        print('\t\t' + str(i))
                        print('\t\t' + str(ind_of_start_sent_in_original_splitfile) + ' <= ' + str(val_ind) + 
                              ' <= ' + str(ind_of_end_sent_in_original_splitfile))"""
                matching_val = val
                num_matching_vals_passed_so_far += 1

        assert matching_val is not None, str((ind_of_start_sent_in_original_splitfile,
                                              ind_of_end_sent_in_original_splitfile,
                                              [get_sentence_ind_in_file(val) for val in all_possible_vals],
                                              sentence + '\n'))

        preceding_sents_as_context = list_of_sentences[max(0, sentence_ind_for_end -
                                                           num_preceding_sents_to_use_as_context): sentence_ind_for_end]
        context = ' '.join(preceding_sents_as_context)
        if len(context) == 0:
            context = ' '

        matching_val.append(document_filename)
        matching_val.append(context)


def write_new_files(train_dict, dev_dict, test_dict, list_of_other_col_names, all_dicts_for_debugging):
    # make one dict that we can turn into a pandas dataframe
    list_of_all_column_names = ['text'] + list_of_other_col_names + ['filename', 'contextbefore']

    def write_split_to_file(info_dict, destination_filename):
        list_of_instances = []
        # list_of_instances should contain tuples: (index in original split file, all relevant info)
        for sentence, listof_otherinfos in info_dict.items():
            for otherinfo in listof_otherinfos:
                sentence_index = otherinfo[0]
                all_other_fields = list(otherinfo[1]) + otherinfo[2:]
                all_other_fields.insert(0, sentence)
                assert len(all_other_fields) == len(list_of_all_column_names), \
                    str(len(all_other_fields)) + ', ' + str(len(list_of_all_column_names)) + '\n' + \
                    str(all_other_fields) + '\n' + str(list_of_all_column_names) + '\n' + \
                    'Num times sent appeared in each dict: ' + \
                    str([0 if sentence not in cand_dict else len(cand_dict[sentence])
                         for cand_dict in all_dicts_for_debugging])
                list_of_instances.append((sentence_index, tuple(all_other_fields)))

        list_of_instances = sorted(list_of_instances, key=(lambda x: x[0]))
        list_of_instances = [tup[1] for tup in list_of_instances]
        data_df = fix_df_format(pd.DataFrame(list_of_instances, columns=list_of_all_column_names))
        data_df.to_csv(destination_filename, index=False)

    write_split_to_file(train_dict, new_train_filename)
    write_split_to_file(dev_dict, new_dev_filename)
    write_split_to_file(test_dict, new_test_filename)


def main():
    train_dict, dev_dict, test_dict, non_text_columns = read_in_existing_csv_files(source_train_filename,
                                                                                   source_dev_filename,
                                                                                   source_test_filename)
    list_of_all_datasplit_dicts = [train_dict, dev_dict, test_dict]

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

    # for each document:
    #     split its sentences
    #     figure out which data split a document (page) is in
    #     add this document's sentences to a file
    for document_tuple in document_text_filename_tuples:
        document_text = document_tuple[0]
        document_filename = document_tuple[1]
        sentence_split_inds = get_sentence_split_inds(document_text)
        list_of_sentences = []
        start_ind = 0
        for split_ind in sentence_split_inds:
            list_of_sentences.append(document_text[start_ind: split_ind].strip())
            start_ind = split_ind
        for i in range(len(list_of_sentences) - 1, -1, -1):
            if len(list_of_sentences[i]) == 0:
                del list_of_sentences[i]

        (dict_corresponding_to_document, ind_of_start_sent_in_original_splitfile,
        ind_of_end_sent_in_original_splitfile) = \
            find_which_split_the_document_got_sorted_into(list_of_sentences, list_of_all_datasplit_dicts,
                                                          len(non_text_columns))
        add_contexts_for_document(list_of_sentences, dict_corresponding_to_document, document_filename,
                                  ind_of_start_sent_in_original_splitfile, ind_of_end_sent_in_original_splitfile,
                                  len(non_text_columns))

    write_new_files(train_dict, dev_dict, test_dict, non_text_columns, list_of_all_datasplit_dicts)


if __name__ == '__main__':
    main()
