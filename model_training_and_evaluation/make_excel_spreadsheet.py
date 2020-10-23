from prep_data import get_sentence_split_inds_spacy, get_sentence_split_inds, \
    extract_file_image_tag_from_relevant_part_of_header_string, \
    extract_and_tag_next_document, get_corresponding_indices_in_document
from config import full_document_filename, positive_sentence_filename
import pandas as pd
import json


use_spacy_to_split_sents = False
tags_to_pull_fname = '../ICR_format_pages.txt'
csv_filename = 'sampledoc_sentences_to_labels' + ('_spacy' if use_spacy_to_split_sents else '') + '.tsv'


def get_list_of_tags_we_want():
    tags = []
    with open(tags_to_pull_fname, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '':
                tags.append(line)
    return tags


def get_dict_of_tags_we_want_to_docs(list_of_uncleaned_tags):
    cleanedtags_to_uncleanedtags = {}
    for tag in list_of_uncleaned_tags:
        cleanedtag = extract_file_image_tag_from_relevant_part_of_header_string(tag)
        cleanedtags_to_uncleanedtags[cleanedtag] = tag

    previously_extracted_header = None
    tags_to_documents_we_want = {}
    with open(full_document_filename, 'r', encoding='utf-8-sig') as f:
        keep_going = True
        while keep_going:
            document, tag, previously_extracted_header = \
                extract_and_tag_next_document(f, previously_extracted_header=previously_extracted_header)
            if document is None:
                keep_going = False
            elif tag in cleanedtags_to_uncleanedtags:
                tags_to_documents_we_want[cleanedtags_to_uncleanedtags[tag]] = document

    assert len(tags_to_documents_we_want) == len(cleanedtags_to_uncleanedtags)
    return tags_to_documents_we_want


def load_in_positive_sentences_with_multilabels(csv_filename):
    dataframe = pd.read_csv(csv_filename)

    def is_problem_sentence(sent):
        # something that follows the following format:
        # Page 1 : (136,153) - (502,293)
        sent = sent.strip()
        if not sent.startswith('Page '):
            return False
        sent = sent[len('Page '):]
        sent_pieces = sent.split(' ')
        if len(sent_pieces) != 5:
            return False

        def string_block_is_all_digits(block):
            for i in range(len(block)):
                if not block[i].isdigit():
                    return False
            return True

        if not string_block_is_all_digits(sent_pieces[0]):
            return False
        if not sent_pieces[1] == ':':
            return False
        if not sent_pieces[3] == '-':
            return False

        def is_number_span(block):
            if len(block) < 2:
                return False
            if block[0] != '(' or block[-1] != ')':
                return False
            block = block[1: -1].split(',')
            if len(block) != 2:
                return False
            return string_block_is_all_digits(block[0]) and string_block_is_all_digits(block[1])

        if not (is_number_span(sent_pieces[2]) and is_number_span(sent_pieces[4])):
            return False

        return True

    positivesentences_tags_isproblemfiller = []
    problemtags = {}
    for index, row in dataframe.iterrows():
        tag = str(row['img_file_orig'])
        label = str(row['justification_cat'])
        sentence = str(row['text'])
        if is_problem_sentence(sentence):
            problemtags[tag] = 0
            continue
        elif tag in problemtags:
            problemtags[tag] += 1
            positivesentences_tags_isproblemfiller.append((sentence, tag, True, label))
        else:
            positivesentences_tags_isproblemfiller.append((sentence, tag, False, label))

    for tag, count in problemtags.items():
        assert count > 0, "Couldn't find transcribed sentences for the following problem tag: " + \
                          str(tag)

    sentencetag_to_alllabelsisproblemtag = {}
    repeat_sents = 0
    for sentence, tag, _, label in positivesentences_tags_isproblemfiller:
        if (sentence, tag) in sentencetag_to_alllabelsisproblemtag:
            repeat_sents += 1
            sentencetag_to_alllabelsisproblemtag[(sentence, tag)][0].append(label)
        else:
            sentencetag_to_alllabelsisproblemtag[(sentence, tag)] = [[label], _]
    print('Found ' + str(repeat_sents) + ' repeat sentences (not necessarily unique) in the ' +
          'same doc with different labels')

    new_list_to_return = []
    for sentencetag, unpack in sentencetag_to_alllabelsisproblemtag.items():
        sentence = sentencetag[0]
        tag = sentencetag[1]
        is_problem_filler = unpack[1]
        labels = unpack[0]
        new_list_to_return.append((sentence, tag, is_problem_filler, labels))

    return new_list_to_return


def get_lists_of_positive_negative_sentences_from_doc_with_all_pos_labels_for_sentence(
        document, list_of_positive_sentence_inds_in_doc):
    if use_spacy_to_split_sents:
        sentence_split_inds = get_sentence_split_inds_spacy(document)
    else:
        sentence_split_inds = get_sentence_split_inds(document)
    list_of_positive_sentence_inds_in_doc = sorted(list_of_positive_sentence_inds_in_doc, key=lambda x: x[0][0])

    negative_spans = []
    positive_spans = []
    corresponding_source_positive_sentences = []
    span_start = 0
    cur_positive_sentence_ind = 0
    for split_ind in sentence_split_inds:
        overlaps_with_positive_sentence = False
        all_relevant_positive_labels = set()
        positive_sentence_overlap_start_ind = None
        while cur_positive_sentence_ind < len(list_of_positive_sentence_inds_in_doc) and \
                ((span_start <= list_of_positive_sentence_inds_in_doc[cur_positive_sentence_ind][0][0] < split_ind) or
                 (span_start < list_of_positive_sentence_inds_in_doc[cur_positive_sentence_ind][0][1] <= split_ind)):
            # this auto-split "sentence" overlaps with a positive one, so it's positive.
            # this is a while loop because it might overlap with multiple positive sentences.
            overlaps_with_positive_sentence = True
            for label in list_of_positive_sentence_inds_in_doc[cur_positive_sentence_ind][1]:
                all_relevant_positive_labels.add(label)
            if positive_sentence_overlap_start_ind is None:
                positive_sentence_overlap_start_ind = cur_positive_sentence_ind
            if span_start < list_of_positive_sentence_inds_in_doc[cur_positive_sentence_ind][0][1] <= split_ind:
                cur_positive_sentence_ind += 1
            else:
                break
        if overlaps_with_positive_sentence:
            positive_spans.append(((span_start, split_ind), all_relevant_positive_labels))
            source_positive_sentences_to_log = list(range(positive_sentence_overlap_start_ind,
                                                          cur_positive_sentence_ind))
            # now decide whether to add cur_positive_sentence_ind to that list as an overlapping sentence
            if cur_positive_sentence_ind < len(list_of_positive_sentence_inds_in_doc) and \
                    ((span_start <= list_of_positive_sentence_inds_in_doc[cur_positive_sentence_ind][0][0] < split_ind) or
                     (span_start < list_of_positive_sentence_inds_in_doc[cur_positive_sentence_ind][0][1] <= split_ind)):
                source_positive_sentences_to_log.append(cur_positive_sentence_ind)
            corresponding_source_positive_sentences.append(document[list_of_positive_sentence_inds_in_doc[
                                                                        source_positive_sentences_to_log[0]][0][0]:
                                                                    list_of_positive_sentence_inds_in_doc[
                                                                        source_positive_sentences_to_log[-1]][0][1]])
        else:
            negative_spans.append((span_start, split_ind))
        span_start = split_ind
    assert cur_positive_sentence_ind == len(list_of_positive_sentence_inds_in_doc)
    positive_sentences = list(zip([document[span[0][0]: span[0][1]].strip() for span in positive_spans],
                                  [span[1] for span in positive_spans],  # these are the label lists
                                  corresponding_source_positive_sentences))
    negative_sentences = [document[span[0]: span[1]].strip() for span in negative_spans]
    for i in range(len(positive_sentences) - 1, -1, -1):
        if len(positive_sentences[i][0]) == 0:
            del positive_sentences[i]
    for i in range(len(negative_sentences) - 1, -1, -1):
        if len(negative_sentences[i]) == 0:
            del negative_sentences[i]
    return [(positive_sentence[0], positive_sentence[1]) for positive_sentence in positive_sentences], \
           negative_sentences, \
           [positive_sentence[2] for positive_sentence in positive_sentences]


def make_csv_file(filename, tags_to_sentslabels):
    headings = ['Obs_number',  # I think this refers to the sentence index in the document?
                'File_Img',
                'Sentence',
                'justification_any',
                'J_Emergency-Policy',
                'J_Legal_Procedure',
                'J_Terrorism',
                'J_Misc',
                'J_Law-and-order',
                'J_Utilitarian-Deterrence',
                'J_Intelligence',
                'J_Intl-Domestic_Precedent',
                'J_Development-Unity',
                'J_Political-Strategic',
                'J_Last-resort',
                'J_Denial']
    file_writing = open(filename, 'w')
    file_writing.write('\t'.join(headings) + '\n')
    with open(tags_to_pull_fname, 'r') as f:
        for tag in f:
            tag = tag.strip()
            if tag == '':
                continue
            sents_to_alllabels = tags_to_sentslabels[tag]
            for i in range(len(sents_to_alllabels)):
                line_fields = [str(i + 1)]
                line_fields.append(tag)
                sent = sents_to_alllabels[i][0]
                all_labels = sents_to_alllabels[i][1]

                sent = sent.replace('\n', '    ')
                sent = sent.replace('\t', '    ')
                line_fields.append(sent)

                line_fields.append('' if len(all_labels) == 0 else '1')
                while len(line_fields) < len(headings):
                    line_fields.append('')

                for label in all_labels:
                    label_ind = headings.index(label)
                    line_fields[label_ind] = '1'

                file_writing.write('\t'.join(line_fields) + '\n')

    file_writing.close()


def main():
    # get tags to list of all distinct sentence inds
    tags_we_want = get_list_of_tags_we_want()
    tags_to_docs = get_dict_of_tags_we_want_to_docs(tags_we_want)
    tags_to_doc_sentence_inds = {}
    for tag, doc in tags_to_docs.items():
        if use_spacy_to_split_sents:
            tags_to_doc_sentence_inds[tag] = get_sentence_split_inds_spacy(doc)
        else:
            tags_to_doc_sentence_inds[tag] = get_sentence_split_inds(doc)

    # get tags to list of (positive_sentence_inds, all_labels_for_sentence)
    sentence_rawtag_isproblemfiller_labels = load_in_positive_sentences_with_multilabels(positive_sentence_filename)
    for i in range(len(sentence_rawtag_isproblemfiller_labels) -1, -1, -1):
        if sentence_rawtag_isproblemfiller_labels[i][1] not in tags_to_doc_sentence_inds:
            del sentence_rawtag_isproblemfiller_labels[i]
    positivesentences_tags, corresponding_indices_in_document = \
        get_corresponding_indices_in_document(sentence_rawtag_isproblemfiller_labels, tags_to_docs,
                                              'problems_writing_excel_sheet' +
                                              ('_spacy' if use_spacy_to_split_sents else '') + '.txt',
                                              'successes_writing_excel_sheet' +
                                              ('_spacy' if use_spacy_to_split_sents else '') + '.txt',
                                              skip_positive_sents_we_have_no_doc_for=True)
    tags_to_list_of_positive_sentence_inds_and_labels = {}
    for i in range(len(positivesentences_tags)):
        tag = positivesentences_tags[i][1]
        corr_labels = positivesentences_tags[i][3]
        corr_inds = corresponding_indices_in_document[i]
        if corr_inds is not None:
            if tag in tags_to_list_of_positive_sentence_inds_and_labels:
                tags_to_list_of_positive_sentence_inds_and_labels[tag].append((corr_inds, corr_labels))
            else:
                tags_to_list_of_positive_sentence_inds_and_labels[tag] = [(corr_inds, corr_labels)]

    tags_to_list_of_foundindoc_positive_sentences_and_labels = {}
    tags_to_list_of_foundindoc_negative_sentences = {}
    for tag in tags_to_docs.keys():
        document = tags_to_docs[tag]
        if tag in tags_to_list_of_positive_sentence_inds_and_labels:
            list_of_positive_sentence_inds_in_doc = tags_to_list_of_positive_sentence_inds_and_labels[tag]
        else:
            list_of_positive_sentence_inds_in_doc = []
        positive_sentences_and_labels, negative_sentences, _ = \
            get_lists_of_positive_negative_sentences_from_doc_with_all_pos_labels_for_sentence(
                document, list_of_positive_sentence_inds_in_doc)
        tags_to_list_of_foundindoc_positive_sentences_and_labels[tag] = positive_sentences_and_labels
        tags_to_list_of_foundindoc_negative_sentences[tag] = negative_sentences

    # now get an ordered list of all sentences in doc with all of their corresponding labels (if any)
    tags_to_sentslabels = {}
    for tag in tags_to_docs:
        list_of_sentencelabels_tuples = []
        document = tags_to_docs[tag]
        ordered_sentence_inds = tags_to_doc_sentence_inds[tag]
        ordered_positive_sents_and_labels = tags_to_list_of_foundindoc_positive_sentences_and_labels[tag]
        ordered_negative_sents = tags_to_list_of_foundindoc_negative_sentences[tag]

        cur_pos_ind = 0
        cur_neg_ind = 0
        sent_start_ind = 0
        for ind_ind, ind in enumerate(ordered_sentence_inds):
            sent_end_ind = ind
            cur_sentence = document[sent_start_ind: sent_end_ind].strip()
            if cur_sentence == '':
                continue
            if cur_pos_ind < len(ordered_positive_sents_and_labels) and \
                    cur_sentence == ordered_positive_sents_and_labels[cur_pos_ind][0].strip():
                list_of_sentencelabels_tuples.append((cur_sentence, ordered_positive_sents_and_labels[cur_pos_ind][1]))
                cur_pos_ind += 1
            elif cur_neg_ind < len(ordered_negative_sents) and \
                    cur_sentence == ordered_negative_sents[cur_neg_ind].strip():
                list_of_sentencelabels_tuples.append((ordered_negative_sents[cur_neg_ind], []))
                cur_neg_ind += 1
            else:
                assert False, '\n'.join(['This should never happen. Next sentences:',
                                         cur_sentence,
                                         ('END' if cur_pos_ind >= len(ordered_positive_sents_and_labels) else
                                          ordered_positive_sents_and_labels[cur_pos_ind][0]),
                                         ('END' if cur_neg_ind >= len(ordered_negative_sents) else
                                          ordered_negative_sents[cur_neg_ind])
                                         ]) + '\n=======================\n' + \
                    str([ps[0] for ps in ordered_positive_sents_and_labels]) + '\n=====================\n' + \
                    str(ordered_negative_sents) + '\n=====================\n' + str(ind_ind)
            sent_start_ind = sent_end_ind
        sent_end_ind = len(document)
        cur_sentence = document[sent_start_ind: sent_end_ind].strip()
        if cur_sentence != '':
            if cur_pos_ind < len(ordered_positive_sents_and_labels) and cur_sentence == \
                    ordered_positive_sents_and_labels[cur_pos_ind][0].strip():
                list_of_sentencelabels_tuples.append(ordered_positive_sents_and_labels[cur_pos_ind])
                cur_pos_ind += 1
            elif cur_neg_ind < len(ordered_negative_sents) and \
                    cur_sentence == ordered_negative_sents[cur_neg_ind].strip():
                list_of_sentencelabels_tuples.append((ordered_negative_sents[cur_neg_ind], []))
                cur_neg_ind += 1
            else:
                assert False, '\n'.join(['This should never happen. Next sentences:',
                                         cur_sentence,
                                         ('END' if cur_pos_ind >= len(ordered_positive_sents_and_labels) else
                                          ordered_positive_sents_and_labels[cur_pos_ind][0]),
                                         ('END' if cur_neg_ind >= len(ordered_negative_sents) else
                                          ordered_negative_sents[cur_neg_ind])
                                         ]) + '\n=======================\n' + \
                              str([ps[0] for ps in ordered_positive_sents_and_labels]) + '\n=====================\n' + \
                              str(ordered_negative_sents) + '\n=====================\n' + \
                              str(len(ordered_sentence_inds))

        tags_to_sentslabels[tag] = list_of_sentencelabels_tuples

    # now assemble CSV file
    make_csv_file(csv_filename, tags_to_sentslabels)


if __name__ == '__main__':
    main()
