import pandas as pd
import os
from math import inf
from random import shuffle
from string import whitespace, punctuation
import torch
from logreg_baseline import run_classification as run_logreg_classification
from roberta import run_classification as run_roberta_classification
from roberta import run_best_model_on
import numpy as np
from detailed_performance_breakdown import get_recall_precision_curve_points, \
    plot_two_precision_recalls_against_each_other, make_multilabel_csv


def get_simplified_filename_string(filename_tag_string):
    first_part = filename_tag_string[:filename_tag_string.index('_') + 1]
    part_to_simplify = filename_tag_string[filename_tag_string.index('_') + 1:]
    if '_' in part_to_simplify:
        simplified = '_'.join([str(int(part)) for part in part_to_simplify.split('_')])
    else:
        simplified = str(int(part_to_simplify))
    return first_part + simplified


def extract_file_image_tag_from_relevant_part_of_header_string(relevant_part):
    # relevant_part will be something that looks like IMG_6444_PREM_15_486 except for a few special cases
    # which we handle separately
    if (relevant_part.startswith('PREM_15_1010_') or relevant_part.startswith('PREM_15_1689_') or
            relevant_part.startswith('PREM_15_478_')) and relevant_part.count('_') == 3:
        image_name = 'IMG_' + relevant_part[relevant_part.rfind('_') + 1:].strip()
        file_name = relevant_part[:relevant_part.rfind('_')]
    else:
        relevant_part = relevant_part.split('_')
        image_name = '_'.join(relevant_part[:2])
        file_name = '_'.join(relevant_part[2:])
    return get_simplified_filename_string(file_name), get_simplified_filename_string(image_name)


def extract_and_tag_next_document(line_iterator, previously_extracted_header=None):
    # returns document, headertag, and any next bit of the header
    if previously_extracted_header is None:
        try:
            header = next(line_iterator)
        except StopIteration:
            return None, None, None  # we're done
    else:
        header = previously_extracted_header

    def is_header_line(line):  # will return tag instead of True if True
        if not line.startswith(r'Files\\'):
            return False
        for i in range(3):
            try:
                line = line[line.index(r'\\') + len(r'\\'):]
            except ValueError:
                return False
        try:
            suspected_relevant_part = line[:line.index(' - § ')]
            line = line[line.index(' - § ') + len(' - § '):]
        except ValueError:
            return False
        # now see if the end of the line is a match, and if it is, then actually extract the tag
        if len(line) == 0 or not line[0].isdigit():
            return False
        while len(line) > 0 and line[0].isdigit():
            line = line[1:]
        if not line.startswith(' references coded [ '):
            return False
        if line.strip().endswith('% Coverage]'):
            return extract_file_image_tag_from_relevant_part_of_header_string(suspected_relevant_part)
        else:
            return False

    header_tag = is_header_line(header)
    assert header_tag, 'Expected following line to be a header, which wasn\'t the case: ' + header

    def is_a_reference_line(line):
        if not line.startswith('Reference '):
            return False
        line = line[len('Reference '):]
        if len(line) == 0 or not line[0].isdigit():
            return False
        while len(line) > 0 and line[0].isdigit():
            line = line[1:]
        if line.startswith(' - ') and line.strip().endswith('% Coverage'):
            return True
        return False

    # now we add the "Reference" subtitles to the header
    try:
        subheader = next(line_iterator)
    except StopIteration:
        return '', header_tag, None
    assert is_a_reference_line(subheader)
    temp_line_1 = ''
    temp_line_2 = ''
    is_first_time_around = True
    while is_first_time_around or is_a_reference_line(temp_line_2):
        is_first_time_around = False
        subheader += temp_line_1 + temp_line_2
        temp_line_1 = ''
        temp_line_2 = ''
        try:
            temp_line_1 = next(line_iterator)
        except StopIteration:
            break
        try:
            temp_line_2 = next(line_iterator)
        except StopIteration:
            break

    assert not is_header_line(temp_line_1), temp_line_1  # if this was True, it would mean we had an empty document
    if is_header_line(temp_line_2):
        return temp_line_1, header_tag, temp_line_2

    document = temp_line_1 + temp_line_2
    while True:
        try:
            next_line = next(line_iterator)
        except StopIteration:
            return document, header_tag, None
        if is_header_line(next_line):
            return document, header_tag, next_line
        else:
            document = document + next_line


def load_in_positive_sentences(csv_filename):
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
        tag = (get_simplified_filename_string(str(row['file_id'])),
               get_simplified_filename_string(str(row['image_id'])))
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
    return positivesentences_tags_isproblemfiller


def get_indices_of_sentencematch_in_document(document, sentence, tag, report_to_file, report_successes_to_file,
                                             is_problem_filler, dont_print_at_all=False):
    sentence = sentence.strip()
    try:
        stuff_to_return = (document.index(sentence), document.index(sentence) + len(sentence))
        if report_successes_to_file:
            report_matches('Successful document: ' + str(tag), report_successes_to_file,
                           dont_print_at_all=dont_print_at_all)
            report_matches(document, report_successes_to_file,
                           dont_print_at_all=dont_print_at_all)
            report_matches('Positive sentence: ' + sentence, report_successes_to_file,
                           dont_print_at_all=dont_print_at_all)
            report_matches('Is hand-transcribed filler for OCR problem match: ' + str(is_problem_filler),
                           report_successes_to_file, dont_print_at_all=dont_print_at_all)
            report_matches('\n', report_successes_to_file, dont_print_at_all=dont_print_at_all)
        return stuff_to_return
    except:
        report_matches('Problem document: ' + str(tag), report_to_file, dont_print_at_all=dont_print_at_all)
        report_matches(document, report_to_file, dont_print_at_all=dont_print_at_all)
        report_matches('Positive sentence: ' + sentence, report_to_file, dont_print_at_all=dont_print_at_all)
        report_matches('Is hand-transcribed filler for OCR problem match: ' + str(is_problem_filler),
                       report_to_file, dont_print_at_all=dont_print_at_all)
        report_matches('\n', report_to_file, dont_print_at_all=dont_print_at_all)
        return None


def report_matches(message, report_to_file, dont_print_at_all=False):
    if not report_to_file and not dont_print_at_all:
        print(message)
    else:
        with open(report_to_file, 'a') as f:
            f.write(message + '\n')


def get_sentence_split_inds(text):
    text = text.rstrip()
    locations_of_punctuation_marks_to_split_on = []

    def token_contains_lowercase_alpha(token):
        for k in range(len(token)):
            if token[k].isalpha() and token[k] == token[k].lower():
                return True
        return False

    for i, char in enumerate(text):
        if char in '!?.':
            if char == '.':
                # extra checking needed
                preceding_text = text[:i].rstrip()
                most_recent_whitespace = max(preceding_text.rfind(' '), preceding_text.rfind('\t'),
                                             preceding_text.rfind('\n'))
                if most_recent_whitespace == -1:
                    # then this is still attached to the first token of the document,
                    # making it not a likely sentence end
                    continue
                preceding_token = text[most_recent_whitespace + 1: len(preceding_text)]
                if len(preceding_token) <= 1:
                    continue  # then this is likely used to denote a subheading
                are_all_digits = True
                for potential_digit in preceding_token:
                    if not potential_digit.isdigit():
                        are_all_digits = False
                        break
                if are_all_digits:
                    continue  # then this is likely used to denote a subheading
                lower_preceding_token = preceding_token.lower()
                if lower_preceding_token == 'mr' or lower_preceding_token == 'mrs' or lower_preceding_token == 'ms' \
                        or lower_preceding_token == 'sr' or lower_preceding_token == 'jr':
                    continue
                if preceding_token[-1] in punctuation:  # we just log the FIRST punctuation mark as the end of the
                    # sentence. we adjust this later at the end of the function.
                    continue
                if len(lower_preceding_token) >= 2 and (lower_preceding_token[-2] == '.' and
                                                        lower_preceding_token[-1].isalpha()):
                    # then this looks like an acronym. the only situation in which we reject this is if a) there
                    # exists a next token and b) it *doesn't* start with an uppercase letter and contain a lowercase
                    # letter afterwards
                    next_token = ''
                    have_started_token = False
                    for j in range(i + 1, len(text)):
                        if (not have_started_token) and text[j] in punctuation:
                            continue
                        if (not have_started_token) and text[j] in whitespace:
                            have_started_token = True
                            continue
                        if have_started_token and text[j] in whitespace:
                            if len(next_token) == 0:
                                continue
                            else:
                                break
                        next_token += text[j]

                    if len(next_token) >= 2 and next_token[0].isalpha() and next_token[0] != next_token[0].lower() and \
                            token_contains_lowercase_alpha(next_token):
                        continue
            locations_of_punctuation_marks_to_split_on.append(i)

    # now judge whether we've got too few sentence breaks or not and, if so, introduce some new ones.
    # any spans of over 500 characters? then look for a comma or ; to split on. for any span that's too long and has
    # no comma, we'll split on newlines.
    extra_locations = []
    span_start = 0
    for i in range(len(locations_of_punctuation_marks_to_split_on)):
        span_end = locations_of_punctuation_marks_to_split_on[i]
        if span_end - span_start >= 500:
            split_inds = []
            for char_ind in range(span_start, span_end):
                if text[char_ind] == ',' or text[char_ind] == ';':
                    split_inds.append(char_ind)
            if len(split_inds) == 0:
                for char_ind in range(span_start, span_end):
                    if text[char_ind] == '\n':
                        split_inds.append(char_ind)
            extra_locations += split_inds
        span_start = span_end
    locations_of_punctuation_marks_to_split_on += extra_locations
    locations_of_punctuation_marks_to_split_on = sorted(locations_of_punctuation_marks_to_split_on)

    # adjust suspected sentence boundaries to start at the beginning of the next sentence's first token
    for i in range(len(locations_of_punctuation_marks_to_split_on) - 1, -1, -1):
        location = locations_of_punctuation_marks_to_split_on[i]
        next_location = (None if i == len(locations_of_punctuation_marks_to_split_on) - 1 else
                         locations_of_punctuation_marks_to_split_on[i + 1])
        if next_location is None:
            locations_of_punctuation_marks_to_split_on[i] = len(text)
            continue

        finished_adjusting = False
        have_started_token = False
        for j in range(location + 1, len(text)):
            if j == next_location:
                del locations_of_punctuation_marks_to_split_on[i]
                finished_adjusting = True
                break

            if text[j] not in punctuation and text[j] not in whitespace:
                locations_of_punctuation_marks_to_split_on[i] = j
                finished_adjusting = True
                break
            if (not have_started_token) and text[j] in punctuation:
                continue
            if (not have_started_token) and text[j] in whitespace:
                have_started_token = True
                continue
            if have_started_token and text[j] not in whitespace:
                locations_of_punctuation_marks_to_split_on[i] = j
                finished_adjusting = True
                break
        if not finished_adjusting:
            locations_of_punctuation_marks_to_split_on[i] = len(text)
            assert False, "We have two sentences purporting to both end at the end of the document:\n" + \
                text[locations_of_punctuation_marks_to_split_on[-2]: locations_of_punctuation_marks_to_split_on[-1]] + \
                '\nAND\n' + \
                text[(0 if i == 0 else locations_of_punctuation_marks_to_split_on[i - 1]):
                     locations_of_punctuation_marks_to_split_on[i] + 10]
    return locations_of_punctuation_marks_to_split_on


def get_lists_of_positive_negative_sentences_from_doc(document, list_of_positive_sentence_inds_in_doc):
    sentence_split_inds = get_sentence_split_inds(document)
    list_of_positive_sentence_inds_in_doc = sorted(list_of_positive_sentence_inds_in_doc, key=lambda x: x[0])

    negative_spans = []
    positive_spans = []
    corresponding_source_positive_sentences = []
    span_start = 0
    cur_positive_sentence_ind = 0
    for split_ind in sentence_split_inds:
        overlaps_with_positive_sentence = False
        positive_sentence_overlap_start_ind = None
        while cur_positive_sentence_ind < len(list_of_positive_sentence_inds_in_doc) and \
                ((span_start <= list_of_positive_sentence_inds_in_doc[cur_positive_sentence_ind][0] < split_ind) or
                 (span_start < list_of_positive_sentence_inds_in_doc[cur_positive_sentence_ind][1] <= split_ind)):
            # this auto-split "sentence" overlaps with a positive one, so it's positive.
            # this is a while loop because it might overlap with multiple positive sentences.
            overlaps_with_positive_sentence = True
            if positive_sentence_overlap_start_ind is None:
                positive_sentence_overlap_start_ind = cur_positive_sentence_ind
            if span_start < list_of_positive_sentence_inds_in_doc[cur_positive_sentence_ind][1] <= split_ind:
                cur_positive_sentence_ind += 1
            else:
                break
        if overlaps_with_positive_sentence:
            positive_spans.append((span_start, split_ind))
            source_positive_sentences_to_log = list(range(positive_sentence_overlap_start_ind,
                                                          cur_positive_sentence_ind))
            # now decide whether to add cur_positive_sentence_ind to that list as an overlapping sentence
            if cur_positive_sentence_ind < len(list_of_positive_sentence_inds_in_doc) and \
                    ((span_start <= list_of_positive_sentence_inds_in_doc[cur_positive_sentence_ind][0] < split_ind) or
                     (span_start < list_of_positive_sentence_inds_in_doc[cur_positive_sentence_ind][1] <= split_ind)):
                source_positive_sentences_to_log.append(cur_positive_sentence_ind)
            corresponding_source_positive_sentences.append(document[list_of_positive_sentence_inds_in_doc[
                                                                        source_positive_sentences_to_log[0]][0]:
                                                                    list_of_positive_sentence_inds_in_doc[
                                                                        source_positive_sentences_to_log[-1]][1]])
        else:
            negative_spans.append((span_start, split_ind))
        span_start = split_ind
    positive_sentences = list(zip([document[span[0]: span[1]].strip() for span in positive_spans],
                                  corresponding_source_positive_sentences))
    negative_sentences = [document[span[0]: span[1]].strip() for span in negative_spans]
    for i in range(len(positive_sentences) - 1, -1, -1):
        if len(positive_sentences[i][0]) == 0:
            del positive_sentences[i]
    for i in range(len(negative_sentences) - 1, -1, -1):
        if len(negative_sentences[i]) == 0:
            del negative_sentences[i]
    return [positive_sentence[0] for positive_sentence in positive_sentences], negative_sentences, \
           [positive_sentence[1] for positive_sentence in positive_sentences]


def make_classification_split(list_of_positive_sentences):
    shuffle(list_of_positive_sentences)
    doctag_to_sentencelabellist = {}
    num_positive_sentences = len(list_of_positive_sentences)
    for positive_sentence, tag, is_problem_filler, label in list_of_positive_sentences:
        if tag not in doctag_to_sentencelabellist:
            doctag_to_sentencelabellist[tag] = []
        doctag_to_sentencelabellist[tag].append((positive_sentence, label))
    test_positive_sentences = []
    dev_positive_sentences = []
    train_positive_sentences = []
    ideal_cutoff_for_test = int(.1 * num_positive_sentences)
    ideal_cutoff_for_dev = 2 * ideal_cutoff_for_test
    num_assigned_so_far = 0
    for doctag, sentence_label_list in doctag_to_sentencelabellist.items():
        if num_assigned_so_far < ideal_cutoff_for_test:
            test_positive_sentences += sentence_label_list
            num_assigned_so_far += len(sentence_label_list)
            if num_assigned_so_far >= ideal_cutoff_for_test:
                ideal_cutoff_for_dev += (num_assigned_so_far - ideal_cutoff_for_test)  # so we don't short-change dev
        elif num_assigned_so_far < ideal_cutoff_for_dev:
            dev_positive_sentences += sentence_label_list
            num_assigned_so_far += len(sentence_label_list)
        else:
            train_positive_sentences += sentence_label_list
            num_assigned_so_far += len(sentence_label_list)
    return train_positive_sentences, dev_positive_sentences, test_positive_sentences


def save_splits_as_csv(train, dev, test, train_filename, dev_filename, test_filename, label_key_filename,
                       split_ex0_into_two_with_second_label: str=None):
    # make strlabel-to-intlabel dict
    strlabels_to_intlabels = {}
    next_available_intlabel = 0
    for i in range(len(train)):
        cur_example = train[i]
        strlabel = cur_example[1].strip()
        if strlabel not in strlabels_to_intlabels:
            strlabels_to_intlabels[strlabel] = next_available_intlabel
            next_available_intlabel += 1
        if split_ex0_into_two_with_second_label is None:
            train[i] = (cur_example[0], strlabel, strlabels_to_intlabels[strlabel])
        else:
            train[i] = (cur_example[0][0], strlabel, strlabels_to_intlabels[strlabel], cur_example[0][1])
    for i in range(len(dev)):
        cur_example = dev[i]
        strlabel = cur_example[1].strip()
        if strlabel not in strlabels_to_intlabels:
            strlabels_to_intlabels[strlabel] = next_available_intlabel
            next_available_intlabel += 1
        if split_ex0_into_two_with_second_label is None:
            dev[i] = (cur_example[0], strlabel, strlabels_to_intlabels[strlabel])
        else:
            dev[i] = (cur_example[0][0], strlabel, strlabels_to_intlabels[strlabel], cur_example[0][1])
    for i in range(len(test)):
        cur_example = test[i]
        strlabel = cur_example[1].strip()
        if strlabel not in strlabels_to_intlabels:
            strlabels_to_intlabels[strlabel] = next_available_intlabel
            next_available_intlabel += 1
        if split_ex0_into_two_with_second_label is None:
            test[i] = (cur_example[0], strlabel, strlabels_to_intlabels[strlabel])
        else:
            test[i] = (cur_example[0][0], strlabel, strlabels_to_intlabels[strlabel], cur_example[0][1])
    strlabel_intlabel_list = sorted(list(strlabels_to_intlabels.items()), key=lambda x: x[1])
    with open(label_key_filename, 'w') as f:
        for strlabel, intlabel in strlabel_intlabel_list:
            f.write(strlabel + '\n')

    column_list = ['text', 'strlabel', 'labels']
    if split_ex0_into_two_with_second_label is not None:
        column_list.append(split_ex0_into_two_with_second_label)
    train_df = fix_df_format(pd.DataFrame(train, columns=column_list))
    train_df.to_csv(train_filename, index=False)
    dev_df = fix_df_format(pd.DataFrame(dev, columns=column_list))
    dev_df.to_csv(dev_filename, index=False)
    test_df = fix_df_format(pd.DataFrame(test, columns=column_list))
    test_df.to_csv(test_filename, index=False)

    return train_df, dev_df, test_df, len(strlabels_to_intlabels)


def read_in_presplit_data(train_filename, dev_filename, test_filename, label_key_filename):
    train_df = fix_df_format(pd.read_csv(train_filename))
    dev_df = fix_df_format(pd.read_csv(dev_filename))
    test_df = fix_df_format(pd.read_csv(test_filename))
    num_labels = 0
    with open(label_key_filename, 'r') as f:
        for line in f:
            if line.strip() != '':
                num_labels += 1
    return train_df, dev_df, test_df, num_labels


def fix_df_format(df):
    df['text'] = df['text'].astype(str)
    df['strlabel'] = df['strlabel'].astype(str)
    df['labels'] = df['labels'].astype(int)
    return df


def get_label_weights_and_report_class_imbalance(train_df, label_file=None):
    df_with_class_counts = train_df['labels'].value_counts()
    labels_and_counts = []
    for label_ind, label_count in df_with_class_counts.iteritems():
        labels_and_counts.append((int(label_ind), int(label_count)))
    labels_and_counts = sorted(labels_and_counts, key=lambda x: x[0])
    corresponding_labels = []
    if label_file is not None:
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line != '':
                    corresponding_labels.append(line)
        assert len(labels_and_counts) == len(corresponding_labels)
    else:
        for label, count in labels_and_counts:
            corresponding_labels.append(str(label))
    median_value = sorted(labels_and_counts, key=lambda x: x[1])
    median_value = median_value[len(median_value) // 2][1]
    label_weights = []
    for label, count in labels_and_counts:
        label_weights.append(median_value / count)
    print('Number of each class in the training data:')
    for i in range(len(labels_and_counts)):
        print('\t' + corresponding_labels[i] + ': ' + str(labels_and_counts[i][1]) + ' => weight = ' +
              str(label_weights[i]))
    return label_weights


def get_binary_classification_data(positivesentences_tags, train_filename, dev_filename, test_filename,
                                   label_key_filename, tags_to_documents, positive_sentences_spot_checking_fname,
                                   negative_sentences_spot_checking_fname, corresponding_indices_in_document):
    if not os.path.isfile(train_filename):


        tags_to_list_of_positive_sentence_inds = {}
        for i in range(len(positivesentences_tags)):
            tag = positivesentences_tags[i][1]
            corr_inds = corresponding_indices_in_document[i]
            if corr_inds is not None:
                if tag in tags_to_list_of_positive_sentence_inds:
                    tags_to_list_of_positive_sentence_inds[tag].append(corr_inds)
                else:
                    tags_to_list_of_positive_sentence_inds[tag] = [corr_inds]

        list_of_individual_sentences = []
        # for positive_sentence, tag, is_problem_filler, label in list_of_positive_sentences:
        for tag in tags_to_documents.keys():
            document = tags_to_documents[tag]
            if tag in tags_to_list_of_positive_sentence_inds:
                list_of_positive_sentence_inds_in_doc = tags_to_list_of_positive_sentence_inds[tag]
            else:
                list_of_positive_sentence_inds_in_doc = []
            positive_sentences, negative_sentences, corresponding_source_positive_sentences = \
                get_lists_of_positive_negative_sentences_from_doc(document, list_of_positive_sentence_inds_in_doc)

            for j in range(len(positive_sentences)):
                sent = positive_sentences[j]
                corr_source_sent = corresponding_source_positive_sentences[j]
                list_of_individual_sentences.append(((sent, corr_source_sent), tag, None, 'Positive'))
            for j in range(len(negative_sentences)):
                sent = negative_sentences[j]
                list_of_individual_sentences.append(((sent, 'n/a'), tag, None, 'Negative'))

        train, dev, test = make_classification_split(list_of_individual_sentences)
        print('Made new binary classification data split.')
        train_df, dev_df, test_df, num_labels = \
            save_splits_as_csv(train, dev, test, train_filename, dev_filename, test_filename, label_key_filename,
                               split_ex0_into_two_with_second_label='source_handcoded_sent')

        if positive_sentences_spot_checking_fname is not None:
            print('Making spot-checking files now...')
            with open(positive_sentences_spot_checking_fname, 'w') as pos_f:
                with open(negative_sentences_spot_checking_fname, 'w') as neg_f:
                    for sent_tuple in list_of_individual_sentences:
                        sent = sent_tuple[0][0]
                        source_sent = sent_tuple[0][1]
                        label = sent_tuple[3]
                        tag = sent_tuple[1]
                        if label == 'Positive':
                            pos_f.write(sent + '\t' + str(tag) + '\n\t' + source_sent + '\n\n')
                        else:
                            neg_f.write(sent + '\t' + str(tag) + '\n\n')
            print('Done making spot-checking files.')
    else:
        train_df, dev_df, test_df, num_labels = \
            read_in_presplit_data(train_filename, dev_filename, test_filename, label_key_filename)
        print('Read in existing binary data split.')
    print('For binary classification:')
    print('\t' + str(int(train_df.shape[0])) + ' training sentences')
    print('\t' + str(int(dev_df.shape[0])) + ' dev sentences')
    print('\t' + str(int(test_df.shape[0])) + ' test sentences')
    return train_df, dev_df, test_df, num_labels


def clean_positive_sentences(positivesentences_tags, corresponding_indices_in_document, tags_to_documents):
    tags_to_positivesents = {}
    for i, tup in enumerate(positivesentences_tags):
        positive_sentence, tag, is_problem_filler, label = tup
        if tag not in tags_to_positivesents:
            tags_to_positivesents[tag] = [(positive_sentence, is_problem_filler, label,
                                           corresponding_indices_in_document[i])]
        else:
            tags_to_positivesents[tag].append((positive_sentence, is_problem_filler, label,
                                               corresponding_indices_in_document[i]))
    all_tags = list(tags_to_positivesents.keys())
    num_removed = 0
    for tag in all_tags:
        denom = max([(0 if tup[3] is None else tup[3][1]) for tup in tags_to_positivesents[tag]]) + 1
        tags_to_positivesents[tag] = sorted(tags_to_positivesents[tag], key=lambda x: inf if x[3] is None else
                                            (denom * x[3][0]) + x[3][1])
        tup_list = tags_to_positivesents[tag]
        for i in range(len(tup_list) - 1, -1, -1):
            if tup_list[i][3] is None:
                continue
            # if the end of this tup is leq the end of any tup that came before it, remove this tup
            for j in range(i):
                if tup_list[i][3][1] <= tup_list[j][3][1]:
                    """assert tup_list[i][2] == tup_list[j][2], ('Found subset sentence with different label:\n' +
                                                              tags_to_documents[tag][tup_list[i][3][0]:
                                                                                     tup_list[i][3][1]] + '\n' +
                                                              'Label: ' + tup_list[i][2] + '\n' +
                                                              tags_to_documents[tag][tup_list[j][3][0]:
                                                                                     tup_list[j][3][1]] + '\n' +
                                                              'Label: ' + tup_list[j][2]
                                                              )"""
                    # counterexample:
                    # Where a breach of the peace is not committed, but the
                    # conduct of the pickets is intimidatory, they should be arrested only if there is no other way to
                    # get them to desist. In these circumstances the power of arrest to be used is SPA Regulation 11,
                    # in accordance with the Blue Card. The words to be used are: "I suspect you of being about to
                    # commit (or of committing) acts prejudicial to the peace".
                    # Label: J_Legal_Procedure AND J_Law-and-order
                    if tup_list[i][2] == tup_list[j][2]:
                        num_removed += 1
                        del tup_list[i]
                        break
        tags_to_positivesents[tag] = tup_list
    print('Removed ' + str(num_removed) + ' handcoded positive sentences that were subsets of another positive ' +
          'sentence and had the same label.')

    # now merge any positive sentences that border/overlap each other and have the same label
    num_merges_made = 0
    for tag in all_tags:
        tup_list = tags_to_positivesents[tag]
        for i in range(len(tup_list) - 1, 0, -1):
            if tup_list[i][3] is None:
                continue
            if tup_list[i][3][0] <= tup_list[i - 1][3][1] and tup_list[i][2] == tup_list[i - 1][2]:
                del tup_list[i]
                num_merges_made += 1
        tags_to_positivesents[tag] = tup_list
    print('Made ' + str(num_merges_made) + ' merges of positive sentences that bordered/overlapped with each other ' +
          'and had the same label.')

    positivesentences_tags = []
    corresponding_indices = []
    for tag in all_tags:
        for sentence, is_problem_filler, label, corr_inds in tags_to_positivesents[tag]:
            positivesentences_tags.append((sentence, tag, is_problem_filler, label))
            corresponding_indices.append(corr_inds)
    return positivesentences_tags, corresponding_indices


def get_multi_way_classification_data(positivesentences_tags, train_filename, dev_filename, test_filename,
                                      label_key_filename):
    if not os.path.isfile(train_filename):
        train, dev, test = make_classification_split(positivesentences_tags)
        print('Made new multi-way classification data split.')
        train_df, dev_df, test_df, num_labels = \
            save_splits_as_csv(train, dev, test, train_filename, dev_filename, test_filename, label_key_filename)
    else:
        train_df, dev_df, test_df, num_labels = \
            read_in_presplit_data(train_filename, dev_filename, test_filename, label_key_filename)
        print('Read in existing multi-way data split.')
    print('For multi-way classification:')
    print('\t' + str(int(train_df.shape[0])) + ' training sentences')
    print('\t' + str(int(dev_df.shape[0])) + ' dev sentences')
    print('\t' + str(int(test_df.shape[0])) + ' test sentences')
    return train_df, dev_df, test_df, num_labels


def report_mismatches_to_files(file_stub, true_labels, baseline_labels, model_labels, test_df, model_name: str=None):
    stub_pieces = file_stub.split('/')
    if len(stub_pieces) > 1:
        dir_so_far = stub_pieces[0]
        if dir_so_far != '' and not os.path.isdir(dir_so_far):
            os.makedirs(dir_so_far)
        for stub_piece in stub_pieces[1: -1]:
            dir_so_far += '/' + stub_piece
            if not os.path.isdir(dir_so_far):
                os.makedirs(dir_so_far)
        dir_so_far += '/' + stub_pieces[-1]

    correct_in_both = 0
    correct_in_both_f = open(file_stub + '_bothcorrect.txt', 'w')
    correct_only_in_model = 0
    correct_only_in_model_f = open(file_stub + '_onlycorrectinmodel.txt', 'w')
    correct_only_in_baseline = 0
    correct_only_in_baseline_f = open(file_stub + '_onlycorrectinbaseline.txt', 'w')
    neither_correct = 0
    neither_correct_f = open(file_stub + '_neithercorrect.txt', 'w')
    for i, row in test_df.iterrows():
        sent = str(row['text'])
        label = str(row['strlabel'])
        if true_labels[i] == model_labels[i]:
            if model_labels[i] == baseline_labels[i]:
                correct_in_both_f.write(str(label) + '\t' + str(sent) + '\n')
                correct_in_both += 1
            else:
                correct_only_in_model_f.write(str(label) + '\t' + str(sent) + '\n')
                correct_only_in_model += 1
        else:
            if true_labels[i] == baseline_labels[i]:
                correct_only_in_baseline_f.write(str(label) + '\t' + str(sent) + '\n')
                correct_only_in_baseline += 1
            else:
                neither_correct_f.write(str(label) + '\t' + str(sent) + '\n')
                neither_correct += 1
    correct_in_both_f.close()
    correct_only_in_model_f.close()
    correct_only_in_baseline_f.close()
    neither_correct_f.close()
    if model_name is None:
        model_name = 'other model'
    print('Comparison of what baseline and ' + model_name + ' got correct:')
    print('\tBoth correct: ' + str(correct_in_both))
    print('\tOnly ' + model_name + ' correct: ' + str(correct_only_in_model))
    print('\tOnly baseline correct: ' + str(correct_only_in_baseline))
    print('\tNeither correct: ' + str(neither_correct))


def clean_roberta_prediction_output(output):
    clean_labels = []
    for arr in output:
        clean_labels.append(int(np.argmax(arr, axis=0)))
    return clean_labels


def main():
    full_document_filename = '../orig_text_data/internment.txt'
    positive_sentence_filename = '../justifications_clean_text_ohe.csv'

    binary_train_filename = 'data/binary_train.csv'
    binary_dev_filename = 'data/binary_dev.csv'
    binary_test_filename = 'data/binary_test.csv'
    binary_label_key_filename = 'data/binary_classes.txt'
    output_binary_model_dir = '../f1-saved_binary_model/'
    binary_output_report_filename_stub = 'output_analysis/f1-binarybest'
    binary_positive_sentences_spot_checking_fname = 'data/binary_extracted_positive_sentences.txt'
    binary_negative_sentences_spot_checking_fname = 'data/binary_extracted_negative_sentences.txt'
    problem_report_filename = 'data/problem_matches.txt'  # or None if you just want to report to the command line
    success_report_filename = 'data/successful_matches.txt'  # or None if you don't want these reported
    dev_precreccurve_plot_filename = 'output_analysis/binarytask_dev_precisionrecallcurve.png'
    test_precreccurve_plot_filename = 'output_analysis/binarytask_test_precisionrecallcurve.png'
    if problem_report_filename and os.path.isfile(problem_report_filename):
        os.remove(problem_report_filename)
    if success_report_filename and os.path.isfile(success_report_filename):
        os.remove(success_report_filename)

    multiway_train_filename = 'data/multiway_train.csv'
    multiway_dev_filename = 'data/multiway_dev.csv'
    multiway_test_filename = 'data/multiway_test.csv'
    multiway_label_key_filename = 'data/multiway_classes.txt'
    output_multiway_model_dir = '../f1-saved_multiway_model/'
    multiway_output_report_filename_stub = 'output_analysis/f1-multiwaybest'
    csv_filename_logreg_on_dev = 'output_analysis/multiwaytask_dev_logregresults.csv'
    csv_filename_roberta_on_dev = 'output_analysis/multiwaytask_dev_robertaresults.csv'
    csv_filename_logreg_on_test = 'output_analysis/multiwaytask_test_logregresults.csv'
    csv_filename_roberta_on_test = 'output_analysis/multiwaytask_test_robertaresults.csv'

    if output_binary_model_dir.endswith('/'):
        output_binary_model_dir = output_binary_model_dir[:-1]
    if output_multiway_model_dir.endswith('/'):
        output_multiway_model_dir = output_multiway_model_dir[:-1]

    previously_extracted_header = None
    tags_to_documents = {}
    with open(full_document_filename, 'r', encoding='utf-8-sig') as f:
        keep_going = True
        while keep_going:
            document, tag, previously_extracted_header = \
                extract_and_tag_next_document(f, previously_extracted_header=previously_extracted_header)
            if document is None:
                keep_going = False
            else:
                tags_to_documents[tag] = document

    if not os.path.isfile(binary_train_filename) or not os.path.isfile(multiway_train_filename):
        positivesentences_tags = load_in_positive_sentences(positive_sentence_filename)

        print('Found ' + str(len(tags_to_documents)) + ' documents in ' + full_document_filename + '.')
        print('Found ' + str(len(positivesentences_tags)) + ' positive sentences in ' + positive_sentence_filename + '.')

        for positivesentence, tag, is_problem_filler, label in positivesentences_tags:
            assert tag in tags_to_documents, "Couldn't find a document corresponding to tag " + str(tag)
        print("All positive sentences' tags have a corresponding document.")

        corresponding_indices_in_document = []
        sentences_with_no_match = 0
        for positive_sentence, tag, is_problem_filler, label in positivesentences_tags:
            index_span = get_indices_of_sentencematch_in_document(tags_to_documents[tag], positive_sentence, tag,
                                                                  problem_report_filename, success_report_filename,
                                                                  is_problem_filler)
            if index_span is None:
                sentences_with_no_match += 1
            corresponding_indices_in_document.append(index_span)
        print('There were ' + str(sentences_with_no_match) + ' out of ' + str(len(positivesentences_tags)) +
              ' positive sentences for which we could not find a match in their corresponding document.')

        positivesentences_tags, corresponding_indices_in_document = \
            clean_positive_sentences(positivesentences_tags, corresponding_indices_in_document, tags_to_documents)
    else:
        positivesentences_tags = []
        corresponding_indices_in_document = []

    if torch.cuda.is_available():
        cuda_device = 0
    else:
        cuda_device = -1

    # binary classification
    train_df, dev_df, test_df, num_labels = \
        get_binary_classification_data(positivesentences_tags, binary_train_filename, binary_dev_filename,
                                       binary_test_filename, binary_label_key_filename, tags_to_documents,
                                       binary_positive_sentences_spot_checking_fname,
                                       binary_negative_sentences_spot_checking_fname, corresponding_indices_in_document)

    label_weights = get_label_weights_and_report_class_imbalance(train_df)

    best_f1 = -1
    best_param = None
    regularization_weights_to_try = [.0001, .001, .01, .1, 1, 10, 100, 1000]
    for regularization_weight in regularization_weights_to_try:
        f1, acc, list_of_all_dev_labels, list_of_all_predicted_dev_labels = \
            run_logreg_classification(train_df, dev_df, regularization_weight=regularization_weight,
                                      label_weights=label_weights, string_prefix='\t', f1_avg='binary')
        if f1 > best_f1:
            best_f1 = f1
            best_param = regularization_weight
    print('For binary case, best baseline logreg model had regularization weight ' + str(best_param) +
          ', and achieved the following performance on the held-out test set:')
    f1, acc, list_of_all_dev_labels, list_of_all_predicted_lr_dev_labels, dev_lr_logits, prec, rec = \
        run_logreg_classification(train_df, dev_df, regularization_weight=best_param,
                                  label_weights=label_weights, string_prefix='(Dev set)  ', f1_avg='binary',
                                  also_output_logits=True, also_report_binary_precrec=True)
    f1, acc, list_of_all_test_labels, list_of_all_predicted_lr_test_labels, test_lr_logits, prec, rec = \
        run_logreg_classification(train_df, test_df, regularization_weight=best_param,
                                  label_weights=label_weights, string_prefix='(Test set) ', f1_avg='binary',
                                  also_output_logits=True, also_report_binary_precrec=True)
    dev_lr_precrec_curve_points = get_recall_precision_curve_points(dev_lr_logits, list_of_all_dev_labels,
                                                                    string_prefix='(Dev for LogReg)  ')
    test_lr_precrec_curve_points = get_recall_precision_curve_points(test_lr_logits, list_of_all_test_labels,
                                                                     string_prefix='(Test for LogReg) ')

    learning_rates_to_try = [1e-5, 2e-5, 3e-5]  # from RoBERTa paper
    batch_sizes_to_try = [32, 16]  # from RoBERTa and BERT papers
    best_f1 = -1
    best_param = None
    for learning_rate in learning_rates_to_try:
        for batch_size in batch_sizes_to_try:
            output_dir = output_binary_model_dir + '_' + str(learning_rate) + '_' + str(batch_size) + '/'
            f1, acc, list_of_all_predicted_dev_labels = \
                run_roberta_classification(train_df, dev_df, num_labels, output_dir, batch_size=batch_size,
                                           learning_rate=learning_rate, label_weights=label_weights,
                                           string_prefix='\t', cuda_device=cuda_device, f1_avg='binary')
            if f1 > best_f1:
                best_f1 = f1
                best_param = (batch_size, learning_rate)
    learning_rate = best_param[1]
    batch_size = best_param[0]
    print('For binary case, best RoBERTa model had lr ' + str(learning_rate) + ' and batch size ' + str(batch_size) +
          '. Performance:')
    output_dir = output_binary_model_dir + '_' + str(learning_rate) + '_' + str(batch_size) + '/'
    dev_f1, dev_acc, list_of_all_predicted_roberta_dev_logits, prec, rec = \
        run_best_model_on(output_dir, dev_df, num_labels, label_weights, lowercase_all_text=True,
                          cuda_device=-1, string_prefix='Dev ', f1_avg='binary', also_report_binary_precrec=True)
    f1, acc, list_of_all_predicted_roberta_test_logits, prec, rec = \
        run_best_model_on(output_dir, test_df, num_labels, label_weights, lowercase_all_text=True,
                          cuda_device=-1, string_prefix='Test ', f1_avg='binary', also_report_binary_precrec=True)
    list_of_all_predicted_roberta_test_labels = \
        clean_roberta_prediction_output(list_of_all_predicted_roberta_test_logits)

    dev_roberta_precrec_curve_points = get_recall_precision_curve_points(list_of_all_predicted_roberta_dev_logits,
                                                                         list_of_all_dev_labels,
                                                                         string_prefix='(Dev for RoBERTa)  ')
    test_roberta_precrec_curve_points = get_recall_precision_curve_points(list_of_all_predicted_roberta_test_logits,
                                                                          list_of_all_test_labels,
                                                                          string_prefix='(Test for RoBERTa) ')

    plot_two_precision_recalls_against_each_other(dev_lr_precrec_curve_points, 'LogReg baseline',
                                                  dev_roberta_precrec_curve_points, 'Finetuned RoBERTa',
                                                  dev_precreccurve_plot_filename,
                                                  plot_title='Precision-recall curve on dev set')
    plot_two_precision_recalls_against_each_other(test_lr_precrec_curve_points, 'LogReg baseline',
                                                  test_roberta_precrec_curve_points, 'Finetuned RoBERTa',
                                                  test_precreccurve_plot_filename,
                                                  plot_title='Precision-recall curve on test set')
    report_mismatches_to_files(binary_output_report_filename_stub, list_of_all_test_labels,
                               list_of_all_predicted_lr_test_labels, list_of_all_predicted_roberta_test_labels,
                               test_df, model_name='RoBERTa')

    print('\n\n')

    # multi-way classification
    train_df, dev_df, test_df, num_labels = \
        get_multi_way_classification_data(positivesentences_tags, multiway_train_filename,
                                          multiway_dev_filename,
                                          multiway_test_filename, multiway_label_key_filename)

    label_weights = get_label_weights_and_report_class_imbalance(train_df)

    best_f1 = -1
    best_param = None
    dev_predictions_of_best_lr_model = None
    regularization_weights_to_try = [.0001, .001, .01, .1, 1, 10, 100, 1000]
    for regularization_weight in regularization_weights_to_try:
        f1, acc, list_of_all_dev_labels, list_of_all_predicted_dev_labels = \
            run_logreg_classification(train_df, dev_df, regularization_weight=regularization_weight,
                                      label_weights=label_weights, string_prefix='\t')
        if f1 > best_f1:
            best_f1 = f1
            best_param = regularization_weight
            dev_predictions_of_best_lr_model = list_of_all_predicted_dev_labels
    print('For multiway case, best baseline logreg model had regularization weight ' + str(best_param) +
          ', and achieved the following performance on the held-out test set:')
    f1, acc, list_of_all_test_labels, list_of_all_predicted_lr_test_labels = \
        run_logreg_classification(train_df, test_df, regularization_weight=best_param,
                                  label_weights=label_weights, string_prefix='')
    make_multilabel_csv(dev_predictions_of_best_lr_model, list_of_all_dev_labels,
                        multiway_label_key_filename, csv_filename_logreg_on_dev,
                        datasplit_label='dev')
    make_multilabel_csv(list_of_all_predicted_lr_test_labels, list_of_all_test_labels,
                        multiway_label_key_filename, csv_filename_logreg_on_test,
                        datasplit_label='test')

    learning_rates_to_try = [1e-5, 2e-5, 3e-5]  # from RoBERTa paper
    batch_sizes_to_try = [32, 16]  # from RoBERTa and BERT papers
    best_f1 = -1
    best_param = None
    for learning_rate in learning_rates_to_try:
        for batch_size in batch_sizes_to_try:
            output_dir = output_multiway_model_dir + '_' + str(learning_rate) + '_' + str(batch_size) + '/'
            f1, acc, list_of_all_predicted_dev_labels = \
                run_roberta_classification(train_df, dev_df, num_labels, output_dir, batch_size=batch_size,
                                           learning_rate=learning_rate, label_weights=label_weights,
                                           string_prefix='\t', cuda_device=cuda_device)
            if f1 > best_f1:
                best_f1 = f1
                best_param = (batch_size, learning_rate)
    learning_rate = best_param[1]
    batch_size = best_param[0]
    print('For multiway case, best RoBERTa model had lr ' + str(learning_rate) + ' and batch size ' + str(batch_size) +
          '. Performance:')
    output_dir = output_multiway_model_dir + '_' + str(learning_rate) + '_' + str(batch_size) + '/'
    f1, acc, list_of_all_predicted_roberta_dev_labels = \
        run_best_model_on(output_dir, dev_df, num_labels, label_weights, lowercase_all_text=True,
                          cuda_device=-1, string_prefix='Dev ', f1_avg='weighted')
    list_of_all_predicted_roberta_dev_labels = \
        clean_roberta_prediction_output(list_of_all_predicted_roberta_dev_labels)
    f1, acc, list_of_all_predicted_roberta_test_labels = \
        run_best_model_on(output_dir, test_df, num_labels, label_weights, lowercase_all_text=True,
                          cuda_device=-1, string_prefix='Test ', f1_avg='weighted')
    list_of_all_predicted_roberta_test_labels = \
        clean_roberta_prediction_output(list_of_all_predicted_roberta_test_labels)

    make_multilabel_csv(list_of_all_predicted_roberta_dev_labels, list_of_all_dev_labels,
                        multiway_label_key_filename, csv_filename_roberta_on_dev,
                        datasplit_label='dev')
    make_multilabel_csv(list_of_all_predicted_roberta_test_labels, list_of_all_test_labels,
                        multiway_label_key_filename, csv_filename_roberta_on_test,
                        datasplit_label='test')
    report_mismatches_to_files(multiway_output_report_filename_stub, list_of_all_test_labels,
                               list_of_all_predicted_lr_test_labels, list_of_all_predicted_roberta_test_labels,
                               test_df, model_name='RoBERTa')


if __name__ == '__main__':
    main()
