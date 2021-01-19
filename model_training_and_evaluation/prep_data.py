import pandas as pd
from math import inf
import random as random_for_seed_setting
from random import shuffle, random
from string import whitespace, punctuation
from util import make_directories_as_necessary
import spacy
from analyze_doc_similarity import get_list_of_filenames_to_cluster_together


use_spacy = False


def set_custom_boundaries(doc):
    # can modify doc's tokens' is_sent_start attribute here, but not their is_sent_end attribute
    return doc


if use_spacy:
    spacy_tools = spacy.load("en_core_web_sm")
    spacy_tools.add_pipe(set_custom_boundaries, before="parser")


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
        assert sentence == document[stuff_to_return[0]: stuff_to_return[1]]
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


def get_sentence_split_inds_spacy(text):
    assert use_spacy  # otherwise we won't have set up spacy_tools yet
    def get_text_ready_for_spacy(some_text):
        return some_text
    text = get_text_ready_for_spacy(text)
    doc = spacy_tools(text)
    sentences = doc.sents
    all_sentence_starts = [doc[sent.start].idx for sent in sentences]
    all_sentence_starts.append(len(text))
    return all_sentence_starts[1:]


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
    if len(text.rstrip()) not in locations_of_punctuation_marks_to_split_on:
        something_in_here = True
        # check that there's something between this location and the last punctuation mark logged
        if len(locations_of_punctuation_marks_to_split_on) > 0:
            text_to_check = text[locations_of_punctuation_marks_to_split_on[-1] + 1: len(text.rstrip())]
            something_in_here = False
            for char in text_to_check:
                if char not in punctuation and char not in whitespace:
                    something_in_here = True
                    break
        if something_in_here:
            locations_of_punctuation_marks_to_split_on.append(len(text.rstrip()))
    return locations_of_punctuation_marks_to_split_on


def get_lists_of_positive_negative_sentences_from_doc(document, list_of_positive_sentence_inds_in_doc):
    if use_spacy:
        sentence_split_inds = get_sentence_split_inds_spacy(document)
    else:
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
                ((list_of_positive_sentence_inds_in_doc[cur_positive_sentence_ind][0] < split_ind)):
            # this auto-split "sentence" overlaps with a positive one, so it's positive.
            # this is a while loop because it might overlap with multiple positive sentences.
            overlaps_with_positive_sentence = True
            if positive_sentence_overlap_start_ind is None:
                positive_sentence_overlap_start_ind = cur_positive_sentence_ind
            if list_of_positive_sentence_inds_in_doc[cur_positive_sentence_ind][1] <= split_ind:
                # only increment if this positive sentence ends here
                cur_positive_sentence_ind += 1
                # IF THE NEW POSITIVE SENTENCE STARTS BEFORE THE START OF THE CURRENT SENT
                # (this can happen if there's a really long sentence logged :P ), we need to keep incrementing
                # cur_positive_sentence_ind until the start of the cur positive sentence falls at, or after, the
                # start of this sentence
                while cur_positive_sentence_ind < len(list_of_positive_sentence_inds_in_doc) and \
                        (list_of_positive_sentence_inds_in_doc[cur_positive_sentence_ind][0] < span_start):
                    cur_positive_sentence_ind += 1
                if cur_positive_sentence_ind == len(list_of_positive_sentence_inds_in_doc):
                    break
            else:
                break

        if overlaps_with_positive_sentence:
            positive_spans.append((span_start, split_ind))
            source_positive_sentences_to_log = list(range(positive_sentence_overlap_start_ind,
                                                          cur_positive_sentence_ind))
            # now decide whether to add cur_positive_sentence_ind to that list as an overlapping sentence
            if cur_positive_sentence_ind < len(list_of_positive_sentence_inds_in_doc) and \
                    ((list_of_positive_sentence_inds_in_doc[cur_positive_sentence_ind][0] < split_ind)):
                source_positive_sentences_to_log.append(cur_positive_sentence_ind)
            corresponding_source_positive_sentences.append(document[list_of_positive_sentence_inds_in_doc[
                                                                        source_positive_sentences_to_log[0]][0]:
                                                                    list_of_positive_sentence_inds_in_doc[
                                                                        source_positive_sentences_to_log[-1]][1]])
        else:
            negative_spans.append((span_start, split_ind))
        span_start = split_ind
    if len(sentence_split_inds) == 0:
        # the whole document is considered one big sentence
        if len(list_of_positive_sentence_inds_in_doc) > 0:
            positive_spans.append((0, len(document)))
            corresponding_source_positive_sentences.append(document[list_of_positive_sentence_inds_in_doc[0][0]:
                                                                    list_of_positive_sentence_inds_in_doc[-1][1]])
            cur_positive_sentence_ind = len(list_of_positive_sentence_inds_in_doc)
        else:
            negative_spans.append((0, len(document)))
    assert cur_positive_sentence_ind == len(list_of_positive_sentence_inds_in_doc), \
        str(cur_positive_sentence_ind) + ', ' + str(len(list_of_positive_sentence_inds_in_doc)) + '\n' + \
        str(list_of_positive_sentence_inds_in_doc) + '\n' + str(sentence_split_inds) + '\n' + document
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
    #shuffle(list_of_positive_sentences)
    doctag_to_sentencelabellist = {}
    num_positive_sentences = len(list_of_positive_sentences)
    for positive_sentence, tag, is_problem_filler, label in list_of_positive_sentences:
        if tag not in doctag_to_sentencelabellist:
            doctag_to_sentencelabellist[tag] = []
        doctag_to_sentencelabellist[tag].append((positive_sentence, label))
    test_positive_sentences = []
    dev_positive_sentences = []
    train_positive_sentences = []

    num_assigned_so_far_in_test = 0
    num_assigned_so_far_in_dev = 0
    num_assigned_so_far_in_train = 0

    ideal_cutoff_for_test = int(.1 * num_positive_sentences)
    ideal_cutoff_for_dev = int(.1 * num_positive_sentences)
    ideal_cutoff_for_train = num_positive_sentences - ideal_cutoff_for_test - ideal_cutoff_for_dev

    list_of_taglists = get_list_of_filenames_to_cluster_together()
    # convert tags in list into same format as other tags-- right now these are like DEFE_13_1358/IMG_9776
    # needs to be (file name, image name) tuple
    new_list_of_taglists = []
    for taglist in list_of_taglists:
        list_to_add = []
        for tag in taglist:
            parts_of_tag = tag.split('/')
            list_to_add.append((parts_of_tag[0], parts_of_tag[1]))
        new_list_of_taglists.append(list_to_add)
    list_of_taglists = new_list_of_taglists
    shuffle(list_of_taglists)

    for list_of_tags_that_need_to_go_together in list_of_taglists:
        # randomly choose where to put this tag cluster
        where_to_put = random()
        if num_assigned_so_far_in_test > ideal_cutoff_for_test and num_assigned_so_far_in_dev > ideal_cutoff_for_dev:
            # put this in train
            for doctag in list_of_tags_that_need_to_go_together:
                sentence_label_list = doctag_to_sentencelabellist[doctag]
                train_positive_sentences += sentence_label_list
                num_assigned_so_far_in_train += len(sentence_label_list)
        elif num_assigned_so_far_in_test > ideal_cutoff_for_test and \
                num_assigned_so_far_in_train > ideal_cutoff_for_train:
            # put this in dev
            for doctag in list_of_tags_that_need_to_go_together:
                sentence_label_list = doctag_to_sentencelabellist[doctag]
                dev_positive_sentences += sentence_label_list
                num_assigned_so_far_in_dev += len(sentence_label_list)
        elif num_assigned_so_far_in_dev > ideal_cutoff_for_dev and \
                num_assigned_so_far_in_train > ideal_cutoff_for_train:
            # put this in test
            for doctag in list_of_tags_that_need_to_go_together:
                sentence_label_list = doctag_to_sentencelabellist[doctag]
                test_positive_sentences += sentence_label_list
                num_assigned_so_far_in_test += len(sentence_label_list)
        elif num_assigned_so_far_in_test > ideal_cutoff_for_test:
            # choose between putting this in dev and train
            num_to_go_for_dev = ideal_cutoff_for_dev - num_assigned_so_far_in_dev
            num_to_go_for_train = ideal_cutoff_for_train - num_assigned_so_far_in_train
            if where_to_put <= (num_to_go_for_dev / (num_to_go_for_dev + num_to_go_for_train)):
                for doctag in list_of_tags_that_need_to_go_together:
                    sentence_label_list = doctag_to_sentencelabellist[doctag]
                    dev_positive_sentences += sentence_label_list
                    num_assigned_so_far_in_dev += len(sentence_label_list)
            else:
                for doctag in list_of_tags_that_need_to_go_together:
                    sentence_label_list = doctag_to_sentencelabellist[doctag]
                    train_positive_sentences += sentence_label_list
                    num_assigned_so_far_in_train += len(sentence_label_list)
        elif num_assigned_so_far_in_dev > ideal_cutoff_for_dev:
            # choose between putting this in test and train
            num_to_go_for_test = ideal_cutoff_for_test - num_assigned_so_far_in_test
            num_to_go_for_train = ideal_cutoff_for_train - num_assigned_so_far_in_train
            if where_to_put <= (num_to_go_for_test / (num_to_go_for_test + num_to_go_for_train)):
                for doctag in list_of_tags_that_need_to_go_together:
                    sentence_label_list = doctag_to_sentencelabellist[doctag]
                    test_positive_sentences += sentence_label_list
                    num_assigned_so_far_in_test += len(sentence_label_list)
            else:
                for doctag in list_of_tags_that_need_to_go_together:
                    sentence_label_list = doctag_to_sentencelabellist[doctag]
                    train_positive_sentences += sentence_label_list
                    num_assigned_so_far_in_train += len(sentence_label_list)
        elif num_assigned_so_far_in_train > ideal_cutoff_for_train:
            # choose between putting this in dev and test
            num_to_go_for_dev = ideal_cutoff_for_dev - num_assigned_so_far_in_dev
            num_to_go_for_test = ideal_cutoff_for_test - num_assigned_so_far_in_test
            if where_to_put <= (num_to_go_for_dev / (num_to_go_for_dev + num_to_go_for_test)):
                for doctag in list_of_tags_that_need_to_go_together:
                    sentence_label_list = doctag_to_sentencelabellist[doctag]
                    dev_positive_sentences += sentence_label_list
                    num_assigned_so_far_in_dev += len(sentence_label_list)
            else:
                for doctag in list_of_tags_that_need_to_go_together:
                    sentence_label_list = doctag_to_sentencelabellist[doctag]
                    test_positive_sentences += sentence_label_list
                    num_assigned_so_far_in_test += len(sentence_label_list)
        else:
            # choose between putting it in all three
            num_to_go_for_dev = ideal_cutoff_for_dev - num_assigned_so_far_in_dev
            num_to_go_for_test = ideal_cutoff_for_test - num_assigned_so_far_in_test
            num_to_go_for_train = ideal_cutoff_for_train - num_assigned_so_far_in_train
            if where_to_put <= (num_to_go_for_test / (num_to_go_for_train + num_to_go_for_test + num_to_go_for_dev)):
                for doctag in list_of_tags_that_need_to_go_together:
                    sentence_label_list = doctag_to_sentencelabellist[doctag]
                    test_positive_sentences += sentence_label_list
                    num_assigned_so_far_in_test += len(sentence_label_list)
            elif where_to_put <= ((num_to_go_for_test + num_to_go_for_dev) /
                                  (num_to_go_for_train + num_to_go_for_test + num_to_go_for_dev)):
                for doctag in list_of_tags_that_need_to_go_together:
                    sentence_label_list = doctag_to_sentencelabellist[doctag]
                    dev_positive_sentences += sentence_label_list
                    num_assigned_so_far_in_dev += len(sentence_label_list)
            else:
                for doctag in list_of_tags_that_need_to_go_together:
                    sentence_label_list = doctag_to_sentencelabellist[doctag]
                    train_positive_sentences += sentence_label_list
                    num_assigned_so_far_in_train += len(sentence_label_list)

        for tag in list_of_tags_that_need_to_go_together:
            del doctag_to_sentencelabellist[tag]

    doctag_to_sentencelabellist = list(doctag_to_sentencelabellist.items())
    shuffle(doctag_to_sentencelabellist)

    ideal_cutoff_for_test -= num_assigned_so_far_in_test
    ideal_cutoff_for_dev = ideal_cutoff_for_test + ideal_cutoff_for_dev
    ideal_cutoff_for_dev -= num_assigned_so_far_in_dev
    num_assigned_so_far = 0

    for doctag, sentence_label_list in doctag_to_sentencelabellist:
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
    make_directories_as_necessary(train_filename)
    make_directories_as_necessary(dev_filename)
    make_directories_as_necessary(test_filename)
    make_directories_as_necessary(label_key_filename)

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


def read_in_presplit_data(train_filename, dev_filename, test_filename, label_key_filename, shuffle_data=True):
    train_df = fix_df_format(pd.read_csv(train_filename))
    if shuffle_data:
        train_df = train_df.sample(frac=1).reset_index(drop=True)
    dev_df = fix_df_format(pd.read_csv(dev_filename))
    if shuffle_data:
        dev_df = dev_df.sample(frac=1).reset_index(drop=True)
    test_df = fix_df_format(pd.read_csv(test_filename))
    if shuffle_data:
        test_df = test_df.sample(frac=1).reset_index(drop=True)
    num_labels = 0
    if label_key_filename is not None:
        with open(label_key_filename, 'r') as f:
            for line in f:
                if line.strip() != '':
                    num_labels += 1
    return train_df, dev_df, test_df, (num_labels if label_key_filename is not None else None)


def fix_df_format(df):
    df['text'] = df['text'].astype(str)
    df['strlabel'] = df['strlabel'].astype(str)
    df['labels'] = df['labels'].astype(int)
    if 'source_handcoded_sent' in df.columns:
        df['source_handcoded_sent'] = df['source_handcoded_sent'].astype(str)
    if 'contextbefore' in df.columns:
        df['contextbefore'] = df['contextbefore'].astype(str)
    if 'filename' in df.columns:
        df['filename'] = df['filename'].astype(str)
    if 'perplexity' in df.columns:
        df['perplexity'] = df['perplexity'].astype(float)
    return df


def clean_positive_sentences(positivesentences_tags, corresponding_indices_in_document):
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


def get_corresponding_indices_in_document(positivesentences_tags, tags_to_documents, problem_report_filename,
                                          success_report_filename, skip_positive_sents_we_have_no_doc_for=False):
    corresponding_indices_in_document = []
    sentences_with_no_match = 0
    for positive_sentence, tag, is_problem_filler, label in positivesentences_tags:
        if not skip_positive_sents_we_have_no_doc_for:
            index_span = get_indices_of_sentencematch_in_document(tags_to_documents[tag], positive_sentence, tag,
                                                                  problem_report_filename, success_report_filename,
                                                                  is_problem_filler)
            if index_span is None:
                sentences_with_no_match += 1
        else:
            try:
                corr_document = tags_to_documents[tag]
                index_span = get_indices_of_sentencematch_in_document(corr_document, positive_sentence, tag,
                                                                      problem_report_filename, success_report_filename,
                                                                      is_problem_filler)
            except KeyError:
                continue
        corresponding_indices_in_document.append(index_span)
    print('There were ' + str(sentences_with_no_match) + ' out of ' + str(len(positivesentences_tags)) +
          ' positive sentences for which we could not find a match in their corresponding document.')

    positivesentences_tags, corresponding_indices_in_document = \
        clean_positive_sentences(positivesentences_tags, corresponding_indices_in_document)
    return positivesentences_tags, corresponding_indices_in_document


def get_positive_sentences_and_tagdocs(full_doc_fname, positive_sent_fname):
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

    positivesentences_tags = load_in_positive_sentences(positive_sent_fname)

    print('Found ' + str(len(tags_to_documents)) + ' documents in ' + full_doc_fname + '.')
    print('Found ' + str(len(positivesentences_tags)) + ' positive sentences in ' + positive_sent_fname + '.')

    for positivesentence, tag, is_problem_filler, label in positivesentences_tags:
        assert tag in tags_to_documents, "Couldn't find a document corresponding to tag " + str(tag)
    print("All positive sentences' tags have a corresponding document.")
    return positivesentences_tags, tags_to_documents


def make_multiway_data_split(multiway_train_filename, multiway_dev_filename,
                             multiway_test_filename, multiway_label_key_filename, positivesentences_tags=None,
                             positive_sentences_filename=None):
    if positivesentences_tags is None:
        # have to load everything from scratch
        assert positive_sentences_filename is not None
        positivesentences_tags = load_in_positive_sentences(positive_sentences_filename)

    #set_of_positive_sents =
    #for positive_sentence, tag, is_problem_filler, label in list_of_positive_sentences:

    train, dev, test = make_classification_split(positivesentences_tags)
    print('Made new multi-way classification data split.')
    train_df, dev_df, test_df, num_labels = \
        save_splits_as_csv(train, dev, test, multiway_train_filename, multiway_dev_filename, multiway_test_filename,
                           multiway_label_key_filename)
    return train_df, dev_df, test_df, num_labels


def make_binary_data_split(binary_train_filename, binary_dev_filename, binary_test_filename, binary_label_key_filename,
                           binary_positive_sentences_spot_checking_fname,
                           binary_negative_sentences_spot_checking_fname, positivesentences_tags=None,
                           tags_to_documents=None, corresponding_indices_in_document=None, full_document_filename=None,
                           positive_sentence_filename=None, problem_report_filename=None,
                           success_report_filename=None):
    if positivesentences_tags is None or tags_to_documents is None:
        # have to load everything from scratch
        assert full_document_filename is not None and positive_sentence_filename is not None
        positivesentences_tags, tags_to_documents = get_positive_sentences_and_tagdocs(full_document_filename,
                                                                                       positive_sentence_filename)
    elif corresponding_indices_in_document is None:
        assert problem_report_filename is not None and success_report_filename is not None
        make_directories_as_necessary(problem_report_filename)
        make_directories_as_necessary(success_report_filename)
        positivesentences_tags, corresponding_indices_in_document = \
            get_corresponding_indices_in_document(positivesentences_tags, tags_to_documents, problem_report_filename,
                                                  success_report_filename)

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
        save_splits_as_csv(train, dev, test, binary_train_filename, binary_dev_filename, binary_test_filename,
                           binary_label_key_filename,
                           split_ex0_into_two_with_second_label='source_handcoded_sent')

    if binary_positive_sentences_spot_checking_fname is not None:
        make_directories_as_necessary(binary_positive_sentences_spot_checking_fname)
        make_directories_as_necessary(binary_negative_sentences_spot_checking_fname)
        print('Making spot-checking files now...')
        with open(binary_positive_sentences_spot_checking_fname, 'w') as pos_f:
            with open(binary_negative_sentences_spot_checking_fname, 'w') as neg_f:
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

    return train_df, dev_df, test_df, num_labels


def main(full_document_filename, binary_train_filename, binary_dev_filename, binary_test_filename,
         binary_label_key_filename, multiway_train_filename, multiway_dev_filename, multiway_test_filename,
         multiway_label_key_filename, positive_sentence_filename, problem_report_filename,
         success_report_filename, binary_positive_sentences_spot_checking_fname,
         binary_negative_sentences_spot_checking_fname):
    positivesentences_tags, tags_to_documents = get_positive_sentences_and_tagdocs(full_document_filename,
                                                                                   positive_sentence_filename)

    """multiway_train_df, multiway_dev_df, multiway_test_df, multiway_num_labels = \
        make_multiway_data_split(multiway_train_filename, multiway_dev_filename, multiway_test_filename,
                                 multiway_label_key_filename, positivesentences_tags=positivesentences_tags)"""

    positivesentences_tags, corresponding_indices_in_document = \
        get_corresponding_indices_in_document(positivesentences_tags, tags_to_documents, problem_report_filename,
                                              success_report_filename)

    binary_train_df, binary_dev_df, binary_test_df, binary_num_labels = \
        make_binary_data_split(binary_train_filename, binary_dev_filename, binary_test_filename,
                               binary_label_key_filename, binary_positive_sentences_spot_checking_fname,
                               binary_negative_sentences_spot_checking_fname,
                               positivesentences_tags=positivesentences_tags, tags_to_documents=tags_to_documents,
                               corresponding_indices_in_document=corresponding_indices_in_document)

    """return multiway_train_df, multiway_dev_df, multiway_test_df, multiway_num_labels, \
           binary_train_df, binary_dev_df, binary_test_df, binary_num_labels"""


if __name__ == '__main__':
    random_for_seed_setting.seed(5)
    from config import full_document_filename, binary_train_filename, binary_dev_filename, \
        binary_test_filename, binary_label_key_filename, multiway_train_filename, multiway_dev_filename, \
        multiway_test_filename, multiway_label_key_filename, positive_sentence_filename, problem_report_filename, \
        success_report_filename, binary_positive_sentences_spot_checking_fname, \
        binary_negative_sentences_spot_checking_fname
    main(full_document_filename, binary_train_filename, binary_dev_filename, binary_test_filename,
         binary_label_key_filename, multiway_train_filename, multiway_dev_filename, multiway_test_filename,
         multiway_label_key_filename, positive_sentence_filename, problem_report_filename,
         success_report_filename, binary_positive_sentences_spot_checking_fname,
         binary_negative_sentences_spot_checking_fname)
