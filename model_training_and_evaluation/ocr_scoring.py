from Bio import pairwise2
from string import whitespace, punctuation
from math import inf


place_a_gap_with_me_char = chr(983041)
open_gap_penalty = -2.0
extend_gap_penalty = -.5


default_block_length = 596
NEITHER_HAS_IT_PREPENDED = 0
BOTH_HAVE_IT_PREPENDED = 1
GOLD_ONLY_HAS_IT_PREPENDED = 2
GUESSED_ONLY_HAS_IT_PREPENDED = 3


def replace_filler_chars_with_none(string_as_string, string_as_list):
    # string_as_list can be longer but not shorter
    cur_stringasstring_index = 0
    for i in range(len(string_as_list)):
        if cur_stringasstring_index >= len(string_as_string):
            string_as_list[i] = None
        elif string_as_list[i] == string_as_string[cur_stringasstring_index]:
            cur_stringasstring_index += 1
        else:
            assert string_as_list[i] == '', str(string_as_string) + '\n' + str(string_as_list)
            string_as_list[i] = None
    return string_as_list


def remove_multiples_of_whitespace(string_to_fix, whitespace_set):
    string_to_fix = string_to_fix.strip()
    whitespace_chars_in_string = set()
    for char in string_to_fix:
        if char in whitespace_set:
            whitespace_chars_in_string.add(char)
    for char in whitespace_chars_in_string:
        if char != ' ':
            string_to_fix = string_to_fix.replace(char, ' ')
    fixed_string = string_to_fix.replace('  ', ' ')
    while fixed_string != string_to_fix:
        string_to_fix = fixed_string
        fixed_string = string_to_fix.replace('  ', ' ')
    return fixed_string


def standardize_whitespace_around_punctuation_in_string(string_to_fix, whitespace_set=None,
                                                        already_set_whitespace_to_spaces=False):
    # we assume multiple consecutive whitespaces have already been removed
    if not already_set_whitespace_to_spaces:
        assert whitespace_set is not None
        whitespace_chars_in_string = set()
        for char in string_to_fix:
            if char in whitespace_set:
                whitespace_chars_in_string.add(char)
        for char in whitespace_chars_in_string:
            if char != ' ':
                string_to_fix = string_to_fix.replace(char, ' ')
    string_to_fix = string_to_fix.replace(' ,', ',')
    string_to_fix = string_to_fix.replace(' .', '.')
    string_to_fix = string_to_fix.replace(' !', '!')
    string_to_fix = string_to_fix.replace(' ?', '?')
    return string_to_fix


def next_to_a_matching_punctuation_mark(list1, list2, index, punctuation_set):
    if index - 1 >= 0:
        list1_char = list1[index - 1]
        list2_char = list2[index - 1]
        if list1_char is not None and list2_char is not None and list1_char == list2_char and \
                list1_char in punctuation_set:
            return True
    if index + 1 < len(list1):
        list1_char = list1[index + 1]
        list2_char = list2[index + 1]
        if list1_char is not None and list2_char is not None and list1_char == list2_char and \
                list1_char in punctuation_set:
            return True
    return False


def match_scorer(char1, char2, whitespace_set, penalize_for_case_errors_in_match):
    if char1 == char2:
        return 1
    elif char1 == place_a_gap_with_me_char or char2 == place_a_gap_with_me_char:
        return min(open_gap_penalty, extend_gap_penalty) - 2
    elif (not penalize_for_case_errors_in_match) and char1.lower() == char2.lower():
        return 1
    elif char1 in whitespace_set and char2 in whitespace_set:
        return 1
    elif char1 not in whitespace_set and char2 not in whitespace_set:
        return -1
    else:
        return -2  # don't confuse characters for whitespace, it's not something that tends to happen much


def get_token_correct_was_token(guessed_aligned_string, gold_aligned_string, start_ind, end_ind,
                                punctuation_set, whitespace_set):
    # returns three things: token_fully_correct, token_correct_case_insensitive, add_to_denom
    # there is guaranteed not to be any whitespace inside these indices of the gold aligned string list
    token_fully_correct = True
    token_correct_case_insensitive = True

    has_gotten_case_sensitive_char_wrong_in_cur_token = False
    has_gotten_case_insensitive_char_wrong_in_cur_token = False
    # find non-punctuation, non-whitespace "start index" of actual token
    start_ind_of_contents = None
    for i in range(start_ind, end_ind):
        if gold_aligned_string[i] is None:
            continue
        elif gold_aligned_string[i] not in punctuation_set:
            start_ind_of_contents = i
            break
    if start_ind_of_contents is not None:
        theres_a_token_here = True
        # find non-punctuation, non-whitespace "end index" of actual token
        end_ind_of_contents_inclusive = None
        for i in range(end_ind - 1, start_ind - 1, -1):
            if gold_aligned_string[i] is None:
                continue
            elif gold_aligned_string[i] not in punctuation_set:
                end_ind_of_contents_inclusive = i
                break
        assert end_ind_of_contents_inclusive is not None

        # now check whether there's whitespace or punctuation on either side of the token contents in the GUESSED string
        closest_guessed_char_before_token_contents = None
        for i in range(start_ind_of_contents - 1, -1, -1):
            if guessed_aligned_string[i] is None:
                continue
            else:
                closest_guessed_char_before_token_contents = guessed_aligned_string[i]
                break
        closest_guessed_char_after_token_contents = None
        for i in range(end_ind_of_contents_inclusive + 1, len(guessed_aligned_string)):
            if guessed_aligned_string[i] is None:
                continue
            else:
                closest_guessed_char_after_token_contents = guessed_aligned_string[i]
                break
        if not ((closest_guessed_char_before_token_contents is None or  # this happens at the ends of the string
                 closest_guessed_char_before_token_contents in whitespace_set or
                 closest_guessed_char_before_token_contents in punctuation_set)
                and
                (closest_guessed_char_after_token_contents is None or  # this happens at the ends of the string
                 closest_guessed_char_after_token_contents in whitespace_set or
                 closest_guessed_char_after_token_contents in punctuation_set)):
            token_fully_correct = False
            token_correct_case_insensitive = False
        else:

            # now check whether there are mistakes inside the gold token contents
            for i in range(start_ind_of_contents, end_ind_of_contents_inclusive + 1):
                gold_char = gold_aligned_string[i]
                guessed_char = guessed_aligned_string[i]
                if gold_char is None and guessed_char is None:
                    continue
                elif gold_char is None and guessed_char is not None:
                    token_fully_correct = False
                    token_correct_case_insensitive = False
                    break
                elif gold_char is not None and guessed_char is None:
                    token_fully_correct = False
                    token_correct_case_insensitive = False
                    break

                if gold_char != guessed_char:
                    token_fully_correct = False
                if gold_char.lower() != guessed_char.lower():
                    token_correct_case_insensitive = False
                    break
    else:
        theres_a_token_here = False

    return int(token_fully_correct and theres_a_token_here), \
           int(token_correct_case_insensitive and theres_a_token_here), \
           int(theres_a_token_here)


def get_word_level_accuracy_from_aligned_strings(guessed_aligned_string, gold_aligned_string, whitespace_set,
                                                 punctuation_set):
    cur_denom = 0
    case_sensitive_words_right = 0
    case_insensitive_words_right = 0
    most_recent_whitespace_index = -1
    for i in range(len(gold_aligned_string)):
        # we count a word as correct if all non-punctuation, non-whitespace tokens inside it are
        # correct, and it's bordered with either punctuation or whitespace on either side
        # steps:
        # - identify the closest whitespace to either side
        # - "strip off" (i.e. don't count mistakes in) punctuation on outsides of whitespace-delimited token
        # - check whether there are mistakes in what's left
        if gold_aligned_string[i] is None:
            continue
        elif gold_aligned_string[i] in whitespace_set:
            # we've found the end of a token
            fully_correct, case_insensitive_correct, add_to_denom = \
                get_token_correct_was_token(guessed_aligned_string, gold_aligned_string,
                                            most_recent_whitespace_index + 1, i, punctuation_set, whitespace_set)
            most_recent_whitespace_index = i
            cur_denom += add_to_denom
            case_sensitive_words_right += fully_correct
            case_insensitive_words_right += case_insensitive_correct
    fully_correct, case_insensitive_correct, add_to_denom = \
        get_token_correct_was_token(guessed_aligned_string, gold_aligned_string,
                                    most_recent_whitespace_index + 1, len(gold_aligned_string), punctuation_set,
                                    whitespace_set)
    cur_denom += add_to_denom
    case_sensitive_words_right += fully_correct
    case_insensitive_words_right += case_insensitive_correct

    case_sensitive_score = 0 if cur_denom == 0 else case_sensitive_words_right / cur_denom
    case_insensitive_score = 0 if cur_denom == 0 else case_insensitive_words_right / cur_denom
    return case_sensitive_score, case_insensitive_score, cur_denom


def score_string_against_gold(guessed_string, gold_string, penalize_for_case_errors_in_match=True,
                              fraction_of_token_correct_to_count_as_attempt_at_right_word=0.2,
                              whitespace_set=None, punctuation_set=None, debugging=False):
    if whitespace_set is None:
        whitespace_set = set([char for char in whitespace])
    if punctuation_set is None:
        punctuation_set = set([char for char in punctuation])

    guessed_string = guessed_string.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
    gold_string = gold_string.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")

    guessed_string = remove_multiples_of_whitespace(guessed_string, whitespace_set)
    gold_string = remove_multiples_of_whitespace(gold_string, whitespace_set)

    guessed_string = standardize_whitespace_around_punctuation_in_string(guessed_string,
                                                                         already_set_whitespace_to_spaces=True)
    gold_string = standardize_whitespace_around_punctuation_in_string(gold_string,
                                                                      already_set_whitespace_to_spaces=True)

    if max(len(gold_string), len(guessed_string)) <= default_block_length:
        alignments = \
            pairwise2.align.globalcs(list(gold_string), list(guessed_string),
                                     match_fn=lambda x, y: match_scorer(x, y, whitespace_set,
                                                                        penalize_for_case_errors_in_match),
                                     open=open_gap_penalty, extend=extend_gap_penalty, gap_char=[''])
        alignments = alignments[0]
        gold_aligned = [char for char in alignments.seqA]
        guessed_aligned = [char for char in alignments.seqB]
        guessed_aligned = replace_filler_chars_with_none(guessed_string, guessed_aligned)
        gold_aligned = replace_filler_chars_with_none(gold_string, gold_aligned)
    else:
        gold_aligned, guessed_aligned = align_long_sequences_by_frankensteining_shorter_alignments(gold_string,
                                                                                                   guessed_string)

    eligible_for_points = [True] * len(guessed_aligned)
    last_gold_whitespace_location = -1
    for i in range(len(gold_aligned)):
        # for each whitespace-delimited token in gold (allowed to have Nones as part), guessed token must match >=
        # fraction_of_token_correct_to_count_as_attempt_at_right_word * len(gold_token including Nones)
        # to be eligible for points for correctness.
        # (this is done to prevent giving partial OCR correctness points for certain methods that post-correct
        #  to different words without paying attention to the visual signal at all.)
        if last_gold_whitespace_location < i - 1 and \
                ((isinstance(gold_aligned[i], str) and gold_aligned[i] == ' ') or
                 (i == len(gold_aligned) - 1)):
            # then we've found the boundaries of a token, so evaluate whether to change the corresponding
            # eligible_for_points values to False
            token_beginning = last_gold_whitespace_location + 1
            token_end_plus_1 = i
            if i == len(gold_aligned) - 1:
                token_end_plus_1 = len(gold_aligned)
            else:
                last_gold_whitespace_location = i
            denom = token_end_plus_1 - token_beginning
            correct = 0
            for j in range(token_beginning, token_end_plus_1):
                if gold_aligned[j] is None and guessed_aligned[j] is None:
                    correct += 1
                elif gold_aligned[j] is not None and guessed_aligned[j] is not None and \
                        gold_aligned[j].lower() == guessed_aligned[j].lower():
                    correct += 1
            if correct / denom < fraction_of_token_correct_to_count_as_attempt_at_right_word:
                for j in range(token_beginning, token_end_plus_1):
                    eligible_for_points[j] = False
        elif gold_aligned[i] == ' ':
            last_gold_whitespace_location = i

    # scoring:
    # denom = # non-null chars in gold + # null chars in gold *where guessed char is a problem*
    # (only case when this would be true: guessed char is whitespace, and there's punctuation bordering it on at
    #  least one side)
    cur_denom = 0
    cur_correct_including_case = 0
    cur_correct_case_insensitive = 0
    for i in range(len(guessed_aligned)):
        cur_gold_char = gold_aligned[i]
        cur_guessed_char = guessed_aligned[i]

        if cur_gold_char is None and cur_guessed_char is None:
            pass
        elif cur_gold_char is None:  # and cur_guessed_char is not None
            if cur_guessed_char in whitespace_set:
                if next_to_a_matching_punctuation_mark(gold_aligned, guessed_aligned, i, punctuation_set):
                    pass  # we skip over this entry as it technically should've been edited out
                else:
                    cur_denom += 1
            else:
                cur_denom += 1
        elif cur_guessed_char is None:  # and cur_gold_char is not None
            if cur_gold_char in whitespace_set:
                if next_to_a_matching_punctuation_mark(gold_aligned, guessed_aligned, i, punctuation_set):
                    pass  # we skip over this entry as it technically should've been edited out
                else:
                    cur_denom += 1
            else:
                cur_denom += 1
        else:
            # doesn't matter if one is whitespace or not, this clearly isn't a case of whitespace gap discrepancy
            cur_denom += 1
            if eligible_for_points[i]:
                if cur_gold_char == cur_guessed_char:
                    cur_correct_including_case += 1
                    cur_correct_case_insensitive += 1
                elif cur_gold_char.lower() == cur_guessed_char.lower():
                    cur_correct_case_insensitive += 1

    case_sensitive_score = cur_correct_including_case / cur_denom
    case_insensitive_score = cur_correct_case_insensitive / cur_denom

    wordlevel_case_sensitive_score, wordlevel_case_insensitive_score, wordlevel_denom = \
        get_word_level_accuracy_from_aligned_strings(guessed_aligned, gold_aligned, whitespace_set,
                                                     punctuation_set)

    if debugging:
        return case_sensitive_score, case_insensitive_score, cur_denom, \
               wordlevel_case_sensitive_score, wordlevel_case_insensitive_score, wordlevel_denom, \
               guessed_aligned, gold_aligned, eligible_for_points
    else:
        return case_sensitive_score, case_insensitive_score, cur_denom, \
               wordlevel_case_sensitive_score, wordlevel_case_insensitive_score, wordlevel_denom


def get_index_of_nth_nonnull_char_in_list(list_of_chars, n):
    total_num_nonnull_chars_passed = 0
    for i, char in enumerate(list_of_chars):
        if total_num_nonnull_chars_passed == n:
            return i - 1
        if char is not None:
            total_num_nonnull_chars_passed += 1
    if total_num_nonnull_chars_passed == n:
        return len(list_of_chars) - 1
    return None


def get_num_nonnull_chars_before_char_index(list_of_chars, i):
    num_nonnull_chars = 0
    for j in range(i):
        if list_of_chars[j] is not None:
            num_nonnull_chars += 1
    return num_nonnull_chars


def align_long_sequences_by_frankensteining_shorter_alignments(full_gold_string, full_guessed_string,
                                                               whitespace_set=None,
                                                               penalize_for_case_errors_in_match=True,
                                                               block_length=default_block_length):
    # assumes roughly equal segment lengths
    assert block_length % 4 == 0
    quarter_block_length = block_length // 4

    if whitespace_set is None:
        whitespace_set = set([char for char in whitespace])

    # do it in overlapping blocks of [block_length] characters
    start_of_next_gold_block = 0
    start_of_next_guessed_block = 0
    frankensteined_gold = []
    frankensteined_guessed = []
    finalized_gold_so_far_ended_with_gap = None
    finalized_guessed_so_far_ended_with_gap = None
    while True:
        cur_block_gold_string = full_gold_string[start_of_next_gold_block: start_of_next_gold_block + block_length]
        cur_block_guessed_string = full_guessed_string[start_of_next_guessed_block:
                                                       start_of_next_guessed_block + block_length]

        gold_has_more_chars_left = (len(full_gold_string) - start_of_next_gold_block) >= \
                                   (len(full_guessed_string) - start_of_next_guessed_block)

        if finalized_gold_so_far_ended_with_gap is not None and finalized_guessed_so_far_ended_with_gap is not None:
            assert not (finalized_gold_so_far_ended_with_gap and finalized_guessed_so_far_ended_with_gap)
            if not (finalized_gold_so_far_ended_with_gap or finalized_guessed_so_far_ended_with_gap):
                # then prepend a character to both that HAS to be paired together (penalty for not pairing is high)
                cur_block_guessed_string = place_a_gap_with_me_char + cur_block_guessed_string
                cur_block_gold_string = place_a_gap_with_me_char + cur_block_gold_string
                prepended_chars = BOTH_HAVE_IT_PREPENDED
            elif finalized_gold_so_far_ended_with_gap:
                # then prepend a character to guessed that HAS to be paired with a gap
                cur_block_guessed_string = place_a_gap_with_me_char + cur_block_guessed_string
                prepended_chars = GUESSED_ONLY_HAS_IT_PREPENDED
            elif finalized_guessed_so_far_ended_with_gap:
                cur_block_gold_string = place_a_gap_with_me_char + cur_block_gold_string
                prepended_chars = GOLD_ONLY_HAS_IT_PREPENDED
        else:
            prepended_chars = NEITHER_HAS_IT_PREPENDED

        alignments = \
            pairwise2.align.globalcs(list(cur_block_gold_string), list(cur_block_guessed_string),
                                     match_fn=lambda x, y: match_scorer(x, y, whitespace_set,
                                                                        penalize_for_case_errors_in_match),
                                     open=open_gap_penalty, extend=extend_gap_penalty, gap_char=[''])
        alignments = alignments[0]  # todo: maybe look through the different alignments to find the least splitty?
        gold_aligned = [char for char in alignments.seqA]
        guessed_aligned = [char for char in alignments.seqB]
        guessed_aligned = replace_filler_chars_with_none(cur_block_guessed_string, guessed_aligned)
        gold_aligned = replace_filler_chars_with_none(cur_block_gold_string, gold_aligned)
        if prepended_chars != NEITHER_HAS_IT_PREPENDED:
            handled_removal_in_special_case = False
            if prepended_chars == BOTH_HAVE_IT_PREPENDED:
                assert gold_aligned[0] == place_a_gap_with_me_char and guessed_aligned[0] == place_a_gap_with_me_char, \
                    str(guessed_aligned[:5]) + '\n' + str(gold_aligned[:5])
            elif prepended_chars == GOLD_ONLY_HAS_IT_PREPENDED:
                if not (gold_aligned[0] == place_a_gap_with_me_char and guessed_aligned[0] is None):
                    for j in range(len(gold_aligned)):
                        if gold_aligned[j] is not None:
                            index_of_char_with_gap = j
                            break
                    assert guessed_aligned[index_of_char_with_gap] is None and \
                           gold_aligned[index_of_char_with_gap] == place_a_gap_with_me_char, \
                        str(guessed_aligned[:5 + index_of_char_with_gap]) + '\n' + \
                        str(gold_aligned[:5 + index_of_char_with_gap])
                    del guessed_aligned[index_of_char_with_gap]
                    del gold_aligned[index_of_char_with_gap]
                    handled_removal_in_special_case = True
            elif prepended_chars == GUESSED_ONLY_HAS_IT_PREPENDED:
                if not (guessed_aligned[0] == place_a_gap_with_me_char and gold_aligned[0] is None):
                    for j in range(len(guessed_aligned)):
                        if guessed_aligned[j] is not None:
                            index_of_char_with_gap = j
                            break
                    assert gold_aligned[index_of_char_with_gap] is None and \
                           guessed_aligned[index_of_char_with_gap] == place_a_gap_with_me_char, \
                        str(guessed_aligned[:6 + index_of_char_with_gap]) + '\n' + \
                        str(gold_aligned[:6 + index_of_char_with_gap])
                    del guessed_aligned[index_of_char_with_gap]
                    del gold_aligned[index_of_char_with_gap]
                    handled_removal_in_special_case = True
            if not handled_removal_in_special_case:
                gold_aligned = gold_aligned[1:]
                guessed_aligned = guessed_aligned[1:]

        # we have one point of interest:
        #    right after first 3 * quarter_block_length characters in whichever sequence that happens first
        gold_third_quarter = get_index_of_nth_nonnull_char_in_list(gold_aligned, 3 * quarter_block_length + 1)
        guessed_third_quarter = get_index_of_nth_nonnull_char_in_list(guessed_aligned, 3 * quarter_block_length + 1)
        if gold_third_quarter is not None and guessed_third_quarter is not None:
            third_quarter = min(gold_third_quarter, guessed_third_quarter)
            came_from_gold = third_quarter == gold_third_quarter
        elif gold_third_quarter is not None:
            third_quarter = gold_third_quarter
            came_from_gold = True
        else:
            third_quarter = guessed_third_quarter
            came_from_gold = False
        if third_quarter is None:
            third_quarter = max(len(gold_aligned), len(guessed_aligned))

        frankensteined_gold += gold_aligned[:third_quarter]
        frankensteined_guessed += guessed_aligned[:third_quarter]
        if frankensteined_gold[-1] is None:
            finalized_gold_so_far_ended_with_gap = True
        else:
            finalized_gold_so_far_ended_with_gap = False
        if frankensteined_guessed[-1] is None:
            finalized_guessed_so_far_ended_with_gap = True
        else:
            finalized_guessed_so_far_ended_with_gap = False

        # adjust next block start and calculate number of characters not added into frankensteined totals yet
        if came_from_gold:
            start_of_next_gold_block += (3 * quarter_block_length)
            num_nonnull_guessed_chars_by_3rdq = get_num_nonnull_chars_before_char_index(guessed_aligned,
                                                                                        third_quarter)
            start_of_next_guessed_block += num_nonnull_guessed_chars_by_3rdq
        else:
            start_of_next_guessed_block += (3 * quarter_block_length)
            num_nonnull_gold_chars_by_3rdq = get_num_nonnull_chars_before_char_index(gold_aligned,
                                                                                     third_quarter)
            start_of_next_gold_block += num_nonnull_gold_chars_by_3rdq

        # check whether we're actually done (i.e., we've run out of characters to process in >=1 of our sequences)
        if start_of_next_guessed_block >= len(full_guessed_string):
            # collect the remaining gold chars
            list_to_append = list(full_gold_string[start_of_next_gold_block:])
            frankensteined_gold += list_to_append
            frankensteined_guessed += ['' for i in range(len(list_to_append))]
            break
        elif start_of_next_gold_block >= len(full_gold_string):
            # collect the remaining guessed chars
            list_to_append = list(full_guessed_string[start_of_next_guessed_block:])
            frankensteined_gold += list_to_append
            frankensteined_guessed += ['' for i in range(len(list_to_append))]
            break

    return frankensteined_gold, frankensteined_guessed


def show_problem_area(alignment_list, original_doc, i, cur_list_ind, halfwindow=5):
    shortened_original_doc = original_doc[max(0, i - halfwindow): i + halfwindow].replace('\n', '\\n')
    ind_of_char_i_in_list = get_index_of_nth_nonnull_char_in_list(alignment_list, i + 1)
    str_list = ['' if val is None else ('\\n' if val == '\n' else val) for val in alignment_list]
    assert cur_list_ind == ind_of_char_i_in_list, str(cur_list_ind) + ', ' + str(ind_of_char_i_in_list) + ', "' + \
                                                  alignment_list[ind_of_char_i_in_list] + '", ' + \
                                                  ''.join(str_list[ind_of_char_i_in_list - halfwindow:
                                                                   ind_of_char_i_in_list + halfwindow]) + ', ' + \
                                                  shortened_original_doc
    if i - halfwindow < 0:
        list_start_ind = 0
    else:
        list_start_ind = get_index_of_nth_nonnull_char_in_list(alignment_list, i - halfwindow + 1)
    list_end_ind = get_index_of_nth_nonnull_char_in_list(alignment_list, i + halfwindow + 1)
    if list_end_ind is None:
        shortened_alignment_list = str_list[list_start_ind:]
    else:
        shortened_alignment_list = str_list[list_start_ind: list_end_ind]
    str_to_return = '\n'
    str_to_return += 'Fraction through doc before this occurs: ' + "{:.2f}".format(i / len(original_doc)) + '\n'
    str_to_return += 'Problem char (in doc):      "' + original_doc[i] + '"\n'
    str_to_return += 'Problem char (in list doc): "' + alignment_list[cur_list_ind] + '"\n'
    str_to_return += '/'.join(list(shortened_original_doc)) + '\n'
    str_to_return += '/'.join(shortened_alignment_list)
    return str_to_return


def check_alignment_list_has_same_sequence_of_characters_as_original_doc(alignment_list, original_doc, halfwindow=5):
    cur_alignment_list_ind = 0
    for i, char in enumerate(original_doc):
        while cur_alignment_list_ind < len(alignment_list) and alignment_list[cur_alignment_list_ind] is None:
            cur_alignment_list_ind += 1
        assert cur_alignment_list_ind < len(alignment_list), show_problem_area(alignment_list, original_doc, i,
                                                                               cur_alignment_list_ind,
                                                                               halfwindow=halfwindow)
        assert char == alignment_list[cur_alignment_list_ind], show_problem_area(alignment_list, original_doc, i,
                                                                                 cur_alignment_list_ind,
                                                                                 halfwindow=halfwindow)
        cur_alignment_list_ind += 1


def print_alignments(list1, list2, eligible=None, max_line_length=140):
    # if eligible is provided, then will also print '.' to indicate that an aligned character pair was eligible
    # to score points for the guessed sequence, or ' ' to indicate that it wasn't (due to not being a convincing
    # enough attempt at the word)
    max_num_chars_per_line = int(max_line_length / 3)
    if eligible is None:
        print_eligible = False
    else:
        assert isinstance(eligible, list) and len(eligible) == len(list1)
        print_eligible = True

    def single_line_join(first_list, second_list, eligible_piece=None):
        if not print_eligible:
            two_lines_to_print = ['  '.join([char if char is not None else '-' for char in line])
                                  for line in [first_list, second_list]]
            for i in range(len(two_lines_to_print)):
                two_lines_to_print[i] = two_lines_to_print[i].replace('\n', '\\')
            two_lines_to_print = '\n'.join(two_lines_to_print)
        else:
            corrected_eligible = ['.' if val else ' ' for val in eligible_piece]
            two_lines_to_print = ['  '.join([char if char is not None else '-' for char in line])
                                  for line in [first_list, second_list, corrected_eligible]]
            for i in range(len(two_lines_to_print)):
                two_lines_to_print[i] = two_lines_to_print[i].replace('\n', '\\')
            two_lines_to_print = '\n'.join(two_lines_to_print)
        return two_lines_to_print

    num_batches_to_run = len(list1) / max_num_chars_per_line
    if num_batches_to_run != int(num_batches_to_run):
        num_batches_to_run = int(num_batches_to_run) + 1
    else:
        num_batches_to_run = int(num_batches_to_run)

    line_pairs = []
    for batch_ind in range(num_batches_to_run):
        if print_eligible:
            line_pairs.append(single_line_join(
                list1[max_num_chars_per_line * batch_ind: max_num_chars_per_line * (batch_ind + 1)],
                list2[max_num_chars_per_line * batch_ind: max_num_chars_per_line * (batch_ind + 1)],
                eligible_piece=eligible[max_num_chars_per_line * batch_ind: max_num_chars_per_line * (batch_ind + 1)]))
        else:
            line_pairs.append(single_line_join(
                list1[max_num_chars_per_line * batch_ind: max_num_chars_per_line * (batch_ind + 1)],
                list2[max_num_chars_per_line * batch_ind: max_num_chars_per_line * (batch_ind + 1)]))

    print(('\n' + ('=' * max_line_length) + '\n').join(line_pairs))


def main():
    whitespace_set = set([char for char in whitespace])
    punctuation_set = set([char for char in punctuation])
    gold = 'Just a "test sentence" to see how many inconsistencies we can find!'
    guessed = 'justa " test sentence " to find how many \t problems we can see. '

    print(gold + '\n' + guessed)
    case_score, uncase_score, denom, \
    wordlevel_case_sensitive_score, wordlevel_case_insensitive_score, wordlevel_denom, \
    guessed_aligned, gold_aligned, eligible = \
        score_string_against_gold(guessed, gold, whitespace_set=whitespace_set, punctuation_set=punctuation_set,
                                  debugging=True)
    print_alignments(gold_aligned, guessed_aligned, eligible=eligible)
    print('Case-sensitive score:   ' + str(round(case_score * denom)) + ' / ' + str(denom) + ' (' +
          str(case_score) + ')')
    print('Case-insensitive score: ' + str(round(uncase_score * denom)) + ' / ' + str(denom) + ' (' +
          str(uncase_score) + ')')
    print('Word-level case-sensitive score:   ' + str(round(wordlevel_case_sensitive_score * wordlevel_denom)) +
          ' / ' + str(wordlevel_denom) + ' (' + str(wordlevel_case_sensitive_score) + ')')
    print('Word-level case-insensitive score: ' + str(round(wordlevel_case_insensitive_score * wordlevel_denom)) +
          ' / ' + str(wordlevel_denom) + ' (' + str(wordlevel_case_insensitive_score) + ')')


if __name__ == '__main__':
    main()
