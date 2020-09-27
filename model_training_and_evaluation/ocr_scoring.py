from Bio import pairwise2
from string import whitespace, punctuation


def replace_filler_chars_with_none(string_as_string, string_as_list):
    # string_as_list can be longer but not shorter
    cur_stringasstring_index = 0
    for i in range(len(string_as_list)):
        if cur_stringasstring_index >= len(string_as_string):
            string_as_list[i] = None
        elif string_as_list[i] == string_as_string[cur_stringasstring_index]:
            cur_stringasstring_index += 1
        else:
            assert string_as_list[i] == '-', str(string_as_string) + '\n' + str(string_as_list)
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


def match_scorer(char1, char2, whitespace_set):
    if char1 == char2:
        return 1
    elif char1 in whitespace_set and char2 in whitespace_set:
        return 1
    elif char1 not in whitespace_set and char2 not in whitespace_set:
        return -1
    else:
        return -2


def score_string_against_gold(guessed_string, gold_string,
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

    alignments = \
        pairwise2.align.globalcs(gold_string, guessed_string, match_fn=lambda x, y: match_scorer(x, y, whitespace_set),
                                 open=-1.0, extend=-.5)
    alignments = alignments[0]  # any optimal alignments will do
    gold_aligned = [char for char in alignments.seqA]
    guessed_aligned = [char for char in alignments.seqB]
    guessed_aligned = replace_filler_chars_with_none(guessed_string, guessed_aligned)
    gold_aligned = replace_filler_chars_with_none(gold_string, gold_aligned)

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

    if debugging:
        return case_sensitive_score, case_insensitive_score, cur_denom, \
               guessed_aligned, gold_aligned, eligible_for_points
    else:
        return case_sensitive_score, case_insensitive_score, cur_denom


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
            two_lines_to_print = '\n'.join(['  '.join([char if char is not None else '-' for char in line])
                                           for line in [first_list, second_list]])
        else:
            corrected_eligible = ['.' if val else ' ' for val in eligible_piece]
            two_lines_to_print = '\n'.join(['  '.join([char if char is not None else '-' for char in line])
                                            for line in [first_list, second_list, corrected_eligible]])
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


if __name__ == '__main__':
    whitespace_set = set([char for char in whitespace])
    punctuation_set = set([char for char in punctuation])
    gold = 'Just a "test sentence" to see how many inconsistencies we can find!'
    guessed = 'justa " test sentence " to find how many \t problems we can see. '

    print(gold + '\n' + guessed)
    case_score, uncase_score, denom, guessed_aligned, gold_aligned, eligible = \
        score_string_against_gold(guessed, gold, whitespace_set=whitespace_set, punctuation_set=punctuation_set,
                                  debugging=True)
    print_alignments(gold_aligned, guessed_aligned, eligible=eligible)
    print('Case-sensitive score:   ' + str(round(case_score * denom)) + ' / ' + str(denom) + ' (' +
          str(case_score) + ')')
    print('Case-insensitive score: ' + str(round(uncase_score * denom)) + ' / ' + str(denom) + ' (' +
          str(uncase_score) + ')')
