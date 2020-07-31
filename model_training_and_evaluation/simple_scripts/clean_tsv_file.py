from math import inf, isnan


filename = '../training_sentence_perplexities.tsv'
new_filename = filename[:filename.rfind('.')] + '-cleaned.tsv'
new_sorted_filename = filename[:filename.rfind('.')] + '-cleanedAndSorted.tsv'


def line_looks_like_new_entry(line):
    try:
        tab_index = line.index('\t')
    except:
        return False
    first_part = line[:tab_index]
    try:
        float(first_part)
        return True
    except:
        return False


perplexities_cleanedsents = []


with open(filename, 'r') as f:
    with open(new_filename, 'w') as new_f:
        firstline = f.readline()
        new_f.write(firstline)

        line = f.readline()
        new_f.write(line[:line.index('\t') + 1])
        most_recent_perplexity = float(line[:line.index('\t')])

        cur_cleaned_line = line[line.index('\t') + 1: -1]

        for line in f:
            if line_looks_like_new_entry(line):
                # time to write the previous line's text, and this line's perplexity
                assert cur_cleaned_line != ''
                cur_cleaned_line = cur_cleaned_line.replace('\t', '\\t')
                if cur_cleaned_line.startswith('""') and cur_cleaned_line.endswith('""'):
                    line_content = cur_cleaned_line[2: -2]
                    if '\\n' in line_content and '"' in line_content:
                        cur_cleaned_line = line_content
                elif cur_cleaned_line.startswith('"') and cur_cleaned_line.endswith('"'):
                    line_content = cur_cleaned_line[1: -1]
                    if '\\n' in line_content:
                        cur_cleaned_line = line_content
                new_f.write(cur_cleaned_line + '\n')
                perplexities_cleanedsents.append((most_recent_perplexity, cur_cleaned_line))

                new_f.write(line[:line.index('\t') + 1])
                most_recent_perplexity = float(line[:line.index('\t')])
                cur_cleaned_line = line[line.index('\t') + 1: -1]
            else:
                cur_cleaned_line += '\\n' + line[:-1]

        assert cur_cleaned_line != ''
        cur_cleaned_line = cur_cleaned_line.replace('\t', '\\t')
        if cur_cleaned_line.startswith('""') and cur_cleaned_line.endswith('""'):
            line_content = cur_cleaned_line[2: -2]
            if '\\n' in line_content and '"' in line_content:
                cur_cleaned_line = line_content
        elif cur_cleaned_line.startswith('"') and cur_cleaned_line.endswith('"'):
            line_content = cur_cleaned_line[1: -1]
            if '\\n' in line_content:
                cur_cleaned_line = line_content
        new_f.write(cur_cleaned_line + '\n')
        perplexities_cleanedsents.append((most_recent_perplexity, cur_cleaned_line))

print('Done writing ' + new_filename)


perplexities_cleanedsents = sorted(perplexities_cleanedsents, key=lambda x: inf if isnan(x[0]) else x[0])
with open(new_sorted_filename, 'w') as f:
    f.write(firstline)
    for perplexity, sentence_without_newline in perplexities_cleanedsents:
        f.write('\t'.join([str(perplexity), str(sentence_without_newline)]) + '\n')
print('Done writing ' + new_sorted_filename)
