import sys
sys.path.append('..')


from prep_data import load_in_positive_sentences


labels_to_tagsets = {}
positivesentences_tags = load_in_positive_sentences('../justifications_clean_text_ohe.csv')
for positivesentence, tag, is_problem_filler, label in positivesentences_tags:
    if not label in labels_to_tagsets:
        labels_to_tagsets[label] = set()
    labels_to_tagsets[label].add(tag)


for label in labels_to_tagsets.keys():
    print(label + ': ' + str(len(labels_to_tagsets[label])) + ' documents')
