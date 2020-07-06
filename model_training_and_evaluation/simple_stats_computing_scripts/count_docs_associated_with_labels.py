import sys
sys.path.append('..')


from prep_data import load_in_positive_sentences


use_sixway_label_set = True
fulllabel_to_sixway = {
    'J_Terrorism' : 'Terrorism',
    'J_Intl-Domestic_Precedent': 'Rights_not_violated',
    'J_Intelligence': 'Security',
    'J_Denial': 'Rights_not_violated',
    'J_Misc': 'Misc',
    'J_Political-Strategic': 'Political',
    'J_Development-Unity': 'Misc',
    'J_Legal_Procedure': 'Legal',
    'J_Emergency-Policy': 'Security',
    'J_Law-and-order': 'Security',
    'J_Utilitarian-Deterrence': 'Security',
    'J_Last-resort': 'Security'
}


labels_to_tagsets = {}
positivesentences_tags = load_in_positive_sentences('../../justifications_clean_text_ohe.csv')
for positivesentence, tag, is_problem_filler, label in positivesentences_tags:
    if use_sixway_label_set:
        label = fulllabel_to_sixway[label]
    if not label in labels_to_tagsets:
        labels_to_tagsets[label] = set()
    labels_to_tagsets[label].add(tag)


for label in labels_to_tagsets.keys():
    print(label + ': ' + str(len(labels_to_tagsets[label])) + ' documents')
