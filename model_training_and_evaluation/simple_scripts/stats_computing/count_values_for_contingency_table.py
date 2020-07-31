import sys


fname_stub = sys.argv[1]


bothcorrect = fname_stub + '_bothcorrect.txt'
neithercorrect = fname_stub + '_neithercorrect.txt'
onlycorrectinbaseline = fname_stub + '_onlycorrectinbaseline.txt'
onlycorrectinmodel = fname_stub + '_onlycorrectinmodel.txt'


if 'condensed' in fname_stub:
    print('Going with 6 labels.')
    labels = ['Political', 'Security', 'Rights_not_violated', 'Terrorism', 'Misc', 'Legal']
else:
    print('Going with 12 labels.')
    labels = ['J_Political-Strategic', 'J_Emergency-Policy', 'J_Denial', 'J_Misc', 'J_Utilitarian-Deterrence',
              'J_Terrorism', 'J_Legal_Procedure', 'J_Law-and-order', 'J_Last-resort', 'J_Development-Unity',
              'J_Intelligence', 'J_Intl-Domestic_Precedent']
starts_of_a_new_examples = set()
for label1 in labels:
    for label2 in labels:
        starts_of_a_new_examples.add(label1 + '\t' + label2 + '\t')


def get_num_examples_in_filename(filename, just_starts_with_1=False):
    counter = 0
    associated_labels_unpinged = set()
    for label in labels:
        associated_labels_unpinged.add(label)
    with open(filename, 'r') as f:
        for line in f:
            if just_starts_with_1:
                try:
                    first_tab_ind = line.index('\t')
                    if line[: first_tab_ind] in labels:
                        counter += 1
                        labels_used = [line[: first_tab_ind]]
                        if labels_used[0] in associated_labels_unpinged:
                            associated_labels_unpinged.remove(labels_used[0])
                except:
                    continue
            else:
                try:
                    first_tab_ind = line.index('\t')
                    second_tab_ind = line[first_tab_ind + 1:].index('\t') + first_tab_ind + 1
                    assert line[second_tab_ind] == '\t' and second_tab_ind != first_tab_ind
                    if line[: second_tab_ind + 1] in starts_of_a_new_examples:
                        counter += 1
                        labels_used = line[:second_tab_ind].split('\t')
                        if labels_used[0] in associated_labels_unpinged:
                            associated_labels_unpinged.remove(labels_used[0])
                        if labels_used[1] in associated_labels_unpinged:
                            associated_labels_unpinged.remove(labels_used[1])
                except:
                    continue
    return counter


print('Num correct in both: ' + str(get_num_examples_in_filename(bothcorrect, just_starts_with_1=True)))
print('Num correct in neither: ' + str(get_num_examples_in_filename(neithercorrect)))
print('Num correct in only baseline: ' + str(get_num_examples_in_filename(onlycorrectinbaseline)))
print('Num correct in only model: ' + str(get_num_examples_in_filename(onlycorrectinmodel)))
