"""
This script makes the assumption that from the stub you provide, the training file will be stub + 'train.csv',
the dev file will be stub + 'dev.csv', the test file will be stub + 'test.csv', and the class key file will
be stub + 'classes.txt'
"""

import sys
from prep_data import read_in_presplit_data, save_splits_as_csv


filename_stub = sys.argv[1]
new_filename_stub = sys.argv[2]


oldlabel_to_newlabel_dict = {
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

####################################################

train_fname = filename_stub + 'train.csv'
dev_fname = filename_stub + 'dev.csv'
test_fname = filename_stub + 'test.csv'
classkey_fname = filename_stub + 'classes.txt'

new_train_fname = new_filename_stub + 'train.csv'
new_dev_fname = new_filename_stub + 'dev.csv'
new_test_fname = new_filename_stub + 'test.csv'
new_classkey_fname = new_filename_stub + 'classes.txt'


train_df, dev_df, test_df, num_labels = \
    read_in_presplit_data(train_fname, dev_fname, test_fname, classkey_fname)


def change_label_of_df_to_new_labels(df):
    list_of_examples = []
    for i, row in df.iterrows():
        oldstrlabel = str(row.loc['strlabel'])
        label = oldlabel_to_newlabel_dict[oldstrlabel]
        text = str(row.loc['text'])
        list_of_examples.append((text, label))
    return list_of_examples


train_list = change_label_of_df_to_new_labels(train_df)
dev_list = change_label_of_df_to_new_labels(dev_df)
test_list = change_label_of_df_to_new_labels(test_df)

save_splits_as_csv(train_list, dev_list, test_list, new_train_fname, new_dev_fname, new_test_fname, new_classkey_fname)
print('Done.')
