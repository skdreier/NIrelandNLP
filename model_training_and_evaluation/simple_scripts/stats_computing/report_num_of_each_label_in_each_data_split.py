"""
This script makes the assumption that from the stub you provide, the training file will be stub + 'train.csv',
the dev file will be stub + 'dev.csv', the test file will be stub + 'test.csv', and the class key file will
be stub + 'classes.txt'
"""

import sys
sys.path.append('..')
from prep_data import read_in_presplit_data


filename_stub = sys.argv[1]


train_fname = filename_stub + 'train.csv'
dev_fname = filename_stub + 'dev.csv'
test_fname = filename_stub + 'test.csv'
classkey_fname = filename_stub + 'classes.txt'


train_df, dev_df, test_df, num_labels = \
    read_in_presplit_data(train_fname, dev_fname, test_fname, classkey_fname)
print('Num of each label in train:')
print(train_df['strlabel'].value_counts())
print()
print('Num of each label in dev:')
print(dev_df['strlabel'].value_counts())
print()
print('Num of each label in test:')
print(test_df['strlabel'].value_counts())
print()
