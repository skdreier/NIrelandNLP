import pandas as pd


fnames_in_expanded_traindata = \
    set(pd.read_csv('../data/binary_mindsduplicates_withcontext_linesupwithmultiway_withfilename_BIGGERDATA_train.csv')[
            'filename'])
for fname in fnames_in_expanded_traindata:
    assert '/' in fname
fnames_in_dev = set(pd.read_csv('../data/multiway_mindsduplicates_withcontext/multiway_withcontext1_dev.csv')[
                        'filename'])
fnames_in_test = set(pd.read_csv('../data/multiway_mindsduplicates_withcontext/multiway_withcontext1_dev.csv')[
                        'filename'])
for fname in fnames_in_dev:
    assert '/' in fname
    assert fname not in fnames_in_expanded_traindata, fname
for fname in fnames_in_test:
    assert '/' in fname
    assert fname not in fnames_in_expanded_traindata, fname
print('No problems.')
