import pandas as pd


full_doc_fname = '../../orig_text_data/internment.txt'


new_train_sents = pd.read_csv('../new_train_sents_withcontext1.csv')
old_train_sents = pd.read_csv('../data/binary_mindsduplicates_withcontext_linesupwithmultiway_withfilename_train.csv')
old_train_sents = old_train_sents.drop(['source_handcoded_sent'], axis=1)
print(new_train_sents.shape)
print(old_train_sents.shape)
new_df = pd.concat([old_train_sents, new_train_sents], ignore_index=True)
print(new_df.shape)
new_df.to_csv('../data/binary_mindsduplicates_withcontext_linesupwithmultiway_withfilename_BIGGERDATA_train.csv', index=False)
