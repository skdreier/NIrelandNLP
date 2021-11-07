import pandas as pd
import sys
sys.path.append('../')
from prep_data import fix_df_format
from copy import deepcopy


splits = ['train', 'dev', 'test']
old_pieces = ('../data/binary_mindsduplicates_withcontext_', '.csv')
new_pieces = ('../new_', '_sents_withcontext1.csv')
destination_pieces = ('../data/binary_full_mindsduplicates_withcontext1_', '.csv')
destinationwithoutcontext_pieces = ('../data/binary_full_mindsduplicates_nocontext_', '.csv')


for split in splits:
    olddata_df = fix_df_format(pd.read_csv(old_pieces[0] + split + old_pieces[1]))
    newdata_df = fix_df_format(pd.read_csv(new_pieces[0] + split + new_pieces[1]))

    bothdata_df = pd.concat([olddata_df, newdata_df])
    bothdata_nocontext = deepcopy(bothdata_df).drop(['contextbefore'], axis=1)
    assert 'contextbefore' in bothdata_df.columns
    if split == 'train':
        print('Dropping following row: ')
        print(bothdata_nocontext[10182:10183])
        # we divide these into two parts because for some reason, dropping row index 10182 would also drop the row with
        # index 24282. no idea why.
        bothdata_df_partafter = deepcopy(bothdata_df[24000:])
        bothdata_df_partbefore = bothdata_df[:24000]
        bothdata_nocontext_partafter = deepcopy(bothdata_nocontext[24000:])
        bothdata_nocontext_partbefore = deepcopy(bothdata_nocontext[:24000])
        print('\t' + str(bothdata_df.shape))
        print('\t' + str(bothdata_nocontext.shape))
        bothdata_df = bothdata_df_partbefore.drop(index=[10182])
        bothdata_df = pd.concat([bothdata_df, bothdata_df_partafter])
        bothdata_nocontext = bothdata_nocontext_partbefore.drop(index=[10182])
        bothdata_nocontext = pd.concat([bothdata_nocontext, bothdata_nocontext_partafter])

    print(olddata_df.shape)
    print(newdata_df.shape)
    if split == 'train':
        print('\t' + str(bothdata_df.shape))
        print('\t' + str(bothdata_nocontext.shape))
    print(str(sum(list(bothdata_df['labels']))) + ' out of ' + str(bothdata_df.shape[0]) +
          ' instances in the ' + split + ' data are positive.')

    bothdata_df.to_csv(destination_pieces[0] + split + destination_pieces[1], index=False)
    bothdata_nocontext.to_csv(destinationwithoutcontext_pieces[0] + split + destinationwithoutcontext_pieces[1],
                              index=False)
