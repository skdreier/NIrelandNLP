import sys
sys.path.append('..')
from prep_data import read_in_presplit_data
from math import isnan, inf

base_data_filename = '../data/binary_'
percentile_of_perplexities_to_keep = 85

percentile_of_perplexities_to_keep = percentile_of_perplexities_to_keep / 100
train_file = base_data_filename + 'train-withperplexities.csv'
dev_file = base_data_filename + 'dev-withperplexities.csv'
test_file = base_data_filename + 'test-withperplexities.csv'
label_key_filename = base_data_filename + 'classes.txt'


train_df, dev_df, test_df, num_labels = read_in_presplit_data(train_file, dev_file, test_file, label_key_filename)


full_list_of_perplexities = train_df['perplexity'].tolist() + \
                            dev_df['perplexity'].tolist() + \
                            test_df['perplexity'].tolist()
full_list_of_perplexities = [float(val) for val in full_list_of_perplexities]
full_list_of_perplexities = sorted(full_list_of_perplexities, key=lambda x: inf if isnan(x) else x)
num_nans = 0
for val in full_list_of_perplexities:
    if isnan(val):
        num_nans += 1
print('Num NaNs: ' + str(num_nans))

print('Quick check that sorting worked:')
print('\tIndices:')
print('\t' + str(int(len(full_list_of_perplexities) * 0)))
print('\t' + str(int(len(full_list_of_perplexities) * .25)))
print('\t' + str(int(len(full_list_of_perplexities) * .5)))
print('\t' + str(int(len(full_list_of_perplexities) * .75)))
print('\t' + str(int(len(full_list_of_perplexities) * 1) - 1))
print()
print('\tValues:')
print('\t' + str(full_list_of_perplexities[int(len(full_list_of_perplexities) * 0)]))
print('\t' + str(full_list_of_perplexities[int(len(full_list_of_perplexities) * .25)]))
print('\t' + str(full_list_of_perplexities[int(len(full_list_of_perplexities) * .5)]))
print('\t' + str(full_list_of_perplexities[int(len(full_list_of_perplexities) * .75)]))
print('\t' + str(full_list_of_perplexities[int(len(full_list_of_perplexities) * 1) - 1]))
print()
for i, val in enumerate(full_list_of_perplexities[:-1]):
    if not isnan(val) and not isnan(full_list_of_perplexities[i + 1]):
        assert val <= full_list_of_perplexities[i + 1], str(i) + ', ' + str(val) + ', ' + \
                                                        str(full_list_of_perplexities[i + 1])

cutoff_ind = int(percentile_of_perplexities_to_keep * len(full_list_of_perplexities))
print('Cutoff ind is ' + str(cutoff_ind) + ' out of ' + str(len(full_list_of_perplexities)))
print(full_list_of_perplexities[cutoff_ind])
print(full_list_of_perplexities[cutoff_ind + 1])
cutoff_val = (full_list_of_perplexities[cutoff_ind] + full_list_of_perplexities[cutoff_ind + 1]) / 2
print('Cutoff perplexity val (must be below this) is ' + str(cutoff_val))
num_perplexities_below = 0
num_perplexities_above = 0
for perplexity in full_list_of_perplexities:
    if perplexity < cutoff_val:
        num_perplexities_below += 1
    else:
        num_perplexities_above += 1
print('There are ' + str(num_perplexities_below) + ' perplexities below cutoff val and ' +
      str(num_perplexities_above) + ' perplexities above it')


def subset_df(df):
    return df[df['perplexity'] < cutoff_val]


train_df = subset_df(train_df)
dev_df = subset_df(dev_df)
test_df = subset_df(test_df)


def get_new_filename(old_filename):
    num_tag = str(percentile_of_perplexities_to_keep)
    if percentile_of_perplexities_to_keep % 1 == 1:
        num_tag = '100'
    else:
        num_tag = num_tag[num_tag.index('.') + 1:]
        if len(num_tag) == 1:
            num_tag += '0'
    return old_filename[:old_filename.rfind('.')] + '-' + num_tag + 'percentile' + \
           old_filename[old_filename.rfind('.'):]


train_df.to_csv(get_new_filename(train_file), index=False)
dev_df.to_csv(get_new_filename(dev_file), index=False)
test_df.to_csv(get_new_filename(test_file), index=False)
print('Wrote subset data csv files.')
