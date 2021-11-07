import pandas as pd
from shutil import copy
import os


file_to_find_positive_pages = '../../justifications_clean_text_ohe.csv'
file_to_augment = '../../../OCRdata/NI_docs/doc_transcription_qualities.csv'
new_fname = file_to_augment[:file_to_augment.rfind('.')] + '_withmatchinfo.csv'
cur_binary_data_files = ['../data/binary_mindsduplicates_withcontext_train.csv',
                         '../data/binary_mindsduplicates_withcontext_dev.csv',
                         '../data/binary_mindsduplicates_withcontext_test.csv',]
fname_for_nonmatch_files_also_in_current_data = \
    '../../../OCRdata/NI_docs/negative_filenames_also_in_current_data.txt'


set_of_positive_fname_ids = set()
dataframe = pd.read_csv(file_to_find_positive_pages)


all_fnames_with_positive_matches = set(dataframe['img_file_orig'])
print('There are ' + str(len(all_fnames_with_positive_matches)) + ' unique filenames with at least one positive match.')


all_old_fnames = {}
with open(file_to_augment, 'r') as old_f:
    old_f.readline()
    for line in old_f:
        fname = line[:line.index(',')]
        if fname in all_old_fnames:
            all_old_fnames[fname].append(line)
        else:
            all_old_fnames[fname] = [line]
print('There are ' + str(len(all_old_fnames)) + ' filenames in total, including the google drive data.')  # 6946
for fname, corr_list in all_old_fnames.items():
    for i in corr_list[1:]:
        assert i == corr_list[0], '\n'.join(corr_list)
print('All duplicate filenames in the file-transcription-quality csv file are actually duplicates.')


# new file won't have duplicates, to avoid confusion in the future.
all_old_fnames_included_so_far = set()
positivematch_fnames_with_corr_fname_found = set()
with open(file_to_augment, 'r') as old_f:
    with open(new_fname, 'w') as new_f:
        new_f.write(old_f.readline()[:-1] + ',has_positive_match\n')
        for line in old_f:
            fname = line[:line.index(',')]
            if fname not in all_old_fnames_included_so_far:
                has_positive_match = fname in all_fnames_with_positive_matches
                if has_positive_match:
                    positivematch_fnames_with_corr_fname_found.add(fname)
                line = line.rstrip()
                new_f.write(line + ',' + str(int(has_positive_match)) + '\n')
                all_old_fnames_included_so_far.add(fname)
assert len(all_old_fnames_included_so_far) == len(all_old_fnames), str(len(all_old_fnames_included_so_far))
assert len(positivematch_fnames_with_corr_fname_found) == len(all_fnames_with_positive_matches), \
    "Only found filenames corresponding to " + str(len(positivematch_fnames_with_corr_fname_found)) + ' of the ' + \
    str(len(all_fnames_with_positive_matches)) + ' identified as containing at least one positive match.'
print(str(len(positivematch_fnames_with_corr_fname_found)) + ' of the files show up in the core positive-match csv.')


def get_simplified_filename_string(filename_tag_string):
    first_part = filename_tag_string[:filename_tag_string.index('_') + 1]
    part_to_simplify = filename_tag_string[filename_tag_string.index('_') + 1:]
    if '_' in part_to_simplify:
        simplified = '_'.join([str(int(part)) for part in part_to_simplify.split('_')])
    else:
        simplified = str(int(part_to_simplify))
    return first_part + simplified


def extract_file_image_tag_from_relevant_part_of_header_string(relevant_part):
    # relevant_part will be something that looks like IMG_6444_PREM_15_486 except for a few special cases
    # which we handle separately
    if (relevant_part.startswith('PREM_15_1010_') or relevant_part.startswith('PREM_15_1689_') or
            relevant_part.startswith('PREM_15_478_')) and relevant_part.count('_') == 3:
        image_name = 'IMG_' + relevant_part[relevant_part.rfind('_') + 1:].strip()
        file_name = relevant_part[:relevant_part.rfind('_')]
    else:
        relevant_part = relevant_part.split('_')
        image_name = '_'.join(relevant_part[:2])
        file_name = '_'.join(relevant_part[2:])
    return get_simplified_filename_string(file_name), get_simplified_filename_string(image_name)


fname_to_slashsepversion = {}
slashsepversion_to_fname = {}
for fname in all_old_fnames:
    fname_str, img_str = extract_file_image_tag_from_relevant_part_of_header_string(fname)
    slashsepvversion = fname_str + '/' + img_str
    fname_to_slashsepversion[fname] = slashsepvversion
    slashsepversion_to_fname[slashsepvversion] = fname


def get_column_from_file(fname, name_of_col):
    dataframe = pd.read_csv(fname)
    to_return = list(dataframe[name_of_col])
    return to_return


# check that the union of all filenames currently showing up in the data exactly overlaps with the set of all filenames
# that show up in the core positive-match csv (this won't be added to the csv though)
all_fnames_in_cur_data = set()
for fname in cur_binary_data_files:
    all_fnames_in_cur_data = all_fnames_in_cur_data.union(set(get_column_from_file(fname, 'filename')))
all_fnames_in_cur_data_fnameformat = set()
for fname in all_fnames_in_cur_data:
    all_fnames_in_cur_data_fnameformat.add(slashsepversion_to_fname[fname])
all_fnames_in_cur_data = all_fnames_in_cur_data_fnameformat


fnames_in_cur_data_not_in_positivematchcsvfile = set()
for fname in all_fnames_in_cur_data:
    if fname not in all_fnames_with_positive_matches:
        fnames_in_cur_data_not_in_positivematchcsvfile.add(fname)
fnames_in_positivematchcsvfile_not_in_cur_data = set()
for fname in all_fnames_with_positive_matches:
    if fname not in all_fnames_in_cur_data:
        fnames_in_positivematchcsvfile_not_in_cur_data.add(fname)
assert len(fnames_in_positivematchcsvfile_not_in_cur_data) == 0, \
       'Filenames in core positive match file that AREN\'T in current data (' + \
       str(len(fnames_in_positivematchcsvfile_not_in_cur_data)) + '):\n' + \
       str(list(fnames_in_positivematchcsvfile_not_in_cur_data))
print('Union of all filenames currently showing up in the data covers all filenames that ' +
      'show up in the core positive-match csv file.')


fnames_in_cur_data_not_in_positivematchcsvfile = sorted(list(fnames_in_cur_data_not_in_positivematchcsvfile))
assert len(all_fnames_in_cur_data) - len(all_fnames_with_positive_matches) == \
       len(fnames_in_cur_data_not_in_positivematchcsvfile)
print('In addition to the ' + str(len(all_fnames_with_positive_matches)) + ' files with at least one documented ' +
      'positive match, there are ' + str(len(fnames_in_cur_data_not_in_positivematchcsvfile)) + ' other filenames ' +
      'already rolled into the current data that don\'t contain any positive matches.')
with open(fname_for_nonmatch_files_also_in_current_data, 'w') as f:
    for fname in fnames_in_cur_data_not_in_positivematchcsvfile:
        f.write(fname + '\n')
# check whether they comprise all of a complete file?
umbrellafname_to_count = {}
for fname in fnames_in_cur_data_not_in_positivematchcsvfile:
    parts = fname_to_slashsepversion[fname].split('/')
    full_fname = parts[0]
    if full_fname in umbrellafname_to_count:
        umbrellafname_to_count[full_fname] = umbrellafname_to_count[full_fname] + 1
    else:
        umbrellafname_to_count[full_fname] = 1
umbrellafname_to_totalcount = {}
for fname in fname_to_slashsepversion.keys():
    parts = fname_to_slashsepversion[fname].split('/')
    full_fname = parts[0]
    if full_fname in umbrellafname_to_totalcount:
        umbrellafname_to_totalcount[full_fname] = umbrellafname_to_totalcount[full_fname] + 1
    else:
        umbrellafname_to_totalcount[full_fname] = 1
print('Those filenames make up the following fractions of the complete files they\'re a part of:')
for umbrellafname in umbrellafname_to_count.keys():
    print(umbrellafname + ': ' + str(umbrellafname_to_count[umbrellafname]) + ' / ' +
          str(umbrellafname_to_totalcount[umbrellafname]))


# TODO: check how many of the non-positive-match files in the current data border a positive-match file


# check how many "positive-match" filenames we couldn't find a location for a positive match in, and therefore didn't
# provide a positive match for the BINARY classification task. (this WILL be added to the csv)
fname_to_logicalor_of_its_labels = {}
for data_fname in cur_binary_data_files:
    fnames_in_file = get_column_from_file(data_fname, 'filename')
    labels_in_file = get_column_from_file(data_fname, 'labels')
    for fname, label in zip(fnames_in_file, labels_in_file):
        if fname in fname_to_logicalor_of_its_labels:
            fname_to_logicalor_of_its_labels[fname] = fname_to_logicalor_of_its_labels[fname] or bool(label)
        else:
            fname_to_logicalor_of_its_labels[fname] = bool(label)
for fname in list(fname_to_logicalor_of_its_labels.keys()):
    fname_to_logicalor_of_its_labels[slashsepversion_to_fname[fname]] = int(fname_to_logicalor_of_its_labels[fname])
    del fname_to_logicalor_of_its_labels[fname]
print('There are ' + str(sum(fname_to_logicalor_of_its_labels.values())) + ' filenames in the current data that have ' +
      'at least one positive-match sentence successfully located within them.')


all_old_fnames_included_so_far = set()
num_ones_dispensed = 0
with open(new_fname, 'r') as old_f:
    with open(new_fname[:new_fname.rfind('.')] + '-temp.csv', 'w') as new_f:
        new_f.write(old_f.readline()[:-1] + ',has_positive_match_script_could_locate\n')
        for line in old_f:
            fname = line[:line.index(',')]
            if fname not in all_old_fnames_included_so_far:
                line = line.rstrip()
                if fname in fname_to_logicalor_of_its_labels:
                    new_f.write(line + ',' + str(fname_to_logicalor_of_its_labels[fname]) + '\n')
                    if fname_to_logicalor_of_its_labels[fname] == 1:
                        num_ones_dispensed += 1
                else:
                    new_f.write(line + ',' + str(0) + '\n')
                all_old_fnames_included_so_far.add(fname)
os.remove(new_fname)
copy(new_fname[:new_fname.rfind('.')] + '-temp.csv', new_fname)
os.remove(new_fname[:new_fname.rfind('.')] + '-temp.csv')
assert sum(fname_to_logicalor_of_its_labels.values()) == num_ones_dispensed, str(num_ones_dispensed)


print('Made it through the entire script successfully.')
