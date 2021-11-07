import sys
sys.path.append('..')
from glob import glob
import os
from prep_data import load_in_positive_sentences, get_simplified_filename_string, extract_and_tag_next_document, \
    extract_file_image_tag_from_relevant_part_of_header_string


all_txt_fnames_fname = 'all_txt_filenames.txt'
fnames_and_dates_dir = '../../orig_text_data/dates_0123/'
all_justifications = '../../justifications_clean_text_ohe.csv'
unaccounted_pages_fname = 'unaccounted_pages.txt'


output_monthyear_to_pagecount_fname = 'dates_to_pagecounts.csv'
output_monthyear_to_occurrencecount_fname = 'dates_to_occurrencecounts.csv'


digital_filenametag_to_monthyear = {}
monthyear_to_pagecount_dict = {}
digital_filenametag_to_occurrencecount = {}
monthyear_to_occurrencecount_dict = {}


def get_tag_for_fname(physical_file_id, image_id):
    tag = (get_simplified_filename_string(str(physical_file_id)),
           get_simplified_filename_string(str(image_id)))
    return tag


def get_str_img_form(img_string):
    part_after_underscore = img_string[img_string.index('_') + 1:]
    for i in range(5 - len(part_after_underscore)):
        part_after_underscore = '0' + part_after_underscore
    return 'IMG_' + part_after_underscore


def img_num(img_string):
    return int(img_string[img_string.index('_') + 1:])


tag_img_physicalfname_date = []
physicalfname_to_earliestimgnum = {}
for fname in glob(fnames_and_dates_dir + '*.txt'):
    if not fname.endswith('No_date.txt'):
        corr_date = fname[fname.rfind('/') + 1: fname.rfind('.')]
        if corr_date == '1967':
            corr_date = '1967-01'

        previously_extracted_header = None
        tags_to_documents = {}
        with open(fname, 'r', encoding='utf-8-sig') as f:
            keep_going = True
            while keep_going:
                document, tag, previously_extracted_header = \
                    extract_and_tag_next_document(f, previously_extracted_header=previously_extracted_header)
                if document is None:
                    keep_going = False
                else:
                    tags_to_documents[tag] = document

            for tag in tags_to_documents.keys():
                if not ((tag == ('DEFE_13_919', 'IMG_1959') and corr_date == '1965-02') or
                        (tag == ('PREM_15_100', 'IMG_4540') and corr_date == '1970-06') or
                        (tag == ('PREM_15_100', 'IMG_4541') and corr_date == '1970-06') or
                        (tag == ('PREM_15_101', 'IMG_4676') and corr_date == '1970-07') or
                        (tag == ('PREM_15_101', 'IMG_4679') and corr_date == '1970-07') or
                        (tag == ('PREM_15_1693', 'IMG_9116') and corr_date == '1973-05') or
                        (tag == ('PREM_15_1693', 'IMG_9117') and corr_date == '1973-05') or
                        (tag == ('PREM_15_1693', 'IMG_9182') and corr_date == '1973-05') or
                        (tag == ('PREM_15_476', 'IMG_5403') and corr_date == '1971-02')):
                    tag_img_physicalfname_date.append((tag, tag[1], tag[0], corr_date))
                    if tag[0] not in physicalfname_to_earliestimgnum:
                        physicalfname_to_earliestimgnum[tag[0]] = img_num(tag[1])
                    else:
                        this_img_num = img_num(tag[1])
                        if this_img_num < physicalfname_to_earliestimgnum[tag[0]]:
                            physicalfname_to_earliestimgnum[tag[0]] = this_img_num


tag_img_physicalfname_date = sorted(tag_img_physicalfname_date, key=(lambda tup: tup[2] + '_' +
                                                                                 get_str_img_form(tup[1])))
prev_physicalfname = None
prev_img = None
for i, tag_img_physicalfname_date_tup in enumerate(tag_img_physicalfname_date):
    if prev_physicalfname is None:
        prev_physicalfname = tag_img_physicalfname_date_tup[2]
        prev_img = int(tag_img_physicalfname_date_tup[1][tag_img_physicalfname_date_tup[1].index('_') + 1:])
    else:
        cur_physicalfname = tag_img_physicalfname_date_tup[2]
        cur_img = int(tag_img_physicalfname_date_tup[1][tag_img_physicalfname_date_tup[1].index('_') + 1:])
        if cur_physicalfname == prev_physicalfname:
            assert cur_img > prev_img, '\n' + '\n'.join([str(tag_img_physicalfname_date[i - 1]),
                                                         str(tag_img_physicalfname_date_tup)])
        prev_physicalfname = cur_physicalfname
        prev_img = cur_img


all_valid_tags = set()
with open(all_txt_fnames_fname, 'r') as f:
    for line in f:
        line = line.strip()
        line = line[:line.rfind('.')]
        file_part, img_part = extract_file_image_tag_from_relevant_part_of_header_string(line)
        earliest_allowed_imgnum_for_file = physicalfname_to_earliestimgnum[file_part]
        if img_num(img_part) < earliest_allowed_imgnum_for_file:
            with open(unaccounted_pages_fname, 'a') as f2:
                f2.write('(' + file_part + ', ' + img_part + ')\n')
        else:
            all_valid_tags.add((file_part, img_part))
print('There are ' + str(len(all_valid_tags)) + ' valid tags with inferrable dates.')


all_valid_tags = list(all_valid_tags)
# physicalfname, then img
all_valid_tags = sorted(all_valid_tags, key=(lambda tup: tup[0] + '_' + get_str_img_form(tup[1])))
prev_physicalfname = None
prev_img = None
for valid_tag in all_valid_tags:
    if prev_physicalfname is None:
        prev_physicalfname = valid_tag[0]
        prev_img = int(valid_tag[1][valid_tag[1].index('_') + 1:])
    else:
        cur_physicalfname = valid_tag[0]
        cur_img = int(valid_tag[1][valid_tag[1].index('_') + 1:])
        if cur_physicalfname == prev_physicalfname:
            assert cur_img > prev_img
        prev_physicalfname = cur_physicalfname
        prev_img = cur_img


assert all_valid_tags[0] == tag_img_physicalfname_date[0][0], str(all_valid_tags[0]) + '\n\n\n' + \
                                                              str(tag_img_physicalfname_date[0])
cur_valid_tag_index = 0
for i, tag_img_physicalfname_date_tup in enumerate(tag_img_physicalfname_date[:-1]):
    assert cur_valid_tag_index < len(all_valid_tags)
    is_last_of_physical_file = (tag_img_physicalfname_date_tup[2] != tag_img_physicalfname_date[i + 1][2])
    if is_last_of_physical_file:
        while all_valid_tags[cur_valid_tag_index][0] == tag_img_physicalfname_date_tup[2]:
            # while the physical fname still matches
            digital_filenametag_to_monthyear[all_valid_tags[cur_valid_tag_index]] = tag_img_physicalfname_date_tup[3]
            cur_valid_tag_index += 1
    else:
        img_num_of_next_doc = tag_img_physicalfname_date[i + 1][1]
        img_num_of_next_doc = int(img_num_of_next_doc[img_num_of_next_doc.index('_') + 1:])
        img_num_of_cur_doc = tag_img_physicalfname_date[i][1]
        img_num_of_cur_doc = int(img_num_of_cur_doc[img_num_of_cur_doc.index('_') + 1:])
        assert img_num_of_cur_doc <= img_num(all_valid_tags[cur_valid_tag_index][1]), \
            str(tag_img_physicalfname_date[i][0]) + '\n' + str(all_valid_tags[cur_valid_tag_index])
        while (all_valid_tags[cur_valid_tag_index][0] == tag_img_physicalfname_date_tup[2] and
               img_num(all_valid_tags[cur_valid_tag_index][1]) < img_num_of_next_doc):
            digital_filenametag_to_monthyear[all_valid_tags[cur_valid_tag_index]] = tag_img_physicalfname_date_tup[3]
            cur_valid_tag_index += 1
assert all_valid_tags[cur_valid_tag_index][0] == tag_img_physicalfname_date[-1][2]
for i in range(cur_valid_tag_index, len(all_valid_tags)):
    # while the physical fname still matches
    digital_filenametag_to_monthyear[all_valid_tags[i]] = tag_img_physicalfname_date[-1][3]


for digitalfnametag, monthyear in digital_filenametag_to_monthyear.items():
    if monthyear in monthyear_to_pagecount_dict:
        monthyear_to_pagecount_dict[monthyear] += 1
    else:
        monthyear_to_pagecount_dict[monthyear] = 1


sent_tuples = load_in_positive_sentences(all_justifications)
for sent_tuple in sent_tuples:
    filenametag = sent_tuple[1]
    if filenametag in digital_filenametag_to_occurrencecount:
        digital_filenametag_to_occurrencecount[filenametag] += 1
    else:
        digital_filenametag_to_occurrencecount[filenametag] = 1


for digitalfnametag, occurrencecount in digital_filenametag_to_occurrencecount.items():
    monthyear = digital_filenametag_to_monthyear[digitalfnametag]
    if monthyear in monthyear_to_occurrencecount_dict:
        monthyear_to_occurrencecount_dict[monthyear] += occurrencecount
    else:
        monthyear_to_occurrencecount_dict[monthyear] = occurrencecount


del monthyear_to_pagecount_dict['1920-1921']
del monthyear_to_occurrencecount_dict['1920-1921']


all_dates = [k for k in monthyear_to_pagecount_dict.keys()] + [k for k in monthyear_to_occurrencecount_dict.keys()]
all_dates = sorted(all_dates)
min_date = all_dates[0]
max_date = all_dates[-1]
min_date = (int(min_date[:min_date.index('-')]), int(min_date[min_date.index('-') + 1:]))
max_date = (int(max_date[:max_date.index('-')]), int(max_date[max_date.index('-') + 1:]))

monthyear_pagecount = []
monthyear_occurrencecount = []
cur_year = min_date[0]
cur_month = min_date[1]
while cur_year < max_date[0] or (cur_month <= max_date[1] and cur_year <= max_date[0]):
    date_string = str(cur_year) + '-' + ('0' if len(str(cur_month)) == 1 else '') + str(cur_month)
    if date_string in monthyear_to_pagecount_dict:
        monthyear_pagecount.append((date_string, monthyear_to_pagecount_dict[date_string]))
    else:
        monthyear_pagecount.append((date_string, 0))

    if date_string in monthyear_to_occurrencecount_dict:
        monthyear_occurrencecount.append((date_string, monthyear_to_occurrencecount_dict[date_string]))
    else:
        monthyear_occurrencecount.append((date_string, 0))

    cur_month += 1
    if cur_month == 13:
        cur_month = 1
        cur_year += 1


with open(output_monthyear_to_pagecount_fname, 'w') as f:
    for tup in monthyear_pagecount:
        f.write(','.join([tup[0], str(tup[1])]) + '\n')
with open(output_monthyear_to_occurrencecount_fname, 'w') as f:
    for tup in monthyear_occurrencecount:
        f.write(','.join([tup[0], str(tup[1])]) + '\n')
