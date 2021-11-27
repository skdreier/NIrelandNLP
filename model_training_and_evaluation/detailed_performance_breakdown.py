from typing import List
import numpy as np
from util import make_directories_as_necessary
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set()
sns.color_palette("colorblind")
from random import random
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
from glob import glob
from prep_data import extract_and_tag_next_document, extract_file_image_tag_from_relevant_part_of_header_string


def make_csv_used_to_compute_mcnemar_bowker(predicted_labels_1, model_name_1, predicted_labels_2, model_name_2,
                                            filename):
    highest_val_to_go_up_to = max(max(predicted_labels_1), max(predicted_labels_2))
    total_num_categories = highest_val_to_go_up_to + 1
    model1label_to_allcorrmodel2labels = {}
    for i in range(highest_val_to_go_up_to + 1):
        model1label_to_allcorrmodel2labels[i] = []
    assert len(predicted_labels_2) == len(predicted_labels_1)
    for i in range(len(predicted_labels_2)):
        label_1 = predicted_labels_1[i]
        label_2 = predicted_labels_2[i]
        model1label_to_allcorrmodel2labels[label_1].append(label_2)
    model1label_to_model2label_to_count = {}
    for i in range(total_num_categories):
        model1label_to_model2label_to_count[i] = {}
        for j in range(total_num_categories):
            model1label_to_model2label_to_count[i][j] = model1label_to_allcorrmodel2labels[i].count(j)
    empty_cells_before_name = 2 + (total_num_categories // 2)
    num_fields_per_line = total_num_categories + 2
    str_to_write = ','.join(([''] * empty_cells_before_name) + [model_name_2] +
                            ([''] * (num_fields_per_line - empty_cells_before_name - 1))) + '\n'
    str_to_write += ','.join(['', ''] + [str(i) for i in range(total_num_categories)]) + '\n'
    for i in range(total_num_categories):
        if i + 2 == empty_cells_before_name:
            initial_field = model_name_1
        else:
            initial_field = ''
        line_fields = [initial_field, str(i)]
        for j in range(total_num_categories):
            line_fields.append(str(model1label_to_model2label_to_count[i][j]))
        str_to_write += ','.join(line_fields) + '\n'

    make_directories_as_necessary(filename)
    with open(filename, 'w') as f:
        f.write(str_to_write)


def plot_two_precision_recalls_against_each_other(recall_precision_points_lists, plot_labels,
                                                  plot_filename, plot_title=None):
    colors = ['#56B4E9', 'black', '#DE8F05']
    make_directories_as_necessary(plot_filename)
    fig = plt.figure()

    for j, recall_precision_points_list in enumerate(recall_precision_points_lists):
        recalls_one = []
        precisions_one = []
        for i in range(len(recall_precision_points_list)):
            recalls_one.append(recall_precision_points_list[i][0])
            precisions_one.append(recall_precision_points_list[i][1])
        plt.plot(recalls_one, precisions_one, label=plot_labels[j], color=colors[j])
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.05)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if plot_title is not None:
        plt.title(plot_title)
    plt.legend()

    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close(fig)

    for j, recall_precision_points_list in enumerate(recall_precision_points_lists):
        just_first_part_filename = plot_filename[:plot_filename.rfind('.')] + '-' + plot_labels[j].replace(' ', '_') + \
                                   plot_filename[plot_filename.rfind('.'):]
        recalls_one = []
        precisions_one = []
        for i in range(len(recall_precision_points_list)):
            recalls_one.append(recall_precision_points_list[i][0])
            precisions_one.append(recall_precision_points_list[i][1])
        fig = plt.figure()
        plt.plot(recalls_one, precisions_one, label=plot_labels[j], color=colors[j])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        if plot_title is not None:
            plt.title(plot_title)
        plt.savefig(just_first_part_filename, bbox_inches='tight')
        plt.close(fig)


def get_fname_to_date_dict(fnames_and_dates_dir='../orig_text_data/dates_0123/',
                           all_txt_fnames_fname='simple_scripts/all_txt_filenames.txt'):
    digital_filenametag_to_monthyear = {}

    def img_num(img_string):
        return int(img_string[img_string.index('_') + 1:])

    def get_str_img_form(img_string):
        part_after_underscore = img_string[img_string.index('_') + 1:]
        for i in range(5 - len(part_after_underscore)):
            part_after_underscore = '0' + part_after_underscore
        return 'IMG_' + part_after_underscore

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
            if img_num(img_part) >= earliest_allowed_imgnum_for_file:
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
                digital_filenametag_to_monthyear[all_valid_tags[cur_valid_tag_index]] = tag_img_physicalfname_date_tup[
                    3]
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
                digital_filenametag_to_monthyear[all_valid_tags[cur_valid_tag_index]] = tag_img_physicalfname_date_tup[
                    3]
                cur_valid_tag_index += 1
    assert all_valid_tags[cur_valid_tag_index][0] == tag_img_physicalfname_date[-1][2]
    for i in range(cur_valid_tag_index, len(all_valid_tags)):
        # while the physical fname still matches
        digital_filenametag_to_monthyear[all_valid_tags[i]] = tag_img_physicalfname_date[-1][3]

    dict_to_return = {tup[0][0] + '/' + tup[0][1]: tup[1] for tup in digital_filenametag_to_monthyear.items()}
    return dict_to_return


def get_file_start_date_end_date(source_fname='file_date.csv'):
    def pad(month):
        if len(month) < 2:
            return '0' + month
        return month

    fname_to_startdateenddate = {}
    with open(source_fname, 'r') as f:
        f.readline()
        for line in f:
            if line.strip() == '':
                continue
            line = line.strip().split(',')
            fname_to_startdateenddate[line[0]] = (line[1] + '-' + pad(line[2]), line[3] + '-' + pad(line[4]))
    return fname_to_startdateenddate


def make_data_file_for_binary_recall_histograms(list_of_logits, data_df, output_filename):
    actual_labels_as_list_of_ints = list(data_df['labels'])
    assert list_of_logits[0].shape[-1] == 2
    assert actual_labels_as_list_of_ints.count(0) + actual_labels_as_list_of_ints.count(1) == \
           len(actual_labels_as_list_of_ints)

    list_of_instancebeingpositive_probs = []
    for i, logit_pair in enumerate(list_of_logits):
        denom = np.log(np.sum(np.exp(logit_pair)))
        if len(list_of_logits[0].shape) == 2:
            list_of_instancebeingpositive_probs.append(float(np.exp(logit_pair[0][1] - denom)))
        else:
            list_of_instancebeingpositive_probs.append(float(np.exp(logit_pair[1] - denom)))

    recalls_to_plot_for = [None, .7, .8, .9]
    list_of_filename_to_recoveredpositivecount_dicts = []
    for recall in recalls_to_plot_for:
        if recall is not None:
            threshold = get_threshold_corresponding_to_recall(list_of_logits, actual_labels_as_list_of_ints, recall)
        else:
            threshold = None
        labels_for_this_threshold = \
            get_labels_according_to_positive_classification_threshold(list_of_instancebeingpositive_probs,
                                                                      threshold=threshold)
        assert len(labels_for_this_threshold) == data_df.shape[0]
        filename_to_recoveredpositivecount_dicts = {}
        for i, row in data_df.iterrows():
            filename = row['filename']
            label = labels_for_this_threshold[i]
            if filename in filename_to_recoveredpositivecount_dicts:
                filename_to_recoveredpositivecount_dicts[filename] = \
                    filename_to_recoveredpositivecount_dicts[filename] + int(label == 1 and row['labels'] == 1)
            else:
                filename_to_recoveredpositivecount_dicts[filename] = int(label == 1 and row['labels'] == 1)

        list_of_filename_to_recoveredpositivecount_dicts.append(filename_to_recoveredpositivecount_dicts)

    filename_to_truepositivecount_dict = {}
    for i, row in data_df.iterrows():
        filename = row['filename']
        label = row['labels']
        if filename in filename_to_truepositivecount_dict:
            filename_to_truepositivecount_dict[filename] = \
                filename_to_truepositivecount_dict[filename] + int(label == 1 and row['labels'] == 1)
        else:
            filename_to_truepositivecount_dict[filename] = int(label == 1 and row['labels'] == 1)

    # filter out all filenames that contribute no true positive sentences
    fnames_to_filter_out = []
    for filename, true_positive_count in filename_to_truepositivecount_dict.items():
        if true_positive_count == 0:
            fnames_to_filter_out.append(filename)
    for fname in fnames_to_filter_out:
        del filename_to_truepositivecount_dict[fname]
        for recovereddict in list_of_filename_to_recoveredpositivecount_dicts:
            del recovereddict[fname]

    # get date corresponding to remaining filenames (date will be a string in format like '1965-02')
    fname_to_date = get_fname_to_date_dict()
    fnames_with_no_date = []
    for fname in filename_to_truepositivecount_dict:
        if fname not in fname_to_date:
            fnames_with_no_date.append(fname)
    fname_to_startdate_enddate = get_file_start_date_end_date()
    for fname in tqdm(filename_to_truepositivecount_dict.keys(), total=len(filename_to_truepositivecount_dict)):
        if fname not in fname_to_date:
            fname_to_date[fname] = fname_to_startdate_enddate[fname.split('/')[0]][0] + '-' + \
                                   fname_to_startdate_enddate[fname.split('/')[0]][1]
    print('Fnames with dates we filled in from parent fname: ' + str(fnames_with_no_date))
    # sort by filename first, then date second
    sorted_fnames_dates = list(fname_to_date.items())
    sorted_fnames_dates = sorted(sorted_fnames_dates, key=lambda x: x[0] + x[1])

    with open(output_filename, 'w') as f:
        f.write('filename,date,truepositivecount,defaultthreshold_positivesrecovered,' +
                (','.join([str(recall) + 'recall_positivesrecovered' for recall in recalls_to_plot_for[1:]])) + '\n')
        for fname, date in sorted_fnames_dates:
            if fname in filename_to_truepositivecount_dict:
                fields = [fname, date]
                fields.append(str(filename_to_truepositivecount_dict[fname]))
                for filename_to_recoveredpositivecount_dict in list_of_filename_to_recoveredpositivecount_dicts:
                    fields.append(str(filename_to_recoveredpositivecount_dict[fname]))
                f.write(','.join(fields) + '\n')
    print('Wrote ' + output_filename)


def get_labels_according_to_positive_classification_threshold(prob_for_each_instance_being_positive, threshold=None):
    list_to_return = []
    for i in range(len(prob_for_each_instance_being_positive)):
        if threshold is None:
            list_to_return.append(int(prob_for_each_instance_being_positive[i] >= .5))
        else:
            list_to_return.append(int(prob_for_each_instance_being_positive[i] >= threshold))
    return list_to_return


def get_threshold_corresponding_to_recall(list_of_logits, actual_labels_as_list_of_ints, desired_recall):
    assert list_of_logits[0].shape[-1] == 2
    assert actual_labels_as_list_of_ints.count(0) + actual_labels_as_list_of_ints.count(1) == \
           len(actual_labels_as_list_of_ints)
    list_of_probs = []
    for i, logit_pair in enumerate(list_of_logits):
        denom = np.log(np.sum(np.exp(logit_pair)))
        list_of_probs.append((np.exp(logit_pair - denom), actual_labels_as_list_of_ints[i]))
    if len(list_of_logits[0].shape) == 2:
        sorted_by_prob = sorted(list_of_probs, key=lambda x: x[0][0][1], reverse=True)
    else:
        sorted_by_prob = sorted(list_of_probs, key=lambda x: x[0][1], reverse=True)

    total_actual_positive_instances = actual_labels_as_list_of_ints.count(1)
    total_true_positives_passed_so_far = 0
    for i in range(len(sorted_by_prob)):
        total_true_positives_passed_so_far += sorted_by_prob[i][1]  # 0 or 1
        if total_true_positives_passed_so_far / total_actual_positive_instances >= desired_recall:
            if i == len(sorted_by_prob) - 1:
                return 0
            else:
                return (sorted_by_prob[i][0][1] + sorted_by_prob[i + 1][0][1]) / 2


def get_recall_precision_curve_points(list_of_logits, actual_labels_as_list_of_ints: List[int], string_prefix=''):
    assert list_of_logits[0].shape[-1] == 2
    assert actual_labels_as_list_of_ints.count(0) + actual_labels_as_list_of_ints.count(1) == \
           len(actual_labels_as_list_of_ints)
    list_of_probs = []
    for i, logit_pair in enumerate(list_of_logits):
        denom = np.log(np.sum(np.exp(logit_pair)))
        list_of_probs.append((np.exp(logit_pair - denom), actual_labels_as_list_of_ints[i]))
    if len(list_of_logits[0].shape) == 2:
        sorted_by_prob = sorted(list_of_probs, key=lambda x: x[0][0][1], reverse=True)
    else:
        sorted_by_prob = sorted(list_of_probs, key=lambda x: x[0][1], reverse=True)

    total_actual_positive_instances = actual_labels_as_list_of_ints.count(1)
    # precision = true_guessed_pos / true_guessed_pos + false_guessed_pos
    # recall = true_guessed_pos / total_actual_positive_instances
    recall_precision_points_to_return = []

    true_guessed_positive_so_far = 0
    best_sum_thresholds_so_far = []
    best_sum_threshold_recprecs = []
    best_sum_of_precrec = 0
    best_euclidean_thresholds_so_far = []
    best_euclidean_threshold_recprecs = []
    best_squared_euclidean_distance = 2
    for total_guessed_positive_so_far_minus_1 in range(len(sorted_by_prob)):
        true_guessed_positive_so_far += sorted_by_prob[total_guessed_positive_so_far_minus_1][-1]  # the true label
        precision_here = true_guessed_positive_so_far / (total_guessed_positive_so_far_minus_1 + 1)
        recall_here = true_guessed_positive_so_far / total_actual_positive_instances
        recall_precision_points_to_return.append((recall_here, precision_here))
        if recall_here + precision_here > best_sum_of_precrec:
            best_sum_of_precrec = recall_here + precision_here
            best_sum_thresholds_so_far = [sorted_by_prob[total_guessed_positive_so_far_minus_1][0][1]]
            best_sum_threshold_recprecs = [(recall_here, precision_here)]
        elif recall_here + precision_here == best_sum_of_precrec:
            best_sum_thresholds_so_far.append(sorted_by_prob[total_guessed_positive_so_far_minus_1][0][1])
            best_sum_threshold_recprecs.append((recall_here, precision_here))
        cur_euclidean_distance_squared = (recall_here * recall_here) - (2 * recall_here) + \
                                         (precision_here * precision_here) - (2 * precision_here) + 2
        if cur_euclidean_distance_squared < best_squared_euclidean_distance:
            best_squared_euclidean_distance = cur_euclidean_distance_squared
            best_euclidean_thresholds_so_far = [sorted_by_prob[total_guessed_positive_so_far_minus_1][0][1]]
            best_euclidean_threshold_recprecs = [(recall_here, precision_here)]
        elif cur_euclidean_distance_squared == best_squared_euclidean_distance:
            best_euclidean_thresholds_so_far.append(sorted_by_prob[total_guessed_positive_so_far_minus_1][0][1])
            best_euclidean_threshold_recprecs.append((recall_here, precision_here))
    print(string_prefix + 'Best thresholds for deciding something is positive:')
    if len(best_sum_thresholds_so_far) > 1:
        print('\tUsing sum of precision and recall, positive probability >= ' +
              'any threshold in range [' + str(best_sum_thresholds_so_far[-1]) + ', ' +
              str(best_sum_thresholds_so_far[0]) + '] (corresponding to (recall, precision) points ' +
              str(best_sum_threshold_recprecs) + ')')
    else:
        print('\tUsing sum of precision and recall, positive probability >= ' +
              str(best_sum_thresholds_so_far[0]) + ' (corresponding to recall ' +
              str(best_sum_threshold_recprecs[0][0]) + ' and precision ' + str(best_sum_threshold_recprecs[0][1]) + ')')
    if len(best_euclidean_thresholds_so_far) > 1:
        print('\tUsing Euclidean distance from point (1, 1), positive probability >= ' +
              'any threshold in range [' + str(best_euclidean_thresholds_so_far[-1]) + ', ' +
              str(best_euclidean_thresholds_so_far[0]) + '] (corresponding to (recall, precision) points ' +
              str(best_euclidean_threshold_recprecs) + ')')
    else:
        print('\tUsing Euclidean distance from point (1, 1), positive probability >= ' +
              str(best_euclidean_thresholds_so_far[0]) + ' (corresponding to recall ' +
              str(best_euclidean_threshold_recprecs[0][0]) + ' and precision ' +
              str(best_euclidean_threshold_recprecs[0][1]) + ')')
    return recall_precision_points_to_return


def bootstrap_f1(list_of_predicted_labels_roberta, list_of_predicted_labels_baseline, list_of_correct_labels,
                 num_times_to_bootstrap, filename_to_write_data_to, num_labels):
    tups_to_draw_from = \
        list(zip(list_of_predicted_labels_roberta, list_of_predicted_labels_baseline, list_of_correct_labels))
    length_of_list = len(list_of_predicted_labels_baseline)

    def bootstrap_once(metric_function_to_apply):
        bootstrapped_data = []
        for i in range(length_of_list):
            ind_to_sample = int(length_of_list * random())
            if ind_to_sample == length_of_list:
                ind_to_sample -= 1
            bootstrapped_data.append(tups_to_draw_from[ind_to_sample])

        true_labels = [tup[2] for tup in bootstrapped_data]

        if num_labels == 2:
            average = 'binary'
        else:
            average = 'weighted'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1_baseline = metric_function_to_apply(true_labels, [tup[1] for tup in bootstrapped_data], average=average)
            f1_roberta = metric_function_to_apply(true_labels, [tup[0] for tup in bootstrapped_data], average=average)
        warnings.filterwarnings('default')

        return f1_roberta, f1_baseline

    list_of_bootstrapped_f1_tups = []
    for j in tqdm(range(num_times_to_bootstrap), total=num_times_to_bootstrap):
        list_of_bootstrapped_f1_tups.append(bootstrap_once(f1_score))

    with open(filename_to_write_data_to, 'w') as f:
        f.write('bootstrapped_f1_roberta,bootstrapped_f1_baseline\n')
        for roberta_val, baseline_val in list_of_bootstrapped_f1_tups:
            f.write(str(roberta_val) + ',' + str(baseline_val) + '\n')

    list_of_bootstrapped_recall_tups = []
    for j in tqdm(range(num_times_to_bootstrap), total=num_times_to_bootstrap):
        list_of_bootstrapped_recall_tups.append(bootstrap_once(recall_score))

    with open(filename_to_write_data_to[:filename_to_write_data_to.rfind('.')] + '-recall' +
              filename_to_write_data_to[filename_to_write_data_to.rfind('.'):], 'w') as f:
        f.write('bootstrapped_recall_roberta,bootstrapped_recall_baseline\n')
        for roberta_val, baseline_val in list_of_bootstrapped_recall_tups:
            f.write(str(roberta_val) + ',' + str(baseline_val) + '\n')

    list_of_bootstrapped_precision_tups = []
    for j in tqdm(range(num_times_to_bootstrap), total=num_times_to_bootstrap):
        list_of_bootstrapped_precision_tups.append(bootstrap_once(precision_score))

    with open(filename_to_write_data_to[:filename_to_write_data_to.rfind('.')] + '-precision' +
              filename_to_write_data_to[filename_to_write_data_to.rfind('.'):], 'w') as f:
        f.write('bootstrapped_precision_roberta,bootstrapped_precision_baseline\n')
        for roberta_val, baseline_val in list_of_bootstrapped_precision_tups:
            f.write(str(roberta_val) + ',' + str(baseline_val) + '\n')
    print('Wrote ' + filename_to_write_data_to)
    print('Wrote ' + filename_to_write_data_to[:filename_to_write_data_to.rfind('.')] + '-recall' +
              filename_to_write_data_to[filename_to_write_data_to.rfind('.'):])
    print('Wrote ' + filename_to_write_data_to[:filename_to_write_data_to.rfind('.')] + '-precision' +
          filename_to_write_data_to[filename_to_write_data_to.rfind('.'):])


def make_multilabel_csv(list_of_predicted_labels, actual_labels_as_list_of_ints, class_key_filename, csv_filename,
                        datasplit_label='test', using_ten_labels_instead = False):
    make_directories_as_necessary(csv_filename)
    precision_recall_f1_numtrulyinlabel_numguessedaslabel = \
        get_classwise_prec_rec_f1_numtrulyinlabel_numguessedaslabel(list_of_predicted_labels,
                                                                    actual_labels_as_list_of_ints)
    class_names = []
    with open(class_key_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '':
                if ',' in line:
                    line = '"' + line + '"'
                class_names.append(line)
    if using_ten_labels_instead:
        class_names = ["J_Terrorism", "J_Intl-Domestic_Precedent", "J_Denial", "J_Political-Strategic",
                       "J_Development-Unity", "J_Legal_Procedure", "J_Emergency-Policy", "J_Law-and-order",
                       "J_Utilitarian-Deterrence", 'J_Combined']
    assert len(class_names) == len(precision_recall_f1_numtrulyinlabel_numguessedaslabel), \
        str(len(class_names)) + ', ' + str(len(precision_recall_f1_numtrulyinlabel_numguessedaslabel))

    with open(csv_filename, 'w') as f:
        f.write(','.join(['label_ind', 'str_label', 'num_of_each_class_in_' + datasplit_label, 'precision',
                          'recall', 'f1', 'num_guessed_as_class']) + '\n')
        for i in range(len(class_names)):
            fields_to_write = [str(i)]
            fields_to_write.append(class_names[i])
            fields_to_write.append(str(precision_recall_f1_numtrulyinlabel_numguessedaslabel[i][3]))
            fields_to_write.append(str(precision_recall_f1_numtrulyinlabel_numguessedaslabel[i][0]))
            fields_to_write.append(str(precision_recall_f1_numtrulyinlabel_numguessedaslabel[i][1]))
            fields_to_write.append(str(precision_recall_f1_numtrulyinlabel_numguessedaslabel[i][2]))
            fields_to_write.append(str(precision_recall_f1_numtrulyinlabel_numguessedaslabel[i][4]))
            f.write(','.join(fields_to_write) + '\n')
    print('Wrote ' + csv_filename)


def get_classwise_prec_rec_f1_numtrulyinlabel_numguessedaslabel(list_of_predicted_labels,
                                                                actual_labels_as_list_of_ints: List[int]):
    assert len(list_of_predicted_labels) == len(actual_labels_as_list_of_ints)
    precision_recall_f1_numtrulyinlabel_numguessedaslabel = []
    highest_label = max(max(list_of_predicted_labels), max(actual_labels_as_list_of_ints))
    for label in range(highest_label + 1):
        num_truly_in_label = actual_labels_as_list_of_ints.count(label)
        num_guessed_as_label = list_of_predicted_labels.count(label)
        true_guessed_positive = 0
        false_guessed_positive = 0
        for i in range(len(list_of_predicted_labels)):
            if list_of_predicted_labels[i] == label:
                if actual_labels_as_list_of_ints[i] == label:
                    true_guessed_positive += 1
                else:
                    false_guessed_positive += 1

        if true_guessed_positive + false_guessed_positive > 0:
            precision = true_guessed_positive / (true_guessed_positive + false_guessed_positive)
        else:
            precision = 'NaN'
        if num_truly_in_label > 0:
            recall = true_guessed_positive / num_truly_in_label
        else:
            recall = 'NaN'
        if recall == 'NaN' or precision == 'NaN' or (precision == 0 and recall == 0):
            f1 = 'NaN'
        else:
            f1 = 2 * precision * recall / (precision + recall)
        precision_recall_f1_numtrulyinlabel_numguessedaslabel.append((precision, recall, f1, num_truly_in_label,
                                                                      num_guessed_as_label))
    return precision_recall_f1_numtrulyinlabel_numguessedaslabel
