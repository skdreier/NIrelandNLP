import os
import torch
from logreg_baseline import run_classification as run_logreg_classification
from roberta import run_classification as run_roberta_classification
from roberta import run_best_model_on
import numpy as np
from util import make_directories_as_necessary
from prep_data import main as prep_data_and_return_necessary_parts
from prep_data import read_in_presplit_data, make_binary_data_split, make_multiway_data_split
from detailed_performance_breakdown import get_recall_precision_curve_points, \
    plot_two_precision_recalls_against_each_other, make_multilabel_csv
from config import full_document_filename, binary_train_filename, binary_dev_filename, \
    binary_test_filename, binary_label_key_filename, multiway_train_filename, multiway_dev_filename, \
    multiway_test_filename, multiway_label_key_filename, positive_sentence_filename, problem_report_filename, \
    success_report_filename, binary_positive_sentences_spot_checking_fname, \
    binary_negative_sentences_spot_checking_fname, output_binary_model_dir, output_multiway_model_dir, \
    csv_filename_logreg_on_test, csv_filename_logreg_on_dev, csv_filename_roberta_on_dev, \
    csv_filename_roberta_on_test, multiway_output_report_filename_stub, binary_output_report_filename_stub, \
    dev_precreccurve_plot_filename, test_precreccurve_plot_filename


def get_binary_classification_data(train_filename, dev_filename, test_filename, label_key_filename):
    train_df, dev_df, test_df, num_labels = \
        read_in_presplit_data(train_filename, dev_filename, test_filename, label_key_filename)
    print('Read in existing binary data split.')
    print('For binary classification:')
    print('\t' + str(int(train_df.shape[0])) + ' training sentences')
    print('\t' + str(int(dev_df.shape[0])) + ' dev sentences')
    print('\t' + str(int(test_df.shape[0])) + ' test sentences')
    return train_df, dev_df, test_df, num_labels


def get_multi_way_classification_data(train_filename, dev_filename, test_filename, label_key_filename):
    train_df, dev_df, test_df, num_labels = \
        read_in_presplit_data(train_filename, dev_filename, test_filename, label_key_filename)
    print('Read in existing multi-way data split.')
    print('For multi-way classification:')
    print('\t' + str(int(train_df.shape[0])) + ' training sentences')
    print('\t' + str(int(dev_df.shape[0])) + ' dev sentences')
    print('\t' + str(int(test_df.shape[0])) + ' test sentences')
    return train_df, dev_df, test_df, num_labels


def report_mismatches_to_files(file_stub, true_labels, baseline_labels, model_labels, test_df, model_name: str=None):
    make_directories_as_necessary(file_stub)

    correct_in_both = 0
    correct_in_both_f = open(file_stub + '_bothcorrect.txt', 'w')
    correct_only_in_model = 0
    correct_only_in_model_f = open(file_stub + '_onlycorrectinmodel.txt', 'w')
    correct_only_in_baseline = 0
    correct_only_in_baseline_f = open(file_stub + '_onlycorrectinbaseline.txt', 'w')
    neither_correct = 0
    neither_correct_f = open(file_stub + '_neithercorrect.txt', 'w')
    for i, row in test_df.iterrows():
        sent = str(row['text'])
        label = str(row['strlabel'])
        if true_labels[i] == model_labels[i]:
            if model_labels[i] == baseline_labels[i]:
                correct_in_both_f.write(str(label) + '\t' + str(sent) + '\n')
                correct_in_both += 1
            else:
                correct_only_in_model_f.write(str(label) + '\t' + str(sent) + '\n')
                correct_only_in_model += 1
        else:
            if true_labels[i] == baseline_labels[i]:
                correct_only_in_baseline_f.write(str(label) + '\t' + str(sent) + '\n')
                correct_only_in_baseline += 1
            else:
                neither_correct_f.write(str(label) + '\t' + str(sent) + '\n')
                neither_correct += 1
    correct_in_both_f.close()
    correct_only_in_model_f.close()
    correct_only_in_baseline_f.close()
    neither_correct_f.close()
    if model_name is None:
        model_name = 'other model'
    print('Comparison of what baseline and ' + model_name + ' got correct:')
    print('\tBoth correct: ' + str(correct_in_both))
    print('\tOnly ' + model_name + ' correct: ' + str(correct_only_in_model))
    print('\tOnly baseline correct: ' + str(correct_only_in_baseline))
    print('\tNeither correct: ' + str(neither_correct))


def clean_roberta_prediction_output(output):
    clean_labels = []
    for arr in output:
        clean_labels.append(int(np.argmax(arr, axis=0)))
    return clean_labels


def get_label_weights_and_report_class_imbalance(train_df, label_file=None):
    df_with_class_counts = train_df['labels'].value_counts()
    labels_and_counts = []
    for label_ind, label_count in df_with_class_counts.iteritems():
        labels_and_counts.append((int(label_ind), int(label_count)))
    labels_and_counts = sorted(labels_and_counts, key=lambda x: x[0])
    corresponding_labels = []
    if label_file is not None:
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line != '':
                    corresponding_labels.append(line)
        assert len(labels_and_counts) == len(corresponding_labels)
    else:
        for label, count in labels_and_counts:
            corresponding_labels.append(str(label))
    median_value = sorted(labels_and_counts, key=lambda x: x[1])
    median_value = median_value[len(median_value) // 2][1]
    label_weights = []
    for label, count in labels_and_counts:
        label_weights.append(median_value / count)
    print('Number of each class in the training data:')
    for i in range(len(labels_and_counts)):
        print('\t' + corresponding_labels[i] + ': ' + str(labels_and_counts[i][1]) + ' => weight = ' +
              str(label_weights[i]))
    return label_weights


def main():
    if not os.path.isfile(binary_train_filename) and not os.path.isfile(multiway_train_filename):
        prep_data_and_return_necessary_parts(full_document_filename, binary_train_filename, binary_dev_filename,
                                             binary_test_filename, binary_label_key_filename, multiway_train_filename,
                                             multiway_dev_filename, multiway_test_filename, multiway_label_key_filename,
                                             positive_sentence_filename, problem_report_filename,
                                             success_report_filename, binary_positive_sentences_spot_checking_fname,
                                             binary_negative_sentences_spot_checking_fname)
    elif not os.path.isfile(binary_train_filename):
        # just make the binary data split
        make_binary_data_split(binary_train_filename, binary_dev_filename, binary_test_filename,
                               binary_label_key_filename,
                               binary_positive_sentences_spot_checking_fname,
                               binary_negative_sentences_spot_checking_fname,
                               full_document_filename=full_document_filename,
                               positive_sentence_filename=positive_sentence_filename,
                               problem_report_filename=problem_report_filename,
                               success_report_filename=success_report_filename)
    elif not os.path.isfile(multiway_train_filename):
        # just make the multiway data split
        make_multiway_data_split(multiway_train_filename, multiway_dev_filename,
                                 multiway_test_filename, multiway_label_key_filename,
                                 positive_sentences_filename=positive_sentence_filename)

    if torch.cuda.is_available():
        cuda_device = 0
    else:
        cuda_device = -1

    # binary classification
    train_df, dev_df, test_df, num_labels = \
        get_binary_classification_data(binary_train_filename, binary_dev_filename,
                                       binary_test_filename, binary_label_key_filename)

    label_weights = get_label_weights_and_report_class_imbalance(train_df)

    best_f1 = -1
    best_param = None
    regularization_weights_to_try = [.0001, .001, .01, .1, 1, 10, 100, 1000]
    for regularization_weight in regularization_weights_to_try:
        f1, acc, list_of_all_dev_labels, list_of_all_predicted_dev_labels = \
            run_logreg_classification(train_df, dev_df, regularization_weight=regularization_weight,
                                      label_weights=label_weights, string_prefix='\t', f1_avg='binary')
        if f1 > best_f1:
            best_f1 = f1
            best_param = regularization_weight
    print('For binary case, best baseline logreg model had regularization weight ' + str(best_param) +
          ', and achieved the following performance on the held-out test set:')
    f1, acc, list_of_all_dev_labels, list_of_all_predicted_lr_dev_labels, dev_lr_logits, prec, rec = \
        run_logreg_classification(train_df, dev_df, regularization_weight=best_param,
                                  label_weights=label_weights, string_prefix='(Dev set)  ', f1_avg='binary',
                                  also_output_logits=True, also_report_binary_precrec=True)
    f1, acc, list_of_all_test_labels, list_of_all_predicted_lr_test_labels, test_lr_logits, prec, rec = \
        run_logreg_classification(train_df, test_df, regularization_weight=best_param,
                                  label_weights=label_weights, string_prefix='(Test set) ', f1_avg='binary',
                                  also_output_logits=True, also_report_binary_precrec=True)
    dev_lr_precrec_curve_points = get_recall_precision_curve_points(dev_lr_logits, list_of_all_dev_labels,
                                                                    string_prefix='(Dev for LogReg)  ')
    test_lr_precrec_curve_points = get_recall_precision_curve_points(test_lr_logits, list_of_all_test_labels,
                                                                     string_prefix='(Test for LogReg) ')

    learning_rates_to_try = [1e-5, 2e-5, 3e-5]  # from RoBERTa paper
    batch_sizes_to_try = [32, 16]  # from RoBERTa and BERT papers
    best_f1 = -1
    best_param = None
    for learning_rate in learning_rates_to_try:
        for batch_size in batch_sizes_to_try:
            output_dir = output_binary_model_dir + '_' + str(learning_rate) + '_' + str(batch_size) + '/'
            f1, acc, list_of_all_predicted_dev_labels = \
                run_roberta_classification(train_df, dev_df, num_labels, output_dir, batch_size=batch_size,
                                           learning_rate=learning_rate, label_weights=label_weights,
                                           string_prefix='\t', cuda_device=cuda_device, f1_avg='binary')
            if f1 > best_f1:
                best_f1 = f1
                best_param = (batch_size, learning_rate)
    learning_rate = best_param[1]
    batch_size = best_param[0]
    print('For binary case, best RoBERTa model had lr ' + str(learning_rate) + ' and batch size ' + str(batch_size) +
          '. Performance:')
    output_dir = output_binary_model_dir + '_' + str(learning_rate) + '_' + str(batch_size) + '/'
    dev_f1, dev_acc, list_of_all_predicted_roberta_dev_logits, prec, rec = \
        run_best_model_on(output_dir, dev_df, num_labels, label_weights, lowercase_all_text=True,
                          cuda_device=-1, string_prefix='Dev ', f1_avg='binary', also_report_binary_precrec=True)
    f1, acc, list_of_all_predicted_roberta_test_logits, prec, rec = \
        run_best_model_on(output_dir, test_df, num_labels, label_weights, lowercase_all_text=True,
                          cuda_device=-1, string_prefix='Test ', f1_avg='binary', also_report_binary_precrec=True)
    list_of_all_predicted_roberta_test_labels = \
        clean_roberta_prediction_output(list_of_all_predicted_roberta_test_logits)

    dev_roberta_precrec_curve_points = get_recall_precision_curve_points(list_of_all_predicted_roberta_dev_logits,
                                                                         list_of_all_dev_labels,
                                                                         string_prefix='(Dev for RoBERTa)  ')
    test_roberta_precrec_curve_points = get_recall_precision_curve_points(list_of_all_predicted_roberta_test_logits,
                                                                          list_of_all_test_labels,
                                                                          string_prefix='(Test for RoBERTa) ')

    plot_two_precision_recalls_against_each_other(dev_lr_precrec_curve_points, 'LogReg baseline',
                                                  dev_roberta_precrec_curve_points, 'Finetuned RoBERTa',
                                                  dev_precreccurve_plot_filename,
                                                  plot_title='Precision-recall curve on dev set')
    plot_two_precision_recalls_against_each_other(test_lr_precrec_curve_points, 'LogReg baseline',
                                                  test_roberta_precrec_curve_points, 'Finetuned RoBERTa',
                                                  test_precreccurve_plot_filename,
                                                  plot_title='Precision-recall curve on test set')
    make_directories_as_necessary(binary_output_report_filename_stub)
    report_mismatches_to_files(binary_output_report_filename_stub, list_of_all_test_labels,
                               list_of_all_predicted_lr_test_labels, list_of_all_predicted_roberta_test_labels,
                               test_df, model_name='RoBERTa')

    print('\n\n')

    # multi-way classification
    train_df, dev_df, test_df, num_labels = \
        get_multi_way_classification_data(multiway_train_filename, multiway_dev_filename,
                                          multiway_test_filename, multiway_label_key_filename)

    label_weights = get_label_weights_and_report_class_imbalance(train_df)

    best_f1 = -1
    best_param = None
    regularization_weights_to_try = [.0001, .001, .01, .1, 1, 10, 100, 1000]
    for regularization_weight in regularization_weights_to_try:
        f1, acc, list_of_all_dev_labels, list_of_all_predicted_dev_labels = \
            run_logreg_classification(train_df, dev_df, regularization_weight=regularization_weight,
                                      label_weights=label_weights, string_prefix='\t')
        if f1 > best_f1:
            best_f1 = f1
            best_param = regularization_weight
    print('For multiway case, best baseline logreg model had regularization weight ' + str(best_param) +
          ', and achieved the following performance on the held-out test set:')
    f1, acc, list_of_all_dev_labels, dev_predictions_of_best_lr_model, prec, rec = \
        run_logreg_classification(train_df, dev_df, regularization_weight=best_param,
                                  label_weights=label_weights, string_prefix='(Dev set)  ',
                                  also_report_binary_precrec=True)
    f1, acc, list_of_all_test_labels, list_of_all_predicted_lr_test_labels, prec, rec = \
        run_logreg_classification(train_df, test_df, regularization_weight=best_param,
                                  label_weights=label_weights, string_prefix='(Test set) ',
                                  also_report_binary_precrec=True)
    make_multilabel_csv(dev_predictions_of_best_lr_model, list_of_all_dev_labels,
                        multiway_label_key_filename, csv_filename_logreg_on_dev,
                        datasplit_label='dev')
    make_multilabel_csv(list_of_all_predicted_lr_test_labels, list_of_all_test_labels,
                        multiway_label_key_filename, csv_filename_logreg_on_test,
                        datasplit_label='test')

    learning_rates_to_try = [1e-5, 2e-5, 3e-5]  # from RoBERTa paper
    batch_sizes_to_try = [32, 16]  # from RoBERTa and BERT papers
    best_f1 = -1
    best_param = None
    for learning_rate in learning_rates_to_try:
        for batch_size in batch_sizes_to_try:
            output_dir = output_multiway_model_dir + '_' + str(learning_rate) + '_' + str(batch_size) + '/'
            f1, acc, list_of_all_predicted_dev_labels = \
                run_roberta_classification(train_df, dev_df, num_labels, output_dir, batch_size=batch_size,
                                           learning_rate=learning_rate, label_weights=label_weights,
                                           string_prefix='\t', cuda_device=cuda_device)
            if f1 > best_f1:
                best_f1 = f1
                best_param = (batch_size, learning_rate)
    learning_rate = best_param[1]
    batch_size = best_param[0]
    print('For multiway case, best RoBERTa model had lr ' + str(learning_rate) + ' and batch size ' + str(batch_size) +
          '. Performance:')
    output_dir = output_multiway_model_dir + '_' + str(learning_rate) + '_' + str(batch_size) + '/'
    f1, acc, list_of_all_predicted_roberta_dev_labels, prec, rec = \
        run_best_model_on(output_dir, dev_df, num_labels, label_weights, lowercase_all_text=True,
                          cuda_device=-1, string_prefix='Dev ', f1_avg='weighted', also_report_binary_precrec=True)
    list_of_all_predicted_roberta_dev_labels = \
        clean_roberta_prediction_output(list_of_all_predicted_roberta_dev_labels)
    f1, acc, list_of_all_predicted_roberta_test_labels, prec, rec = \
        run_best_model_on(output_dir, test_df, num_labels, label_weights, lowercase_all_text=True,
                          cuda_device=-1, string_prefix='Test ', f1_avg='weighted', also_report_binary_precrec=True)
    list_of_all_predicted_roberta_test_labels = \
        clean_roberta_prediction_output(list_of_all_predicted_roberta_test_labels)

    make_multilabel_csv(list_of_all_predicted_roberta_dev_labels, list_of_all_dev_labels,
                        multiway_label_key_filename, csv_filename_roberta_on_dev,
                        datasplit_label='dev')
    make_multilabel_csv(list_of_all_predicted_roberta_test_labels, list_of_all_test_labels,
                        multiway_label_key_filename, csv_filename_roberta_on_test,
                        datasplit_label='test')
    report_mismatches_to_files(multiway_output_report_filename_stub, list_of_all_test_labels,
                               list_of_all_predicted_lr_test_labels, list_of_all_predicted_roberta_test_labels,
                               test_df, model_name='RoBERTa')


if __name__ == '__main__':
    main()
