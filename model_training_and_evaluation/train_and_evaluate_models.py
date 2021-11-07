import os
import torch
from logreg_baseline import run_classification as run_logreg_classification
from nn_baseline import run_classification as run_word2vec_classification
from nn_baseline import run_best_model_on as run_best_word2vec_model_on
from roberta_fromtransformers import run_classification as run_roberta_classification
from roberta_fromtransformers import run_best_model_on as run_best_roberta_model_on
import numpy as np
from util import make_directories_as_necessary
from prep_data import main as prep_data_and_return_necessary_parts
from prep_data import read_in_presplit_data, make_binary_data_split, make_multiway_data_split
from detailed_performance_breakdown import get_recall_precision_curve_points, \
    plot_two_precision_recalls_against_each_other, make_multilabel_csv, make_csv_used_to_compute_mcnemar_bowker, \
    bootstrap_f1
from config import full_document_filename, binary_train_filename, binary_dev_filename, \
    binary_test_filename, binary_label_key_filename, multiway_train_filename, multiway_dev_filename, \
    multiway_test_filename, multiway_label_key_filename, positive_sentence_filename, problem_report_filename, \
    success_report_filename, binary_positive_sentences_spot_checking_fname, \
    binary_negative_sentences_spot_checking_fname, output_binary_model_dir, output_multiway_model_dir, \
    csv_filename_logreg_on_test, csv_filename_logreg_on_dev, csv_filename_roberta_on_dev, \
    csv_filename_roberta_on_test, multiway_output_report_filename_stub, binary_output_report_filename_stub, \
    dev_precreccurve_plot_filename, test_precreccurve_plot_filename, csv_filename_logregtest_vs_robertatest, \
    use_ten_labels_instead, binary_dev_bootstrapped_f1_filename, binary_test_bootstrapped_f1_filename, \
    multiway_dev_bootstrapped_f1_filename, multiway_test_bootstrapped_f1_filename, \
    csv_filename_word2vec_on_dev, csv_filename_word2vec_on_test, use_context, lowercase_all_text
print('Everything has been imported.')


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


def report_mismatches_to_files(file_stub, true_labels, baseline_labels, model_labels, test_df,
                               label_key_filename, model_name: str=None):
    make_directories_as_necessary(file_stub)

    counter = 0
    inds_to_labels = {}
    with open(label_key_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '':
                inds_to_labels[counter] = line
                counter += 1

    correct_in_both = 0
    correct_in_both_f = open(file_stub + '_bothcorrect.txt', 'w')
    correct_only_in_model = 0
    correct_only_in_model_f = open(file_stub + '_onlycorrectinmodel.txt', 'w')
    correct_only_in_model_f.write('true_label\tincorrect_baseline_label\ttext\n')
    correct_only_in_baseline = 0
    correct_only_in_baseline_f = open(file_stub + '_onlycorrectinbaseline.txt', 'w')
    correct_only_in_baseline_f.write('true_label\tincorrect_model_label\ttext\n')
    neither_correct = 0
    neither_correct_f = open(file_stub + '_neithercorrect.txt', 'w')
    neither_correct_f.write('true_label\tincorrect_baseline_label\tincorrect_model_label\ttext\n')
    for i, row in test_df.iterrows():
        sent = str(row['text'])
        label = str(row['strlabel'])
        if true_labels[i] == model_labels[i]:
            if model_labels[i] == baseline_labels[i]:
                correct_in_both_f.write(str(label) + '\t' + str(sent) + '\n')
                correct_in_both += 1
            else:
                correct_only_in_model_f.write(str(label) + '\t' + inds_to_labels[baseline_labels[i]] + '\t' +
                                              str(sent) + '\n')
                correct_only_in_model += 1
        else:
            if true_labels[i] == baseline_labels[i]:
                correct_only_in_baseline_f.write(str(label) + '\t' + inds_to_labels[model_labels[i]] + '\t' +
                                                 str(sent) + '\n')
                correct_only_in_baseline += 1
            else:
                neither_correct_f.write(str(label) + '\t' + inds_to_labels[baseline_labels[i]] + '\t' +
                                        inds_to_labels[model_labels[i]] + '\t' + str(sent) + '\n')
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


def clean_neuralmodel_prediction_output(output):
    clean_labels = []
    for arr in output:
        if isinstance(arr, int):
            clean_labels.append(arr)
        elif np.size(arr) == 1:
            arr = arr.flatten()
            element = arr[0]
            if int(element) == element:
                clean_labels.append(int(element))
            else:
                assert False, "Don't know how to interpret array of floats into labels: " + str(output)
        else:
            clean_labels.append(int(np.argmax(arr, axis=0)))
    return clean_labels


def get_label_weights_and_report_class_imbalance(train_df, label_file=None, datasplit='training'):
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
    print('Number of each class in the ' + datasplit + ' data:')
    for i in range(len(labels_and_counts)):
        print('\t' + corresponding_labels[i] + ': ' + str(labels_and_counts[i][1]) + ' => weight = ' +
              str(label_weights[i]))
    return label_weights


def convert_labels_to_ten_label_setup(df_to_convert):
    def convert_strlabel_to_new_strlabel(old_strlabel, old_intlabel):
        if old_strlabel == "J_Last-resort" or old_strlabel == "J_Misc" or old_strlabel == "J_Intelligence":
            return "J_Combined", 9
        elif old_strlabel == "J_Terrorism":
            return old_strlabel, 0
        elif old_strlabel == "J_Intl-Domestic_Precedent":
            return old_strlabel, 1
        elif old_strlabel == "J_Denial":
            return old_strlabel, 2
        elif old_strlabel == "J_Political-Strategic":
            return old_strlabel, 3
        elif old_strlabel == "J_Development-Unity":
            return old_strlabel, 4
        elif old_strlabel == "J_Legal_Procedure":
            return old_strlabel, 5
        elif old_strlabel == "J_Emergency-Policy":
            return old_strlabel, 6
        elif old_strlabel == "J_Law-and-order":
            return old_strlabel, 7
        elif old_strlabel == "J_Utilitarian-Deterrence":
            return old_strlabel, 8
        return None

    for i, row in df_to_convert.iterrows():
        old_strlabel = row['strlabel']
        old_label = row['labels']
        new_strlabel, new_label = convert_strlabel_to_new_strlabel(old_strlabel, old_label)
        df_to_convert.loc[i, 'strlabel'] = new_strlabel
        df_to_convert.loc[i, 'labels'] = new_label
    return df_to_convert


def run_logreg_training_testing(train_df, dev_df, test_df, label_weights, use_context, lowercase_all_text,
                                is_multiway, best_param=None):
    best_f1 = -1
    if best_param is None:
        regularization_weights_to_try = [.0001, .001, .01, .1, 1, 10, 100, 1000]
        if use_context:
            values_for_double_features = [True, False]
        else:
            values_for_double_features = [False]
        for regularization_weight in regularization_weights_to_try:
            for use_double_feature_val in values_for_double_features:
                f1, acc, list_of_all_dev_labels, list_of_all_predicted_dev_labels = \
                    run_logreg_classification(train_df, dev_df, regularization_weight=regularization_weight,
                                              label_weights=label_weights, string_prefix='\t',
                                              use_context=use_context, double_context_features=use_double_feature_val,
                                              f1_avg=('binary' if not is_multiway else 'weighted'),
                                              lowercase_all_text=lowercase_all_text)
                if f1 > best_f1:
                    best_f1 = f1
                    best_param = (regularization_weight, use_double_feature_val)
    print('For ' + ('multiway' if is_multiway else 'binary') +
          ' case, best baseline logreg model had regularization weight ' + str(best_param[0]) +
          ' and ' + ('NO ' if not best_param[1] else '') + 'doubled features, ' +
          'and achieved the following performance on the held-out test set:')

    outputs = \
        run_logreg_classification(train_df, dev_df, regularization_weight=best_param[0],
                                  label_weights=label_weights, string_prefix='(Dev set)  ',
                                  also_report_binary_precrec=True, use_context=use_context,
                                  double_context_features=best_param[1], also_output_logits=(not is_multiway),
                                  f1_avg=('binary' if not is_multiway else 'weighted'),
                                  lowercase_all_text=lowercase_all_text)
    if is_multiway:
        f1, acc, list_of_all_dev_labels, dev_predictions_of_best_lr_model, prec, rec = \
            outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]
    else:
        f1, acc, list_of_all_dev_labels, list_of_all_predicted_lr_dev_labels, dev_lr_logits, prec, rec = \
            outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6]

    outputs = \
        run_logreg_classification(train_df, test_df, regularization_weight=best_param[0],
                                  label_weights=label_weights, string_prefix='(Test set) ',
                                  also_report_binary_precrec=True, use_context=use_context,
                                  double_context_features=best_param[1], also_output_logits=(not is_multiway),
                                  f1_avg=('binary' if not is_multiway else 'weighted'),
                                  lowercase_all_text=lowercase_all_text)
    if is_multiway:
        f1, acc, list_of_all_test_labels, list_of_all_predicted_lr_test_labels, prec, rec = \
            outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]
    else:
        f1, acc, list_of_all_test_labels, list_of_all_predicted_lr_test_labels, test_lr_logits, prec, rec = \
            outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6]

    if is_multiway:
        make_multilabel_csv(dev_predictions_of_best_lr_model, list_of_all_dev_labels,
                            multiway_label_key_filename, csv_filename_logreg_on_dev,
                            datasplit_label='dev', using_ten_labels_instead=use_ten_labels_instead)
        make_multilabel_csv(list_of_all_predicted_lr_test_labels, list_of_all_test_labels,
                            multiway_label_key_filename, csv_filename_logreg_on_test,
                            datasplit_label='test', using_ten_labels_instead=use_ten_labels_instead)
        return dev_predictions_of_best_lr_model, list_of_all_predicted_lr_test_labels, \
               list_of_all_dev_labels, list_of_all_test_labels
    else:
        dev_lr_precrec_curve_points = get_recall_precision_curve_points(dev_lr_logits, list_of_all_dev_labels,
                                                                        string_prefix='(Dev for LogReg)  ')
        test_lr_precrec_curve_points = get_recall_precision_curve_points(test_lr_logits, list_of_all_test_labels,
                                                                         string_prefix='(Test for LogReg) ')
        return list_of_all_predicted_lr_dev_labels, list_of_all_predicted_lr_test_labels, \
               list_of_all_dev_labels, list_of_all_test_labels, dev_lr_precrec_curve_points, \
               test_lr_precrec_curve_points


def run_roberta_training_testing(train_df, dev_df, test_df, num_labels, label_weights, cuda_device,
                                 list_of_all_dev_labels, list_of_all_test_labels, use_context,
                                 is_multiway, best_param=None, lowercase_all_text=True):
    print("Following cuda device for roberta: " + str(cuda_device))
    if is_multiway:
        output_model_dir_withextension = output_multiway_model_dir + '_roberta'
    else:
        output_model_dir_withextension = output_binary_model_dir + '_roberta'
    learning_rates_to_try = [1e-5, 2e-5, 3e-5]  # from RoBERTa paper
    batch_sizes_to_try = [32, 16]  # from RoBERTa and BERT papers
    if not is_multiway:
        batch_sizes_to_try.append(136)
    best_f1 = -1
    if best_param is None:
        for learning_rate in learning_rates_to_try:
            for batch_size in batch_sizes_to_try:
                output_dir = output_model_dir_withextension + '_' + str(learning_rate) + '_' + \
                             str(batch_size) + '/'
                f1, acc, list_of_all_predicted_dev_labels = \
                    run_roberta_classification(train_df, dev_df, num_labels, output_dir, batch_size=batch_size,
                                               learning_rate=learning_rate, label_weights=label_weights,
                                               string_prefix='\t', cuda_device=cuda_device,
                                               f1_avg=('weighted' if is_multiway else 'binary'),
                                               use_context=use_context, lowercase_all_text=lowercase_all_text)
                if f1 > best_f1:
                    best_f1 = f1
                    best_param = (batch_size, learning_rate)
    learning_rate = best_param[1]
    batch_size = best_param[0]
    print('For ' + ('multiway' if is_multiway else 'binary') +
          ' case, best RoBERTa model had lr ' + str(learning_rate) + ' and batch size ' + str(batch_size) +
          '. Performance:')
    output_dir = output_model_dir_withextension + '_' + str(learning_rate) + '_' + str(batch_size) + '/'
    dev_f1, dev_acc, list_of_all_predicted_roberta_dev_logits, prec, rec = \
        run_best_roberta_model_on(output_dir, dev_df, num_labels, label_weights, lowercase_all_text=lowercase_all_text,
                                  cuda_device=cuda_device, string_prefix='Dev ',
                                  f1_avg=('weighted' if is_multiway else 'binary'), also_report_binary_precrec=True,
                                  use_context=use_context)
    list_of_all_predicted_roberta_dev_labels = \
        clean_neuralmodel_prediction_output(list_of_all_predicted_roberta_dev_logits)
    test_f1, test_acc, list_of_all_predicted_roberta_test_logits, prec, rec = \
        run_best_roberta_model_on(output_dir, test_df, num_labels, label_weights, lowercase_all_text=lowercase_all_text,
                                  cuda_device=cuda_device, string_prefix='Test ',
                                  f1_avg=('weighted' if is_multiway else 'binary'), also_report_binary_precrec=True,
                                  use_context=use_context)
    list_of_all_predicted_roberta_test_labels = \
        clean_neuralmodel_prediction_output(list_of_all_predicted_roberta_test_logits)

    if is_multiway:
        make_multilabel_csv(list_of_all_predicted_roberta_dev_labels, list_of_all_dev_labels,
                            multiway_label_key_filename, csv_filename_roberta_on_dev,
                            datasplit_label='dev', using_ten_labels_instead=use_ten_labels_instead)
        make_multilabel_csv(list_of_all_predicted_roberta_test_labels, list_of_all_test_labels,
                            multiway_label_key_filename, csv_filename_roberta_on_test,
                            datasplit_label='test', using_ten_labels_instead=use_ten_labels_instead)
        return list_of_all_predicted_roberta_dev_labels, list_of_all_predicted_roberta_test_labels
    else:
        dev_roberta_precrec_curve_points = get_recall_precision_curve_points(list_of_all_predicted_roberta_dev_logits,
                                                                             list_of_all_dev_labels,
                                                                             string_prefix='(Dev for RoBERTa)  ')
        test_roberta_precrec_curve_points = get_recall_precision_curve_points(list_of_all_predicted_roberta_test_logits,
                                                                              list_of_all_test_labels,
                                                                              string_prefix='(Test for RoBERTa) ')
        return list_of_all_predicted_roberta_dev_labels, list_of_all_predicted_roberta_test_labels, \
            dev_roberta_precrec_curve_points, test_roberta_precrec_curve_points


def run_word2vecbaseline_training_testing(train_df, dev_df, test_df, num_labels, label_weights, cuda_device,
                                          list_of_all_dev_labels, list_of_all_test_labels, use_context,
                                          is_multiway, lowercase_all_text, use_lstm, best_param=None):
    print('Following cuda device for neural baseline: ' + str(cuda_device))
    if is_multiway:
        output_model_dir_withextension = output_multiway_model_dir + '_word2vecbaselinemoredataembeds'
        dir_with_pretrained_embeddings = 'multiway_word2vec_biggerdata_' + ('lowercase' if lowercase_all_text else 'fullcase') + '/'  # 'multiway_word2vec_justpositivesentences/'
    else:
        output_model_dir_withextension = output_binary_model_dir + '_word2vecbaselinemoredataembeds'
        dir_with_pretrained_embeddings = 'binary_word2vec_biggerdata_' + ('lowercase' if lowercase_all_text else 'fullcase') + '/'
    output_model_dir_withextension += ('_lstm' if use_lstm else '_feedforward')

    learning_rates_to_try = [1e-4, 1e-3, 5e-5]
    batch_sizes_to_try = [32, 16, 136]  # from RoBERTa and BERT papers
    if use_context and not use_lstm:
        separate_features_for_contexts = [True, False]
    else:
        separate_features_for_contexts = [False]
    best_f1 = -1
    if best_param is None:
        for learning_rate in learning_rates_to_try:
            for batch_size in batch_sizes_to_try:
                for separate_features_for_context in separate_features_for_contexts:
                    if not use_lstm:
                        output_dir = output_model_dir_withextension + '_' + str(learning_rate) + '_' + str(
                            batch_size) + '_' + str(separate_features_for_context) + '/'
                        print('Using feedforward architecture.')
                    else:
                        output_dir = output_model_dir_withextension + '_' + str(learning_rate) + '_' + \
                                     str(batch_size) + '/'
                        print('Using LSTM architecture.')
                    f1, acc, list_of_all_predicted_dev_labels = \
                        run_word2vec_classification(train_df, dev_df, lowercase_all_text=lowercase_all_text,
                                                    num_labels=num_labels, output_dir=output_dir, batch_size=batch_size,
                                                    learning_rate=learning_rate, label_weights=label_weights,
                                                    string_prefix='\t', cuda_device=cuda_device,
                                                    process_context_separately=separate_features_for_context,
                                                    f1_avg=('weighted' if is_multiway else 'binary'),
                                                    pretrained_word2vec_dir=dir_with_pretrained_embeddings,
                                                    use_context=use_context, use_lstm=use_lstm)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_param = (batch_size, learning_rate, separate_features_for_context)
    learning_rate = best_param[1]
    batch_size = best_param[0]
    separate_features_for_context = best_param[2]
    print('For ' + ('multiway' if is_multiway else 'binary') +
          ' case, best ' + ('lstm' if use_lstm else 'feedforward') + ' word2vec baseline model had lr ' +
          str(learning_rate) + ', batch size ' + str(batch_size) +
          ', and ' + ('DOUBLED' if separate_features_for_context else 'NO doubled') +
          ' features for context. Performance:')
    if use_lstm:
        output_dir = output_model_dir_withextension + '_' + str(learning_rate) + '_' + str(batch_size) + '/'
    else:
        output_dir = output_model_dir_withextension + '_' + str(learning_rate) + '_' + str(batch_size) + '_' +\
                     str(separate_features_for_context) + '/'
    dev_f1, dev_acc, list_of_all_predicted_word2vec_dev_logits, prec, rec = \
        run_best_word2vec_model_on(output_dir, dev_df, num_labels, label_weights, lowercase_all_text=lowercase_all_text,
                                   process_context_separately=separate_features_for_context,
                                   cuda_device=cuda_device, string_prefix='Dev ',
                                   f1_avg=('weighted' if is_multiway else 'binary'), also_report_binary_precrec=True,
                                   use_context=use_context, use_lstm=use_lstm)
    list_of_all_predicted_word2vec_dev_labels = \
        clean_neuralmodel_prediction_output(list_of_all_predicted_word2vec_dev_logits)

    test_f1, test_acc, list_of_all_predicted_word2vec_test_logits, prec, rec = \
        run_best_word2vec_model_on(output_dir, test_df, num_labels, label_weights, lowercase_all_text=lowercase_all_text,
                                   process_context_separately=separate_features_for_context,
                                   cuda_device=cuda_device, string_prefix='Test ',
                                   f1_avg=('weighted' if is_multiway else 'binary'), also_report_binary_precrec=True,
                                   use_context=use_context, use_lstm=use_lstm)
    list_of_all_predicted_word2vec_test_labels = \
        clean_neuralmodel_prediction_output(list_of_all_predicted_word2vec_test_logits)

    if is_multiway:
        make_multilabel_csv(list_of_all_predicted_word2vec_dev_labels, list_of_all_dev_labels,
                            multiway_label_key_filename, csv_filename_word2vec_on_dev,
                            datasplit_label='dev', using_ten_labels_instead=use_ten_labels_instead)
        make_multilabel_csv(list_of_all_predicted_word2vec_test_labels, list_of_all_test_labels,
                            multiway_label_key_filename, csv_filename_word2vec_on_test,
                            datasplit_label='test', using_ten_labels_instead=use_ten_labels_instead)
        return dev_f1, list_of_all_predicted_word2vec_dev_labels, list_of_all_predicted_word2vec_test_labels
    else:
        dev_word2vec_precrec_curve_points = \
            get_recall_precision_curve_points(list_of_all_predicted_word2vec_dev_logits, list_of_all_dev_labels,
                                              string_prefix='(Dev for RoBERTa)  ')
        test_word2vec_precrec_curve_points = \
            get_recall_precision_curve_points(list_of_all_predicted_word2vec_test_logits, list_of_all_test_labels,
                                              string_prefix='(Test for RoBERTa) ')
        return dev_f1, list_of_all_predicted_word2vec_dev_labels, list_of_all_predicted_word2vec_test_labels, \
            dev_word2vec_precrec_curve_points, test_word2vec_precrec_curve_points


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

    use_best_nocontext_params = False
    use_best_withcontext_params = False
    assert not (use_best_nocontext_params and use_best_withcontext_params)
    not_using_best_params = not (use_best_withcontext_params or use_best_nocontext_params)

    # multi-way classification
    train_df, dev_df, test_df, num_labels = \
        get_multi_way_classification_data(multiway_train_filename, multiway_dev_filename,
                                          multiway_test_filename, multiway_label_key_filename)
    if use_ten_labels_instead:
        print('Switching to ten-label setup instead.')
        num_labels = 10
        train_df = convert_labels_to_ten_label_setup(train_df)
        dev_df = convert_labels_to_ten_label_setup(dev_df)
        test_df = convert_labels_to_ten_label_setup(test_df)

    label_weights = get_label_weights_and_report_class_imbalance(train_df, datasplit='training')
    get_label_weights_and_report_class_imbalance(dev_df, datasplit='dev')
    get_label_weights_and_report_class_imbalance(test_df, datasplit='test')

    # for no-context: (1, False) # for with-context: (10, False)
    best_param = None if not_using_best_params else ((1, False) if use_best_nocontext_params else (10, False))
    dev_predictions_of_best_lr_model, list_of_all_predicted_lr_test_labels, \
    list_of_all_dev_labels, list_of_all_test_labels = \
        run_logreg_training_testing(train_df, dev_df, test_df, label_weights, use_context, is_multiway=True,
                                    best_param=best_param, lowercase_all_text=lowercase_all_text)

    # for no-context: (32, 1e-3, False)  # for with-context: (16, 1e-3, True)
    best_param = None
    dev_multiway_ff_f1, list_of_all_predicted_word2vec_ff_dev_labels, list_of_all_predicted_word2vec_ff_test_labels = \
        run_word2vecbaseline_training_testing(train_df, dev_df, test_df, num_labels, label_weights, cuda_device,
                                              list_of_all_dev_labels, list_of_all_test_labels, use_context,
                                              is_multiway=True, lowercase_all_text=lowercase_all_text,
                                              best_param=best_param, use_lstm=False)

    # for no-context: (32, 1e-3, False)  # for with-context: (16, 1e-3, False)
    best_param = None
    dev_multiway_lstm_f1, list_of_all_predicted_word2vec_lstm_dev_labels, \
    list_of_all_predicted_word2vec_lstm_test_labels = \
        run_word2vecbaseline_training_testing(train_df, dev_df, test_df, num_labels, label_weights, cuda_device,
                                              list_of_all_dev_labels, list_of_all_test_labels, use_context,
                                              is_multiway=True, lowercase_all_text=lowercase_all_text,
                                              best_param=best_param, use_lstm=True)

    # for no-context: (32, 3e-5) # for with-context: (32, 2e-5)
    best_param = None if not_using_best_params else ((32, 3e-5) if use_best_nocontext_params else (32, 2e-5))
    list_of_all_predicted_roberta_dev_labels, list_of_all_predicted_roberta_test_labels = \
        run_roberta_training_testing(train_df, dev_df, test_df, num_labels, label_weights, cuda_device,
                                     list_of_all_dev_labels, list_of_all_test_labels, use_context,
                                     is_multiway=True, best_param=best_param, lowercase_all_text=lowercase_all_text)

    bootstrap_f1(list_of_all_predicted_roberta_dev_labels, dev_predictions_of_best_lr_model,
                 list_of_all_dev_labels, 500, multiway_dev_bootstrapped_f1_filename, num_labels)
    bootstrap_f1(list_of_all_predicted_roberta_test_labels, list_of_all_predicted_lr_test_labels,
                 list_of_all_test_labels, 500, multiway_test_bootstrapped_f1_filename, num_labels)

    report_mismatches_to_files(multiway_output_report_filename_stub, list_of_all_test_labels,
                               list_of_all_predicted_lr_test_labels, list_of_all_predicted_roberta_test_labels,
                               test_df, multiway_label_key_filename, model_name='RoBERTa')
    make_csv_used_to_compute_mcnemar_bowker(list_of_all_predicted_roberta_test_labels, 'RoBERTa',
                                            list_of_all_predicted_lr_test_labels, 'LogReg',
                                            csv_filename_logregtest_vs_robertatest)
    print('\n\n')


    # binary classification
    train_df, dev_df, test_df, num_labels = \
        get_binary_classification_data(binary_train_filename, binary_dev_filename,
                                       binary_test_filename, binary_label_key_filename)

    label_weights = get_label_weights_and_report_class_imbalance(train_df, datasplit = 'training')
    get_label_weights_and_report_class_imbalance(dev_df, datasplit='dev')
    get_label_weights_and_report_class_imbalance(test_df, datasplit='test')

    # for no-context: (100, False) # for with-context: (100, False)
    best_param = None if not_using_best_params else ((100, False) if use_best_nocontext_params else (100, False))
    list_of_all_predicted_lr_dev_labels, list_of_all_predicted_lr_test_labels, \
    list_of_all_dev_labels, list_of_all_test_labels, dev_lr_precrec_curve_points, test_lr_precrec_curve_points = \
        run_logreg_training_testing(train_df, dev_df, test_df, label_weights, use_context, is_multiway=False,
                                    best_param=best_param, lowercase_all_text=lowercase_all_text)

    # for no-context: (16, 1e-3, False)  # for with-context: (32, 5e-5, False)
    best_param = None if not_using_best_params else ((16, 1e-3, False) if use_best_nocontext_params else
                                                     (32, 5e-5, False))
    dev_binary_ff_f1, list_of_all_predicted_word2vec_ff_dev_labels, list_of_all_predicted_word2vec_ff_test_labels, \
    dev_word2vec_ff_precrec_curve_points, test_word2vec_ff_precrec_curve_points = \
        run_word2vecbaseline_training_testing(train_df, dev_df, test_df, num_labels, label_weights, cuda_device,
                                              list_of_all_dev_labels, list_of_all_test_labels, use_context,
                                              is_multiway=False, lowercase_all_text=lowercase_all_text,
                                              best_param=best_param, use_lstm=False)

    # for no-context: (32, 5e-5, False)  # for with-context: (16, 1e-4, False)
    best_param = None if not_using_best_params else ((32, 5e-5, False) if use_best_nocontext_params else
                                                     (16, 1e-4, False))
    dev_binary_lstm_f1, list_of_all_predicted_word2vec_lstm_dev_labels, \
    list_of_all_predicted_word2vec_lstm_test_labels, dev_word2vec_lstm_precrec_curve_points, \
    test_word2vec_lstm_precrec_curve_points = \
        run_word2vecbaseline_training_testing(train_df, dev_df, test_df, num_labels, label_weights, cuda_device,
                                              list_of_all_dev_labels, list_of_all_test_labels, use_context,
                                              is_multiway=False, lowercase_all_text=lowercase_all_text,
                                              best_param=best_param, use_lstm=True)

    # for no-context: (32, 1e-5) # for with-context: (32, 1e-5)
    best_param = None if not_using_best_params else ((32, 1e-5) if use_best_nocontext_params else (32, 1e-5))
    list_of_all_predicted_roberta_dev_labels, list_of_all_predicted_roberta_test_labels, \
    dev_roberta_precrec_curve_points, test_roberta_precrec_curve_points = \
        run_roberta_training_testing(train_df, dev_df, test_df, num_labels, label_weights, cuda_device,
                                     list_of_all_dev_labels, list_of_all_test_labels, use_context,
                                     is_multiway=False, best_param=best_param, lowercase_all_text=lowercase_all_text)

    bootstrap_f1(list_of_all_predicted_roberta_dev_labels, list_of_all_predicted_lr_dev_labels,
                 list_of_all_dev_labels, 500, binary_dev_bootstrapped_f1_filename, num_labels)
    bootstrap_f1(list_of_all_predicted_roberta_test_labels, list_of_all_predicted_lr_test_labels,
                 list_of_all_test_labels, 500, binary_test_bootstrapped_f1_filename, num_labels)

    dev_precrec_points = [dev_lr_precrec_curve_points, dev_roberta_precrec_curve_points]
    dev_precrec_labels = ['LogReg baseline', 'Finetuned RoBERTa']
    test_precrec_points = [test_lr_precrec_curve_points, test_roberta_precrec_curve_points]
    test_precrec_labels = ['LogReg baseline', 'Finetuned RoBERTa']
    if dev_binary_ff_f1 > dev_binary_lstm_f1:
        dev_precrec_points.insert(1, dev_word2vec_ff_precrec_curve_points)
        dev_precrec_labels.insert(1, 'FeedForward baseline')
        test_precrec_points.insert(1, test_word2vec_ff_precrec_curve_points)
        test_precrec_labels.insert(1, 'FeedForward baseline')
    else:
        dev_precrec_points.insert(1, dev_word2vec_lstm_precrec_curve_points)
        dev_precrec_labels.insert(1, 'LSTM baseline')
        test_precrec_points.insert(1, test_word2vec_lstm_precrec_curve_points)
        test_precrec_labels.insert(1, 'LSTM baseline')
    plot_two_precision_recalls_against_each_other(dev_precrec_points,
                                                  dev_precrec_labels,
                                                  dev_precreccurve_plot_filename,
                                                  plot_title='Precision-recall curve on dev set')
    plot_two_precision_recalls_against_each_other(test_precrec_points,
                                                  test_precrec_labels,
                                                  test_precreccurve_plot_filename,
                                                  plot_title='Precision-recall curve on test set')
    make_directories_as_necessary(binary_output_report_filename_stub)
    report_mismatches_to_files(binary_output_report_filename_stub, list_of_all_test_labels,
                               list_of_all_predicted_lr_test_labels, list_of_all_predicted_roberta_test_labels,
                               test_df, binary_label_key_filename, model_name='RoBERTa')


if __name__ == '__main__':
    main()
