import os


multiway_dir_extension = '_nocontextNEWEMBEDSFIXEDINDEX_10class'  # '_condensed'  # set to '_condensed' for 6-way multiway, or '' for full 12-way


full_document_filename = '../orig_text_data/internment.txt'
positive_sentence_filename = '../justifications_clean_text_ohe.csv'

use_context = False
# must be at least 1; use_context will control whether it actually gets used or not
num_context_sents_to_use = 0
lowercase_all_text = True
ignore_params_given_above_and_use_best_param_instead = False
additional_numcontextsents_to_restrict_to_if_ignoring_other_params = None

actual_minibatch_size = 4
use_ten_labels_instead = True

if ignore_params_given_above_and_use_best_param_instead:
    if additional_numcontextsents_to_restrict_to_if_ignoring_other_params is not None:
        if additional_numcontextsents_to_restrict_to_if_ignoring_other_params == 0:
            use_context = False
            num_context_sents_to_use = 0
        else:
            use_context = True
            num_context_sents_to_use = additional_numcontextsents_to_restrict_to_if_ignoring_other_params
elif num_context_sents_to_use == 0:
    assert not use_context
context_sents_for_data_fnames = max(1, num_context_sents_to_use)

if not ignore_params_given_above_and_use_best_param_instead:
    subdir_of_expresults = str(num_context_sents_to_use) + '_sents_as_context' + ('' if lowercase_all_text else '_CASED')
else:
    if additional_numcontextsents_to_restrict_to_if_ignoring_other_params is None:
        subdir_of_expresults = 'best_hparams'
    else:
        subdir_of_expresults = 'best_hparams_with_' + str(num_context_sents_to_use) + '_sents_as_context'
while subdir_of_expresults.endswith('/'):
    subdir_of_expresults = subdir_of_expresults[:-1]


def get_model_dir(task, num_context_sents, all_text_lowercased):
    if task == 'binary':
        return 'experiment_results/' + str(num_context_sents) + '_sents_as_context' + ('' if all_text_lowercased else '_CASED') + '/binary_model'
    else:
        return 'experiment_results/' + str(num_context_sents) + '_sents_as_context' + ('' if all_text_lowercased else '_CASED') + '/multiway_model' + multiway_dir_extension


binary_train_filename = "data/binary_full_mindsduplicates_withcontext"+str(context_sents_for_data_fnames)+"_train.csv"  # 'data/binary_train-withperplexities-85percentile.csv'
binary_dev_filename = "data/binary_full_mindsduplicates_withcontext"+str(context_sents_for_data_fnames)+"_dev.csv"  # 'data/binary_dev-withperplexities-85percentile.csv'
binary_test_filename = "data/binary_full_mindsduplicates_withcontext"+str(context_sents_for_data_fnames)+"_test.csv"  # 'data/binary_test-withperplexities-85percentile.csv'
binary_label_key_filename = 'data/binary_mindsduplicates_classes.txt'
output_binary_model_dir = get_model_dir('binary', num_context_sents_to_use, lowercase_all_text)
binary_output_report_filename_stub = 'output_analysis/nocontextFIXEDINDEX-binarybest'
binary_dev_bootstrapped_f1_filename = 'binaryDEV_nocontextFIXEDINDEX_bootstrappedf1s.csv'
binary_test_bootstrapped_f1_filename = 'binaryTEST_nocontextFIXEDINDEX_bootstrappedf1s.csv'

# relevant for making the data
binary_positive_sentences_spot_checking_fname = 'data/binary_extracted_positive_sentences_withcontextmasking_correctedsavelocation.txt'
binary_negative_sentences_spot_checking_fname = 'data/binary_extracted_negative_sentences_withcontextmasking_correctedsavelocation.txt'
problem_report_filename = 'data/problem_matches_withcontextmasking_correctedsavelocation.txt'  # or None if you just want to report to the command line
success_report_filename = 'data/successful_matches_withcontextmasking_correctedsavelocation.txt'  # or None if you don't want these reported

dev_precreccurve_plot_filename = 'experiment_results/' + subdir_of_expresults + '/output_analysis/binary_nocontextFIXEDINDEX_dev_precisionrecallcurve.png'
test_precreccurve_plot_filename = 'experiment_results/' + subdir_of_expresults + '/output_analysis/output_analysis/binary_nocontextFIXEDINDEX_test_precisionrecallcurve.png'
if problem_report_filename and os.path.isfile(problem_report_filename):
    os.remove(problem_report_filename)
if success_report_filename and os.path.isfile(success_report_filename):
    os.remove(success_report_filename)

multiway_train_filename = 'data/multiway_mindsduplicates_withcontext/multiway_withcontext' + str(context_sents_for_data_fnames) + '_train.csv'
multiway_dev_filename = 'data/multiway_mindsduplicates_withcontext/multiway_withcontext' + str(context_sents_for_data_fnames) + '_dev.csv'
multiway_test_filename = 'data/multiway_mindsduplicates_withcontext/multiway_withcontext' + str(context_sents_for_data_fnames) + '_test.csv'
multiway_label_key_filename = 'data/multiway_mindsduplicates_withcontext/multiway_classes.txt'
multiway_dev_bootstrapped_f1_filename = 'experiment_results/' + subdir_of_expresults + '/output_analysis/multiwayDEV' + multiway_dir_extension + str(use_ten_labels_instead) + '_bootstrappedf1s.csv'
multiway_test_bootstrapped_f1_filename = 'experiment_results/' + subdir_of_expresults + '/output_analysis/multiwayTEST' + multiway_dir_extension + str(use_ten_labels_instead) + '_bootstrappedf1s.csv'
output_multiway_model_dir = get_model_dir('multiway', num_context_sents_to_use, lowercase_all_text)
multiway_output_report_filename_stub = 'experiment_results/' + subdir_of_expresults + '/output_analysis/multiway_mindsduplicates' + multiway_dir_extension + '/f1-multiwaybest'

csv_filename_logreg_on_dev = 'experiment_results/' + subdir_of_expresults + '/output_analysis/multiway_mindsduplicates' + multiway_dir_extension + '/multiwaytask_dev_logregresults.csv'
csv_filename_roberta_on_dev = 'experiment_results/' + subdir_of_expresults + '/output_analysis/multiway_mindsduplicates' + multiway_dir_extension + '/multiwaytask_dev_robertaresults.csv'
csv_filename_logreg_on_test = 'experiment_results/' + subdir_of_expresults + '/output_analysis/multiway_mindsduplicates' + multiway_dir_extension + '/multiwaytask_test_logregresults.csv'
csv_filename_roberta_on_test = 'experiment_results/' + subdir_of_expresults + '/output_analysis/multiway_mindsduplicates' + multiway_dir_extension + '/multiwaytask_test_robertaresults.csv'
csv_filename_logregtest_vs_robertatest = 'experiment_results/' + subdir_of_expresults + '/output_analysis/multiway_mindsduplicates' + multiway_dir_extension + '/multiwaytask_testroberta_x_testlogreg.csv'
csv_filename_word2vec_on_dev = 'experiment_results/' + subdir_of_expresults + '/output_analysis/multiway_mindsduplicates' + multiway_dir_extension + '/multiwaytask_dev_word2vecresults_postembedchange.csv'
csv_filename_word2vec_on_test = 'experiment_results/' + subdir_of_expresults + '/output_analysis/multiway_mindsduplicates' + multiway_dir_extension + '/multiwaytask_test_word2vecresults_postembedchange.csv'

if output_binary_model_dir.endswith('/'):
    output_binary_model_dir = output_binary_model_dir[:-1]
if output_multiway_model_dir.endswith('/'):
    output_multiway_model_dir = output_multiway_model_dir[:-1]
