import os


multiway_dir_extension = '_withcontextmasking_10class'  # '_condensed'  # set to '_condensed' for 6-way multiway, or '' for full 12-way


full_document_filename = '../orig_text_data/internment.txt'
positive_sentence_filename = '../justifications_clean_text_ohe.csv'

binary_train_filename = 'data/binary_mindsduplicates_withcontext_train.csv'  # 'data/binary_train-withperplexities-85percentile.csv'
binary_dev_filename = 'data/binary_mindsduplicates_withcontext_dev.csv'  # 'data/binary_dev-withperplexities-85percentile.csv'
binary_test_filename = 'data/binary_mindsduplicates_withcontext_test.csv'  # 'data/binary_test-withperplexities-85percentile.csv'
binary_label_key_filename = 'data/binary_mindsduplicates_classes.txt'
output_binary_model_dir = '../f1-saved_binary_model_withcontextmasking_correctedsavelocation/'
binary_output_report_filename_stub = 'output_analysis/withcontextmasking_correctedsavelocation-f1-binarybest'
binary_dev_bootstrapped_f1_filename = 'binaryDEV_withcontextmasking_bootstrappedf1s.csv'
binary_test_bootstrapped_f1_filename = 'binaryTEST_withcontextmasking_bootstrappedf1s.csv'

# relevant for making the data
binary_positive_sentences_spot_checking_fname = 'data/binary_extracted_positive_sentences_withcontextmasking_correctedsavelocation.txt'
binary_negative_sentences_spot_checking_fname = 'data/binary_extracted_negative_sentences_withcontextmasking_correctedsavelocation.txt'
problem_report_filename = 'data/problem_matches_withcontextmasking_correctedsavelocation.txt'  # or None if you just want to report to the command line
success_report_filename = 'data/successful_matches_withcontextmasking_correctedsavelocation.txt'  # or None if you don't want these reported
dev_precreccurve_plot_filename = 'output_analysis/binarytaskwithcontextmasking_correctedsavelocation_dev_precisionrecallcurve.png'
test_precreccurve_plot_filename = 'output_analysis/binarytaskwithcontextmasking_correctedsavelocation_test_precisionrecallcurve.png'
if problem_report_filename and os.path.isfile(problem_report_filename):
    os.remove(problem_report_filename)
if success_report_filename and os.path.isfile(success_report_filename):
    os.remove(success_report_filename)

use_ten_labels_instead = True
multiway_train_filename = 'data/multiway_mindsduplicates_withcontext/multiway_train.csv'
multiway_dev_filename = 'data/multiway_mindsduplicates_withcontext/multiway_dev.csv'
multiway_test_filename = 'data/multiway_mindsduplicates_withcontext/multiway_test.csv'
multiway_label_key_filename = 'data/multiway_mindsduplicates_withcontext/multiway_classes.txt'
multiway_dev_bootstrapped_f1_filename = 'multiwayDEV_' + multiway_dir_extension + str(use_ten_labels_instead) + '_bootstrappedf1s.csv'
multiway_test_bootstrapped_f1_filename = 'multiwayTEST_' + multiway_dir_extension + str(use_ten_labels_instead) + '_bootstrappedf1s.csv'
output_multiway_model_dir = '../f1-saved_multiway_model_mindsduplicates' + multiway_dir_extension + '/'
multiway_output_report_filename_stub = 'output_analysis/multiway_mindsduplicates' + multiway_dir_extension + '/f1-multiwaybest'
csv_filename_logreg_on_dev = 'output_analysis/multiway_mindsduplicates' + multiway_dir_extension + '/multiwaytask_dev_logregresults.csv'
csv_filename_roberta_on_dev = 'output_analysis/multiway_mindsduplicates' + multiway_dir_extension + '/multiwaytask_dev_robertaresults.csv'
csv_filename_logreg_on_test = 'output_analysis/multiway_mindsduplicates' + multiway_dir_extension + '/multiwaytask_test_logregresults.csv'
csv_filename_roberta_on_test = 'output_analysis/multiway_mindsduplicates' + multiway_dir_extension + '/multiwaytask_test_robertaresults.csv'
csv_filename_logregtest_vs_robertatest = 'output_analysis/multiway_mindsduplicates' + multiway_dir_extension + '/multiwaytask_testroberta_x_testlogreg.csv'

if output_binary_model_dir.endswith('/'):
    output_binary_model_dir = output_binary_model_dir[:-1]
if output_multiway_model_dir.endswith('/'):
    output_multiway_model_dir = output_multiway_model_dir[:-1]
