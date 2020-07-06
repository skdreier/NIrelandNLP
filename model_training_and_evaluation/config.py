import os


full_document_filename = '../orig_text_data/internment.txt'
positive_sentence_filename = '../justifications_clean_text_ohe.csv'

binary_train_filename = 'data/binary_train.csv'
binary_dev_filename = 'data/binary_dev.csv'
binary_test_filename = 'data/binary_test.csv'
binary_label_key_filename = 'data/binary_classes.txt'
output_binary_model_dir = '../f1-saved_binary_model/'
binary_output_report_filename_stub = 'output_analysis/f1-binarybest'
binary_positive_sentences_spot_checking_fname = 'data/binary_extracted_positive_sentences.txt'
binary_negative_sentences_spot_checking_fname = 'data/binary_extracted_negative_sentences.txt'
problem_report_filename = 'data/problem_matches.txt'  # or None if you just want to report to the command line
success_report_filename = 'data/successful_matches.txt'  # or None if you don't want these reported
dev_precreccurve_plot_filename = 'output_analysis/binarytask_dev_precisionrecallcurve.png'
test_precreccurve_plot_filename = 'output_analysis/binarytask_test_precisionrecallcurve.png'
if problem_report_filename and os.path.isfile(problem_report_filename):
    os.remove(problem_report_filename)
if success_report_filename and os.path.isfile(success_report_filename):
    os.remove(success_report_filename)

multiway_train_filename = 'data/multiway_second_split_condensed/multiway_train.csv'
multiway_dev_filename = 'data/multiway_second_split_condensed/multiway_dev.csv'
multiway_test_filename = 'data/multiway_second_split_condensed/multiway_test.csv'
multiway_label_key_filename = 'data/multiway_second_split_condensed/multiway_classes.txt'
output_multiway_model_dir = '../f1-saved_multiway_model_secondsplit_condensed/'
multiway_output_report_filename_stub = 'output_analysis/multiway_second_split_condensed/f1-multiwaybest'
csv_filename_logreg_on_dev = 'output_analysis/multiway_second_split_condensed/multiwaytask_dev_logregresults.csv'
csv_filename_roberta_on_dev = 'output_analysis/multiway_second_split_condensed/multiwaytask_dev_robertaresults.csv'
csv_filename_logreg_on_test = 'output_analysis/multiway_second_split_condensed/multiwaytask_test_logregresults.csv'
csv_filename_roberta_on_test = 'output_analysis/multiway_second_split_condensed/multiwaytask_test_robertaresults.csv'

if output_binary_model_dir.endswith('/'):
    output_binary_model_dir = output_binary_model_dir[:-1]
if output_multiway_model_dir.endswith('/'):
    output_multiway_model_dir = output_multiway_model_dir[:-1]
