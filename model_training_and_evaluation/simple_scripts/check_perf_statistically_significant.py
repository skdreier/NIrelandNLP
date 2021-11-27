from scipy.stats import ttest_ind
import sys
sys.path.append('../')
from config import binary_test_bootstrapped_f1_filename, multiway_test_bootstrapped_f1_filename
binary_test_bootstrapped_f1_filename = '../' + binary_test_bootstrapped_f1_filename
multiway_test_bootstrapped_f1_filename = '../' + multiway_test_bootstrapped_f1_filename


def load_stats_from_file(fname):
    roberta_list = []
    othermodel_list = []
    with open(fname, 'r') as f:
        f.readline()
        for line in f:
            pieces = line.strip().split(',')
            roberta_list.append(float(pieces[0]))
            othermodel_list.append(float(pieces[1]))
    return roberta_list, othermodel_list


def run_test_for(fname):
    roberta_scores, other_model_scores = load_stats_from_file(fname)
    tstat, pval = ttest_ind(roberta_scores, other_model_scores, axis=None, nan_policy='omit', alternative='greater')
    print(fname + ': p-val is ' + str(pval))
    return tstat, pval


def get_recall_fname(fname):
    return fname[:fname.rfind('.')] + '-recall' + fname[fname.rfind('.'):]


def get_precision_fname(fname):
    return fname[:fname.rfind('.')] + '-precision' + fname[fname.rfind('.'):]


def append_vsneuralbaseline_to_fname(fname):
    return fname[:fname.rfind('.')] + '-vsneuralbaseline' + fname[fname.rfind('.'):]


print('Multiway compared-to-logreg F1:')
run_test_for(multiway_test_bootstrapped_f1_filename)
print('Multiway compared-to-logreg recall:')
run_test_for(get_recall_fname(multiway_test_bootstrapped_f1_filename))
print('Multiway compared-to-logreg precision:')
run_test_for(get_precision_fname(multiway_test_bootstrapped_f1_filename))

multiway_test_bootstrapped_f1_filename_nn = append_vsneuralbaseline_to_fname(multiway_test_bootstrapped_f1_filename)
print('Multiway compared-to-neural F1:')
run_test_for(multiway_test_bootstrapped_f1_filename_nn)
print('Multiway compared-to-neural recall:')
run_test_for(get_recall_fname(multiway_test_bootstrapped_f1_filename_nn))
print('Multiway compared-to-neural precision:')
run_test_for(get_precision_fname(multiway_test_bootstrapped_f1_filename_nn))

print()

print('Binary compared-to-logreg F1:')
run_test_for(binary_test_bootstrapped_f1_filename)
print('Binary compared-to-logreg recall:')
run_test_for(get_recall_fname(binary_test_bootstrapped_f1_filename))
print('Binary compared-to-logreg precision:')
run_test_for(get_precision_fname(binary_test_bootstrapped_f1_filename))

binary_test_bootstrapped_f1_filename_nn = append_vsneuralbaseline_to_fname(binary_test_bootstrapped_f1_filename)
print('Binary compared-to-neural F1:')
run_test_for(binary_test_bootstrapped_f1_filename_nn)
print('Binary compared-to-neural recall:')
run_test_for(get_recall_fname(binary_test_bootstrapped_f1_filename_nn))
print('Binary compared-to-neural precision:')
run_test_for(get_precision_fname(binary_test_bootstrapped_f1_filename_nn))
