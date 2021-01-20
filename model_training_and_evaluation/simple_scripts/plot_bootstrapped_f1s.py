import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')


test_f1s_filename = '/Users/sofiaserrano/Downloads/paperResults/multiwayTEST_withcontext_bootstrappedf1s.csv'
dev_f1s_filename = '/Users/sofiaserrano/Downloads/paperResults/multiwayDEV_withcontext_bootstrappedf1s.csv'
tag = 'multiway'


sns.set()


def read_in_data(fname):
    roberta_f1s = []
    baseline_f1s = []
    with open(fname, 'r') as f:
        field_names = f.readline().strip().split(',')
        roberta_is_field_0 = (field_names[0] == 'bootstrapped_f1_roberta')
        if not roberta_is_field_0:
            assert field_names[0] == 'bootstrapped_f1_baseline'
        for line in f:
            string_vals = line.strip().split(',')
            if roberta_is_field_0:
                roberta_f1s.append(float(string_vals[0]))
                baseline_f1s.append(float(string_vals[1]))
            else:
                roberta_f1s.append(float(string_vals[1]))
                baseline_f1s.append(float(string_vals[0]))
    return roberta_f1s, baseline_f1s


dev_roberta_f1s, dev_baseline_f1s = read_in_data(dev_f1s_filename)
test_roberta_f1s, test_baseline_f1s = read_in_data(test_f1s_filename)


list_of_row_dicts = []
for data_point in dev_roberta_f1s:
    row_dict = {'Data split': 'Dev',
                "Model": "RoBERTa",
                "Bootstrapped F1 score": data_point}
    list_of_row_dicts.append(row_dict)
for data_point in test_roberta_f1s:
    row_dict = {'Data split': 'Test',
                "Model": "RoBERTa",
                "Bootstrapped F1 score": data_point}
    list_of_row_dicts.append(row_dict)
for data_point in dev_baseline_f1s:
    row_dict = {'Data split': 'Dev',
                "Model": "Baseline",
                "Bootstrapped F1 score": data_point}
    list_of_row_dicts.append(row_dict)
for data_point in test_baseline_f1s:
    row_dict = {'Data split': 'Test',
                "Model": "Baseline",
                "Bootstrapped F1 score": data_point}
    list_of_row_dicts.append(row_dict)
data_to_plot = pd.DataFrame(list_of_row_dicts)


fig = plt.figure(figsize=(12, 4))
#plt.ylim(0, 1)
ax = sns.boxplot(x="Data split", y="Bootstrapped F1 score", hue="Model", data=data_to_plot, palette='PuOr')
plt.title('Bootstrapped F1 scores for ' + tag + ' held-out data')
plt.savefig('/Users/sofiaserrano/Downloads/paperResults/BootstrappedF1s' + tag + '.png', bbox_inches='tight')
plt.close(fig)
