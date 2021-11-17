import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


df = pd.read_csv('../data_for_recallFnameDateHist.csv')
toplevel_fname_to_truecounts = {}
toplevel_fname_to_recalllists = {}
for i, row in df.iterrows():
    toplevel_fname = row['filename'].split('/')[0]
    if toplevel_fname in toplevel_fname_to_truecounts:
        toplevel_fname_to_truecounts[toplevel_fname] += row['truepositivecount']
        recalllists_tup = toplevel_fname_to_recalllists[toplevel_fname]
        toplevel_fname_to_recalllists[toplevel_fname] = (row['defaultthreshold_positivesrecovered'] + recalllists_tup[0],
                                                         row['0.7recall_positivesrecovered'] + recalllists_tup[1],
                                                         row['0.8recall_positivesrecovered'] + recalllists_tup[2],
                                                         row['0.9recall_positivesrecovered'] + recalllists_tup[3])
    else:
        toplevel_fname_to_truecounts[toplevel_fname] = row['truepositivecount']
        toplevel_fname_to_recalllists[toplevel_fname] = (row['defaultthreshold_positivesrecovered'],
                                                         row['0.7recall_positivesrecovered'],
                                                         row['0.8recall_positivesrecovered'],
                                                         row['0.9recall_positivesrecovered'])
value_to_listsofrecalls = {}
for fname, truecount in toplevel_fname_to_truecounts.items():
    if truecount in value_to_listsofrecalls:
        value_to_listsofrecalls[truecount].append(toplevel_fname_to_recalllists[fname])
    else:
        value_to_listsofrecalls[truecount] = [toplevel_fname_to_recalllists[fname]]

for val, lists in sorted(list(value_to_listsofrecalls.items()), key=lambda x: x[0]):
    print(val)
    print(lists)

lists_to_order = []
for val, lists in list(value_to_listsofrecalls.items()):
    for recalltup in lists:
        lists_to_order.append((recalltup[0], recalltup[1], recalltup[2], recalltup[3], val))

lists_to_order = sorted(lists_to_order,
                        key=lambda tup: tup[4] * 21**4 + tup[3] * 21**3 + tup[2] * 21**2 + tup[1] * 21 + tup[0])
print(lists_to_order)

# order the values
recall_str = 'RoBERTa recall after thresholding'
list_of_dicts = []  # each dict corresponds to a particular tuple list
cur_dummy_xval = 0.0
for tup in lists_to_order:
    # add this dict_to_add tup[-1] times
    for i in range(tup[0]):
        dict_to_add = {'File': cur_dummy_xval}
        dict_to_add[recall_str] = '61%'
        list_of_dicts.append(dict_to_add)
    for i in range(tup[1] - tup[0]):
        dict_to_add = {'File': cur_dummy_xval}
        dict_to_add[recall_str] = '70%'
        list_of_dicts.append(dict_to_add)
    for i in range(tup[2] - tup[1]):
        dict_to_add = {'File': cur_dummy_xval}
        dict_to_add[recall_str] = '80%'
        list_of_dicts.append(dict_to_add)
    for i in range(tup[3] - tup[2]):
        dict_to_add = {'File': cur_dummy_xval}
        dict_to_add[recall_str] = '90%'
        list_of_dicts.append(dict_to_add)
    for i in range(tup[4] - tup[3]):
        dict_to_add = {'File': cur_dummy_xval}
        dict_to_add[recall_str] = '100%'
        list_of_dicts.append(dict_to_add)
    cur_dummy_xval += 1

same_list_of_dicts_but_reordered = []
skip_these_inds = []
for i in range(len(list_of_dicts)):
    if list_of_dicts[i][recall_str] == '100%':
        skip_these_inds.append(i)
        same_list_of_dicts_but_reordered.append(list_of_dicts[i])
        break
for i in range(len(list_of_dicts)):
    if list_of_dicts[i][recall_str] == '90%' and i not in skip_these_inds:
        skip_these_inds.append(i)
        same_list_of_dicts_but_reordered.append(list_of_dicts[i])
        break
for i in range(len(list_of_dicts)):
    if list_of_dicts[i][recall_str] == '80%' and i not in skip_these_inds:
        skip_these_inds.append(i)
        same_list_of_dicts_but_reordered.append(list_of_dicts[i])
        break
for i in range(len(list_of_dicts)):
    if list_of_dicts[i][recall_str] == '70%' and i not in skip_these_inds:
        skip_these_inds.append(i)
        same_list_of_dicts_but_reordered.append(list_of_dicts[i])
        break
for i in range(len(list_of_dicts)):
    if list_of_dicts[i][recall_str] == '61%' and i not in skip_these_inds:
        skip_these_inds.append(i)
        same_list_of_dicts_but_reordered.append(list_of_dicts[i])
        break
for i in range(len(list_of_dicts)):
    if i not in skip_these_inds:
        same_list_of_dicts_but_reordered.append(list_of_dicts[i])

dummy_df = pd.DataFrame(same_list_of_dicts_but_reordered)

f = plt.figure(figsize=(7,5))
ax = f.add_subplot(1,1,1)

sns.histplot(data=dummy_df, ax=ax, stat="count", multiple="stack",
             x="File", kde=False,
             palette="rocket", hue=recall_str,
             element="bars", legend=True, bins=[i for i in range(int(cur_dummy_xval) + 1)])
ax.set_title("RoBERTa's recall in different files post-thresholding")
ax.set_xlabel("Files in test set with at least one true positive sentence\n(each bar representing a different, possibly multi-page file)")
ax.set_ylabel("Total true positives recovered from file")

ax = plt.gca()
ax.axes.xaxis.set_ticks([])

plt.savefig('RecallHist.png', bbox_inches='tight')
plt.close(f)
