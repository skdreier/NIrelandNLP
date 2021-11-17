import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from tqdm import tqdm


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

def prime_factorization(n):
    """
    from https://scientific-python-101.readthedocs.io/python/exercises/prime_factorization.html

    Return the prime factorization of `n`.

    Parameters
    ----------
    n : int
        The number for which the prime factorization should be computed.

    Returns
    -------
    dict[int, int]
        List of tuples containing the prime factors and multiplicities of `n`.

    """
    prime_factors = {}

    i = 2
    while i**2 <= n:
        if n % i:
            i += 1
        else:
            n /= i
            try:
                prime_factors[i] += 1
            except KeyError:
                prime_factors[i] = 1

    if n > 1:
        try:
            prime_factors[n] += 1
        except KeyError:
            prime_factors[n] = 1
    return prime_factors

accumulated_prime_factors = {}
for val in value_to_listsofrecalls:
    prime_factors = prime_factorization(val)
    for factor, num_times in prime_factors.items():
        if factor not in accumulated_prime_factors:
            accumulated_prime_factors[factor] = num_times
        elif accumulated_prime_factors[factor] < num_times:
            accumulated_prime_factors[factor] = num_times
total_num_rows_per_dummy = 1
for pf, num_times in accumulated_prime_factors.items():
    total_num_rows_per_dummy *= (pf ** num_times)
total_num_rows_per_dummy = int(total_num_rows_per_dummy)
print(total_num_rows_per_dummy)

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
prev_total = None
for tup in tqdm(lists_to_order):
    num_times_to_add = total_num_rows_per_dummy // tup[4]
    # add this dict_to_add tup[-1] times
    cur_total = 0
    if tup[0] > 0:
        dict_to_add = {'File': cur_dummy_xval}
        dict_to_add[recall_str] = '61%'
        cur_total += tup[0] * num_times_to_add
        for i in range(tup[0]):
            for _ in range(num_times_to_add):
                list_of_dicts.append(dict_to_add)
    if tup[1] > tup[0]:
        dict_to_add = {'File': cur_dummy_xval}
        dict_to_add[recall_str] = '70%'
        cur_total += (tup[1] - tup[0]) * num_times_to_add
        for i in range(tup[1] - tup[0]):
            for _ in range(num_times_to_add):
                list_of_dicts.append(dict_to_add)
    if tup[2] > tup[1]:
        dict_to_add = {'File': cur_dummy_xval}
        dict_to_add[recall_str] = '80%'
        cur_total += (tup[2] - tup[1]) * num_times_to_add
        for i in range(tup[2] - tup[1]):
            for _ in range(num_times_to_add):
                list_of_dicts.append(dict_to_add)
    if tup[3] > tup[2]:
        dict_to_add = {'File': cur_dummy_xval}
        dict_to_add[recall_str] = '90%'
        cur_total += (tup[3] - tup[2]) * num_times_to_add
        for i in range(tup[3] - tup[2]):
            for _ in range(num_times_to_add):
                list_of_dicts.append(dict_to_add)
    if tup[4] > tup[3]:
        dict_to_add = {'File': cur_dummy_xval}
        dict_to_add[recall_str] = '100%'
        cur_total += (tup[4] - tup[3]) * num_times_to_add
        for i in range(tup[4] - tup[3]):
            for _ in range(num_times_to_add):
                list_of_dicts.append(dict_to_add)
    cur_dummy_xval += 1
    if prev_total is not None:
        assert cur_total == prev_total, str(prev_total) + ', ' + str(cur_total)
    prev_total = cur_total

same_list_of_dicts_but_reordered = []
skip_these_inds = set()
for i in range(len(list_of_dicts)):
    if list_of_dicts[i][recall_str] == '100%':
        skip_these_inds.add(i)
        same_list_of_dicts_but_reordered.append(list_of_dicts[i])
        break
for i in range(len(list_of_dicts)):
    if list_of_dicts[i][recall_str] == '90%' and i not in skip_these_inds:
        skip_these_inds.add(i)
        same_list_of_dicts_but_reordered.append(list_of_dicts[i])
        break
for i in range(len(list_of_dicts)):
    if list_of_dicts[i][recall_str] == '80%' and i not in skip_these_inds:
        skip_these_inds.add(i)
        same_list_of_dicts_but_reordered.append(list_of_dicts[i])
        break
for i in range(len(list_of_dicts)):
    if list_of_dicts[i][recall_str] == '70%' and i not in skip_these_inds:
        skip_these_inds.add(i)
        same_list_of_dicts_but_reordered.append(list_of_dicts[i])
        break
for i in range(len(list_of_dicts)):
    if list_of_dicts[i][recall_str] == '61%' and i not in skip_these_inds:
        skip_these_inds.add(i)
        same_list_of_dicts_but_reordered.append(list_of_dicts[i])
        break
for i in range(len(list_of_dicts)):
    if i not in skip_these_inds:
        same_list_of_dicts_but_reordered.append(list_of_dicts[i])
assert len(same_list_of_dicts_but_reordered) == len(list_of_dicts)

dummy_df = pd.DataFrame(same_list_of_dicts_but_reordered)

f = plt.figure(figsize=(7,5))
ax = f.add_subplot(1,1,1)

sns.histplot(data=dummy_df, ax=ax, stat="count", multiple="stack",
             x="File", kde=False,
             palette="rocket", hue=recall_str,
             element="bars", legend=True, bins=[i for i in range(int(cur_dummy_xval) + 1)])
ax.set_title("RoBERTa's recall in different files post-thresholding")
ax.set_xlabel("Files in test set with at least one true positive sentence\n(each bar representing a different, possibly multi-page file,\nnumber of true positives in file annotated underneath)")
ax.set_ylabel("Fraction of true positives recovered from file")
x_ticks_labels = [str(tup[4]) for tup in lists_to_order]
ax.set_xticklabels(x_ticks_labels)
ax.set_yticklabels(['0.00', '0.25', '0.50', '0.75', '1.00'])

ax = plt.gca()
ax.axes.xaxis.set_ticks([i + 0.5 for i in range(int(cur_dummy_xval) + 1)])
ax.axes.yaxis.set_ticks([0, total_num_rows_per_dummy // 4, total_num_rows_per_dummy // 2,
                         3 * total_num_rows_per_dummy // 4, total_num_rows_per_dummy])

plt.savefig('RecallHistNormalized.png', bbox_inches='tight')
plt.close(f)
