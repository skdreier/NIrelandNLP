import matplotlib.pyplot as plt
import numpy as np


date_to_pagecounts_file = 'dates_to_pagecounts.csv'
date_to_occurrences_file = 'dates_to_occurrencecounts.csv'
make_plot_only_in_busier_section = True
color_for_vertline = (1, 0.387, 0, 1)


def read_in_file(fname):
    xs = []
    ys = []
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            fields = line.split(',')
            xs.append(fields[0])
            ys.append(int(fields[1]))
    return xs, ys

dates, pagecounts = read_in_file(date_to_pagecounts_file)
new_dates, occurrencecounts = read_in_file(date_to_occurrences_file)

assert len(dates) == len(new_dates)
for i in range(len(dates)):
    assert dates[i] == new_dates[i]


if make_plot_only_in_busier_section:
    for i in range(len(dates) - 1, -1, -1):
        year = dates[i].split('-')
        month = int(year[1])
        year = int(year[0])
        if year < 1970:
            del dates[i]
            del occurrencecounts[i]
            del pagecounts[i]
        elif year == 1970 and month < 4:
            del dates[i]
            del occurrencecounts[i]
            del pagecounts[i]


def convert_date_string(date_string):
    fields = date_string.split('-')
    year = fields[0]
    month = int(fields[1])
    if month == 1:
        strmonth = 'Jan'
    elif month == 2:
        strmonth = 'Feb'
    elif month == 3:
        strmonth = 'Mar'
    elif month == 4:
        strmonth = 'Apr'
    elif month == 5:
        strmonth = 'May'
    elif month == 6:
        strmonth = 'Jun'
    elif month == 7:
        strmonth = 'Jul'
    elif month == 8:
        strmonth = 'Aug'
    elif month == 9:
        strmonth = 'Sep'
    elif month == 10:
        strmonth = 'Oct'
    elif month == 11:
        strmonth = 'Nov'
    elif month == 12:
        strmonth = 'Dec'
    return strmonth + ' ' + year


dates = [convert_date_string(date) for date in dates]


if make_plot_only_in_busier_section:
    fig, ax = plt.subplots(figsize=(16, 7))
else:
    fig, ax = plt.subplots(figsize=(16, 4.5))

# We need to draw the canvas, otherwise the labels won't be positioned and
# won't have values yet.
fig.canvas.draw()


for i in range(len(dates)):
    if dates[i] == 'Aug 1971':
        index_to_use = i + 1
        break
plt.axvline(x=index_to_use, color=color_for_vertline)
ax.set_axisbelow(True)
plt.grid(visible=True)



modified_occurrence_counts = []
for i, count in enumerate(occurrencecounts):
    for _ in range(count):
        modified_occurrence_counts.append(i + 1)

plt.hist(modified_occurrence_counts, bins=[i + 0.5 for i in range(len(occurrencecounts) + 1)], ec='white',
         label='Justification sentences in month', zorder=10)
plt.xlim(0.5, len(occurrencecounts) + 0.5)
plt.ylim(0, 650)
plt.xticks(np.arange(1, len(occurrencecounts) + 1, 2.0))


plt.plot([i + 1 for i in range(len(pagecounts))], pagecounts, color='black', label='Pages in month',
         zorder=11)#, marker='.')


labels = [item.get_text() for item in ax.get_xticklabels()]
for i in range(len(dates)):
    if i % 2 == 0:
        labels[i // 2] = dates[i]

ax.tick_params(pad=0)
if make_plot_only_in_busier_section:
    ax.set_xticklabels(labels, rotation=90)  # , horizontalalignment='right')
else:
    ax.set_xticklabels(labels, rotation=90, fontsize=6)#, horizontalalignment='right')

plt.legend()
if make_plot_only_in_busier_section:
    fontsize = 18
else:
    fontsize = 12
plt.title('Page and Justification Frequencies Over Time', fontsize=fontsize)
if make_plot_only_in_busier_section:
    plt.text(index_to_use + 0.2, 550, 'Internment initiated:\n9-10 August 1971', color=color_for_vertline,
             fontweight='bold')
else:
    plt.text(index_to_use - 0.2, 530, 'Internment initiated:\n9-10 August 1971', color=color_for_vertline,
             fontweight='bold', horizontalalignment='right')


plt.show()
