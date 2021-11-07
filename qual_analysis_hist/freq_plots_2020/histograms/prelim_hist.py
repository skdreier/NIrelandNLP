import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

this_file_path = os.path.abspath(__file__)
folder_root = os.path.split(this_file_path)[0]
repo_root = os.path.split(folder_root)[0]
repo_path = os.path.join(repo_root)

data_raw = pd.read_csv(os.path.join(repo_path, 'justifications_clean_text_ohe.csv'))

#print("Number of rows in data =",data_raw.shape[0])
#print("Number of columns in data =",data_raw.shape[1])
#data_raw.head()

df = pd.DataFrame(data_raw['justification_cat'].value_counts())
#df['just_name'] = df.index
df.reset_index(level=0, inplace=True)

sns.set(font_scale = 2)
plt.figure(figsize=(15,8))

ax= sns.barplot(df['index'],df['justification_cat']) 

plt.title("Codes in each category", fontsize=24)
plt.ylabel('Number of codes', fontsize=18)
plt.xlabel('Justification Type ', fontsize=18)
ax.set_xticklabels([])

#adding the text labels
rects = ax.patches
labels = df['index']
for rect, label in zip(rects, labels): 
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=12)
plt.savefig('just_12.png')
plt.show()

# For analysis, start with: Terrorism, Emergency, Legal_procedure, Political-strategic, Denial