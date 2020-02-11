import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "/Users/sarahdreier/OneDrive/Incubator/NIreland_NLP/justifications_long_training.csv"
data_raw = pd.read_csv(data_path)
print("Number of rows in data =",data_raw.shape[0])
print("Number of columns in data =",data_raw.shape[1])
print("\n")
print("**Sample data:**")
data_raw.head()


df = pd.DataFrame(data_raw['justification'].value_counts())
#df['just_name'] = df.index
df.reset_index(level=0, inplace=True)

sns.set(font_scale = 2)
plt.figure(figsize=(15,8))

ax= sns.barplot(df['index'],df['justification']) 

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
plt.show()
plt.savefig('just_hist.png')



# For analysis, start with: Terrorism, Emergency, Legal_procedure, Political-strategic, Denial