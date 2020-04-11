#####################################################
##### Plot justifications over time #################
##### (based on PREM_15 files) ######################
#####################################################

import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

this_file_path = os.path.abspath(__file__)
folder_root = os.path.split(this_file_path)[0]
folder_path = os.path.join(folder_root) + '/'
repo_root = os.path.split(folder_root)[0]
repo_path = os.path.join(repo_root) + '/'

sys.path # make sure the repo is in the path 

# Download justification file
df_just = pd.read_csv(os.path.join(repo_path, "justifications_clean_text_ohe.csv"))

# Build a date var for visualization
df_just['file_start_year'] = df_just['file_start_year'].fillna(0.0).astype(int)
df_just['file_start_month'] = df_just['file_start_month'].fillna(0.0).astype(int)
df_just['date'] = df_just.file_start_month.map(str) + "_" + df_just.file_start_year.map(str)

# Transform file var for sorting
df = df_just[df_just['file_id'].str.contains(r'PREM_15')] # Subset to PREM_15 files

df_1 = df[~df['file_id'].str.contains(r'PREM_15_\d{4}')] # Pull PREM files < 1000 to add "0" placeholder
df_1["file_id"] = df_1["file_id"].str.replace(r"PREM_15_", "PREM_15_0").str.strip()
df_2 = df[df['file_id'].str.contains(r'PREM_15_\d{4}')] # Pull PREM files >= 1000
df = pd.concat([df_1, df_2]) # Recombine df_1 and df_2
df = df[df['file_id']!=('PREM_15_0485' or 'PREM_15_0486')] # Remove 485 and 486 (not Int Sit files)
df['prem_15'] = df['file_id'].str.replace(r'PREM_15_', '') # Remove PREM_15 for ease of visualization

# PREM_15_100 (Int Sit NI part 1): No Internment references
# PREM_15_485 and PREM_15_486: Remove (not Int Sit files)
# PREM_15_1014 lumped in w PREM_15_1013 -- Keep together for now (only re 5 docs)

# Plots all justifications over time

plt.figure(figsize=(10,10))

plt.subplot(2,1,1)
plt.title("Justification frequency per PREM 15 file")
df.groupby('prem_15').text.count().plot.bar(ylim=0)
plt.xlabel('File ID (proxy for date)')
plt.ylabel('Count')
plt.xticks(rotation=60)
    
plt.subplot(2,1,2)
(df.groupby('prem_15').text.count() / sum(df.groupby('prem_15').text.count())).plot.bar(ylim=0)
plt.xlabel('File ID (proxy for date)')
plt.ylabel('Proportion of overall justifications')
plt.xticks(rotation=60)

plt.savefig(folder_path + 'freq_plots/J_All_time.png')


# Count overall justifications / file
df_count = df.groupby('prem_15').text.count()
df_count = pd.DataFrame(df_count)
df_count.index.name = 'prem_15'
df_count.reset_index(inplace=True)
df_count['total_just_per_file'] = df_count['text']
df_count = df_count[['prem_15', 'total_just_per_file']]

# Loop to plot each category: count and proportion of justifications in that file
for cat in df['justification_cat'].unique():

    df_cat = df[df['justification_cat']==cat]
    df_cat_count = df_cat.groupby('prem_15').clean_text.count()
    df_cat_merge = pd.DataFrame.merge(df_count, df_cat_count, 'left', on='prem_15')
    df_cat_merge.columns = ['prem_15', 'total_just_per_file', 'cat_just_per_file']
    df_cat_merge = df_cat_merge.fillna(0)
    df_cat_merge['prop'] = df_cat_merge['cat_just_per_file'] / df_cat_merge['total_just_per_file']
    
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    plt.bar(df_cat_merge['prem_15'], df_cat_merge['cat_just_per_file'])
    plt.ylabel('Count')
    plt.title(cat)
    plt.xlabel('File (Proxy for date)')
    plt.xticks(rotation=60)
    
    plt.subplot(2,1,2)
    plt.bar(df_cat_merge['prem_15'], df_cat_merge['prop'])
    plt.ylabel('Proportion')
    #plt.xticks('')
    plt.xticks(rotation=60)
    plt.xlabel('File (Proxy for date)')

    plt.savefig(folder_path + 'freq_plots/count_prop' + cat + '_time.png')
    plt.close()


# Loop to plot each category: raw count only

for cat in df['justification_cat'].unique():

    df_cat = df[df['justification_cat']==cat]
    df_cat_count = df_cat.groupby('prem_15').clean_text.count()
    df_cat_merge = pd.DataFrame.merge(df_count, df_cat_count, 'left', on='prem_15')
    df_cat_merge.columns = ['prem_15', 'total_just_per_file', 'cat_just_per_file']
    df_cat_merge = df_cat_merge.fillna(0)
    
    plt.figure(figsize=(10,5))
    plt.bar(df_cat_merge['prem_15'], df_cat_merge['cat_just_per_file'])
    plt.ylabel('Count')
    plt.title(cat)
    #plt.xlabel('File (Proxy for date)')
    plt.xticks("")
    
    plt.savefig(folder_path + 'freq_plots/raw_count' + cat + '_time.png')
    plt.close()

##########################################################################################
##########################################################################################

# Subset to category
df_cat = df[df['justification_cat']=="J_Political-Strategic"]
df_cat_count = df_cat.groupby('prem_15').clean_text.count()
df_cat_merge = pd.DataFrame.merge(df_count, df_cat_count, 'left', on='prem_15')
df_cat_merge.columns = ['prem_15', 'total_just_per_file', 'cat_just_per_file']
df_cat_merge = df_cat_merge.fillna(0)
df_cat_merge['prop'] = df_cat_merge['cat_just_per_file'] / df_cat_merge['total_just_per_file']

plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.bar(df_cat_merge['prem_15'], df_cat_merge['cat_just_per_file'])
plt.ylabel('Count')
plt.title("Political/Strategic")
plt.xticks(rotation=90)
plt.xlabel('File (Proxy for date)')
plt.xticks(rotation=60)

plt.subplot(2,1,2)
plt.bar(df_cat_merge['prem_15'], df_cat_merge['prop'])
plt.ylabel('Proportion')
#plt.xticks('')
plt.xticks(rotation=60)
plt.xlabel('File (Proxy for date)')

plt.show()
plt.savefig(folder_path + 'freq_plots/J_Political-Strategic_time.png')

