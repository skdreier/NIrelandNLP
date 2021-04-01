#####################################################
##### Plot justifications per category ##############
##### Plot justifications over time #################
##### (based on PREM_15 files) ######################
#####################################################
##### Feb 2021 ######################################
#####################################################

# Plot fequencies per category (Feb 2021)

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
df_just = pd.read_csv(os.path.join(repo_path, "justifications_clean_text_ohe.csv")) # This is for everything else
#df_just = pd.read_csv(os.path.join(repo_path, "justifications_clean_text_ohe2.csv")) # Plots Internal Sit Parts 1-35
# (Includes placeholder for 100 and 1014)

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

df_just = df_just.set_index('justification_cat')
df_just = df_just.rename(index={'J_Misc':'Misc'})
df_just = df_just.rename(index={'J_Terrorism':'Terrorism'})
df_just = df_just.rename(index={'J_Legal_Procedure':'Legal Authorization'})
df_just = df_just.rename(index={'J_Emergency-Policy':'Emergency Situation'})
df_just = df_just.rename(index={'J_Legal_Procedure':'Legal Authorization'})
df_just = df_just.rename(index={'J_Denial':'Denial'})
df_just = df_just.rename(index={'J_Political-Strategic':'Political'})
df_just = df_just.rename(index={'J_Development-Unity':'Development and Unity'})
df_just = df_just.rename(index={'J_Utilitarian-Deterrence':'Utilitarian/Deterrence'})
df_just = df_just.rename(index={'J_Intl-Domestic_Precedent':'Precedent (Intl/Dom)'})
df_just = df_just.rename(index={'J_Law-and-order':'Law and Order'})
df_just = df_just.rename(index={'J_Last-resort':'Misc'})
df_just = df_just.rename(index={'J_Intelligence':'Misc'})
df_just = df_just.rename(index={'Misc':'Misc (incl Military)'})

df_just.head

#### Plot frequencies by rationalization category over all files (not just PREM 15) ####
plt.figure(figsize=(12,10))

plt.title("Category Frequencies", fontsize=20)
df_just.groupby('justification_cat').text.count().sort_values(ascending=False).plot.bar(ylim=0)
plt.xlabel('')
plt.ylabel('')
plt.xticks(rotation=60, ha='right', fontsize=16)
plt.yticks(fontsize=12)
plt.subplots_adjust(bottom=.27) 

plt.savefig(folder_path + 'freq_plots_2021/2021_Just_freq.png')

######### Plot frequencies of justifications over time (All PREM 15) ###########
plt.figure(figsize=(15,10))

plt.title("Frequencies Over Time", fontsize=20)
df.groupby('prem_15').text.count().plot.bar(ylim=0)
plt.xlabel('PREM 15 File ID (proxy for date)', fontsize=16)
plt.ylabel('')
plt.xticks(rotation=60, fontsize=14)
plt.yticks(fontsize=14)
plt.axvline(x=6.2, c='k')
plt.text(6.8, 250, 'Internment Initiated:', c='k', fontsize=16)
plt.text(6.8, 235, '9-10 August 1971', c='k', fontsize=16)
plt.subplots_adjust(bottom=.1) # or whatever
    
plt.savefig(folder_path + 'freq_plots_2021/2021_J_All_time.png')

######### Plot frequencies of justifications over time ###########
######### Only Plots PREM 15 Internal Situation 1-35 #############
#### With document frequencies ####

# Remove 0486
df = df[df.prem_15 != "0486"]

# Code to count in terminal:
# ls *100.pdf | wc -l #Counts Prem 15 100 files
# ls *100*.pdf | wc -l

# Manually enter (agh) number of pages per file  
data = [
    [0, 143], #100
    [1, 195], #101
    [2, 145], #474
    [3, 140], #475 #139+1 -- counted as 305??? ####
    [4, 72],  #476
    [5, 115], #477
    [6, 275], #478 
    [7, 149], #479
    [8, 96], #480
    [9, 95], #481
    [10, 122], #482
    [11, 180], #483
    [12, 105], #484
    [13, 134], #1000
    [14, 144], #1001
    [15, 156], #1002
    [16, 133], #1003
    [17, 110], #1004
    [18, 128], #1005
    [19, 94], #1006
    [20, 144], #1007
    [21, 118], #1008
    [22, 127], #1009
    [23, 194], #1010    
    [24, 92], #1011    
    [25, 192], #1012
    [26, 132], #1013 # 131+1
    [27, 125], #1014 #labeled as 1013, 124+1
    [28, 188], #1015 
    [29, 217], #1016     
    [30, 188], #1689   
    [31, 139], #1690     
    [32, 118], #1691     
    [33, 159], #1692     
    [34, 145]  #1693     
]
x, y = zip(*data)

plt.figure(figsize=(15,8))

plt.title("Document and Justification Frequencies Over Time", fontsize=20)
df.groupby('prem_15').text.count().plot.bar(ylim=0, label="Justification sentences per file")
plt.plot(x, y, c="k", label="Pages per file")
plt.scatter(x, y, c="k")
plt.xlabel('PREM 15 Files: Internal Situation in Northern Ireland, Parts 1-35 (06/1970 - 07/1973)\nIncremental files cover between 1-5 months. Files provide proxy for date.', fontsize=16)
plt.ylabel('')
plt.xticks(rotation=60, fontsize=14)
plt.yticks(fontsize=14)
plt.axvline(x=6.2, c='k')
plt.text(6.8, 350, 'Internment Initiated:', c='k', fontsize=16)
plt.text(6.8, 335, '9-10 August 1971', c='k', fontsize=16)
plt.subplots_adjust(bottom=.1) # or whatever
#ax.legend([line1, line2, line3], ['label1', 'label2', 'label3'])
plt.legend(loc="upper right", fontsize=16)
plt.tight_layout()
    
plt.savefig(folder_path + 'freq_plots_2021/2021_J_All_time_pg_count.png')


######### Plot absolute and relative frequencies of justifications per category ##########

# Count overall justifications / file
df_count = df.groupby('prem_15').text.count()
df_count = pd.DataFrame(df_count)
df_count.index.name = 'prem_15'
df_count.reset_index(inplace=True)
df_count['total_just_per_file'] = df_count['text']
df_count = df_count[['prem_15', 'total_just_per_file']]

df['justification_cat'] = df['justification_cat'].replace("J_Misc", "Misc (incl Military)")
df['justification_cat'] = df['justification_cat'].replace("J_Terrorism", "Terrorism")
df['justification_cat'] = df['justification_cat'].replace("J_Denial", "Denial")
df['justification_cat'] = df['justification_cat'].replace("J_Development-Unity", "Development and Unity")
df['justification_cat'] = df['justification_cat'].replace("J_Emergency-Policy", "Emergency Situation")
df['justification_cat'] = df['justification_cat'].replace("J_Intl-Domestic_Precedent", "Precedent (Intl or Dom)")
df['justification_cat'] = df['justification_cat'].replace("J_Last-resort", "Misc (incl Military)")
df['justification_cat'] = df['justification_cat'].replace("J_Law-and-order", "Law and Order")
df['justification_cat'] = df['justification_cat'].replace("J_Legal_Procedure", "Legal Authorization")
df['justification_cat'] = df['justification_cat'].replace("J_Political-Strategic", "Political")
df['justification_cat'] = df['justification_cat'].replace("J_Utilitarian-Deterrence", "Utilitarianism or Deterrence")

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
    plt.title(cat, fontsize=20)
    #plt.xlabel('File (Proxy for date)')
    plt.ylim([0, 75])
    plt.xticks(rotation=60, fontsize=14)
    plt.yticks(fontsize=14)
    plt.axvline(x=5.2, c='k')
    plt.text(5.8, 50, 'Internment', c='k', fontsize=16)
    plt.text(5.8, 45, 'Initiated', c='k', fontsize=16)

   # plt.text(5.8, 30, 'SKD:', c='k', fontsize=16)
    #plt.text(5.8, 235, '9-10 August 1971', c='k', fontsize=16)
    
    plt.subplot(2,1,2)
    plt.bar(df_cat_merge['prem_15'], df_cat_merge['prop'])
    plt.ylabel('Proportion')
    plt.ylim([0, 1])
    plt.xticks(rotation=60, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('PREM 15 file (Proxy for date)', fontsize=16)
    plt.axvline(x=5.2, c='k')
    #plt.text(5.8, 50, 'Internment', c='k', fontsize=16)
    #plt.text(5.8, 40, 'Initiated', c='k', fontsize=16)

    #plt.text(5.8, 235, '9-10 August 1971', c='k', fontsize=16)

    plt.savefig(folder_path + 'freq_plots_2021/2021_count_prop_' + cat + '_time.png')
    plt.close()
