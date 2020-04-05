###############################################
## NI NLP Project
## Merge date with justification codes
## NOTE: This is a temporary output. Once we get the whole data file as a .txt, we can: 
## 1) run a loop that assigns every date to the previous date, then
## 2) merge this to the justification file output.
## 
###############################################

import os
from pathlib import Path
import re
import pandas as pd

## Date range (based on archive file records)

this_file_path = os.path.abspath(__file__)
subfolder_root = os.path.split(this_file_path)[0]
folder_root = os.path.split(subfolder_root)[0]
repo_root = os.path.split(folder_root)[0]

# Upload justification file
df_just = pd.read_csv(os.path.join(folder_root, 'justifications_long_training_w_transcriptions.csv'))

# Upload dates coded file
df_dates = pd.read_csv(os.path.join(subfolder_root, "dates_long_parsed.csv"))
df_dates['date_coded'] = df_dates['date']
df_dates['img_file_orig'] = df_dates['img_file']
df_dates = df_dates[['date_coded', 'img_file', 'img_file_orig']]

# Upload date range (based on file archive record) file
df_date_range = pd.read_csv(os.path.join(subfolder_root, "date_range.csv"), dtype=str)
df_date_range.columns
df_date_range["file_id"] = df_date_range["file1"] + "_" + df_date_range["file2"] + "_" + df_date_range["file3"]
df_date_range = df_date_range[['file_start_year', 'file_start_month', 'file_end_year', 'file_end_month', 'file_id']]

df_date_range.shape
df_dates.shape
df_just.shape

# Merge justification and date ranges
df_merge = pd.DataFrame.merge(df_just, df_date_range, 'left', on='file_id')

# Merge justification/date ranges w date codes
df_merge_final = pd.DataFrame.merge(df_merge, df_dates, 'left', on="img_file_orig")
df_merge_final.shape
df_merge_final.columns

# output csv
#df_merge_final = df_merge_final[['img_file', 'img_file_orig', 'file_id_orig', 'file_id', 'image_id', 'date', 'start_yr', 'start_mo', 'end_year', 'end_mo', 'justification', 'raw_text', 'just_plain_text_all_ref', 'just_text_all_ref', 'ref_count', 'coded_refs']]

df_merge_final.to_csv(os.path.join(folder_root, 'justifications_complete.csv'))