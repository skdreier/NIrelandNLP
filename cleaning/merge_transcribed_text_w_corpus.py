
import os
from pathlib import Path
import re
import pandas as pd
import numpy as np

this_file_path = os.path.abspath(__file__)
folder_root = os.path.split(this_file_path)[0]
folder_path = os.path.join(folder_root) + '/'

repo_root = os.path.split(folder_root)[0]
repo_path = os.path.join(repo_root)

df_trans = pd.read_csv(os.path.join(folder_path, "page_ref_transcribed_04-2020.csv"))
df_just = pd.read_csv(os.path.join(folder_path, "justifications_long_training.csv"))

df_trans.columns
df_just.columns

df_just['occurances'] = df_just['level_4']
df_trans['text'] = df_trans['transcribed_text']

df_just_merge = df_just[['image_id', 'file_id', 'img_file_orig', 'justification', 'occurances', 'text']]
df_trans_merge = df_trans[['image_id', 'file_id', 'img_file_orig', 'justification', 'occurances', 'text']]

all_just = pd.concat([df_just_merge, df_trans_merge])

#### Cross-reference number of justification codes w Nvivo record:
all_just.shape #2277 rows
df_just.shape #2063 rows 
df_trans.shape #214 rows 
176+77+327+36+48+23+101+338+377+148+351+60 #Nvivo number of codes / justification: 2062 unique codes

### NOTES:
### Resulting DF has 2277 lines. This includes:
# 1849 text-coded justifications
# 214 place holders for the screenshot refs to be transcribed (these will be omitted later) 
# 214 the transcribed references
### Result: 2063 lines, 2062 unique coded justifications; this aligns w Nvivo record (2062)
### CONCLUSION: THIS WORKED; NOTHING APPEARS TO HAVE BEEN DROPPED OR DOUBLE-COUNTED

all_just.head

# Save it as a csv
all_just.to_csv(os.path.join(folder_root, 'justifications_long_training_w_transcriptions.csv'))
