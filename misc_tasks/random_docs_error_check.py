#########################################
## Pull 10 random docs for error check ##
#########################################

import os
from pathlib import Path
import re
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocess as pr

## Set project path 
this_file_path = os.path.abspath(__file__)
folder_root = os.path.split(this_file_path)[0]
repo_root = os.path.split(folder_root)[0]
all_path = os.path.join(repo_root, "orig_text_data/NI_docs/") 

#sys.path # make sure the repo is in the path 

## Load txt documents (all)
raw = pr.text_preprocess(all_path)
raw.files

## Randomly draw 10 documents
random_draw = np.random.choice(raw.files, 10)

## Loop through 10 documents to count characters (not including spaces)
words2={}
characters2={}
characters3=[]

for f in random_draw:
        # Extract text and parse into df
    with open(f) as text:
        words = 0
        characters = 0
        file_name = Path(f).name 
        for lineno, line in enumerate(text, 1):  
            wordslist = line.split()
            words += len(wordslist)
            characters += sum(len(word) for word in wordslist)
        words2.update({file_name: words})
        characters2.update({file_name: characters})
        characters3.append(characters)

print(words2)
print(characters2)
print(characters3)
sum(characters3)

## Build character count as dataframe
random_df = pd.DataFrame.from_dict(characters2, orient='index')
random_df.reset_index(level=0, inplace=True)
random_df = random_df.rename(columns = {'index':'document', 0:"char_count"})
np.sort(random_df['document'])

## Output random count csv
random_df.to_csv(os.path.join(folder_root, 'random_docs_error_check.csv'))