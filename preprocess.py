import os
from pathlib import Path
import re
import pandas as pd

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]
c_test_path = os.path.join(project_root, 'data') + '/'

# Steps
# 1. Load all file paths
# 2. Filter option for what is found in the training file 
# 3. parse filename from path and raw text into dict
# individual 
## Justifications + DATE  
# 1. GET CATEGORY + RAW TEXT FROM PARSED CATEGORY
## CORPUS
# 1. LOAD RAW TEXT + DOCUMENT ID FROM FILE NAME INTO OUTPUT 

class text_preprocess:
    """
    Parameters:
    """
    # Initialize
    def __init__(self, f_path):
        self.files = []
        self.text = []
        self.cat = []
        for r, d, f in os.walk(f_path):
            for file in f:
                if '.txt' in file:
                    self.files.append(os.path.join(r, file))
    # single files with clumped nvivo codes
    def nvivo_clumps(self):
        for f in self.files:
            docs = open(f, "r")
            text = docs.read()
            docs.close()
            text = re.split(r'.*(?=Files)', text) 
            # check to see if this makes sense for the date codes
            cat_code = Path(f).name 
            self.cat.append(re.sub('.txt', '', cat_code))
            self.text.append(list(filter(None, text)))
        return dict(zip(self.cat, self.text))
    # Individual files
    # create dict with document id and corresponding text 
    def nvivo_ocr(self, img_id = None):
        for f in self.files: 
            print(f)


for img_id in all_files: # get all files 
if op.exists(img_id):






# create dict that we can use to create a df
cat_text = dict(zip(cat, p_text))

parse = os.path.join(project_root, 'data/')

test = text_preprocess(f_path = parse)

test.nvivo_clumps() 

test.nvivo_ocr()

path = os.path.join(project_root, 'data/')

# Test The whole doc NEW
keywords = ['IMG_1247_DEFE_24_876', 'IMG_1246_DEFE_24_876', 'PREM_15_1010_022']
result = {}  # dict store our results

for filename in os.listdir(path):
    for keyword in keywords:
        if keyword in filename:
            docs = open(os.path.join(path, filename), "r")
            text = docs.read()
            docs.close()
            result[keyword] = text

            return(result)
result
#this works next incorporate


# create method to extract relevant text and appropriate categories from file name
for f in files:
    if "Nodes" not in f:
        print(f)
        # Extract text and parse into df
        docs = open(f, "r")
        text = docs.read()
        docs.close()
        text = re.split(r'.*(?=Files)', text)
        # get the file name 
        cat_code = Path(f).name 
        cat.append(re.sub('.txt', '', cat_code))
        p_text.append(list(filter(None, text)))
# create dict that we can use to create a df
cat_text = dict(zip(cat, p_text))

for key, value in cat_text.items() :
    print (key)