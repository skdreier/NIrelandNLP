
import os
from pathlib import Path
import re

test_file = '/Users/josehernandez/Documents/eScience/projects/NIreland_NLP/justifications_01-06/justifications_txt/J_Denial.txt'

f = open(test_file, "r")
text = f.read()
f.close()

text

path = '/Users/josehernandez/Documents/eScience/projects/NIreland_NLP/'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))
            
for f in files:
    print(f)
    

files[1]
name = Path(files[1]).name 
# two different ways  
name.replace('.txt', '')
re.sub('.txt', '', name)


for f in files:
    print(f)

