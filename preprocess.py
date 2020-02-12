import os
from pathlib import Path
import re
import pandas as pd

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]
c_test_path = os.path.join(project_root, 'data') + '/'

class text_preprocess:
    """
    Parameters
    -----------
    f_path: string
        path to your data directory
        Assumes all files are in a single directory in .txt format 
        and follow the typical Nvivo file structure.
    Initialize values for classes

    """
    # Initialize
    def __init__(self, f_path):
        self.files = []
        self.text = []
        self.cat = []
        self.doc_text = {}
        for r, d, f in os.walk(f_path):
            for file in f:
                if '.txt' in file:
                    self.files.append(os.path.join(r, file))
    # single files with clumped nvivo codes
    def nvivo_clumps(self):
        """"
        Parse nvivo coding code and text txt file into dict of category and accompanied text and file information.
        It uses the nvivo file outputs where each txt file represents all the codings and source text in a particular category
        
        Returns
        ----------
        It will output a dictionary where the keys correspond to a specific category and accompanying text.   
        
        """"
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
        """"
        Load nvivo OCRd txt files into dict where key is the imgage or pdf id.
        To be used in the case where you need the overall document of the individual 
        coded sentences.

        Parameters
        ----------
        img_id: list
        List of strings of a subset of documents you need to upload

        Returns
        ----------
        dictionary with keys corresponding to the file ID and values the text corpus for that document

        """"
        if img_id is not None:
            if not isinstance(img_id, list):
                raise ValueError('img_id must be a list of strings')
            for f in self.files:
                for keyword in img_id:
                    if keyword in f:
                        docs = open(f, "r")
                        text = docs.read()
                        docs.close()
                        self.doc_text.update({keyword: text})
            return self.doc_text
        else:
            for f in self.files:
                docs = open(f, "r")
                doc_text = docs.read()
                docs.close()
                cat_code = Path(f).name 
                self.cat.append(re.sub('.txt', '', cat_code))
                self.text.append(doc_text)
        self.doc_text = dict(zip(self.cat, self.text))
        return self.doc_text
