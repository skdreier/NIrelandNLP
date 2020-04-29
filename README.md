# NIreland_NLP
## British Justifications for Internment without Trial: NLP Approaches to Analyzing Government Archives. 

**[2020 Incubator project](https://escience.washington.edu/winter-2020-incubator-projects/) with the [eScience Institute](https://escience.washington.edu/)**

**Project Lead:** [Sarah Dreier](https://escience.washington.edu/people/sarah-k-dreier) ([email](skdreier@uw.edu))

**eScience Liaison:** [Jose Hernandez](https://escience.washington.edu/people/jose-hernandez/)

**Repo Updated:** April 2020

*This project is funded by NSF Award [#1823547](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1823547&HistoricalAwards=false); Principal Investigators: [Emily Gade](http://emilykgade.com/), [Noah Smith](https://homes.cs.washington.edu/~nasmith/), and [Michael McCann](https://www.polisci.washington.edu/people/michael-w-mccann)*

___

### Analysis

- [preprocess.py](https://github.com/skdreier/NIreland_NLP/tree/master/preprocess.py): ***Functions to concatenate Nvivo outputs or original .txt data for analysis***
  + A) Parse nvivo coding code and text txt file into dictionary of category and accompanied text and file information. It uses the nvivo file outputs where each txt file represents all the codings and source text in a particular category.

  + B) Load Nvivo OCRd txt files into dict where key is the imgage or pdf id To be used in the case where you need the overall document of the individual coded sentences.
  
- [justifications_clean_text_ohe.csv](https://github.com/skdreier/NIreland_NLP/tree/master/justifications_clean_text_ohe.csv): ***Cleaned justification text data used in all analysis scripts*** 

- [class_biclass](https://github.com/skdreier/NIreland_NLP/tree/master/class_biclass): Binary classification analysis

  + [text_classification_intro.py](https://github.com/skdreier/NIreland_NLP/tree/master/class_biclass/text_classification_intro.py): Runs binary classifiers for each category. 

  + [biclass_logreg](https://github.com/skdreier/NIreland_NLP/tree/master/class_biclass/biclass_logreg): Plots words with the most weight in classifying each category.

- [class_multiclass](https://github.com/skdreier/NIreland_NLP/tree/master/class_multiclass/): Multi-class classification analysis

  + [grid_search.py](https://github.com/skdreier/NIreland_NLP/tree/master/class_multiclass/grid_search.py): Uses grid search to examine and identify best-performing multi-class models and parameters for predicting classification. Produces: [histograms](https://github.com/skdreier/NIreland_NLP/tree/master/histograms/) of distribution of categories by training/testing data, box plots demonstrating accuracy for various model approaches ([model_accuracy](https://github.com/skdreier/NIreland_NLP/tree/master/class_multiclass/model_accuracy)), confusion matrices, and most "important" unigram/bigrams for each justification category. 

  + [multiclass.py](https://github.com/skdreier/NIreland_NLP/tree/master/class_multiclass/multiclass.py): Initial development script for multi-class analysis. Produces multi-class Naive Bayes confusion matrices ([multiclass_NB](https://github.com/skdreier/NIreland_NLP/tree/master/class_multiclass/multiclass_NB/)). Code can be integrated into [grid_search.py](https://github.com/skdreier/NIreland_NLP/tree/master/class_multiclass/grid_search.py). Consider building confusion matrices for LR. 

- [class_nn](https://github.com/skdreier/NIreland_NLP/tree/master/class_nn/): Neural Network classification analysis

  + [baseline_neural_network.py](https://github.com/skdreier/NIreland_NLP/tree/master/class_nn/baseline_neural_network.py): Builds a shallow, one-layer neural network (with just a few hidden nodes). Takes justification sentences as inputs to train a classifier. Uses stemmed vocabulary (tokenized) but no word embeddings.
  
  + [embeddings_google.py](https://github.com/skdreier/NIreland_NLP/tree/master/class_nn/embeddings_google.py): Builds a NN that uses an embedded layer populated by the Google word2vec embeddings). 

  + [build_archive_corpus_w2v_model.py](https://github.com/skdreier/NIreland_NLP/tree/master/class_nn/build_archive_corpus_w2v_model.py): Builds a word vector model based on our archive data. [archive_corpus_embedding_w2v.txt](https://github.com/skdreier/NIreland_NLP/tree/master/class_nn/wordvec/archive_corpus_embedding_w2v.txt) is trained based on cleaned corpus (15.5k stemmed words); [archive_corpus_embedding_w2v_big.txt](https://github.com/skdreier/NIreland_NLP/tree/master/class_nn/wordvec/archive_corpus_embedding_w2v_big.txt) is trained based on uncleaned complete corpus (67.9k stemmed words). Also explores scatter-plot visualization of word embeddings.
  
  . ([archive_corpus_w2v_model.bin](https://github.com/skdreier/NIreland_NLP/tree/master/class_nn/archive_corpus_w2v_model.bin)). **Action:** Save visualized output to GitHub, address/improve/jettison model cleaning tactics, ***add as an embedded layer to NN.***
  
  + [pretrained_models_init.py](https://github.com/skdreier/NIreland_NLP/tree/master/class_nn/pretrained_models_init.py): Uploads pretrained word vector models (e.g., Stanford GloVe model, Google Word2Vec model). 

___

### Cleaning, Preliminary Scripts, and Miscellaneous Tasks

- [cleaning](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning): Uploads, appends, cleans Nvivo data ([orig_text_data](https://github.com/skdreier/NIreland_NLP/tree/master/orig_text_data/)).

  + [justifications_compile.py](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/justifications_compile.py): Prepares justification code files from Nvivo for text analysis. 1) Appends and prepares justifications from Nvivo ([orig_text_data/just_0404](https://github.com/skdreier/NIreland_NLP/tree/master/orig_text_data/just_0404)), parses text, addresses file image name issue, and creates file of Nvivo page captures (rather than text codes) for hand transcription ([page_ref.csv](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/page_ref.csv)). 2) Merges text justifications with hand-transcribed screenshot justifications (after hand-transcription is completed: [page_ref_transcribed_04-2020.csv](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/page_ref_transcribed_04-2020.csv)). 3) Merges justifications with [date_files](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/date_files). 4) Saves justification text corpus ([justifications_long_training.csv](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/justifications_long_training.csv)); this resulting .csv contains Nvivo text justifications, hand-transcribed justifications from Nvivo screenshot codes, file date range, and coded range (where possible).

  + [function_clean_text.py](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/function_clean_text.py): Builds a function to clean text, one-hot-encode justifications, and save text for main analysis [justifications_clean_text_ohe.csv](https://github.com/skdreier/NIreland_NLP/tree/master/justifications_clean_text_ohe.csv). **Action:** Clean script.

  + [date_files](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/date_files): [date_compile.py](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/date_files/date_compile.py): Appends each of 48 date code .txt files and cleans text; creates [dates_long_parsed.csv](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/date_files/dates_long_parsed.csv). [date_range.csv](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/date_files/date_range.csv) represents date range for all archive files that contain an "internment" code; document developed by hand. 

- [histograms](https://github.com/skdreier/NIreland_NLP/tree/master/histograms/): Plots frequencies for justification categories.
  
- [misc_tasks](https://github.com/skdreier/NIreland_NLP/tree/master/misc_tasks/):

  + [random_docs_error_check.py](https://github.com/skdreier/NIreland_NLP/tree/master/misc_tasks/random_docs_error_check.py): Pulls a random sample of .txt files (from NI_docs) and counts characters appearing on each document. Used for error check task (completed 03/2020). Creates: [random_docs_error_check.csv](https://github.com/skdreier/NIreland_NLP/tree/master/misc_tasks/random_docs_error_check.csv).

- [orig_text_data](https://github.com/skdreier/NIreland_NLP/tree/master/orig_text_data/):

  + [dates_0123](https://github.com/skdreier/NIreland_NLP/tree/master/orig_text_data/dates_0123): Contains Nvivo output text for each specific "date" code in Nviov (Jan 2020). **Used by:** [date_compile.py](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/date_compile.py).
  
  + [just_0404](https://github.com/skdreier/NIreland_NLP/tree/master/orig_text_data/just_0404): Contains Nvivo output text for each specific "justification" code in Nvivo (April 2020). **Used by:** [justifications_compile.py](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/justifications_compile.py).
 
  + [sample_data](https://github.com/skdreier/NIreland_NLP/tree/master/orig_text_data/sample_data): Contains 7 sample .txt files from the original document PDFs. 
  
  + [internment.txt](https://github.com/skdreier/NIreland_NLP/tree/master/orig_text_data/internment.txt): Nvivo output text for "Internment" code (first round -- coded at the document level)
  
  + [terrorism.txt](https://github.com/skdreier/NIreland_NLP/tree/master/orig_text_data/terrorism.txt): Nvivo output text for "Internment" code (first round -- coded at the document level)

- [old_files](https://github.com/skdreier/NIreland_NLP/tree/master/old_files): Old and/or exploratory scripts and text data corpora.
