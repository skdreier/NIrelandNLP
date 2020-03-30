# NIreland_NLP
## British Justifications for Internment without Trial: NLP Approaches to Analyzing Government Archives. 

**[2020 Incubator project](https://escience.washington.edu/winter-2020-incubator-projects/) with the [eScience Institute](https://escience.washington.edu/)**

**Project Lead:** [Sarah Dreier](https://escience.washington.edu/people/sarah-k-dreier) ([email](skdreier@uw.edu))

**eScience Liaison:** [Jose Hernandez](https://escience.washington.edu/people/jose-hernandez/)

**Repo Updated:** March 2020

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

  + [baseline_neural_network.py](https://github.com/skdreier/NIreland_NLP/tree/master/class_nn/baseline_neural_network.py): Builds a shallow, one-layer neural network. This uses the justification training data; takes sentences as inputs and tries to train a model. Uses stemmed vocabulary (tokenized) but no word-to-vector embeddings (just a few hidden nodes trying to train a classifier).
  
  + [embeddings_google.py](https://github.com/skdreier/NIreland_NLP/tree/master/class_nn/embeddings_google.py): Builds NN using google embeddings and parameters. This has two NN models: 1) One with an embedded layer but trained based on our own vocabulary (uses an embedded layer as a layer); 2) One which uses google word2vec embeddings as an added layer in the NN model (uses an embedded layer populated by the google word2vec model). 

  + [build_archive_corpus_w2v_model.py](https://github.com/skdreier/NIreland_NLP/tree/master/class_nn/build_archive_corpus_w2v_model.py): Builds Word vector model based on our archive data ([archive_corpus_w2v_model.bin](https://github.com/skdreier/NIreland_NLP/tree/master/class_nn/archive_corpus_w2v_model.bin)). **Action:** Save visualized output to GitHub, address/improve/jettison model cleaning tactics, ***add as an embedded layer to NN.***
  
  + [pretrained_models_init.py](https://github.com/skdreier/NIreland_NLP/tree/master/class_nn/pretrained_models_init.py): Uploads pretrained word vector models (e.g., Stanford GloVe model, Google Word2Vec model). 

___

### Cleaning, Preliminary Scripts, and Miscellaneous Tasks (file-management branch)

- [cleaning](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning): Uploads, appends, cleans Nvivo data ([orig_text_data](https://github.com/skdreier/NIreland_NLP/tree/master/orig_text_data/)).

  + [justifications_compile.py](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/justifications_compile.py): Prepares justification code files from Nvivo for text analysis: Appends each of 12 justification.txt files (from Nvivo, [orig_text_data/just_0106](https://github.com/skdreier/NIreland_NLP/tree/master/orig_text_data/just_0106)), parses text into relevant components, fixes image naming issue, saves justification text corpus ([justifications_long_training.csv](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/justifications_long_training.csv)), creates file of Nvivo page captures (rather than text codes) for hand transcription ([page_ref.csv](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/page_ref.csv)). **Actions:** Clean script.
  
  + [function_clean_text.py](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/function_clean_text.py): Builds a function to clean text, one-hot-encode justifications, and save text for main analysis [justifications_clean_text_ohe.csv](https://github.com/skdreier/NIreland_NLP/tree/master/justifications_clean_text_ohe.csv). **Action:** Clean script.

  + [date_files](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/date_files): [date_compile.py](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/date_files/date_compile.py): Appends each of 48 date code .txt files and cleans text; creates [dates_long_parsed.csv](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/date_files/dates_long_parsed.csv). [merge_codes.py](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/date_files/merge_codes.py) merges justification and data codes ([justifications_dates_long_parsed.csv](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/date_files/justifications_dates_long_parsed.csv)). None of these text files are currently used in analysis. [date_range.csv](https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/date_files/date_range.csv) represents date range for all archive files that contain an "internment" code; document developed by hand. 

- [histograms](): Plots frequencies for justification categories.
  
- [misc_tasks]():

  + [random_docs_error_check.py](): Pulls a random sample of .txt files (from NI_docs) and counts characters appearing on each document. Used for error check task (completed 03/2020). Creates: [random_docs_error_check.csv]().

- [orig_text_data](https://github.com/skdreier/NIreland_NLP/tree/master/orig_text_data/):

  + [dates_0123](https://github.com/skdreier/NIreland_NLP/tree/master/orig_text_data/dates_0123): Contains Nvivo output text for each specific "date" code in Nviov (codes as of Jan 2020). **Action:** Update files fron Nvivo after coding is updated, rename file, update scripts (**Used by:** [date_compile.py]((https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/date_compile.py))).
  
  + [just_0106](https://github.com/skdreier/NIreland_NLP/tree/master/orig_text_data/just_0106): Contains Nvivo output text for each specific "justification" code in Nvivo (codes as of Jan 2020). **Action:** Update files fron Nvivo after coding is updated, rename file, update scripts (**Used by:** [justifications_compile.py]((https://github.com/skdreier/NIreland_NLP/tree/master/cleaning/justifications_compile.py))).
  
  + [sample_data](https://github.com/skdreier/NIreland_NLP/tree/master/orig_text_data/sample_data): Contains 7 sample .txt files from the original document PDFs. 
  
  + [internment.txt](https://github.com/skdreier/NIreland_NLP/tree/master/orig_text_data/internment.txt): Nvivo output text for "Internment" code (first round -- coded at the document level)
  
  + [terrorism.txt](https://github.com/skdreier/NIreland_NLP/tree/master/orig_text_data/terrorism.txt): Nvivo output text for "Internment" code (first round -- coded at the document level)

- [old_files](https://github.com/skdreier/NIreland_NLP/tree/master/old_files): Old and/or exploratory scripts and text data corpora.
