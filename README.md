# NIreland_NLP
## British Justifications for Internment without Trial: NLP Approaches to Analyzing Government Archives. 

**[2020 Incubator project](https://escience.washington.edu/winter-2020-incubator-projects/) with the [eScience Institute](https://escience.washington.edu/)**

**Project Lead:** [Sarah Dreier](https://escience.washington.edu/people/sarah-k-dreier) ([email](skdreier@uw.edu))

**eScience Liaison:** [Jose Hernandez](https://escience.washington.edu/people/jose-hernandez/)

**Repo Updated:** March 2020

*This project is funded by NSF Award [#1823547](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1823547&HistoricalAwards=false); Principal Investigators: [Emily Gade](http://emilykgade.com/), [Noah Smith](https://homes.cs.washington.edu/~nasmith/), and [Michael McCann](https://www.polisci.washington.edu/people/michael-w-mccann)*

___

### Analysis

- [preprocess.py](): ***Functions to concatenate Nvivo outputs or original .txt data for analysis***
  + A) Parse nvivo coding code and text txt file into dictionary of category and accompanied text and file information. It uses the nvivo file outputs where each txt file represents all the codings and source text in a particular category.

  + B) Load Nvivo OCRd txt files into dict where key is the imgage or pdf id To be used in the case where you need the overall document of the individual coded sentences.
  
- [justifications_clean_text_ohe.csv](): ***Cleaned justification text data used in all analysis scripts*** 

- [class_biclass](): Binary classification analysis

  + [text_classification_intro.py](): Runs binary classifiers for each category. 

  + [biclass_logreg](): Plots words with the most weight in classifying each category.

- [class_multiclass](): Multi-class classification analysis

  + [grid_search.py](): Uses grid search to examine and identify best-performing multi-class models and parameters for predicting classification. Produces: [histograms]() of distribution of categories by training/testing data, box plots demonstrating accuracy for various model approaches ([model_accuracy]()), confusion matrices, and most "important" unigram/bigrams for each justification category. 

  + [multiclass.py](): Initial development script for multi-class analysis. Produces multi-class Naive Bayes confusion matrices ([multiclass_NB]()). Code can be integrated into [grid_search.py](). Consider building confusion matrices for LR. 

- [class_nn](): Neural Network classification analysis

  + [baseline_neural_network.py](): Builds a shallow, one-layer neural network. This uses the justification training data; takes sentences as inputs and tries to train a model. Uses stemmed vocabulary (tokenized) but no word-to-vector embeddings (just a few hidden nodes trying to train a classifier).
  
  + [embeddings_google.py](): Builds NN using google embeddings and parameters. This has two NN models: 1) One with an embedded layer but trained based on our own vocabulary (uses an embedded layer as a layer); 2) One which uses google word2vec embeddings as an added layer in the NN model (uses an embedded layer populated by the google word2vec model). 

  + [build_archive_corpus_w2v_model.py](): Builds Word vector model based on our archive data ([archive_corpus_w2v_model.bin]()). **Action:** Save visualized output to GitHub, address/improve/jettison model cleaning tactics, ***add as an embedded layer to NN.***
  
  + [pretrained_models_init.py](): Uploads pretrained word vector models (e.g., Stanford GloVe model, Google Word2Vec model). 

___

### Cleaning, Preliminary Scripts, and Miscellaneous Tasks (file-management branch)

- [cleaning](): Uploads, appends, cleans Nvivo data ([orig_text_data]().

  + [justifications_compile.py](): Prepares justification code files from Nvivo for text analysis: Appends each of 12 justification.txt files (from Nvivo, [orig_text_data/just_0106]()), parses text into relevant components, fixes image naming issue, saves justification text corpus ([justifications_long_training.csv]()), creates file of Nvivo page captures (rather than text codes) for hand transcription ([misc_tasks/page_ref.csv]()). **Actions:** Clean script.
  
  + [function_clean_text.py](): Builds a function to clean text, one-hot-encode justifications, and save text for main analysis [justifications_clean_text_ohe.csv](). **Action:** Clean script.

  + [date_files](): [date_compile.py](): Appends each of 48 date code .txt files and cleans text; creates [dates_long_parsed.csv](). [merge_codes.py]() merges justification and data codes ([justifications_dates_long_parsed.csv]()). None of these text files are currently used in analysis. [date_range.csv]() represents date range for all archive files that contain an "internment" code; document developed by hand. 

- [histograms](): Plots frequencies for justification categories.
  
## SKD must clean rest:
  
- [misc_tasks]():

  + [random_docs_error_check.py](): Pulls a random sample of .txt files (from NI_docs) and counts characters appearing on each document. Used for error check exercise (completed by Arica 03/2020). Creates: [random_docs_error_check.csv]().
  
  + [page_ref.csv](): Page captures (rather than text captures -- for hand transcription). 

- [orig_text_data]():

  + [dates_0123](): Contains Nvivo output text for each specific "date" code in Nviov (codes as of Jan 2020). **Used by:** [date_compile.py](). **Action:** Update files fron Nvivo after coding is updated, rename file, update scripts.
  
  + [just_0106](): Contains Nvivo output text for each specific "justification" code in Nvivo (codes as of Jan 2020). **Action:** Update files fron Nvivo after coding is updated, rename file, update scripts (**Used by:** justifications_compile.py).
  
  + [sample_data](): Contains 7 sample .txt files from the original document PDFs. 
  
  + [internment.txt]()
  
  + [terrorism.txt]()


- [archive](): Old and/or exploratory scripts and text data corpora.


# OLD README (from last week):

# NIreland_NLP
Ongoing repo for NLP analysis of N.Ireland archive text (UW eScience Incubator Project)

## File Summaries (prelim-analysis branch)

### Folders:
- [biclass_logreg](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/biclass_logreg "biclass_logreg"): Contains plots for each of justification (12) of the 40 words with the largest coefficients (absolute value) for each justification category. **Created by:** [text_classification_intro.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/text_classification_intro.py).

- [data](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/data "sample data"): Contains 7 sample .txt files from the original document PDFs. **Action:** Rename as "sample_data"
 
 - [dates_0123](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/dates_0123 "dates_0123"): Contains Nvivo output text for each specific "date" code in Nviov (codes as of Jan 2020). **Used by:** [date_compile.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/date_compile.py:). **Action:** Update files fron Nvivo after coding is updated, rename file, update scripts.
 
 - [just_0106](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/just_0106 "just_0106"): Contains Nvivo output text for each specific "justification" code in Nvivo (codes as of Jan 2020). **Action:** Update files fron Nvivo after coding is updated, rename file, update scripts (**Used by:** justifications_compile.py).
 
- [multiclass_LR](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/multiclass_LR "multiclass_LR"): Distribution plots for justification categories (6, 7) and performance (accuracy) for each of several regression/ML models. **Created by:** [grid_search.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/grid_search.py). **Action:** Move justification histograms in unique folder.
   
- [multiclass_NB](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/multiclass_NB "multiclass_NB"): Multi-class Naive Bayes confusion matrices. **Created by:** [multiclass.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/multiclass.py). **Action:** Draft description. Consider doing this for LR (not just NB), maybe integrate script into grid_search.py file.

- [old_docs](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/old_docs "old_docs"): Files/scripts no longer being used.  


### Files (python scripts):

- [baseline_neural_network.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/baseline_neural_network.py): Builds a shallow, one-layer NN. **Uses:** [justifications_clean_text_ohe.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_clean_text_ohe.csv). **Action:** Talk through code w Jose. What is the difference between this and "embeddings_google.py"? From Jose: Using the justification training data; take sentences as inputs and tries to train. Uses stemmed vocabulary (tokenized) but no word-to-vector embeddings. Just a few hidden nodes trying to train a classifier.

- [date_compile.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/date_compile.py): Appends each of 48 date code .txt files and cleans text. **Uses:** [dates_0123](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/dates_0123). **Creates:** [dates_long_parsed.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/dates_long_parsed.csv). **Action:** Integrated with justifications_compile.py; DELETE SCRIPT.

- [embeddings_google.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/embeddings_google.py): Builds NN using google embeddings and parameters. **Uses:** [justifications_clean_text_ohe.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_clean_text_ohe.csv). **Action:** What is the difference between this and "baseline_neural_network.py"? From Jose: This has two NN models. First: has an embedded layer but trained based on our own vocabulary (uses an embedded layer as a layer); Second: Uses google word2vec embeddings as an added layer in the NN model (uses an embedded layer populated by the google word2vec model). 

- [function_clean_text.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/function_clean_text.py): Builds a function to clean text and one-hot-encode justifications. **Uses:** [justifications_long_training.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_long_training.csv). **Creates:** [justifications_clean_text_ohe.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_clean_text_ohe.csv) **Action:** Clean script.

- [gensim_example.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/gensim_example.py): Builds Word vector model based on our archive data. **Uses:** NI_docs, [preprocessing.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/preprocessing.py). **Creates:** [archive_corpus_w2v_model.bin](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/archive_corpus_w2v_model.bin). **Action:** Move NI_docs to Github (?), save visualized output to new GitHub folder, talk through "cleaning" tactics with Jose, add as an embedded layer to NN.

- [grid_search.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/grid_search.py): Examines various multi-class models and parameters for predicting classification (includes: histograms of distribution of categories by training/testing data, plots accuracy for various model approaches, confusion matrices, grid search to output best performing model/parameters, most "important" unigram/bigrams for each justification category. **Uses:** [justifications_clean_text_ohe.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_clean_text_ohe.csv). **Creates:** Histograms for justification categories and box plots to show which model (Random forest, SVC, Multinom NB, LR) performs best (in [multiclass_LR](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/multiclass_LR "multiclass_LR")). **Actions: Clean script**

- [justifications_compile.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_compile.py): Prepares justification code files from Nvivo for text analysis. Specifically: Appends each of 12 justification.txt files (from Nvivo), parses text into relevant components, fixes image naming issue, creates file of Nvivo page captures (rather than text codes) for hand transcription. **Uses:** [just_0106](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/just_0106) and [preprocess.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/preprocess.py). **Creates:** [justifications_long_training.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_long_training.csv), [page_ref.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/page_ref.csv). **Actions: Clean script**

- [merge_codes.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/merge_codes.py): Code merges justification and data codes. **Uses:** [justifications_long_parsed.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_long_parsed.csv), [dates_long_parsed.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/dates_long_parsed.csv). **Creates:** [justifications_dates_long_parsed.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_dates_long_parsed.csv). **Actions:** None of these text files are used in analysis. If we want to have date included in text, we should add date to the "justifications_long_training.csv". *Otherwise, this code and its outputs are obsolete.*

- [mlc_plots.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/mlc_plots.py): Creates basic histogram of frequency for each justification category (plot not saved anywhere). **Uses:** [justifications_long_training.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_long_training.csv) **Actions:** Rename file and/or join this with justificaiton histograms created in grid_search.py. 

- [multiclass.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/multiclass.py): Looks like this is the first cut at a multi-class analysis. **Uses:** [justifications_clean_text_ohe.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_clean_text_ohe.csv). **Creates:** Confusion matrices (stored in [multiclass_NB](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/multiclass_NB)) **Actions:** Integrate this code into "grid_search.py" and delete this as a stand-alone script. 

- [nn_start.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/nn_start.py): This is old code (adapted from on online resource) that starts to build a NN; this is obsolete. **Actions:** MOVE TO "old_docs" folder.

- [preprocess.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/preprocess.py): Functions to: 
  + A) Parse nvivo coding code and text txt file into dict of category and accompanied text and file information. It uses the nvivo file outputs where each txt file represents all the codings and source text in a particular category.

  + B) Load Nvivo OCRd txt files into dict where key is the imgage or pdf id To be used in the case where you need the overall document of the individual coded sentences.

  **Used by:** [gensim_example.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/gensim_example.py), [justifications_compile.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_compile.py)

- [pretrained_models_init.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/pretrained_models_init.py): Code to use pretrained word vector models (e.g., Stanford GloVe model, Google Word2Vec model). Code also contains old code to train based on our archive corpus, but this code does not work; see [gensim_example.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/gensim_example.py) for working code. **Uses:** Glove and Word2Vec files from online. **Action:** Move old, defunct code to "old_docs" file. Save GloVe and Word2Vec original files onto Repo?

- [random_docs_error_check.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/random_docs_error_check.py): Pulls a random sample of .txt files (from NI_docs) and counts characters appearing on each document. Used for error check exercise (completed by Arica 03/2020). **Uses:** NI_docs (not currently on GitHub). **Creates:** [random_docs_error_check.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/random_docs_error_check.csv). **Action:** Refile under a "Misc" folder.

- [text_classification_intro.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/text_classification_intro.py): Runs binary classifiers for each category, plots the words that have the most weight in classifying categories (biclass_logreg). Can use code for all words or just stopwords. **Uses:** [justifications_clean_text_ohe.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_clean_text_ohe.csv). **Creates:** [biclass_logreg](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/biclass_logreg "biclass_logreg"). **Action:** Code can be cleaned/streamlined with a few functions. 


### Files (data and models):

- [archive_corpus_w2v_model.bin](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/archive_corpus_w2v_model.bin): Word2Vec model created using archive document. **Created by:** gensim_example.py.

- [date_range.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/date_range.csv): Date range for all archive files that contain an "internment" code (Round 1). Document developed by hand. **Action:** Move to "old_docs" or "Misc docs."

- [dates_long_parsed.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/dates_long_parsed.csv): All date codes in entire corpus (from Nvivo codes). *Perhaps obsolete.* **Created by:** date_compile.py, **Used by:** merge_codes.py. **Action:** PUT ALL DATE SCRIPTS/CSV FILES IN A SPECIFIC FOLDER.

- [justifications_clean_text_ohe.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_clean_text_ohe.csv): ***Used for all of the analysis scripts.*** **Created by:** function_clean_text.py. **Used by:** baseline_neural_network.py, embeddings_google.py, grid_search.py, multiclass.py, text_classification_intro.py. 
   
- [justifications_dates_long_parsed.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_dates_long_parsed.csv): **Perhaps obsolete** (created by merge_codes.py)

- [justifications_long_parsed.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_long_parsed.csv): **Perhaps obsolete** (originally created by justifications_compile.py, used by merge_codes.py).

- [justifications_long_training.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_long_training.csv): **Created by:** justifications_compile.py
  **Used by:** function_clean_text.py (to create justifications_clean_text_ohe.csv), mlc_plots.py

- [page_ref.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/page_ref_csv): Page captures (rather than text captures -- for hand transcription). **Created by:** justifications_compile.py. **Action:** Move to "Misc" folder.

- [random_docs_error_check.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis): For random error check RA tasks. **Created by:** random_docs_error_check.py. **Action:** Move to "Misc" folder.
