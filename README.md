# NIreland_NLP
Ongoing repo for NLP analysis of N.Ireland archive text (UW eScience Incubator Project)

## File Summaries (prelim-analysis branch)

### Folders:
- [biclass_logreg](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/biclass_logreg "biclass_logreg"): Contains plots for each of justification (12) of the 40 words with the largest coefficients (absolute value) for each justification category. **Action:** Link to python script

- [data](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/data "sample data"): Contains 7 sample .txt files from the original document PDFs. **Action:** Rename as "sample_data"
 
 - [dates_0123](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/dates_0123 "dates_0123"): Contains Nvivo output text for each specific "date" code in Nviov (codes as of Jan 2020). **Action:** Update files fron Nvivo after coding is updated, rename file, update scripts.
 
 - [just_0106](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/just_0106 "just_0106"): Contains Nvivo output text for each specific "justification" code in Nvivo (codes as of Jan 2020). **Action:** Update files fron Nvivo after coding is updated, rename file, update scripts (**Used by:** justifications_compile.py).
 
- [multiclass_LR](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/multiclass_LR "multiclass_LR"): Distribution plots for justification categories (6, 7) and performance (accuracy) for each of several regression/ML models. Scripts: [grid_search.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/grid_search.py) **Action:** Move justification histograms in unique folder.
   
- [multiclass_NB](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/multiclass_NB "multiclass_NB"): Multi-class Naive Bayes confusion matrices. **Action:** Link to python script, draft description. Consider doing this for LR (not just NB).

- [old_docs](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/old_docs "old_docs"): Files/scripts no longer being used.  


### Files (python scripts):

- [baseline_neural_network.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/baseline_neural_network.py): Builds a shallow, one-layer NN. **Uses:** [justifications_clean_text_ohe.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_clean_text_ohe.csv). **Action:** Talk through code w Jose. What is the difference between this and "embeddings_google.py"?

- [date_compile.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/date_compile.py): Appends each of 48 date code .txt files and cleans text. **Uses:** [dates_0123](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/dates_0123). **Action:** Integrated with justifications_compile.py; DELETE SCRIPT.

- [embeddings_google.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/embeddings_google.py): Builds NN using google embeddings and parameters. **Uses:** [justifications_clean_text_ohe.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_clean_text_ohe.csv). **Action:** What is the difference between this and "baseline_neural_network.py"?

- [function_clean_text.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/function_clean_text.py): Builds a function to clean text and one-hot-encode justifications. **Uses:** [justifications_long_training.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_long_training.csv). **Creates:** [justifications_clean_text_ohe.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_clean_text_ohe.csv) **Action:** Clean script.

- [gensim_example.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/gensim_example.py): Builds Word vector model based on our archive data. **Uses:** NI_docs, [preprocessing.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/preprocessing.py). **Creates:** [archive_corpus_w2v_model.bin](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/archive_corpus_w2v_model.bin). **Action:** Move NI_docs to Github (?), save visualized output to new GitHub folder, talk through "cleaning" tactics with Jose, add as an embedded layer to NN.

- [grid_search.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/grid_search.py): Examines various multi-class models and parameters for predicting classification (includes: histograms of distribution of categories by training/testing data, plots accuracy for various model approaches, confusion matrices, grid search to output best performing model/parameters, most "important" unigram/bigrams for each justification category. **Uses:** [justifications_clean_text_ohe.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_clean_text_ohe.csv). **Creates:** Histograms for justification categories and box plots to show which model (Random forest, SVC, Multinom NB, LR) performs best (in [multiclass_LR](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/multiclass_LR "multiclass_LR")). **Actions: Clean script**

- [justifications_compile.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_compile.py): Prepares justification code files from Nvivo for text analysis. Specifically: Appends each of 12 justification.txt files (from Nvivo), parses text into relevant components, fixes image naming issue, creates file of Nvivo page captures (rather than text codes) for hand transcription. **Uses:** [just_0106](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/just_0106) and [preprocess.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/preprocess.py). **Creates:** [justifications_long_training.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_long_training.csv), [page_ref.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/page_ref.csv). **Actions: Clean script**

- [merge_codes.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/merge_codes.py): Code merges justification and data codes. **Uses:** [justifications_long_parsed.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_long_parsed.csv), [dates_long_parsed.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/dates_long_parsed.csv). **Creates:** [justifications_dates_long_parsed.csv](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/justifications_dates_long_parsed.csv). **Actions:** None of these text files are used in analysis. If we want to have date included in text, we should add date to the "justifications_long_training.csv". *Otherwise, this code and its outputs are obsolete.*

- [mlc_plots.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/mlc_plots.py): **Uses:** **Creates:** **Actions:**

- [multiclass.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/multiclass.py): **Uses:** **Creates:** **Actions:**

- [nn_start.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/nn_start.py): **Uses:** **Creates:** **Actions:**

- [preprocess.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/preprocess.py): **Uses:** **Creates:** **Actions:**

   Used by: gensim_example.py, justifications_compile.py

- [pretrained_models_init.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/pretrained_models_init.py):
- [random_docs_error_check.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/random_docs_error_check.py):
- [text_classification_intro.py](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/text_classification_intro.py):

### Files (data and models):

- archive_corpus_w2v_model.bin

   Created by: gensim_example.py

- date_range.csv

- dates_long_parsed.csv: Perhaps obsolete (created by XXX, used by merge_codes.py)

- justifications_clean_text_ohe.csv: 

   Created by: function_clean_text.py
   Used by: baseline_neural_network.py, embeddings_google.py, grid_search.py
   
- justifications_dates_long_parsed.csv: Perhaps obsolete (created by merge_codes.py)

- justifications_long_parsed.csv: Perhaps obsolete (originally created by justifications_compile.py, used by merge_codes.py)

- justifications_long_training.csv

  Created by: justifications_compile.py
  Used by: function_clean_text.py

- page_ref.csv: Page captures (rather than text captures -- for hand transcription).

  Created by: justifications_compile.py

- random_docs_error_check.csv

   **Action:** None.

- [ ](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/FILE ""):  

   **Action:** .

- [ ](https://github.com/skdreier/NIreland_NLP/tree/prelim-analysis/FILE ""):  

   **Action:** .


