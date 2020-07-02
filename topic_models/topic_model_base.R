library(tidyverse)
library(tidytext)
library(quanteda)
library(stm)

home <- '/Users/josehernandez/Documents/eScience/projects/NIreland_NLP' 
data_loc <- paste0(home, '/justifications_clean_text_ohe.csv')
  
ni_df <- read.csv(file = data_loc, stringsAsFactors = F)
names(ni_df)

ni_df %>% 
  group_by(justification_cat) %>%
  summarise(n = n())

# using the category as the "document" not necessarily needed for topic modeling but good for exploration
# You might also just try to use the actual document when doing it on the larger corpus by document 
ni_tm <- ni_df %>%
  select(clean_text,justification_cat) %>%
  mutate(cat = factor(justification_cat, levels = unique(justification_cat))) %>%
  mutate(line = row_number()) %>%
  unnest_tokens(word, clean_text) %>%
  anti_join(stop_words)

# look at most common words (no surprise)
ni_tm %>%
  count(word, sort = T)

# exploration of tf-idf leveraging the category as "document"
ni_tf_idf <- ni_tm %>%
  count(cat, word, sort = T) %>%
  bind_tf_idf(word, cat, n) %>%
  arrange(-tf_idf) %>%
  group_by(cat) %>%
  top_n(10) %>%
  ungroup

ni_tf_idf %>%
  mutate(word = reorder_within(word, tf_idf, cat)) %>%
  ggplot(aes(word, tf_idf, fill= cat)) +
  geom_col(alpha = .8, show.legend = F) +
  facet_wrap(~ cat, scales = "free", ncol = 3) +
  scale_x_reordered() +
  coord_flip()

# data prep for model
ni_dtm <- ni_tm %>%
  count(cat, word, sort = T) %>%
  cast_dfm(cat, word, n)
  
ni_topic_model <- stm(ni_dtm, K = 10, 
                      verbose = FALSE, init.type = "Spectral")

###
summary(ni_topic_model)

# How to select K 
# iteration! and Coherent Score + exclusivity 
# you will need to read up on the best approach currently being used 
# see selectModel() in documentation and also searchK
