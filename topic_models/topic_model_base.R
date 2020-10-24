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
  anti_join(stop_words) %>%
  add_count(word) %>%
  filter(n > 50) %>%
  select(-n)

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

# use ni_tm
ni_tm_sparse <- ni_tm %>%
  count(cat, word) %>%
  cast_sparse(cat, word, n)

# Code adapted from Julia's fabulous blog: https://juliasilge.com/blog/evaluating-stm/ 
library(furrr)
plan(multiprocess)

# create a number of different models to run
many_models <- data_frame(K = c(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)) %>%
  mutate(topic_model = future_map(K, ~stm(ni_tm_sparse, K = .,
                                          verbose = FALSE)))

heldout <- make.heldout(ni_tm_sparse)

k_result <- many_models %>%
  mutate(exclusivity = map(topic_model, exclusivity),
         semantic_coherence = map(topic_model, semanticCoherence, ni_tm_sparse),
         eval_heldout = map(topic_model, eval.heldout, heldout$missing),
         residual = map(topic_model, checkResiduals, ni_tm_sparse),
         bound =  map_dbl(topic_model, function(x) max(x$convergence$bound)),
         lfact = map_dbl(topic_model, function(x) lfactorial(x$settings$dim$K)),
         lbound = bound + lfact,
         iterations = map_dbl(topic_model, function(x) length(x$convergence$bound)))

k_result

# Diagnostic plots
k_result %>%
  transmute(K,
            `Lower bound` = lbound,
            Residuals = map_dbl(residual, "dispersion"),
            `Semantic coherence` = map_dbl(semantic_coherence, mean),
            `Held-out likelihood` = map_dbl(eval_heldout, "expected.heldout")) %>%
  gather(Metric, Value, -K) %>%
  ggplot(aes(K, Value, color = Metric)) +
  geom_line(size = 1.5, alpha = 0.7, show.legend = FALSE) +
  facet_wrap(~Metric, scales = "free_y") +
  labs(x = "K (number of topics)",
       y = NULL,
       title = "Model diagnostics by number of topics",
       subtitle = "These diagnostics indicatre 10 to be a good number")

# Held-out likelihood is highest between 8 and 10 topics and the residuals are lowest around 10.
# I say 10 is a good number for K
# Semantic coherence is maximized when the most probable words in a given topic 
# frequently co-occur together, and it’s a metric that correlates well with human judgment of topic quality.
# see https://dl.acm.org/doi/10.5555/2145432.2145462

# looking at the k=10 model

topic_model <- k_result %>% 
  filter(K == 10) %>% 
  pull(topic_model) %>% 
  .[[1]]

topic_model

# attach beta matrix probability that each word is generated from each topic
td_beta <- tidy(topic_model)

td_beta

# attach gamma matrix or probability that each "document" 
# is generated from each topic (might not make sense for 
# classification not actual documents)
td_gamma <- tidy(topic_model, matrix = "gamma",
                 document_names = rownames(ni_tm_sparse))

td_gamma

# combine these and plot 
library(ggthemes)

top_terms <- td_beta %>%
  arrange(beta) %>%
  group_by(topic) %>%
  top_n(7, beta) %>%
  arrange(-beta) %>%
  select(topic, term) %>%
  summarise(terms = list(term)) %>%
  mutate(terms = map(terms, paste, collapse = ", ")) %>% 
  unnest()

gamma_terms <- td_gamma %>%
  group_by(topic) %>%
  summarise(gamma = mean(gamma)) %>%
  arrange(desc(gamma)) %>%
  left_join(top_terms, by = "topic") %>%
  mutate(topic = paste0("Topic ", topic),
         topic = reorder(topic, gamma))

gamma_terms %>%
  ggplot(aes(topic, gamma, label = terms, fill = topic)) +
  geom_col(show.legend = FALSE) +
  geom_text(hjust = 0, nudge_y = 0.0005, size = 2) +
  coord_flip() +
  scale_y_continuous(expand = c(0,0),
                     limits = c(0, 0.23),
                     labels = percent_format()) +
  theme_tufte(ticks = FALSE) +
  theme(plot.title = element_text(size = 16),
        plot.subtitle = element_text(size = 13)) +
  labs(x = NULL, y = expression(gamma),
       title = "10 topics by prevalence in the NI Sentence Corpus",
       subtitle = "With the top words that contribute to each topic")

## look at a table
gamma_terms %>%
  select(topic, gamma, terms) 
