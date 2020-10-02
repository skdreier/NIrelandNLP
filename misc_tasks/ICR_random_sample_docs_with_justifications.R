## Random select docs for IRC among docs that contain a justification
## Oct 2, 2020


test <- read.csv("OneDrive/Incubator/NIreland_NLP/justifications_clean_text_ohe.csv", 
                 header=T, stringsAsFactors = F)

head(test)
dim(test)
colnames(test)

unique_docs <- unique(test$img_file_orig)
length(unique_docs)
x <- sample(unique_docs, 10, replace = F)

sort(x)

write.csv(x, "OneDrive/Incubator/NIreland_NLP/misc_tasks/random_sample_justification_ICR_among_docs_w_justifications.csv")
