rm(list=ls())

library(readr)
library(tidyverse)
library(stringr)

jose <- F

if(jose == T) {
path <- "/Users/josehernandez/Documents/eScience/projects/NIreland_NLP/justifications_01-06/justifications_txt/"
text_fn <- "J_Denial.txt"
} else {setwd("/Users/sarahdreier/OneDrive/Academic/repositories/NIreland_NLP/justifications_01-06/justifications_txt/")
text_fn <- "J_Denial.txt"
}

mystring <- read_file(text_fn)

mystring

#string <- "Files\\\\DNV_round1\\\\DEFE 13 1358\\\\IMG_9711_DEFE_13_1358 - § 1 reference coded [ 4.20% Coverage]\nReference 1 - 4.20% Coverage\nInformation about the tribunal should emphasise the steps that will be \ntaken to separate the decision to detain a person from the executive.\nFiles\\\\DNV_round1\\\\PREM 15 1002\\\\IMG_6736_PREM_15_1002 - § 1 reference coded [ 3.46% Coverage]\nReference 1 - 3.46% Coverage\nMcAuley’s father, who is an internee at Long esh, was allowed out on parole for the funeral he had not reported back by 4 p.m.\nFiles\\\\DNV_round1\\\\PREM 15 1002\\\\IMG_6773_PREM_15_1002 - § 1 reference coded [ 5.23% Coverage]\nReference 1 - 5.23% Coverage\nTHE DISCUSSION IN JSC MOVED ON TO A SUGGESTION BY BLOOMFIELD THAT THERE SHOULD BE A PUBLIC STATEMENT WHICH WOULD BE DESIGNED TO CLEAR UP SOME MISCONCEPTIONS ABOUT INTERNMENT EG THE LEGAL BASIS FOR IT AND THE RIGHT OF INTERNEES TO SECURE LEGAL ADVICE IN PRESENTING THEMSELVES TO JUDGE BROWN’S COMMITTEE.\n"
 
img_number = str_extract_all(mystring, "\\IMG_\\d{4}") # this is an example we need to find our genex... 
file_number = str_extract_all(mystring, "(PREM_|DEFE_|CJ_)\\w+" )
# \\ signals start of a reg expression command in R
# d{4}: indicates 4 digits together

### GOAL for/before Thursday: figure out how to ID the mis-named cases. From here, we separate the file and image name and paste back together

# Regular expression backwards lookup
# Conditional on specific words/aspects

x = str_extract_all(mystring, "(PREM_15_478|PREM_15_1689|PREM_15_1010)")
x = str_extract_all(mystring, "\\PREM_\\d{7}")
x = str_extract_all(mystring, "\sPREM_")


mystring
