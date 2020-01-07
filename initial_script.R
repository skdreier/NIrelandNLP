rm(list=ls())

library(readr)

if(jose == T) {
path <- "/Users/josehernandez/Documents/eScience/projects/NIreland_NLP/justifications_01-06/justifications_txt/"
text_fn <- "J_Denial.txt"
} else {
  
}

mystring <- read_file("/Users/josehernandez/Documents/eScience/projects/NIreland_NLP/justifications_01-06/justifications_txt/J_Denial.txt")

mystring
library(tidyverse)
library(stringr)

string <- "Files\\\\DNV_round1\\\\DEFE 13 1358\\\\IMG_9711_DEFE_13_1358"
 
values = str_extract(string, "\\d+(?=_[a-zA-Z]+.+$)") # this is an example we need to find our genex...
