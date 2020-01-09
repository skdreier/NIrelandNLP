rm(list=ls())

library(readr)
library(tidyverse)
library(stringr)

jose <- F

if(jose == T) {
path <- "/Users/josehernandez/Documents/eScience/projects/NIreland_NLP/justifications_01-06/justifications_txt/"
text_fn <- "J_Denial.txt"
} else { path <- "/Users/sarahdreier/OneDrive/Academic/repositories/NIreland_NLP/justifications_01-06/justifications_txt/"
text_fn <- "J_Denial.txt"
}

setwd("/Users/sarahdreier/OneDrive/Academic/repositories/NIreland_NLP/justifications_01-06/justifications_txt/")

mystring <- read_file(text_fn)

mystring

#string <- "Files\\\\DNV_round1\\\\DEFE 13 1358\\\\IMG_9711_DEFE_13_1358 - § 1 reference coded [ 4.20% Coverage]\nReference 1 - 4.20% Coverage\nInformation about the tribunal should emphasise the steps that will be \ntaken to separate the decision to detain a person from the executive.\nFiles\\\\DNV_round1\\\\PREM 15 1002\\\\IMG_6736_PREM_15_1002 - § 1 reference coded [ 3.46% Coverage]\nReference 1 - 3.46% Coverage\nMcAuley’s father, who is an internee at Long esh, was allowed out on parole for the funeral he had not reported back by 4 p.m.\nFiles\\\\DNV_round1\\\\PREM 15 1002\\\\IMG_6773_PREM_15_1002 - § 1 reference coded [ 5.23% Coverage]\nReference 1 - 5.23% Coverage\nTHE DISCUSSION IN JSC MOVED ON TO A SUGGESTION BY BLOOMFIELD THAT THERE SHOULD BE A PUBLIC STATEMENT WHICH WOULD BE DESIGNED TO CLEAR UP SOME MISCONCEPTIONS ABOUT INTERNMENT EG THE LEGAL BASIS FOR IT AND THE RIGHT OF INTERNEES TO SECURE LEGAL ADVICE IN PRESENTING THEMSELVES TO JUDGE BROWN’S COMMITTEE.\n"
 
img_number = str_extract_all(mystring, "\\IMG_\\d{4}") # this is an example we need to find our regex... 
file_number = str_extract_all(mystring, "(PREM_|DEFE_|CJ_)\\w+" )
# \\ signals start of a reg expression command in R
# d{4}: indicates 4 digits together

### GOAL for/before Thursday: figure out how to ID the mis-named cases. From here, we separate the file and image name and paste back together

# Regular expression backwards lookup
# Conditional on specific words/aspects

x = str_extract_all(mystring, "(PREM_15_478|PREM_15_1689|PREM_15_1010)")
x = str_extract_all(mystring, "\\PREM_\\d{10}" )
x = str_extract_all(mystring, "w(PREM_)" )

test478 <- "Files\\\\SKD_round1\\\\PREM 15 1035\\\\IMG_5083_PREM_15_475 - § 1 reference coded [ 6.65% Coverage]\nReference 1 - 6.65% Coverage\nArrests \n- 342 men were arrested on August 9 in a highly successful \noperation in very difficult circumstances. No force but the minimum necessary was used to achieve arrest. The men concerned were likely to resist arrest and many of them were likely to be armed; in these circumstances the army’s performance was highly creditable.\nFiles\\\\SKD_round1\\\\PREM 15 478\\\\PREM_15_478_045 - § 1 reference coded [ 3.90% Coverage]\nReference 1 - 3.90% Coverage\nHE DID NOT BELIEVE THAT THE BRITISH GOVERNMENT WOULD PERMIT SITUATION IN WHICH THEIR ARMY WOULD NOT PROTECT PEOPLE WHOM THEY CALMED TO BE THEIR CITIZENS.\nFiles\\\\SKD_round1\\\\PREM 15 478\\\\PREM_15_478_050 - § 2 references coded [ 6.78% Coverage]\nReference 1 - 5.80% Coverage\nThis will be done only after a careful scrutiny of information furnished to me by the polic in respect of each such person, sufficient to convince me that the individual in question is a threat to present peace and maintenance of order.\nReference 2 - 0.98% Coverage\nAny such person will then have the right\n"

x <- str_extract_all(test478, "PREM_15_478_\\d{3}")
x <- str_extract_all(mystring, "(PREM_15_478_\\d{3})")

mystring
