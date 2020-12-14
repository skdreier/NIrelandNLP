############################################
##### ICR: SKD and DNV (Nov/Dec 2020) ######
############################################

rm(list=ls())

##load needed packages
library(foreign)
#library(Hmisc)
library(irr)
library(tidyverse)
library(stringr)
library(tidytext)
library(dplyr)
library(tidyverse)

setwd("~/OneDrive/Incubator/NIreland_NLP/icr/")

#### CODING ROUND 0 (practice among docs containing a justification) ####

skd_dnv_practice <- read.csv(file="DNV_practice_justifications.csv", header=TRUE)
skd_dnv_practice[is.na(skd_dnv_practice)] <- 0
skd_dnv_practice$Round <- 0
skd_dnv_practice$Obs_number <- seq(1, dim(skd_dnv_practice)[1]) + 1000

#### CODING ROUND 1, first five (docs containing a justification) ####
dir()
skd_data <- read.csv(file="hand_coded_icr_skd.csv", header=TRUE)
skd_data[is.na(skd_data)] <- 0

table(skd_data$skd.J_Emergency_Policy)

#dnv_data <- read.csv(file="DNV_icr_first_5.csv", header=TRUE)
dnv_data <- read.csv(file="DNV_icr_first_10.csv", header=TRUE)
dnv_data[is.na(dnv_data)] <- 0

skd_dnv_r1 <- merge(skd_data, dnv_data, by="Obs_number") 

# Make sure all sentences are equal and check where wrong -- all ok #
with(skd_dnv_r1, all.equal(Sentence.x, Sentence.y))
skd_dnv_r1$Sentence.x == skd_dnv_r1$Sentence.y

skd_dnv_r1$Sentence.x[143]
skd_dnv_r1$Sentence.y[143]
skd_dnv_r1$Sentence.x[215]
skd_dnv_r1$Sentence.y[215]

names(skd_dnv_r1)[names(skd_dnv_r1) == "File_Img.x"] <- "File_Img"
names(skd_dnv_r1)[names(skd_dnv_r1) == "Sentence.x"] <- "Sentence"
names(skd_dnv_r1)[names(skd_dnv_r1) == "Round.x"] <- "Round"

skd_dnv_r1 = subset(skd_dnv_r1, select = - c(Date_coded, File_Img.y, Sentence.y, Round.y) )
skd_dnv_practice = subset(skd_dnv_practice, select = - DNV.Notes) 

# Subset to first 5 in R1

#skd_dnv_r1 <- subset(skd_dnv_r1, File_Img == "1507_DEFE_24_877" | File_Img == "6497_PREM_15_1000" |
#                     File_Img == "7508_PREM_15_1008" | File_Img == "7595_PREM_15_1008" | 
#                     File_Img == "2105_DEFE_13_919" 
#)

# Subset to first 10 in R1
skd_dnv_r1b <- subset(skd_dnv_r1, Round == 1)

# Just a coincidencde that both practice and Round 1 (first 5) each have 80 observations

skd_dnv_all <- rbind(skd_dnv_practice, skd_dnv_r1b)

# Subset to Practice and Round 1
skd_dnv <- skd_dnv_all
skd_dnv <- skd_dnv_r1b

#skd_dnv <- subset(skd_dnv, File_Img!="7508_PREM_15_1008") #test to see if we improve if we drop this one

kf_any <- kappam.fleiss(cbind(skd_dnv$skd.justification_any, skd_dnv$c2.justification_any))
k2_any <- kappa2(cbind(skd_dnv$skd.justification_any, skd_dnv$c2.justification_any))
any <- cbind("any", kf_any$value, k2_any$value)

kf_em <- kappam.fleiss(cbind(skd_dnv$skd.J_Emergency_Policy, skd_dnv$c2.J_Emergency_Policy))
k2_em <- kappa2(cbind(skd_dnv$skd.J_Emergency_Policy, skd_dnv$c2.J_Emergency_Policy))
em <- cbind("emergency", kf_em$value, k2_em$value)

kf_legal <- kappam.fleiss(cbind(skd_dnv$skd.J_Legal_Procedure, skd_dnv$c2.J_Legal_Procedure))
k2_legal <- kappa2(cbind(skd_dnv$skd.J_Legal_Procedure, skd_dnv$c2.J_Legal_Procedure))
legal <- cbind("legal_procedure", kf_legal$value, k2_legal$value)

kf_ter <- kappam.fleiss(cbind(skd_dnv$skd.J_Terrorism, skd_dnv$c2.J_Terrorism))
k2_ter <- kappa2(cbind(skd_dnv$skd.J_Terrorism, skd_dnv$c2.J_Terrorism))
ter <- cbind("terrorism", kf_ter$value, k2_ter$value)

kf_misc <- kappam.fleiss(cbind(skd_dnv$skd.J_Misc, skd_dnv$c2.J_Misc))
k2_misc <- kappa2(cbind(skd_dnv$skd.J_Misc, skd_dnv$c2.J_Misc))
misc <- cbind("misc", kf_misc$value, k2_misc$value)

kf_order <- kappam.fleiss(cbind(skd_dnv$skd.J_Law_Order, skd_dnv$c2.J_Law_Order))
k2_order <- kappa2(cbind(skd_dnv$skd.J_Law_Order, skd_dnv$c2.J_Law_Order))
order <- cbind("law_order", kf_order$value, k2_order$value)

kf_util <- kappam.fleiss(cbind(skd_dnv$skd.J_Utilitarian_Deterrence, skd_dnv$c2.J_Utilitarian_Deterrence))
k2_util <- kappa2(cbind(skd_dnv$skd.J_Utilitarian_Deterrence, skd_dnv$c2.J_Utilitarian_Deterrence))
util <- cbind("utilitarian_deterrence", kf_util$value, k2_util$value)

kf_prec <- kappam.fleiss(cbind(skd_dnv$skd.J_Intl_Domestic_Precedent, skd_dnv$c2.J_Intl_Domestic_Precedent))
k2_prec <- kappa2(cbind(skd_dnv$skd.J_Intl_Domestic_Precedent, skd_dnv$c2.J_Intl_Domestic_Precedent))
prec <- cbind("intl_dom_precedent", kf_prec$value, k2_prec$value)

kf_dev <- kappam.fleiss(cbind(skd_dnv$skd.J_Development_Unity, skd_dnv$c2.J_Development_Unity))
k2_dev <- kappa2(cbind(skd_dnv$skd.J_Development_Unity, skd_dnv$c2.J_Development_Unity))
dev <- cbind("development_social_unity", kf_dev$value, k2_dev$value)

kf_pol <- kappam.fleiss(cbind(skd_dnv$skd.J_Political_Strategic, skd_dnv$c2.J_Political_Strategic))
k2_pol <- kappa2(cbind(skd_dnv$skd.J_Political_Strategic, skd_dnv$c2.J_Political_Strategic))
pol <- cbind("political", kf_pol$value, k2_pol$value)

kf_denial <- kappam.fleiss(cbind(skd_dnv$skd.J_Denial, skd_dnv$c2.J_Denial))
k2_denial <- kappa2(cbind(skd_dnv$skd.J_Denial, skd_dnv$c2.J_Denial))
denial <- cbind("denial", kf_denial$value, k2_denial$value)


res <- data.frame(rbind(any, em, legal, ter, misc, order, util, prec, dev, pol, denial))
names(res)[1] <- "var"
names(res)[2] <- "kappa_fleiss"
names(res)[3] <- "kappa2"
res

write.csv(res, "kappas_dnv_r1_Dec2020.csv")
write.csv(res, "kappas_dnv_r1_Dec2020_no7508.csv")
write.csv(res, "kappas_dnv_practice_Nov2020.csv")


mean(as.numeric(res$kappa_fleiss), na.rm = TRUE)
mean(as.numeric(res$kappa2), na.rm = TRUE)

# Round 1 codes (all): Kappa_fleiss mean: .63; Kappa2: .72
# Round 1 codes (removing 7508): Kappa_fleiss mean: .71; Kappa2: .80


# Check to see where things go wrong

data <- with(skd_dnv,
             cbind(File_Img, Sentence, Round,
                   skd.justification_any, c2.justification_any,
                   skd.J_Emergency_Policy, c2.J_Emergency_Policy,
                   skd.J_Legal_Procedure, c2.J_Legal_Procedure,
                   skd.J_Terrorism, c2.J_Terrorism,
                   skd.J_Misc, c2.J_Misc, skd.J_Law_Order, c2.J_Law_Order,
                   skd.J_Utilitarian_Deterrence, c2.J_Utilitarian_Deterrence,
                   skd.J_Intl_Domestic_Precedent, c2.J_Intl_Domestic_Precedent,
                   skd.J_Development_Unity, c2.J_Development_Unity,
                   skd.J_Political_Strategic, c2.J_Political_Strategic,
                   skd.J_Denial, c2.J_Denial
             ))

head(data)

write.csv(data, "skd_dnv_first_10.csv")























#### REAL CODING ROUND ####
rm(list=ls())

skd_data <- read.csv(file="hand_coded_icr_skd.csv", header=TRUE)
skd_data[is.na(skd_data)] <- 0

dnv_data <- read.csv(file="DNV_icr_first_5.csv", header=TRUE)
dnv_data[is.na(dnv_data)] <- 0

skd <- subset(skd_data, File_Img == "1507_DEFE_24_877" | File_Img == "6497_PREM_15_1000" |
                File_Img == "7508_PREM_15_1008" | File_Img == "7595_PREM_15_1008" | 
                File_Img == "2105_DEFE_13_919" 
                )

dnv <- subset(dnv_data, File_Img == "1507_DEFE_24_877" | File_Img == "6497_PREM_15_1000" |
                File_Img == "7508_PREM_15_1008" | File_Img == "7595_PREM_15_1008" | 
                File_Img == "2105_DEFE_13_919" 
)


kf_any <- kappam.fleiss(cbind(skd$skd.justification_any, dnv$c2.justification_any))
k2_any <- kappa2(cbind(skd$skd.justification_any, dnv$c2.justification_any))
any <- cbind("any", kf_any$value, k2_any$value)

kf_em <- kappam.fleiss(cbind(skd$skd.J_Emergency_Policy, dnv$c2.J_Emergency_Policy))
k2_em <- kappa2(cbind(skd$skd.J_Emergency_Policy, dnv$c2.J_Emergency_Policy))
em <- cbind("emergency", kf_em$value, k2_em$value)

kf_legal <- kappam.fleiss(cbind(skd$skd.J_Legal_Procedure, dnv$c2.J_Legal_Procedure))
k2_legal <- kappa2(cbind(skd$skd.J_Legal_Procedure, dnv$c2.J_Legal_Procedure))
legal <- cbind("legal_procedure", kf_legal$value, k2_legal$value)

kf_ter <- kappam.fleiss(cbind(skd$skd.J_Terrorism, dnv$c2.J_Terrorism))
k2_ter <- kappa2(cbind(skd$skd.J_Terrorism, dnv$c2.J_Terrorism))
ter <- cbind("terrorism", kf_ter$value, k2_ter$value)

kf_misc <- kappam.fleiss(cbind(skd$skd.J_Misc, dnv$c2.J_Misc))
k2_misc <- kappa2(cbind(skd$skd.J_Misc, dnv$c2.J_Misc))
misc <- cbind("misc", kf_misc$value, k2_misc$value)

kf_order <- kappam.fleiss(cbind(skd$skd.J_Law_Order, dnv$c2.J_Law_Order))
k2_order <- kappa2(cbind(skd$skd.J_Law_Order, dnv$c2.J_Law_Order))
order <- cbind("law_order", kf_order$value, k2_order$value)

kf_util <- kappam.fleiss(cbind(skd$skd.J_Utilitarian_Deterrence, dnv$c2.J_Utilitarian_Deterrence))
k2_util <- kappa2(cbind(skd$skd.J_Utilitarian_Deterrence, dnv$c2.J_Utilitarian_Deterrence))
util <- cbind("utilitarian_deterrence", kf_util$value, k2_util$value)

kf_prec <- kappam.fleiss(cbind(skd$skd.J_Intl_Domestic_Precedent, dnv$c2.J_Intl_Domestic_Precedent))
k2_prec <- kappa2(cbind(skd$skd.J_Intl_Domestic_Precedent, dnv$c2.J_Intl_Domestic_Precedent))
prec <- cbind("intl_dom_precedent", kf_prec$value, k2_prec$value)

kf_dev <- kappam.fleiss(cbind(skd$skd.J_Development_Unity, dnv$c2.J_Development_Unity))
k2_dev <- kappa2(cbind(skd$skd.J_Development_Unity, dnv$c2.J_Development_Unity))
dev <- cbind("development_social_unity", kf_dev$value, k2_dev$value)

kf_pol <- kappam.fleiss(cbind(skd$skd.J_Political_Strategic, dnv$c2.J_Political_Strategic))
k2_pol <- kappa2(cbind(skd$skd.J_Political_Strategic, dnv$c2.J_Political_Strategic))
pol <- cbind("political", kf_pol$value, k2_pol$value)

kf_denial <- kappam.fleiss(cbind(skd$skd.J_Denial, dnv$c2.J_Denial))
k2_denial <- kappa2(cbind(skd$skd.J_Denial, dnv$c2.J_Denial))
denial <- cbind("denial", kf_denial$value, k2_denial$value)

# To check to see where things go wrong # 
data_merge <- merge(skd, dnv, by="Obs_number")
data <- with(data_merge,
             cbind(File_Img.x, Sentence.x,
                   skd.justification_any, c2.justification_any,
                   skd.J_Emergency_Policy, c2.J_Emergency_Policy,
                   skd.J_Legal_Procedure, c2.J_Legal_Procedure,
                   skd.J_Terrorism, c2.J_Terrorism,
                   skd.J_Misc, c2.J_Misc, skd.J_Law_Order, c2.J_Law_Order,
                   skd.J_Utilitarian_Deterrence, c2.J_Utilitarian_Deterrence,
                   skd.J_Intl_Domestic_Precedent, c2.J_Intl_Domestic_Precedent,
                   skd.J_Development_Unity, c2.J_Development_Unity,
                   skd.J_Political_Strategic, c2.J_Political_Strategic,
                   skd.J_Denial, c2.J_Denial
             ))
all.equal(data_merge$Sentence.x, data_merge$Sentence.y)

head(data)

write.csv(data, "skd_dnv_first_5.csv")

