############################################
##### ICR: SKD and DNV (Nov/Dec 2020) ######
############################################

rm(list=ls())

library(irr)
library(xtable)

setwd("~/OneDrive/Incubator/NIreland_NLP/icr/")

#### ICR ANALYSIS: Ten documents (docs containing a justification, randomly selected) ####

# load lead coder data
skd_data <- read.csv(file="hand_coded_icr_skd.csv", header=TRUE)
skd_data[is.na(skd_data)] <- 0

# load icr coder data
dnv_data <- read.csv(file="DNV_icr_first_10.csv", header=TRUE)
dnv_data[is.na(dnv_data)] <- 0

# merge lead and icr coder data
skd_dnv <- merge(skd_data, dnv_data, by="Obs_number") 

# Make sure all sentences are equal and check where wrong -- all ok #
with(skd_dnv, all.equal(Sentence.x, Sentence.y))
skd_dnv$Sentence.x == skd_dnv$Sentence.y

# two sentences that are not equal are effectively the same -- no problems here
skd_dnv$Sentence.x[143]
skd_dnv$Sentence.y[143]
skd_dnv$Sentence.x[215]
skd_dnv$Sentence.y[215]

# clean up column names
names(skd_dnv)[names(skd_dnv) == "File_Img.x"] <- "File_Img"
names(skd_dnv)[names(skd_dnv) == "Sentence.x"] <- "Sentence"
names(skd_dnv)[names(skd_dnv) == "Round.x"] <- "Round"

# remove unnecessary columns (to avoid confusion)
skd_dnv = subset(skd_dnv, select = - c(Date_coded, File_Img.y, Sentence.y, Round.y) )

# Subset to 10 documents coded (Round 1) -- Round 2 not executed
skd_dnv_r1 <- subset(skd_dnv, Round == 1)

# Calculate Kappa Scores
skd_dnv <- skd_dnv_r1

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

xtable(res)
write.csv(res, "kappas_icr_Dec2020.csv")

mean(as.numeric(res$kappa_fleiss), na.rm = TRUE)
mean(as.numeric(res$kappa2), na.rm = TRUE)

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
write.csv(data, "skd_dnv_merged_check.csv")

#### ICR ANALYSIS: Remove 7508 (badly coded) ####

skd_dnv_no7508 <- subset(skd_dnv, File_Img!="7508_PREM_15_1008") #test to see if we improve if we drop this one

# Calculate Kappa scores

skd_dnv <- skd_dnv_no7508

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

xtable(res)
write.csv(res, "kappas_icr_Dec2020_no7508.csv")


#### ICR ANALYSIS: Practice documents documents (5, docs containing a justification, 
#### selected for their extensive justifications among 10 randomly selected documents) ####

# load data from lead and ICR coder (all one file)
skd_dnv_practice <- read.csv(file="DNV_practice_justifications.csv", header=TRUE)
skd_dnv_practice[is.na(skd_dnv_practice)] <- 0
skd_dnv_practice$Round <- 0
skd_dnv_practice$Obs_number <- seq(1, dim(skd_dnv_practice)[1]) + 1000
skd_dnv_practice = subset(skd_dnv_practice, select = - DNV.Notes) 

# Merge practice and round 1
# skd_dnv_all <- rbind(skd_dnv_practice, skd_dnv)

# Calculate Kappa scores

skd_dnv <- skd_dnv_practice

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

xtable(res)
write.csv(res, "kappas_dnv_practice_Nov2020.csv")
