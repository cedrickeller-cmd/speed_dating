# Predictive Modeling of Speed Dating Outcomes

#install.packages(c("farff", "corrplot", "caret", "e1071"))

library(farff)
library(mlbench)
library(corrplot)
library(caret)
library(e1071)
library(dplyr)
library(VIM)

################################################################################
# Data import
################################################################################

# data source: https://www.openml.org/search?type=data&status=active&id=40536
data = readARFF("speeddating.arff") # requires "farff" package

dim(data)
names(data)
head(data)

################################################################################
# Dropping columns not used in prediction, storing response variable separately
################################################################################

##############################
# response variable
response = data[, length(data)]

##############################
# we assume that the model will infer decision and decision_o to conclude a match (1, 1) vs no match (any 0)
head(data[, c("decision", "decision_o", "match")])

# remove has_null, decision, decision_o, and match columns (first/last 3)
data_pred = data[, -c(1, length(data)-2, length(data)-1, length(data))]

############################## # drop bins (maybe drop d_age as well => drop all columns starting with "d_")
# bins are incorrect (NA are treated as 0 in bins)
lapply(data_pred, table, useNA = "ifany")

# all column names
all_cols = names(data_pred)

# find all d_ columns
d_cols = grep("^d_", all_cols, value = TRUE)

# strip one "d_"
base1 = function(x) sub("^d_", "", x)

# find all d_d_ columns (second-level diffs)
dd_cols = grep("^d_d_", all_cols, value = TRUE)
dd_bases = sub("^d_d_", "d_", dd_cols) # their corresponding d_base

# rule 1 & 2: drop d_base if base exists, unless it's in dd_bases
drop_single_d = d_cols[
  base1(d_cols) %in% all_cols & # base exists
    !(d_cols %in% dd_bases)     # but no d_d_base for it
]

# rule 3: drop all d_d_* if their d_base exists
drop_double_d = dd_cols[dd_bases %in% all_cols]

# final drop list (bins)
cols_bin = c(drop_single_d, drop_double_d)
# keep everything else (continuous)
cols_cont = setdiff(all_cols, cols_bin)

length(cols_cont)
length(cols_bin) ############################## #  d_d_age is duplicated somehow
cols_cont
cols_bin

# keep only non-bin data
data_no_bin = data[, cols_cont]

dim(data_no_bin)
head(data_no_bin)

################################################################################
# Clean $field   # need to address NA values (add to other?)
################################################################################
# exploration 
length(unique(data_no_bin$field))
#table(data_no_bin$field, useNA = "ifany")

# group fields
data_no_bin$field = case_when(
  grepl("law", data_no_bin$field, ignore.case = TRUE) ~ "Law",
  grepl("business|mba|finance|finanace|marketing|consulting|real estate|money|economics|fundraising", data_no_bin$field, ignore.case = TRUE) ~ "Business/Finance",
  grepl("economics", data_no_bin$field, ignore.case = TRUE) ~ "Economics",
  grepl("psychology|counseling|sociology|social|human rights|religion|anthropology|urban planning|QMSS", data_no_bin$field, ignore.case = TRUE) ~ "Social Sciences",
  grepl("arts|music|history|literature|philosophy|film|creative writing|theater|arts administration|architecture|english|french|polish|american studies|classics", data_no_bin$field, ignore.case = TRUE) ~ "Arts/Humanities",
  grepl("computer|engineering|mechanical|electrical|industrial|operations research|mathematics|math|statistics|stats|computational|physics|quantitative|chemistry|climate|earth|SOA", data_no_bin$field, ignore.case = TRUE) ~ "STEM",
  grepl("medicine|health|medical|biology|biomedical|neuroscience|nutrition|genetics|epidemiology|public health|preMed|nutrition|nutritiron|biotechnology", data_no_bin$field, ignore.case = TRUE) ~ "Health/Medical",
  grepl("education|teaching|childhood|curriculum|pedagogy|school|instruction|early childhood|literacy|TESOL|higher ed", data_no_bin$field, ignore.case = TRUE) ~ "Education",
  grepl("political|international|public policy|public administration|SIPA|international affairs|Intrernational", data_no_bin$field, ignore.case = TRUE) ~ "Politics/International Affairs",
  grepl("Acting|MFA|Nonfiction|Writing|Theatre Management|Speech|journalism|communications|GSAS", data_no_bin$field, ignore.case = TRUE) ~ "Arts/Humanities",
  grepl("theory|Undergrad - GS|working", data_no_bin$field, ignore.case = TRUE) ~ "Other",
  TRUE ~ data_no_bin$field  # OR USE "Other" -- currently shows NA
)

length(table(data_no_bin$field, useNA = "ifany"))
#table(data_no_bin$field, useNA = "ifany")

################################################################################
# KNN Imputation
################################################################################
data_imputed <- kNN(data_no_bin, k = 5, imp_var = FALSE)
dim(data_imputed)

################################################################################
# Separating continuous and categorical data
################################################################################
cont_data = data_imputed[, sapply(data_imputed, is.numeric)]
cat_data = data_imputed[, sapply(data_imputed, function(x) is.factor(x) | is.character(x))]
names(cont_data)
names(cat_data)


################################################################################
# Creating dummy variables for Categorical data
################################################################################
dummies <- dummyVars(~ ., data = cat_data, fullRank=TRUE)

# Apply transformation
cat_dummies <- data.frame(predict(dummies, newdata = cat_data))
dim(cat_dummies)
names(cat_dummies)

################################################################################
# Applying correlations on Continuous Data
################################################################################

correlations <- cor(cont_data)
dim(correlations)
correlations[1:4, 1:4]
corrplot(correlations, order = "hclust")
findCorrelation(correlations, cutoff = 0.85)
colnames(cont_data)[c(40)]

################################################################################
# Applying NZV on Categorical Data
################################################################################

nearZeroVar(cat_dummies)
colnames(cat_dummies)[c(12,16)]
cat_dummies$fieldOther


















