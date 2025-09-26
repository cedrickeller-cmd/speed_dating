# Predictive Modeling of Speed Dating Outcomes

#install.packages(c("farff", "corrplot", "caret", "e1071", "VIM"))

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
cols_no_bin = setdiff(all_cols, cols_bin)

length(cols_no_bin)
length(cols_bin)-1 ############################## #  d_d_age is duplicated somehow
cols_no_bin
cols_bin

##############################
# keep only non-bin data
data_no_bin = data[, cols_no_bin]

dim(data_no_bin)
head(data_no_bin)

################################################################################
# Clean $field
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
  TRUE ~ data_no_bin$field  # keeping NA
)

length(table(data_no_bin$field, useNA = "ifany"))
#table(data_no_bin$field, useNA = "ifany")

################################################################################
# Proportion of missing data in each column
################################################################################
##############################
# Missing values -- overall and by obs
sum(is.na(data_no_bin))
mean(is.na(data_no_bin))
num_missing_obs = sum(!complete.cases(data_no_bin))
num_missing_obs
prop_missing_obs = num_missing_obs / nrow(data_no_bin)
prop_missing_obs # 87.5%

##############################
# Proportion of missing values per column
missing_proportions = colSums(is.na(data_no_bin)) / nrow(data_no_bin)

sort(missing_proportions, decreasing = TRUE)

##############################
# Drop predictors with missing values > 50%
data_no_bin = data_no_bin[, missing_proportions <= 0.75]

dim(data_no_bin)

##############################
# Missing values -- overall and by obs (after drop)
sum(is.na(data_no_bin))
mean(is.na(data_no_bin))
num_missing_obs = sum(!complete.cases(data_no_bin))
num_missing_obs
prop_missing_obs = num_missing_obs / nrow(data_no_bin)
prop_missing_obs # 42.0%

################################################################################
# KNN Imputation   # may consider interpolation instead   # do splitting before to avoid data leakage
################################################################################
data_imputed = kNN(data_no_bin, k = 5, imp_var = FALSE)
dim(data_imputed)
colSums(is.na(data_imputed))

################################################################################
# Separating continuous and categorical data
################################################################################
cont_data = data_imputed[, sapply(data_imputed, is.numeric)]
cat_data = data_imputed[, sapply(data_imputed, function(x) is.factor(x) | is.character(x))]
names(cont_data)
names(cat_data)

################################################################################
# Creating dummy variables for categorical data
################################################################################
dummies = dummyVars(~ ., data = cat_data, fullRank=TRUE)

# Apply transformation
cat_dummies = data.frame(predict(dummies, newdata = cat_data))
dim(cat_dummies)
names(cat_dummies)

################################################################################
# Applying correlations on continuous data
################################################################################

correlations_cont = cor(cont_data)
dim(correlations_cont)
#correlations_cont[1:4, 1:4]
?corrplot(correlations_cont, order = "hclust", title = "Correlation Matrix of Continuous Predictors", mar = c(0, 0, 2, 0))
findCorrelation(correlations_cont, cutoff = 0.85)
#colnames(cont_data)[c(40)] # "museums"

# columns not dropped because we will let PCA handle it

################################################################################
# Applying NZV on categorical data
################################################################################
# Find NZV predictors
nzv_cat = nearZeroVar(cat_dummies)
colnames(cat_dummies)[nzv_cat] # "fieldEcology" "fieldOther"

# columns not dropped because we will let PCA handle it

################################################################################
# Histograms of continuous predictors
################################################################################
#length(cont_data) # 59

# Calculate the number of plot pages needed
num_cols = length(names(cont_data))
num_pages = ceiling(num_cols / 9)  # 9 plots per page (3x3 grid)

# Loop through pages
for (page in 1:num_pages) {
  # Set up 3x3 plotting area
  par(mfrow = c(3, 3), 
      mar = c(4, 4, 2, 1),   # Adjust margins to give some space
      oma = c(0, 0, 2, 0))   # Outer margin for overall title
  
  # Calculate start and end columns for this page
  start_col = (page - 1) * 9 + 1
  end_col = min(start_col + 8, num_cols)
  
  # Create histograms for columns on this page
  for (i in start_col:end_col) {
    col = names(cont_data)[i]
    hist(cont_data[[col]], 
         main = col, 
         xlab = "Value",
         col = "skyblue", 
         border = "black")
  }
  
  # Add an overall title for the page
  mtext(paste0("Histograms (", page, "/", num_pages, ")"), outer = TRUE, cex = 1.5)
}

# Reset the plotting parameters
par(mfrow = c(1, 1))

################################################################################
# Transformation -- Box-Cox
################################################################################
##############################
# Get all BoxCox lambda values for predictors
lambdas = sapply(cont_data, function(x) { 
  BoxCoxTrans(x + 1)$lambda # requires "caret" package; value must be added to only have positive numbers (for the Box Cox Transformation to converge)
})
#lambdas

skew_values = apply(cont_data, 2, skewness)
#skewValues

# Only retain highly skewed predictors
cont_skewed = names(skew_values[abs(skew_values) > 1])
#cont_skewed

##############################
# Box-Cox transformed continuous list (initialization)
cont_data_bc = cont_data

# Loop through variables for transformation
for (col in names(cont_skewed)) {
  x = cont_data[[col]]
  
  # Box-Cox transform
  bct = BoxCoxTrans(x + 0.000001)  # tiny offset to only have positive numbers (for the Box Cox Transformation to converge)
  x_trans = predict(bct, x + 0.000001)
  
  # Replace original values with transformed values
  cont_data_bc[[col]] = x_trans
}

##############################
# Histograms of box-cox transformed data

#length(cont_data) # 59

# Calculate the number of plot pages needed
num_cols = length(names(cont_data_bc))
num_pages = ceiling(num_cols / 9)  # 9 plots per page (3x3 grid)

# Loop through pages
for (page in 1:num_pages) {
  # Set up 3x3 plotting area
  par(mfrow = c(3, 3), 
      mar = c(4, 4, 2, 1),   # Adjust margins to give some space
      oma = c(0, 0, 2, 0))   # Outer margin for overall title
  
  # Calculate start and end columns for this page
  start_col = (page - 1) * 9 + 1
  end_col = min(start_col + 8, num_cols)
  
  # Create histograms for columns on this page
  for (i in start_col:end_col) {
    col = names(cont_data_bc)[i]
    hist(cont_data_bc[[col]], 
         main = col, 
         xlab = "Value",
         col = "skyblue", 
         border = "black")
  }
  
  # Add an overall title for the page
  mtext(paste0("Histograms Box-Cox transformed (", page, "/", num_pages, ")"), outer = TRUE, cex = 1.5)
}

# Reset the plotting parameters
par(mfrow = c(1, 1))

################################################################################
# Merging continuous and categorical data
################################################################################
# Merge continuous + categorical (dummy vars after nzv)
dim(cont_data_bc) # 8378 rows
dim(cat_dummies) # 8378 rows
combined_data = cbind(cont_data_bc, cat_dummies)
dim(combined_data)
names(combined_data)

################################################################################
# PCA (incl. center + scale) -- only on continuous data
################################################################################
pcaObject = prcomp(combined_data, center = TRUE, scale. = TRUE)

# Cumulative percentage of variance which each component accounts for
percentVariance = pcaObject$sd^2/sum(pcaObject$sd^2)*100
percentVariance[1:5]

cumpercentVariance = cumsum(pcaObject$sd^2)/sum(pcaObject$sd^2)*100
cumpercentVariance[1:20]

# Transformed values are stored in pcaObject as a sub-object called x:
head(pcaObject$x[, 1:5])

##############################
# Cutoff values
cutoffs = c(65, 80, 90, 95) # total variance explained

# Scree plot
plot(percentVariance, type = "o",
     xlab = "Principal Component",
     ylab = "Percent of Total Variance Explained (%)",
     main = "Scree Plot")
# Add lines and labels
for (c in cutoffs) {
  # Find the first PC that reaches the cutoff
  k = which(cumpercentVariance >= c)[1]
  
  # Vertical line at k
  abline(v = k, col = "darkblue", lty = 2)
  # Text label
  text(x = k+3.5, y = percentVariance[k]+.6, labels = paste0(c, " %"), col = "darkred", cex = 0.8)
  text(x = k+4.5, y = percentVariance[k]+.25, labels = paste0(k, " PCs"), col = "darkblue", cex = 0.8)
}

##############################
# Cumulative variance plot
plot(cumpercentVariance, type = "o",
     xlab = "# of Principal Components",
     ylab = "Percent of Total Variance Explained (%)",
     main = "Cumulative Variance Explained by PCs")
# Add lines and labels
for (c in cutoffs) {
  # Find the first PC that reaches the cutoff
  k = which(cumpercentVariance >= c)[1]
  
  # Add horizontal line
  abline(h = c, col = "darkred", lty = 2)
  # Vertical line at k
  abline(v = k, col = "darkblue", lty = 2)
  # Text label
  text(x = 1, y = c+2, labels = paste0(c, " %"), col = "darkred", cex = 0.8)
  text(x = k+4.5, y = c-2, labels = paste0(k, " PCs"), col = "darkblue", cex = 0.8)
}

##############################
# The reduction of predictors with high enough variance explained is limited
# Before PCA: 76, After PCA: 95%->61 PCs, 90%->52PCs

c = 95 # cutoff percentage
k = which(cumpercentVariance >= c)[1] # first k PC exceeding the cum pct var explained

data_pca = as.data.frame(pcaObject$x[, 1:k])

################################################################################
# Boxplots of Box-Cox of k PCs
################################################################################

# Calculate the number of plot pages needed
num_cols = length(names(data_pca))
num_pages = ceiling(num_cols / 9)  # 9 plots per page (3x3 grid)

# Loop through pages
for (page in 1:num_pages) {
  # Set up 3x3 plotting area
  par(mfrow = c(3, 3), 
      mar = c(4, 4, 2, 1),   # Adjust margins to give some space
      oma = c(0, 0, 2, 0))   # Outer margin for overall title
  
  # Calculate start and end columns for this page
  start_col = (page - 1) * 9 + 1
  end_col = min(start_col + 8, num_cols)
  
  # Create boxplots for columns on this page
  for (i in start_col:end_col) {
    col = names(data_pca)[i]
    boxplot(data_pca[[col]], 
         main = col, 
         xlab = "Value",
         col = "skyblue", 
         border = "black")
  }
  
  # Add an overall title for the page
  mtext(paste0("Boxplots PCA transformed (", page, "/", num_pages, ")"), outer = TRUE, cex = 1.5)
}

# Reset the plotting parameters
par(mfrow = c(1, 1))


################################################################################
# Spatial Sign transformation
################################################################################
# Spacial sign transform the PCs
ss_trans = preProcess(data_pca, method = c("spatialSign")) # requires "caret" package
ss_trans

# Apply transformation
data_pca_ss = predict(ss_trans, data_pca)  # all (k) centered, scaled, and spatial sign transformed
dim(data_pca)
dim(data_pca_ss)

# Check different values
head(data_pca[1:3, 1:5])
head(data_pca_ss[1:3, 1:5])
min(data_pca_ss); max(data_pca_ss) # health check: all between -1 and 1

################################################################################
# Box-Plot of spatial sign transformed data
################################################################################
# Calculate the number of plot pages needed
num_cols = length(names(data_pca_ss))
num_pages = ceiling(num_cols / 9)  # 9 plots per page (3x3 grid)

# Loop through pages
for (page in 1:num_pages) {
  # Set up 3x3 plotting area
  par(mfrow = c(3, 3), 
      mar = c(4, 4, 2, 1),   # Adjust margins to give some space
      oma = c(0, 0, 2, 0))   # Outer margin for overall title
  
  # Calculate start and end columns for this page
  start_col = (page - 1) * 9 + 1
  end_col = min(start_col + 8, num_cols)
  
  # Create boxplots for columns on this page
  for (i in start_col:end_col) {
    col = names(data_pca_ss)[i]
    boxplot(data_pca_ss[[col]], 
            main = col, 
            xlab = "Value",
            col = "skyblue", 
            border = "black")
  }
  
  # Add an overall title for the page
  mtext(paste0("Boxplots Spatial Sign transformed (", page, "/", num_pages, ")"), outer = TRUE, cex = 1.5)
}

# Reset the plotting parameters
par(mfrow = c(1, 1))

################################################################################
# Bar plot of response variable
################################################################################
# Calculate proportions
response_prop = prop.table(table(response))
barplot(response_prop,
        main = "Proportion of Match Classes",
        xlab = "Match",
        col = c("tomato","skyblue"),
        legend.text = c(
          paste0("0 (no match): ", 
                 sprintf("%.1f%%", response_prop[1] * 100)),
          paste0("1 (match): ", 
                 sprintf("%.1f%%", response_prop[2] * 100))
        ))

# Imbalanced dataset --> stratified sampling