library(tidyverse)
library(xgboost)
library(magrittr)
library(readr)
library(dplyr)

set.seed(0)

# Load data
bbalance <- read_csv("../data/raw/bureau_balance.csv") 
bureau <- read_csv("../data/raw/bureau.csv")
cc_balance <- read_csv("../data/raw/credit_card_balance.csv")
payments <- read_csv("../data/raw/installments_payments.csv") 
pc_balance <- read_csv("../data/raw/POS_CASH_balance.csv")
prev <- read_csv("../data/raw/previous_application.csv")
tr <- read_csv("../data/raw/application_train.csv") 
te <- read_csv("../data/raw/application_test.csv")

# Generate features
fn <- funs(mean, sd, min, max, sum, n_distinct, .args = list(na.rm = TRUE))

sum_bbalance <- bbalance %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  group_by(SK_ID_BUREAU) %>% 
  summarise_all(fn) 

# Get home related features
home_related_features <- c("APARTMENTS_AVG", "APARTMENTS_MEDI", "APARTMENTS_MODE", "BASEMENTAREA_AVG", "BASEMENTAREA_MEDI",
                           "BASEMENTAREA_MODE", "COMMONAREA_AVG", "COMMONAREA_MEDI", "COMMONAREA_MODE", "ELEVATORS_AVG",
                           "ELEVATORS_MEDI", "ELEVATORS_MODE", "EMERGENCYSTATE_MODE", "ENTRANCES_AVG", "ENTRANCES_MEDI", 
                           "ENTRANCES_MODE", "FLOORSMAX_AVG", "FLOORSMAX_MEDI", "FLOORSMAX_MODE", "FLOORSMIN_AVG", 
                           "FLOORSMIN_MEDI", "FLOORSMIN_MODE", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "LANDAREA_AVG", 
                           "LANDAREA_MEDI", "LANDAREA_MODE", "LIVINGAPARTMENTS_AVG", "LIVINGAPARTMENTS_MEDI", 
                           "LIVINGAPARTMENTS_MODE", "LIVINGAREA_AVG", "LIVINGAREA_MEDI", "LIVINGAREA_MODE", 
                           "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAPARTMENTS_MODE", 
                           "NONLIVINGAREA_AVG", "NONLIVINGAREA_MEDI", "NONLIVINGAREA_MODE", "TOTALAREA_MODE", 
                           "WALLSMATERIAL_MODE", "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BEGINEXPLUATATION_MEDI", 
                           "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BUILD_AVG", "YEARS_BUILD_MEDI", "YEARS_BUILD_MODE")

tr2 <- select(tr, home_related_features)

count_df <- data.frame(feature = colnames(tr2))
count_lst <- c()
for (col in colnames(tr2)) {
  count_lst <- c(count_lst, length(unique(tr2[[col]])))
}
count_df$count <- count_lst

# Feature selection
library(dplyr)
library(readr)

previous_application_stats <- read_csv("E:/previous_application_stats.csv")
colnames(previous_application_stats) <- c("feature", "nb_nas", "nb_levels", "target_correlation")
previous_application_stats <- subset(previous_application_stats, nb_levels > 1)
previous_application_stats <- subset(previous_application_stats, feature != "SK_ID_CURR")

extract <- read_csv("E:/extract.csv")
previous_application_stats <- left_join(previous_application_stats, extract, by = "feature")

