#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the Home Credit Default Risk Challenge                   #
#                                                                             #
# This is the entry point of the solution.                                    #
# Developped using Python 3.6.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-03-06                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import os
import time
import numpy as np
import pandas as pd
import pickle
import gc
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

from pipeline.efficient_pipeline import EfficientPipeline
from data_preprocessing.PreprocessingStep import PreprocessingStep
from data_preprocessing.AdditionalFilesPreprocessingStep import AdditionalFilesPreprocessingStep
from feature_processing.categorical_encoders import CategoricalFeaturesEncoder, OrdinalEncoder, GroupingEncoder, LeaveOneOutEncoder, TargetAvgEncoder
from feature_processing.missing_values_imputation import MissingValuesImputer
from feature_engineering.features_selectors import VarianceFeatureSelector, L1NormFeatureSelector, LGBMFeatureSelector, ConstantFeaturesRemover, DuplicatedFeaturesRemover
from feature_engineering.features_generators import PairwiseNumericalInteractionsGenerator
from wrappers.lightgbm_wrapper import LGBMClassifier

from load_data import load_data
from files_paths import *

pd.set_option("display.max_columns", 100)

# Call to main
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    
    # Set the seed of numpy's PRNG
    np.random.seed(2017)

    enable_validation = True

    # Load the data; y_test is None when 'enable_validation' is False
    X_train, X_test, y_train, y_test, bureau_data_df, bureau_balance_data_df, credit_card_balance_data_df, installments_payments_data_df, pos_cash_balance_data_df, previous_application_data_df = load_data(TRAINING_DATA_str, TESTING_DATA_str, BUREAU_DATA_str, BUREAU_BALANCE_DATA_str, CREDIT_CARD_BALANCE_DATA_str, INSTALLMENTS_PAYMENTS_DATA_str, POS_CASH_BALANCE_DATA_str, PREVIOUS_APPLICATION_DATA_str, enable_validation, "TARGET", CACHE_DIR_str)
    target_df = y_train.reset_index()
    target_df.columns = ["SK_ID_CURR", "target"]
    
    print("Train shape: ", X_train.shape)
    print("Test shape: ", X_test.shape)

    if not os.path.exists("E:/final_dataset_df.pkl"):
        print("Processing additional datasets...")
        additional_files_preprocessor = AdditionalFilesPreprocessingStep()
        final_dataset_df = additional_files_preprocessor.fit_transform(target_df, bureau_data_df, bureau_balance_data_df, credit_card_balance_data_df, installments_payments_data_df, pos_cash_balance_data_df, previous_application_data_df)

        with open("E:/final_dataset_df.pkl", "wb") as f:
            pickle.dump(final_dataset_df, f, protocol = 4)
    else:
        print("Loading processed additional datasets from cache...")
        with open("E:/final_dataset_df.pkl", "rb") as f:
            final_dataset_df = pickle.load(f)

    columns_to_be_encoded_lst = ["NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "EMERGENCYSTATE_MODE", "CODE_GENDER", "CODE_GENDER", 
                                 "HOUSETYPE_MODE", "FONDKAPREMONT_MODE", "NAME_EDUCATION_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
                                 "NAME_TYPE_SUITE", "WEEKDAY_APPR_PROCESS_START", "WALLSMATERIAL_MODE", "NAME_INCOME_TYPE", "NAME_INCOME_TYPE", "OCCUPATION_TYPE", 
                                 "ORGANIZATION_TYPE"]#, "ORGANIZATION_TYPE"]
    encoders_lst = [LabelBinarizer(), OrdinalEncoder(), OrdinalEncoder(), OrdinalEncoder(), OrdinalEncoder(), TargetAvgEncoder(),
                    LabelBinarizer(), LabelBinarizer(), LabelBinarizer(), TargetAvgEncoder(), LabelBinarizer(), LabelBinarizer(),
                    GroupingEncoder(LabelBinarizer(), 3), OrdinalEncoder(), OrdinalEncoder(), LabelBinarizer(), TargetAvgEncoder(), OrdinalEncoder(),
                    GroupingEncoder(LabelBinarizer(), 25)]#, GroupingEncoder(TargetAvgEncoder(), 20)] =>  Generates infinity / nans (cf variance selector)

    """lgb_params = {
        "learning_rate": 0.015,
        "application": "binary",
        "max_depth": 7,
        "num_leaves": 70,
        "verbosity": -1,
        "metric": "auc",
        "subsample": 0.9,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_split_gain": 0.01,
        "min_child_weight": 19
    }"""

    lgb_params = {"boosting_type": "gbdt",
          "max_depth" : -1,
          "objective": "binary",
          "num_leaves": 30,
          "learning_rate": 0.010,
          "max_bin": 255,
          "subsample": 1.0,
          "subsample_freq": 1,
          "colsample_bytree": 0.05,
          "reg_alpha": 0,
          "reg_lambda": 100,
          "min_split_gain": 0.5,
          "min_child_weight": 19,
          "min_child_samples": 70,
          "scale_pos_weight": 1, # or is_unbalance = True
          "metric" : "auc",
          "verbosity": -1,
          "device": "gpu"
    }
    
    """
    lgb_params = {"boosting_type": "gbdt",
          "max_depth" : 8,
          "objective": "binary",
          "num_leaves": 32,
          "learning_rate": 0.010,
          "max_bin": 255,
          "subsample_for_bin": 200,
          "subsample": 0.8715623,
          "subsample_freq": 1,
          "colsample_bytree": 0.9497036,
          "reg_alpha": 0.04,
          "reg_lambda": 0.073,
          "min_split_gain": 0.0222415,
          "min_child_weight": 40,
          "metric" : "auc",
          "verbosity": -1,
          "device": "gpu"
    } # 0.791
    """

    # ("ConstantFeaturesRemover", ConstantFeaturesRemover()), ("DuplicatedFeaturesRemover", DuplicatedFeaturesRemover()),
    # ("PairwiseNumericalInteractionsGenerator", PairwiseNumericalInteractionsGenerator(columns_names_lst = ["AMT_GOODS_PRICE", "FLOORSMAX_MEDI", "EXT_SOURCE_2", "CNT_FAM_MEMBERS", "NONLIVINGAREA_MEDI", "AMT_REQ_CREDIT_BUREAU_MON", "CNT_CHILDREN", "COMMONAREA_AVG", "APARTMENTS_MEDI", "COMMONAREA_MODE", "NONLIVINGAREA_MODE", "ENTRANCES_MEDI", "NONLIVINGAREA_AVG", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_WEEK", "APARTMENTS_MODE", "AMT_ANNUITY", "YEARS_BEGINEXPLUATATION_AVG", "ELEVATORS_MEDI", "APARTMENTS_AVG", "BASEMENTAREA_AVG", "FLOORSMIN_MEDI", "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BEGINEXPLUATATION_MEDI", "FLOORSMIN_AVG"])),

    # Put EfficientPipeline instead of Pipeline
    main_pipeline = Pipeline([
                                       ("PreprocessingStep", PreprocessingStep(additional_data_lst = [final_dataset_df])),
                                       ("MissingValuesImputer", MissingValuesImputer(num_col_imputation = -999, cat_col_imputation = "NA")),
                                       ("CategoricalFeaturesEncoder", CategoricalFeaturesEncoder(columns_to_be_encoded_lst, encoders_lst)),
                                       ("VarianceFeatureSelector", VarianceFeatureSelector(3e-5)),
                                       ("LGBMFeatureSelector", LGBMFeatureSelector(threshold = 0.762, problem_type = "classification", enable_cv = False, lgbm_params = lgb_params)),
                                       ("LightGBM", LGBMClassifier(lgb_params, early_stopping_rounds = 150, test_size = 0.15, verbose_eval = 100, nrounds = 10000, enable_cv = True))
                                      ])
    
    # Train the model
    main_pipeline.fit(X_train, y_train)
    
    # Make predictions
    predictions_npa = main_pipeline.predict(X_test)

    # Evaluate the model
    if enable_validation:
        print("Validation AUC:", roc_auc_score(y_test, predictions_npa))
    else:
        predictions_df = pd.DataFrame({"SK_ID_CURR": X_test.index, "TARGET": predictions_npa})
        predictions_df.to_csv(PREDICTIONS_DIR_str + "first_solution_submission.csv", index = False)

    # Stop the timer and print the exectution time
    print("*** Test finished : Executed in:", time.time() - start_time, "seconds ***")

    # Plot features importance
    feature_importance_df = main_pipeline._final_estimator.get_features_importance()
    feature_importance_df.to_excel("E:/lgbm_feature_importance.xlsx")

    # Last submission: 21/05/2018, Public LB score: 0.761, local validation score: 0.7687976114367754
    # Last submission: 21/05/2018, Public LB score: 0.770, local validation score: 0.7775741641376764
    # Last submission: 21/05/2018, Public LB score: 0.771, local validation score: 0.7776394514924672
    # Last submission: 24/05/2018, Public LB score: 0.773, local validation score: 0.7794927755985116
    # Last submission: 24/05/2018, Public LB score: 0.775, local validation score: 0.7793674315947481, best iteration: [1052]	training's auc: 0.860267	valid_1's auc: 0.782403
    # Last submission: 24/05/2018, Public LB score: 0.771, local validation score: 0., best iteration: [1216]	training's auc: 0.867785	valid_1's auc: 0.783003
    # Last submission: 25/05/2018, Public LB score: 0.775, local validation score: 0.7804498196917823, best iteration: [927]	training's auc: 0.874475	valid_1's auc: 0.783752
    # Last submission: 25/05/2018, Public LB score: 0.775, local validation score: 0.7830585913439122, best iteration: [1286]	training's auc: 0.900563	valid_1's auc: 0.786186
    # Last submission: 25/05/2018, Public LB score: 0.778, local validation score: 0.7839737433431141, best iteration: [1731]	training's auc: 0.887433	valid_1's auc: 0.786673
    # Last submission: 03/06/2018, Public LB score: 0.778, local validation score: 0.78543687251583, best iteration: [3149]	training's auc: 0.886978	valid_1's auc: 0.787719
    # Last submission: 03/06/2018, Public LB score: 0.779, local validation score: 0.7861441575520606, best iteration: [3151]	training's auc: 0.887514	valid_1's auc: 0.788613
    # Last submission: 03/06/2018, Public LB score: 0.782, local validation score: 0.786935358973229, best iteration: [3200]	cv_agg's auc: 0.787716 + 0.00251364
    # Last submission: 03/06/2018, Public LB score: 0.786, local validation score: 0.7872588516887172, best iteration: [3300]	cv_agg's auc: 0.788296 + 0.00260346
    # Last submission: 24/06/2018, Public LB score: 0.785, local validation score: 0.7890342189844852, best iteration: [3300]	cv_agg's auc: 0.789042 + 0.0023899
    # Last submission: 30/06/2018, Public LB score: 0.791, local validation score: 0.7913583030412702, best iteration: [3000]  cv_agg's auc: 0.791811 + 0.00229
    # Last submission: 10/07/2018, Public LB score: 0.794, local validation score: 0.792572705721141, best iteration: [3300]  cv_agg's auc: 0.792713 + 0.00250958
    # Last submission: 15/07/2018, Public LB score: 0.797, local validation score: 0.7930225821503608, best iteration: [3300]  cv_agg's auc: 0.793265 + 0.0024848
    # Last submission: 22/07/2018, Public LB score: 0.802, local validation score: 0.7939531011343464, best iteration: [6200]  cv_agg's auc: 0.794916 + 0.00251919
    # Last submission: 26/07/2018, Public LB score: 0.802, local validation score: 0.7942585900748229, best iteration: [6600]  cv_agg's auc: 0.795471 + 0.00262586
    # Last submission: 21/08/2018, Public LB score: 0.802, local validation score: 0.79408108997792, best iteration: [5800]  cv_agg's auc: 0.795097 + 0.00244536

"""
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

#X["target"] = y
with open("E:/full_dataset_22072018.pkl", "rb") as f:
    X = pickle.load(f)

X.replace(-999, np.nan, inplace = True)

X2 = X.select_dtypes(include = [np.number])

# Remove categorical features
occurences_lst = []
for feat in X2.columns.tolist():
    if X2[feat].nunique() < 10:
        occurences_lst.append(feat)

X2.drop(occurences_lst, axis = 1, inplace = True)

X2["target"] = X["target"]
cormat = X2.corr()

duplicates_features_lst = [] #duplicates_features_lst = ['bureau_nb_bureau_records', 'bureau_AMT_CREDIT_SUM_DEBT_sum', 'previous_application_PRODUCT_COMBINATION_NA_mean', 'previous_application_credit_length_months_nunique', 'credit_card_balance_AVG_DPD', 'credit_card_balance_NO_DRAWINGS', 'credit_card_balance_AMT_DRAWINGS_CURRENT_sum', 'bureau_CNT_CREDIT_PROLONG_mean', 'bureau_DAYS_ENDDATE_DIFF_mean', 'credit_card_balance_CNT_DRAWINGS_CURRENT_sum', 'credit_card_balance_DRAWINGS_ATM', 'previous_application_PRODUCT_COMBINATION_NA_std', 'bureau_TOTAL_CUSTOMER_OVERDUE', 'credit_card_balance_TOTAL_DRAWINGS', 'bureau_CREDIT_ENDDATE_BINARY_mean', 'previous_application_NAME_CONTRACT_TYPE_XNA_std', 'bureau_AVG_ENDDATE_FUTURE', 'installments_payments_installment_overdue_ratio', 'previous_application_credit_length_nunique', 'previous_application_NAME_CONTRACT_TYPE_XNA_mean', 'bureau_CREDIT_ENDDATE_PERCENTAGE', 'bureau_BUREAU_LOAN_COUNT', 'previous_application_CODE_REJECT_REASON_CLIENT', 'credit_card_balance_SK_DPD_mean', 'bureau_AVG_CREDITDAYS_PROLONGED', 'bureau_bureau_count', 'bureau_AMT_CREDIT_SUM_OVERDUE_sum', 'previous_application_NAME_CONTRACT_TYPE_Cash loans_std', 'bureau_TOTAL_CUSTOMER_DEBT', 'previous_application_NAME_CASH_LOAN_PURPOSE_XAP_std', 'bureau_TOTAL_CUSTOMER_CREDIT', 'bureau_AMT_CREDIT_SUM_sum', 'credit_card_balance_AMT_DRAWINGS_ATM_CURRENT_sum', 'previous_application_NAME_CONTRACT_STATUS_Unused offer', 'installments_payments_is_installment_overdue_mean', 'credit_card_balance_DRAWINGS_TOTAL']
highly_correlated_features_dict = {"feature1" : [], "feature2" : [], "correlation value" : []}

for i in range(X2.shape[1]):
    for j in range(i + 1, X2.shape[1]):
        if cormat.iloc[i, j] == 1: # If features are duplicated
            duplicates_features_lst.append(X2.columns.tolist()[i])
            duplicates_features_lst.append(X2.columns.tolist()[j])
        elif np.abs(cormat.iloc[i, j]) > 0.6:
            highly_correlated_features_dict["feature1"].append(X2.columns.tolist()[i])
            highly_correlated_features_dict["feature2"].append(X2.columns.tolist()[j])
            highly_correlated_features_dict["correlation value"].append(cormat.iloc[i, j])

highly_correlated_features_df = pd.DataFrame(highly_correlated_features_dict)

i = 0
for feat1, feat2 in zip(highly_correlated_features_df["feature1"], highly_correlated_features_df["feature2"]):
    df[[feat1, feat2]].plot.scatter(x = feat1, y = feat2, figsize = (16, 9), alpha = 0.2)
    plt.title("Scatter plot of " + feat1  + " and " + feat2)
    plt.savefig("E:/plots2/scatter_plot_" + str(i) + ".png")
    plt.close()
    i += 1

highly_correlated_features_df["abs correlation value"] = np.abs(highly_correlated_features_df["correlation value"])
linear_corr_df = highly_correlated_features_df.loc[highly_correlated_features_df["abs correlation value"] > 0.95]

df = X2[list(set(linear_corr_df["feature1"].tolist() + linear_corr_df["feature2"].tolist()))]
df.fillna(-999, inplace = True)

from sklearn.decomposition import PCA
pca = PCA(n_components = 7)
tmp = pca.fit_transform(df)

"""
    
"""
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

#X["target"] = y
with open("E:/full_dataset_22072018.pkl", "rb") as f:
    X = pickle.load(f)

X.replace(-999, np.nan, inplace = True)

X2 = X.select_dtypes(include = [np.number])

# Remove categorical features
occurences_lst = []
for feat in X2.columns.tolist():
    if X2[feat].nunique() < 10:
        occurences_lst.append(feat)

X2.drop(occurences_lst, axis = 1, inplace = True)

for i, feat in enumerate(X2.columns.tolist()):
    if i % 50 == 0:
        print("[" + str(i) + "/" + str(len(X2.columns.tolist())) + "] Saving feature histogram")

    X2[feat].plot.hist(bins = 100)
    plt.title("Histogram of " + feat)
    plt.savefig("E:/plots/histogram_" + str(i) + ".png")
    plt.close()
            
# Compute influence of log
X2["target"] = X["target"]
cormat = X2.corr()

X2 = X.select_dtypes(include = [np.number])
for feat in X2.columns:
    X2[feat + "_log"] = np.log1p(X2[feat])

X2 = X2.filter(regex = ".*_log")
X2["target"] = X["target"]
cormat_log = X2.corr()
cormat_log.index = [item.replace("_log", "") for item in cormat_log.index.tolist()]
tmp = cormat["target"]
tmp2 = cormat_log["target"]
tmp2.name = "target_log"
df = tmp.reset_index().merge(tmp2.reset_index(), how = "inner", on = "index")
df.columns = ["feature", "target", "target_log"]
df.to_csv("E:/log_effect.csv", index = False)
"""

"""
NEED TO REMOVE THESE DUPLICATES!!!
Duplicates: previous_application_credit_length_nunique ; previous_application_credit_length_months_nunique
Duplicates: previous_application_NAME_CONTRACT_TYPE_Cash loans_std ; previous_application_NAME_CASH_LOAN_PURPOSE_XAP_std
Duplicates: previous_application_CODE_REJECT_REASON_CLIENT ; previous_application_NAME_CONTRACT_STATUS_Unused offer
Duplicates: bureau_AMT_CREDIT_SUM_sum ; bureau_TOTAL_CUSTOMER_CREDIT
Duplicates: bureau_AMT_CREDIT_SUM_DEBT_sum ; bureau_TOTAL_CUSTOMER_DEBT
Duplicates: bureau_AMT_CREDIT_SUM_OVERDUE_sum ; bureau_TOTAL_CUSTOMER_OVERDUE
Duplicates: bureau_CREDIT_ENDDATE_BINARY_mean ; bureau_CREDIT_ENDDATE_PERCENTAGE
Duplicates: bureau_DAYS_ENDDATE_DIFF_mean ; bureau_AVG_ENDDATE_FUTURE
Duplicates: bureau_bureau_count ; bureau_nb_bureau_records
Duplicates: bureau_bureau_count ; bureau_BUREAU_LOAN_COUNT
Duplicates: bureau_nb_bureau_records ; bureau_BUREAU_LOAN_COUNT
Duplicates: credit_card_balance_AMT_DRAWINGS_ATM_CURRENT_sum ; credit_card_balance_DRAWINGS_ATM
Duplicates: credit_card_balance_AMT_DRAWINGS_CURRENT_sum ; credit_card_balance_DRAWINGS_TOTAL
Duplicates: credit_card_balance_AMT_DRAWINGS_CURRENT_sum ; credit_card_balance_TOTAL_DRAWINGS
Duplicates: credit_card_balance_CNT_DRAWINGS_CURRENT_sum ; credit_card_balance_NO_DRAWINGS
Duplicates: credit_card_balance_SK_DPD_mean ; credit_card_balance_AVG_DPD
Duplicates: credit_card_balance_SK_DPD_min ; credit_card_balance_SK_DPD_DEF_min
Duplicates: credit_card_balance_DRAWINGS_TOTAL ; credit_card_balance_TOTAL_DRAWINGS
Duplicates: installments_payments_is_installment_overdue_mean ; installments_payments_installment_overdue_ratio
"""


"""from sklearn.manifold import TSNE

nb_samples = 5000
df = X_train[["APARTMENTS_AVG", "APARTMENTS_MEDI", "APARTMENTS_MODE", "BASEMENTAREA_AVG", "BASEMENTAREA_MEDI",
              "BASEMENTAREA_MODE", "COMMONAREA_AVG", "COMMONAREA_MEDI", "COMMONAREA_MODE", "ELEVATORS_AVG",
              "ELEVATORS_MEDI", "ELEVATORS_MODE", "EMERGENCYSTATE_MODE", "ENTRANCES_AVG", "ENTRANCES_MEDI", 
              "ENTRANCES_MODE", "FLOORSMAX_AVG", "FLOORSMAX_MEDI", "FLOORSMAX_MODE", "FLOORSMIN_AVG", 
              "FLOORSMIN_MEDI", "FLOORSMIN_MODE", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "LANDAREA_AVG", 
              "LANDAREA_MEDI", "LANDAREA_MODE", "LIVINGAPARTMENTS_AVG", "LIVINGAPARTMENTS_MEDI", 
              "LIVINGAPARTMENTS_MODE", "LIVINGAREA_AVG", "LIVINGAREA_MEDI", "LIVINGAREA_MODE", 
              "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAPARTMENTS_MODE", 
              "NONLIVINGAREA_AVG", "NONLIVINGAREA_MEDI", "NONLIVINGAREA_MODE", "TOTALAREA_MODE", 
              "WALLSMATERIAL_MODE", "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BEGINEXPLUATATION_MEDI", 
              "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BUILD_AVG", "YEARS_BUILD_MEDI", "YEARS_BUILD_MODE"]]
df = df.select_dtypes(include = np.number)
df.fillna(-1, inplace = True)
tsne = TSNE(learning_rate = 25, perplexity = 40, n_iter = 4000, early_exaggeration = 5.0)
tsne_output_df = pd.DataFrame(tsne.fit_transform(df.head(nb_samples), y_train.head(nb_samples)), index = y_train.head(nb_samples).index, columns = ["F1", "F2"])
tsne_output_df["target"] = y_train.head(nb_samples)
tsne_output_df.plot.scatter(x = "F1", y = "F2", c = tsne_output_df["target"], cmap = "PiYG", alpha = 0.8)
plt.show()
"""

"""
X_train["target"] = y_train
X_train2 = pd.concat([X_train.loc[X_train["target"] == 0].sample(5), X_train.loc[X_train["target"] == 1].sample(5)], axis = 0)

bureau_data_df.columns = ["bureau_" + c if c != "SK_ID_CURR" and c != "SK_ID_BUREAU" else c for c in bureau_data_df.columns]
X_train2 = X_train2.merge(bureau_data_df, how = "left", on = "SK_ID_CURR")
bureau_balance_data_df.columns = ["bureau_balance_" + c if c != "SK_ID_CURR" and c != "SK_ID_BUREAU" else c for c in bureau_balance_data_df.columns]
X_train2 = X_train2.merge(bureau_balance_data_df, how = "left", on = "SK_ID_BUREAU")

X_train2 = X_train2.merge(previous_application_data_df, how = "left", on = "SK_ID_CURR")
X_train2 = X_train2.merge(pos_cash_balance_data_df, how = "left", on = "SK_ID_CURR")
X_train2 = X_train2.merge(installments_payments_data_df, how = "left", on = "SK_ID_CURR")
X_train2 = X_train2.merge(credit_card_balance_data_df, how = "left", on = "SK_ID_CURR")"""

"""
X_train["is_train"] = 1
X_test["is_train"] = 0
df = pd.concat([X_train[["is_train", "NAME_CONTRACT_TYPE_y"]], X_test[["is_train", "NAME_CONTRACT_TYPE_y"]]], axis = 0)
pd.crosstab(df["is_train"], df["NAME_CONTRACT_TYPE_y"]).div(df["is_train"].value_counts(ascending = True).values, axis = 0)
"""

"""
i = 0
for feat1 in final_dataset_df.columns:
    nb_levels = final_dataset_df[feat1].nunique()

    if nb_levels > 10:
        print("Plotting", feat1, "...")
        try:
            final_dataset_df[feat1].plot.hist(bins = 100, figsize = (16, 9))
        except:
            #sns.countplot(final_dataset_df[feat1])
            print(feat1, "failed!")

        plt.title("Histogram plot of " + feat1)
        plt.savefig("E:/plots2/histogram_plot_" + str(i) + ".png")
        plt.close()
        i += 1
    else:
        print(feat1, "only have", nb_levels, "levels")
"""
