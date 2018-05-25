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
# Date: 2017-12-09                                                            #
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

# Call to main
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    
    # Set the seed of numpy's PRNG
    np.random.seed(2017)

    enable_validation = False

    # Load the data; y_test is None when 'enable_validation' is False
    X_train, X_test, y_train, y_test, bureau_data_df, bureau_balance_data_df, credit_card_balance_data_df, installments_payments_data_df, pos_cash_balance_data_df, previous_application_data_df = load_data(TRAINING_DATA_str, TESTING_DATA_str, BUREAU_DATA_str, BUREAU_BALANCE_DATA_str, CREDIT_CARD_BALANCE_DATA_str, INSTALLMENTS_PAYMENTS_DATA_str, POS_CASH_BALANCE_DATA_str, PREVIOUS_APPLICATION_DATA_str, enable_validation, "TARGET", CACHE_DIR_str)
    
    print("Train shape: ", X_train.shape)
    print("Test shape: ", X_test.shape)

    additional_files_preprocessor = AdditionalFilesPreprocessingStep()
    final_dataset_df = additional_files_preprocessor.fit_transform(bureau_data_df, bureau_balance_data_df, credit_card_balance_data_df, installments_payments_data_df, pos_cash_balance_data_df, previous_application_data_df)

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
          "max_depth" : 9,
          "objective": "binary",
          "num_leaves": 70,
          "learning_rate": 0.015,
          "max_bin": 512,
          "subsample_for_bin": 200,
          "subsample": 1,
          "subsample_freq": 1,
          "colsample_bytree": 0.7,
          "reg_alpha": 5,
          "reg_lambda": 10,
          "min_split_gain": 0.5,
          "min_child_weight": 19,
          "min_child_samples": 5,
          "scale_pos_weight": 1,
          "metric" : "auc",
          "verbosity": -1
    }

    # ("ConstantFeaturesRemover", ConstantFeaturesRemover()), ("DuplicatedFeaturesRemover", DuplicatedFeaturesRemover()),
    # ("PairwiseNumericalInteractionsGenerator", PairwiseNumericalInteractionsGenerator(columns_names_lst = ["AMT_GOODS_PRICE", "FLOORSMAX_MEDI", "EXT_SOURCE_2", "CNT_FAM_MEMBERS", "NONLIVINGAREA_MEDI", "AMT_REQ_CREDIT_BUREAU_MON", "CNT_CHILDREN", "COMMONAREA_AVG", "APARTMENTS_MEDI", "COMMONAREA_MODE", "NONLIVINGAREA_MODE", "ENTRANCES_MEDI", "NONLIVINGAREA_AVG", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_WEEK", "APARTMENTS_MODE", "AMT_ANNUITY", "YEARS_BEGINEXPLUATATION_AVG", "ELEVATORS_MEDI", "APARTMENTS_AVG", "BASEMENTAREA_AVG", "FLOORSMIN_MEDI", "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BEGINEXPLUATATION_MEDI", "FLOORSMIN_AVG"])),
    main_pipeline = Pipeline([
                                       ("PreprocessingStep", PreprocessingStep(additional_data_lst = [final_dataset_df])),
                                       ("MissingValuesImputer", MissingValuesImputer(num_col_imputation = -999, cat_col_imputation = "NA")),
                                       ("CategoricalFeaturesEncoder", CategoricalFeaturesEncoder(columns_to_be_encoded_lst, encoders_lst)),
                                       ("VarianceFeatureSelector", VarianceFeatureSelector(3e-5)),
                                       ("LGBMFeatureSelector", LGBMFeatureSelector(threshold = 0.762, problem_type = "classification", enable_cv = False, lgbm_params = lgb_params)),
                                       ("LightGBM", LGBMClassifier(lgb_params, early_stopping_rounds = 150, test_size = 0.15, verbose_eval = 100, nrounds = 10000, enable_cv = False))
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