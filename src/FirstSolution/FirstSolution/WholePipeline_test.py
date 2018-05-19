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

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from data_preprocessing.PreprocessingStep import PreprocessingStep
from feature_processing.categorical_encoders import CategoricalFeaturesEncoder, OrdinalEncoder, GroupingEncoder, LeaveOneOutEncoder, TargetAvgEncoder
from feature_engineering.features_selectors import VarianceFeatureSelector, L1NormFeatureSelector, LGBMFeatureSelector, ConstantFeaturesRemover, DuplicatedFeaturesRemover
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
    X_train, X_test, y_train, y_test = load_data(TRAINING_DATA_str, TESTING_DATA_str, BUREAU_DATA_str, BUREAU_BALANCE_DATA_str, CREDIT_CARD_BALANCE_DATA_str, INSTALLMENTS_PAYMENTS_DATA_str, POS_CASH_BALANCE_DATA_str, PREVIOUS_APPLICATION_DATA_str, enable_validation, "TARGET")
    
    print("Train shape: ", X_train.shape)
    print("Test shape: ", X_test.shape)

    columns_to_be_encoded_lst = ["NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "EMERGENCYSTATE_MODE", 
                                 "CODE_GENDER", "HOUSETYPE_MODE", "FONDKAPREMONT_MODE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "NAME_TYPE_SUITE", "WEEKDAY_APPR_PROCESS_START", "WALLSMATERIAL_MODE", "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"]
    encoders_lst = [LabelBinarizer() for _ in columns_to_be_encoded_lst]

    lgb_params = {
        "learning_rate": 0.005,
        "application": "binary",
        "max_depth": 7,
        "num_leaves": 80,
        "verbosity": -1,
        "metric": "auc",
        "subsample": 0.9,
        "colsample_bytree": 0.8
    }

    main_pipeline = Pipeline([("ConstantFeaturesRemover", ConstantFeaturesRemover()),
                              ("PreprocessingStep", PreprocessingStep()),
                              ("CategoricalFeaturesEncoder", CategoricalFeaturesEncoder(columns_to_be_encoded_lst, encoders_lst)),
                              ("LightGBM", LGBMClassifier(lgb_params, early_stopping_rounds = 20, random_state = 144, test_size = 0.15, verbose_eval = 100, nrounds = 10000, enable_cv = False))
                             ])
    
    # Train the model
    main_pipeline.fit(X_train, y_train)
    
    # Make predictions
    predictions_npa = main_pipeline.predict(X_test)

    # Evaluate the model
    if enable_validation:
        print("Validation AUC:",roc_auc_score(y_test, predictions_npa))
    else:
        predictions_df = pd.DataFrame({"SK_ID_CURR": X_test.index, "TARGET": predictions_npa})
        predictions_df.to_csv(PREDICTIONS_DIR_str + "first_solution_submission.csv", index = False)

    # Stop the timer and print the exectution time
    print("*** Test finished : Executed in:", time.time() - start_time, "seconds ***")