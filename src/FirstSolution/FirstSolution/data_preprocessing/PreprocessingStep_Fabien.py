#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the Home Credit Default Risk Challenge                   #
#                                                                             #
# This file contains the code needed for the second preprocessing step.       #
# Developped using Python 3.6.                                                #
#                                                                             #
# Author: Fabien VAVRAND                                                      #
# e-mail: <unknown>                                                           #
# Date: 2018-05-27                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import numpy as np
import pandas as pd
import time

from sklearn.base import BaseEstimator, TransformerMixin

from feature_processing.categorical_encoders import CategoricalFeaturesEncoder, OrdinalEncoder, GroupingEncoder, LeaveOneOutEncoder, TargetAvgEncoder

class PreprocessingStep(BaseEstimator, TransformerMixin):
    """
    This class defines the first preprocessing step.
    """

    def __init__(self, additional_data_lst):
        """
        This is the class' constructor.

        Parameters
        ----------
        additional_data_lst : list
                List containing additional dataset we will use for data preprocessing.

        Returns
        -------
        None
        """

        # Unpack datasets
        """self.bureau_data_df = additional_data_lst[0]
        self.bureau_balance_data_df = additional_data_lst[1]
        self.credit_card_balance_data_df = additional_data_lst[2]
        self.installments_payments_data_df = additional_data_lst[3]
        self.pos_cash_balance_data_df = additional_data_lst[4]
        self.previous_application_data_df = additional_data_lst[5]"""
        self._final_dataset_df = additional_data_lst[0]
        self._useful_features_lst = None
                                      
    def fit(self, X, y = None):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series (optional)
                This is the target associated with the X data.

        Returns
        -------
        None
        """
        
        return self

    def transform(self, X):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.
                
        Returns
        -------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.
        """
                    
        print("Preprocessing data...")    
        
        # Count number of missing values by row
        X["missing_values_count"] = X.isnull().sum(axis = 1)

        # When AMT_ANNUITY, CNT_FAM_MEMBERS and DAYS_LAST_PHONE_CHANGE are missing, then the target is always 0. 
        X["AMT_ANNUITY_is_missing"] = X["AMT_ANNUITY"].isnull().astype(np.int8)
        X["CNT_FAM_MEMBERS_is_missing"] = X["CNT_FAM_MEMBERS"].isnull().astype(np.int8)
        X["DAYS_LAST_PHONE_CHANGE_is_missing"] = X["DAYS_LAST_PHONE_CHANGE"].isnull().astype(np.int8)

        # Compute age from "DAYS_BIRTH"
        X["age"] = -X["DAYS_BIRTH"] // 365.25
        X["age_lt_25"] = (X["age"] < 25).astype(np.int8)
        X["age_25_30"] = ((X["age"] >= 25) & (X["age"] < 30)).astype(np.int8)
        X["age_30_40"] = ((X["age"] >= 30) & (X["age"] < 40)).astype(np.int8)
        X["age_40_50"] = ((X["age"] >= 40) & (X["age"] < 50)).astype(np.int8)
        X["age_50_60"] = ((X["age"] >= 50) & (X["age"] < 60)).astype(np.int8)
        #X["age_ge_60"] = (X["age"] >= 60).astype(np.int8)
        X["binned_age"] = 6 * X["age_lt_25"] + 5 * X["age_25_30"] + 4 * X["age_30_40"] + 3 * X["age_40_50"] + 2 * X["age_50_60"] #+ X["age_ge_60"]

        # Compute number working years
        X["nb_working_years"] = -X["DAYS_EMPLOYED"] // 365.25
        #X["is_working_for_1000_years"] = (X["DAYS_EMPLOYED"] > 300000).astype(np.int8)
        X["nb_working_years_lt_5"] = (X["nb_working_years"] < 5).astype(np.int8)
        X["nb_working_years_5_10"] = ((X["nb_working_years"] >= 5) & (X["nb_working_years"] < 10)).astype(np.int8)
        X["nb_working_years_10_20"] = ((X["nb_working_years"] >= 10) & (X["nb_working_years"] < 20)).astype(np.int8)
        X["nb_working_years_20_30"] = ((X["nb_working_years"] >= 20) & (X["nb_working_years"] < 30)).astype(np.int8)
        #X["nb_working_years_30_40"] = ((X["nb_working_years"] >= 30) & (X["nb_working_years"] < 40)).astype(np.int8)
        #X["nb_working_years_ge_40"] = (X["nb_working_years"] >= 40).astype(np.int8)
        X["binned_nb_working_years"] = 6 * X["nb_working_years_lt_5"] + 5 * X["nb_working_years_5_10"] + 4 * X["nb_working_years_10_20"] + 3 * X["nb_working_years_20_30"] #+ 2 * X["nb_working_years_30_40"] + X["nb_working_years_ge_40"]

        # Compute interactions between income and annuity
        X["diff_income_annuity"] = X["AMT_INCOME_TOTAL"] - X["AMT_ANNUITY"]
        X["annuity_income_ratio"] = X["AMT_ANNUITY"] / X["AMT_INCOME_TOTAL"]

        # How much times income does the credit represents
        X["credit_income_ratio"] = X["AMT_CREDIT"] / X["AMT_INCOME_TOTAL"]

        # Is your income < 700k? => In this case you have ~10% default rate
        #X["income_lt_700k"] = (X["AMT_INCOME_TOTAL"] < 700000).astype(np.int8)

        # How many adult in the family?
        X["nb_adults"] = X["CNT_FAM_MEMBERS"] - X["CNT_CHILDREN"]

        # Try to deanonymize ELEVATORS features
        X["ELEVATORS_AVG"] = X["ELEVATORS_AVG"] // 0.04
        X["ELEVATORS_MEDI"] = X["ELEVATORS_MEDI"] // 0.04
        X["ELEVATORS_MODE"] = X["ELEVATORS_MODE"] // 0.04

        # Try to deanonymize ENTRANCES features
        X["ENTRANCES_AVG"] = X["ENTRANCES_AVG"] // 0.0345
        X["ENTRANCES_MEDI"] = X["ENTRANCES_MEDI"] // 0.0345
        X["ENTRANCES_MODE"] =X["ENTRANCES_MODE"] // 0.0345

        # Number of documents the client gave
        X["number_of_provided_documents"] = X.filter(regex = "FLAG_DOCUMENT_.*").sum(axis = 1)

        # Ratio between age and years of work
        X["age_to_work_ratio"] = X["DAYS_EMPLOYED"] / X["DAYS_BIRTH"]

        # Merge additional data to main dataframe
        #X = X.merge(self._final_dataset_df, how = "left", left_index = True, right_on = "SK_ID_CURR")

        #X["SK_ID_CURR"] = X.index
        X = X.reset_index()
        X = X.merge(self._final_dataset_df, how = "left", on = "SK_ID_CURR")
        X.index = X["SK_ID_CURR"] # => This influences model AUC. Why ?
                
        # Drop ID        
        X.drop("SK_ID_CURR", axis = 1, inplace = True)

        # Remove features with many missing values
        print("    Removing features with more than 85% missing...")
        if self._useful_features_lst == None:
            self._useful_features_lst = X.columns[X.isnull().mean() < 0.85].tolist()

        X = X[self._useful_features_lst]
        
        print("Preprocessing data... done")

        return X