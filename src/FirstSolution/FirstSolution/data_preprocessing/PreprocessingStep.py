#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the Home Credit Default Risk Challenge                   #
#                                                                             #
# This file contains the code needed for the second preprocessing step.       #
# Developped using Python 3.6.                                                #
#                                                                             #
# Authors: Thomas SELECK and Fabien VAVRAND                                   #
# e-mail: thomas.seleck@outlook.fr and fabien.vavrand@gmail.com               #
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

        # Generating some features from EXT_SOURCE_*
        X["EXT_SOURCE_1_*_EXT_SOURCE_2_*_EXT_SOURCE_3"] = X["EXT_SOURCE_1"] * X["EXT_SOURCE_2"] * X["EXT_SOURCE_3"]
        X["ext_sources_mean"] = np.nanmean(X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]], axis = 1)
        X["ext_sources_median"] = np.nanmedian(X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]], axis = 1)
        X["ext_sources_std"] = np.nanstd(X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]], axis = 1)
        X["ext_sources_sum"] = np.nansum(X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]], axis = 1)
        X["ext_sources_nb_missing"] = X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isnull().sum(axis = 1)
        X["ext_sources_range"] = np.abs(X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].max(axis = 1) - X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis = 1))

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
        X["age_ge_60"] = (X["age"] >= 60).astype(np.int8)
        X["binned_age"] = 6 * X["age_lt_25"] + 5 * X["age_25_30"] + 4 * X["age_30_40"] + 3 * X["age_40_50"] + 2 * X["age_50_60"] + X["age_ge_60"]
        X.drop("age_ge_60", axis = 1, inplace = True)

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
        X.drop(["nb_working_years_5_10", "nb_working_years_10_20", "nb_working_years_20_30"], axis = 1, inplace = True)

        # Create a dummy indicating if the client has a job
        #X["is_employed"] = (X["DAYS_EMPLOYED"] >= 0).astype(np.int8)

        # Create some interactions related to 'DAYS_LAST_PHONE_CHANGE'
        X["phone_to_birth_ratio"] = X["DAYS_LAST_PHONE_CHANGE"] / X["DAYS_BIRTH"]
        X["phone_to_employ_ratio"] = X["DAYS_LAST_PHONE_CHANGE"] / X["DAYS_EMPLOYED"]

        X["employment_0_1000"] = ((X["DAYS_EMPLOYED"] > -1000) & (X["DAYS_EMPLOYED"] < 0)).astype(np.int8)
        X["employment_1000_2000"] = ((X["DAYS_EMPLOYED"] > -2000) & (X["DAYS_EMPLOYED"] < -1000)).astype(np.int8)
        X["employment_2000_5000"] = ((X["DAYS_EMPLOYED"] > -5000) & (X["DAYS_EMPLOYED"] < -2000)).astype(np.int8)
        X["employment_gt_5000"] = (X["DAYS_EMPLOYED"] < -5000).astype(np.int8)
        X["employment_group"] = 1 * X["employment_0_1000"] + 2 * X["employment_1000_2000"] + 3 * X["employment_2000_5000"] + 4 * X["employment_gt_5000"]
        X["employment_group"] = X["employment_group"].map({0: 0.054307, 1: 0.111332, 2: 0.095376, 3: 0.068404, 4: 0.047204}) # Mean value of the target for each group

        # Compute interactions between income and annuity
        X["diff_income_annuity"] = X["AMT_INCOME_TOTAL"] - X["AMT_ANNUITY"]
        X["annuity_income_ratio"] = X["AMT_ANNUITY"] / X["AMT_INCOME_TOTAL"]

        # How much times income does the credit represents
        X["credit_income_ratio"] = X["AMT_CREDIT"] / X["AMT_INCOME_TOTAL"]

        # Bin AMT_CREDIT
        X["AMT_CREDIT_lt_500k"] = (X["AMT_CREDIT"] < 500000).astype(np.int8)
        X["AMT_CREDIT_between_500k_1500k"] = ((X["AMT_CREDIT"] > 500000) & (X["AMT_CREDIT"] < 1500000)).astype(np.int8)
        X["AMT_CREDIT_gt_1500k"] = (X["AMT_CREDIT"] > 1500000).astype(np.int8)

        # How many adult in the family?
        X["nb_adults"] = X["CNT_FAM_MEMBERS"] - X["CNT_CHILDREN"]
        X["children_ratio"] = X["CNT_CHILDREN"] / X["CNT_FAM_MEMBERS"]
        X["income_per_child"] = X["AMT_INCOME_TOTAL"] / (1 + X["CNT_CHILDREN"])

        # Income per person
        X["income_per_person"] = X["AMT_INCOME_TOTAL"] / X["CNT_FAM_MEMBERS"]

        # Credit amount per person
        X["credit_amount_per_person"] = X["AMT_CREDIT"] / X["CNT_FAM_MEMBERS"]
        X["credit_annuity_per_person"] = X["AMT_ANNUITY"] / X["CNT_FAM_MEMBERS"]

        """
        # Remaining cash per person
        X["remaining_cash_per_person"] = (X["AMT_INCOME_TOTAL"] - X["AMT_ANNUITY"]) / X["CNT_FAM_MEMBERS"]
        """

        # Number of annuities
        X["nb_annuities"] = X["AMT_CREDIT"] / X["AMT_ANNUITY"]

        # Age of person at the end of the credit
        X["credit_end_age"] = X["age"] + X["nb_annuities"]

        # Try to deanonymize ELEVATORS features
        X["ELEVATORS_AVG"] = X["ELEVATORS_AVG"] // 0.04
        X["ELEVATORS_MEDI"] = X["ELEVATORS_MEDI"] // 0.04
        X["ELEVATORS_MODE"] = X["ELEVATORS_MODE"] // 0.04

        # Try to deanonymize ENTRANCES features
        X["ENTRANCES_AVG"] = X["ENTRANCES_AVG"] // 0.0345
        X["ENTRANCES_MEDI"] = X["ENTRANCES_MEDI"] // 0.0345
        X["ENTRANCES_MODE"] =X["ENTRANCES_MODE"] // 0.0345

        """
        # Deanonymize floors-related features ; Strange: FLOORSMIN > FLOORSMAX in most cases
        X["FLOORSMIN_MODE"] = round(X["FLOORSMIN_MODE"] / 0.0208, 0).astype(np.int8)
        X["FLOORSMAX_MODE"] = round(X["FLOORSMAX_MODE"] / 0.0208, 0).astype(np.int8)
        X["FLOORSMIN_MEDI"] = round(X["FLOORSMIN_MEDI"] / 0.0208, 0).astype(np.int8)
        X["FLOORSMAX_MEDI"] = round(X["FLOORSMAX_MEDI"] / 0.0208, 0).astype(np.int8)
        X["FLOORSMIN_MEDI"] = round(X["FLOORSMIN_MEDI"] / 0.0208, 0).astype(np.int8)
        X["FLOORSMAX_MEDI"] = round(X["FLOORSMAX_MEDI"] / 0.0208, 0).astype(np.int8)

        # Deanonymize ELEVATORS features
        X["ELEVATORS_AVG"] = round(X["ELEVATORS_AVG"] / 0.0403, 0).astype(np.int8)
        X["ELEVATORS_MEDI"] = round(X["ELEVATORS_MEDI"] / 0.0403, 0).astype(np.int8)
        X["ELEVATORS_MODE"] = round(X["ELEVATORS_MODE"] / 0.0403, 0).astype(np.int8)

        # Deanonymize ENTRANCES features
        X["ENTRANCES_AVG"] = round(X["ENTRANCES_AVG"] / 0.0172, 0).astype(np.int8)
        X["ENTRANCES_MEDI"] = round(X["ENTRANCES_MEDI"] / 0.0172, 0).astype(np.int8)
        X["ENTRANCES_MODE"] = round(X["ENTRANCES_MODE"] / 0.0172, 0).astype(np.int8)

        # Deanonymize YEARS_BUILD
        X["YEARS_BUILD_AVG"] = ((round(X["YEARS_BUILD_AVG"] / 0.0004, 0) - 1) / 17).astype(np.int16)
        X["YEARS_BUILD_MEDI"] = ((round(X["YEARS_BUILD_MEDI"] / 0.0003, 0) - 1) / 22.66667).astype(np.int16)
        X["YEARS_BUILD_MODE"] = ((round(X["YEARS_BUILD_MODE"] / 0.0003, 0) - 1) / 22.66667).astype(np.int16)

        # Deanonymize NONLIVINGAPARTMENTS
        X["NONLIVINGAPARTMENTS_MODE"] = round(X["NONLIVINGAPARTMENTS_MODE"] / 0.0039, 0).astype(np.int16)
        X["NONLIVINGAPARTMENTS_MEDI"] = round(X["NONLIVINGAPARTMENTS_MEDI"] / 0.0019, 0).astype(np.int16)
        X["NONLIVINGAPARTMENTS_AVG"] = round(X["NONLIVINGAPARTMENTS_AVG"] / 0.0019, 0).astype(np.int16)

        # Deanonymize TOTALAREA
        X["TOTALAREA_MODE"] = round(X["TOTALAREA_MODE"] / 0.0001, 0).astype(np.int16)

        # Find something for YEARS_BEGINEXPLUATATION_MODE and YEARS_BEGINEXPLUATATION_MEDI
        # Deanonymize YEARS_BEGINEXPLUATATION features
        X["YEARS_BEGINEXPLUATATION_AVG"] = round(X["YEARS_BEGINEXPLUATATION_AVG"] / 0.0179, 0).astype(np.int8)
        X["YEARS_BEGINEXPLUATATION_MEDI"] = round(X["YEARS_BEGINEXPLUATATION_MEDI"] / 0.0179, 0).astype(np.int8)
        X["YEARS_BEGINEXPLUATATION_MODE"] = ((round(X["YEARS_BEGINEXPLUATATION_MODE"] / 0.0005, 0) - 1) / 35.8).astype(np.int8)
        """

        # Number of documents the client gave
        X["number_of_provided_documents"] = X.filter(regex = "FLAG_DOCUMENT_.*").sum(axis = 1)

        # Ratio between age and years of work
        X["age_to_work_ratio"] = X["DAYS_EMPLOYED"] / X["DAYS_BIRTH"]

        # Did the client subscribe to an insurance?
        X["subscribed_to_insurance"] = (X["AMT_CREDIT"] > X["AMT_GOODS_PRICE"]).astype(np.int8)
        X["insurance_amount"] = X["AMT_CREDIT"] - X["AMT_GOODS_PRICE"]
        X["insurance_percentage_goods_price"] = (X["insurance_amount"] / X["AMT_GOODS_PRICE"]) * 100
        X["insurance_percentage_total_amount"] = (X["insurance_amount"] / X["AMT_CREDIT"]) * 100
        #X["insurance_percentage_goods_price_lt_0_or_gt_100"] = ((X["insurance_percentage_goods_price"] < 0) | (X["insurance_percentage_goods_price"] > 100)).astype(np.int8)

        # Distribution of cars older than 60 years are strange. Maybe is a default value
        #X["is_car_older_than_60y"] = (X["OWN_CAR_AGE"] > 60).astype(np.int8)

        X["car_to_birth_ratio"] = X["OWN_CAR_AGE"] / X["DAYS_BIRTH"]
        X["car_to_employ_ratio"] = X["OWN_CAR_AGE"] / X["DAYS_EMPLOYED"]

        # Generate some interactions
        X["OWN_CAR_AGE_-_DAYS_EMPLOYED"] = X["OWN_CAR_AGE"] * 365.25 - X["DAYS_EMPLOYED"]
        X["OWN_CAR_AGE_*_DAYS_EMPLOYED"] = X["OWN_CAR_AGE"] * X["DAYS_EMPLOYED"]
        X["DAYS_ID_PUBLISH_*_DAYS_LAST_PHONE_CHANGE"] = X["DAYS_ID_PUBLISH"] * X["DAYS_LAST_PHONE_CHANGE"]
        X["DAYS_ID_PUBLISH_-_DAYS_LAST_PHONE_CHANGE"] = X["DAYS_ID_PUBLISH"] - X["DAYS_LAST_PHONE_CHANGE"]
        X["DAYS_ID_PUBLISH_*_DAYS_LAST_PHONE_CHANGE_*_DAYS_REGISTRATION"] = X["DAYS_ID_PUBLISH"] * X["DAYS_LAST_PHONE_CHANGE"] * X["DAYS_REGISTRATION"]
        X["age_*_nb_annuities"] = X["age"] * X["nb_annuities"]

        # Merge additional data to main dataframe
        print("    Merging additional data to main dataframe...")
        #X["SK_ID_CURR"] = X.index
        X = X.reset_index()
        X = X.merge(self._final_dataset_df, how = "left", on = "SK_ID_CURR")
        X.index = X["SK_ID_CURR"] # => This influences model AUC. Why ?
                
        # Drop ID        
        X.drop("SK_ID_CURR", axis = 1, inplace = True)

        """
        X["total_credit_amount"] = X["bureau_TOTAL_ACTIVE_CUSTOMER_CREDIT"] + X["AMT_CREDIT"]
        X["total_annuity"] = X["bureau_TOTAL_ACTIVE_AMT_ANNUITY"] + X["AMT_ANNUITY"]
        X["AMT_INCOME_TOTAL_-_total_annuity"] = X["AMT_INCOME_TOTAL"] - X["total_annuity"]
        X["total_annuity_/_AMT_INCOME_TOTAL"] = X["total_annuity"] / X["AMT_INCOME_TOTAL"]

        # Remaining cash per person (after removing CB annuities)
        X["remaining_cash_per_person_after_CB"] = (X["AMT_INCOME_TOTAL"] - X["total_annuity"]) / X["CNT_FAM_MEMBERS"]
        """

        # Remove features with many missing values
        print("    Removing features with more than 85% missing...")
        if self._useful_features_lst == None:
            self._useful_features_lst = X.columns[X.isnull().mean() < 0.85].tolist()

        X = X[self._useful_features_lst]
        
        print("Preprocessing data... done")

        return X