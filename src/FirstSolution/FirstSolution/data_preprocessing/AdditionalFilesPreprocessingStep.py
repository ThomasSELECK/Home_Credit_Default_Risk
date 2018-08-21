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
import gc
import multiprocessing as mp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression

from feature_processing.categorical_encoders import CategoricalFeaturesEncoder, OrdinalEncoder, GroupingEncoder, LeaveOneOutEncoder, TargetAvgEncoder

class AdditionalFilesPreprocessingStep(BaseEstimator, TransformerMixin):
    """
    This class defines the first preprocessing step.
    """

    def __init__(self):
        """
        This is the class' constructor.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.n_jobs = 16

    def create_trend_feature(self, feature_sr):
        y = feature_sr.values
        try:
            x = np.arange(0, len(y)).reshape(-1, 1)
            lr = LinearRegression()
            lr.fit(x, y)
            trend = lr.coef_[0]
        except:
            trend = np.nan

        return trend

    def _flatten_columns_names(self, df):
        """
        This method flattens the columns names of a data frame by concatening the MultiIndex entry.

        Parameters
        ----------
        df : Pandas DataFrame
                Data Frame that must have its columns names flattened.

        Returns
        -------
        df : Pandas Data Frame
                Data Frame with flattened columns names.
        """

        flatten_columns_names = []
        for feature in df.columns.levels[0]:
            for agg_method in df.columns.levels[1]:
                flatten_columns_names.append(feature + "_" + agg_method)

        df.columns = flatten_columns_names

        return df
                                      
    def fit(self, bureau_data_df, bureau_balance_data_df, credit_card_balance_data_df, installments_payments_data_df, pos_cash_balance_data_df, previous_application_data_df):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        bureau_data_df : pd.DataFrame
                Additional data frame.
                
        bureau_balance_data_df : pd.DataFrame
                Additional data frame.
                
        credit_card_balance_data_df : pd.DataFrame
                Additional data frame.

        installments_payments_data_df : pd.DataFrame
                Additional data frame.

        pos_cash_balance_data_df : pd.DataFrame
                Additional data frame.

        previous_application_data_df : pd.DataFrame
                Additional data frame.
                
        Returns
        -------
        None
        """

        # Get list of categorical columns for each dataset and create encoders for encoding categorical variables
        self._previous_application_categ_feats_lst = ["NAME_CONTRACT_TYPE", "WEEKDAY_APPR_PROCESS_START", "FLAG_LAST_APPL_PER_CONTRACT", "NAME_CASH_LOAN_PURPOSE", "NAME_PAYMENT_TYPE", 
                                                      "NAME_TYPE_SUITE", "NAME_CLIENT_TYPE", "NAME_GOODS_CATEGORY", "NAME_PORTFOLIO", "NAME_PRODUCT_TYPE", "CHANNEL_TYPE", 
                                                      "NAME_SELLER_INDUSTRY", "NAME_YIELD_GROUP", "PRODUCT_COMBINATION"]
        self._previous_application_encoders_lst = [LabelBinarizer(), LabelBinarizer(), OrdinalEncoder(), LabelBinarizer(), LabelBinarizer(), 
                                                   LabelBinarizer(), LabelBinarizer(), LabelBinarizer(), LabelBinarizer(), LabelBinarizer(), LabelBinarizer(), 
                                                   LabelBinarizer(), LabelBinarizer(), LabelBinarizer()]

        self._bureau_categ_feats_lst = ["CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE"]
        self._bureau_encoders_lst = [LabelBinarizer(), OrdinalEncoder(), GroupingEncoder(LabelBinarizer(), 4)]

        self._credit_card_balance_categ_feats_lst = credit_card_balance_data_df.select_dtypes(["object"]).columns.tolist()
        self._credit_card_balance_encoders_lst = [OrdinalEncoder() for _ in self._credit_card_balance_categ_feats_lst]

        self._pos_cash_balance_categ_feats_lst = pos_cash_balance_data_df.select_dtypes(["object"]).columns.tolist()
        self._pos_cash_balance_encoders_lst = [OrdinalEncoder() for _ in self._pos_cash_balance_categ_feats_lst]

        self._previous_application_cfe = CategoricalFeaturesEncoder(self._previous_application_categ_feats_lst, self._previous_application_encoders_lst)
        self._bureau_cfe = CategoricalFeaturesEncoder(self._bureau_categ_feats_lst, self._bureau_encoders_lst, drop_initial_features = False)
        self._credit_card_balance_cfe = CategoricalFeaturesEncoder(self._credit_card_balance_categ_feats_lst, self._credit_card_balance_encoders_lst, drop_initial_features = False)
        self._pos_cash_balance_cfe = CategoricalFeaturesEncoder(self._pos_cash_balance_categ_feats_lst, self._pos_cash_balance_encoders_lst)

        # Processing 'bureau.csv'

        # Encode categorical features
        self._bureau_cfe.fit(bureau_data_df)

        # Processing 'previous_application.csv'

        # Impute missing values
        previous_application_data_df["NAME_TYPE_SUITE"].fillna("missing", inplace = True)

        # Encode categorical features
        self._previous_application_cfe.fit(previous_application_data_df)

        # Processing 'credit_card_balance_data_df.csv'

        # Encode categorical features
        self._credit_card_balance_cfe.fit(credit_card_balance_data_df)

        # Processing 'pos_cash_balance_data_df.csv'

        # Encode categorical features
        self._pos_cash_balance_cfe.fit(pos_cash_balance_data_df)
        
        return self

    def transform_bureau_balance_data(self, bureau_data_df, bureau_balance_data_df):
        st = time.time()
        print("    Pre-processing 'bureau_balance.csv'...")
        bureau_balance_stats_df = self._flatten_columns_names(bureau_balance_data_df.select_dtypes(include = np.number).groupby("SK_ID_BUREAU").agg(["mean", "std", "min", "max", "sum", "nunique"]))
        bureau_balance_stats2_df = self._flatten_columns_names(bureau_balance_data_df[bureau_balance_data_df.select_dtypes(include = "object").columns.tolist() + ["SK_ID_BUREAU"]].groupby("SK_ID_BUREAU").agg(["nunique"]))
        bureau_balance_status_df = bureau_balance_data_df.groupby("SK_ID_BUREAU")["STATUS"].value_counts(normalize = False)
        bureau_balance_status_df = bureau_balance_status_df.unstack("STATUS")
        bureau_balance_status_df.columns = ["STATUS_0", "STATUS_1", "STATUS_2", "STATUS_3", "STATUS_4", "STATUS_5", "STATUS_C", "STATUS_X"]

        bureau_balance_features_df = pd.concat([bureau_balance_stats_df, bureau_balance_stats2_df, bureau_balance_status_df], axis = 1)
        
        # Add prefix to columns
        bureau_balance_features_df.columns = ["bureau_balance_" + c for c in bureau_balance_features_df.columns.tolist()]

        bureau_data_df = bureau_data_df.join(bureau_balance_features_df, how = "left", on = "SK_ID_BUREAU")

        print("    Pre-processing 'bureau_balance.csv'... done in", round(time.time() - st, 3), "secs")

        return bureau_data_df

    def transform_bureau_data(self, bureau_data_df, bureau_balance_data_df):
        # Processing 'bureau_balance.csv'
        bureau_data_df = self.transform_bureau_balance_data(bureau_data_df, bureau_balance_data_df)

        st = time.time()
        print("    Pre-processing 'bureau.csv'...")

        # Replace outliers with NaNs
        bureau_data_df["DAYS_CREDIT_ENDDATE"].loc[bureau_data_df["DAYS_CREDIT_ENDDATE"] < -5000] = np.nan
        bureau_data_df["DAYS_CREDIT_UPDATE"].loc[bureau_data_df["DAYS_CREDIT_UPDATE"] < -40000] = np.nan
        bureau_data_df["DAYS_ENDDATE_FACT"].loc[bureau_data_df["DAYS_ENDDATE_FACT"] < -40000] = np.nan
        bureau_data_df["AMT_CREDIT_MAX_OVERDUE"].loc[bureau_data_df["AMT_CREDIT_MAX_OVERDUE"] > 200000] = np.nan
        bureau_data_df["AMT_ANNUITY"].loc[bureau_data_df["AMT_ANNUITY"] > 1000000] = np.nan

        # Count number of missing values by row
        bureau_data_df["missing_values_count"] = bureau_data_df.isnull().sum(axis = 1)

        # Encode categorical features
        bureau_data_df = self._bureau_cfe.transform(bureau_data_df)

        # Impute NAs
        bureau_data_df[["AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_MAX_OVERDUE", "DAYS_CREDIT_ENDDATE", "DAYS_CREDIT_UPDATE"]].fillna(0, inplace = True)

        ## Credit duration
        bureau_data_df["credit_duration"] = bureau_data_df["DAYS_CREDIT_ENDDATE"] - bureau_data_df["DAYS_CREDIT"]

        ## Number of days credit ends after initial end date
        bureau_data_df["nb_days_after_enddate"] = bureau_data_df["DAYS_ENDDATE_FACT"] - bureau_data_df["DAYS_CREDIT_ENDDATE"]

        ## Credit usage
        bureau_data_df["credit_usage"] = bureau_data_df["AMT_CREDIT_SUM_DEBT"] / bureau_data_df["AMT_CREDIT_SUM_LIMIT"]

        # How was credit consumed in the past
        ## Average number of days between successive past applications for each customer
        ### Groupby each customer and sort values of DAYS_CREDIT in ascending order
        grp = bureau_data_df[["SK_ID_CURR", "SK_ID_BUREAU", "DAYS_CREDIT"]].groupby(by = ["SK_ID_CURR"])
        grp1 = grp.apply(lambda x: x.sort_values(["DAYS_CREDIT"], ascending = False)).reset_index(drop = True)#rename(index = str, columns = {"DAYS_CREDIT": "DAYS_CREDIT_DIFF"})

        ### Calculate Difference between the number of days 
        grp1["DAYS_CREDIT1"] = grp1["DAYS_CREDIT"] * -1
        grp1["DAYS_DIFF"] = grp1.groupby(by = ["SK_ID_CURR"])["DAYS_CREDIT1"].diff()
        grp1["DAYS_DIFF"] = grp1["DAYS_DIFF"].fillna(0).astype("uint32")
        grp1.drop(["DAYS_CREDIT1", "DAYS_CREDIT", "SK_ID_CURR"], axis = 1, inplace = True)
        gc.collect()

        bureau_data_df = bureau_data_df.merge(grp1, on = ["SK_ID_BUREAU"], how = "left")

        # Potential future delinquencies
        ## Average number of days in which credit expires in the future        
        ### We take only positive values of ENDDATE since we are looking at Bureau Credit VALID IN FUTURE 
        ### as of the date of the customer"s loan application with Home Credit 
        bureau_data_df["CREDIT_ENDDATE_BINARY"] = (bureau_data_df["DAYS_CREDIT_ENDDATE"].apply(lambda x: int(x >= 0))).astype(np.int8)
        B1 = bureau_data_df[bureau_data_df["CREDIT_ENDDATE_BINARY"] == 1]

        ### Calculate Difference in successive future end dates of CREDIT 
        ### Groupby Each Customer ID 
        grp = B1[["SK_ID_CURR", "SK_ID_BUREAU", "DAYS_CREDIT_ENDDATE"]].groupby(by = ["SK_ID_CURR"])
        # Sort the values of CREDIT_ENDDATE for each customer ID 
        grp1 = grp.apply(lambda x: x.sort_values(["DAYS_CREDIT_ENDDATE"], ascending = True)).reset_index(drop = True)
        del grp
        gc.collect()

        # Calculate the Difference in ENDDATES and fill missing values with zero 
        grp1["DAYS_ENDDATE_DIFF"] = grp1.groupby(by = ["SK_ID_CURR"])["DAYS_CREDIT_ENDDATE"].diff()
        grp1["DAYS_ENDDATE_DIFF"] = grp1["DAYS_ENDDATE_DIFF"].fillna(0).astype("uint32")
        grp1.drop(["SK_ID_CURR", "DAYS_CREDIT_ENDDATE"], axis = 1, inplace = True)
        gc.collect()

        ### Merge new feature "DAYS_ENDDATE_DIFF" with original Data frame for BUREAU DATA
        bureau_data_df = bureau_data_df.merge(grp1, on = ["SK_ID_BUREAU"], how = "left")
        del grp1
        gc.collect()

        # Compute average values and counts
        bureau_stats_df = self._flatten_columns_names(bureau_data_df.select_dtypes(include = np.number).drop("SK_ID_BUREAU", axis = 1).groupby("SK_ID_CURR").agg(["mean", "std", "min", "max", "sum", "nunique"]))
        bureau_stats_df["bureau_count"] = bureau_data_df[["SK_ID_BUREAU", "SK_ID_CURR"]].groupby("SK_ID_CURR").count()["SK_ID_BUREAU"]
        bureau_stats_df["nb_bureau_records"] = bureau_data_df.groupby("SK_ID_CURR").size()

        # Computing credit profile for each customer
        ## Number of past loans per customer
        grp = bureau_data_df[["SK_ID_CURR", "DAYS_CREDIT"]].groupby(by = ["SK_ID_CURR"])["DAYS_CREDIT"].count().reset_index().rename(index = str, columns = {"DAYS_CREDIT": "BUREAU_LOAN_COUNT"})
        bureau_stats_df = bureau_stats_df.merge(grp, on = ["SK_ID_CURR"], how = "left")

        ## Number of types of past loans per customer
        grp = bureau_data_df[["SK_ID_CURR", "CREDIT_TYPE"]].groupby(by = ["SK_ID_CURR"])["CREDIT_TYPE"].nunique().reset_index().rename(index = str, columns = {"CREDIT_TYPE": "BUREAU_LOAN_TYPES"})
        bureau_stats_df = bureau_stats_df.merge(grp, on = ["SK_ID_CURR"], how = "left")

        ## Average number of past loans per type of loan per customer
        ### Average Number of Loans per Loan Type
        bureau_stats_df["AVERAGE_LOAN_TYPE"] = bureau_stats_df["BUREAU_LOAN_COUNT"] / bureau_stats_df["BUREAU_LOAN_TYPES"]        
        gc.collect()

        ## Percentage of loans that are active per customer
        ### Create a new dummy column for whether CREDIT is ACTIVE OR CLOSED 
        bureau_data_df["CREDIT_ACTIVE_BINARY"] = (bureau_data_df["CREDIT_ACTIVE"].apply(lambda x: int(x != "Closed"))).astype(np.int8)
        
        ### Calculate mean number of loans that are ACTIVE per CUSTOMER 
        grp = bureau_data_df.groupby(by = ["SK_ID_CURR"])["CREDIT_ACTIVE_BINARY"].mean().reset_index().rename(index = str, columns = {"CREDIT_ACTIVE_BINARY": "ACTIVE_LOANS_PERCENTAGE"})
        bureau_stats_df = bureau_stats_df.merge(grp, on = ["SK_ID_CURR"], how = "left")
        #bureau_stats_df.drop("CREDIT_ACTIVE_BINARY", axis = 1, inplace = True)
        gc.collect()

        # How was credit consumed in the past
        ## Percentage of loans per customer where end data for credit is past
        bureau_data_df["CREDIT_ENDDATE_BINARY"] = (bureau_data_df["DAYS_CREDIT_ENDDATE"].apply(lambda x: int(x >= 0))).astype(np.int8)
        grp = bureau_data_df.groupby(by = ["SK_ID_CURR"])["CREDIT_ENDDATE_BINARY"].mean().reset_index().rename(index = str, columns = {"CREDIT_ENDDATE_BINARY": "CREDIT_ENDDATE_PERCENTAGE"})
        bureau_stats_df = bureau_stats_df.merge(grp, on = ["SK_ID_CURR"], how = "left")
        gc.collect()

        # Potential future delinquencies
        ## Average number of days in which credit expires in the future      
        ### Calculate Average of DAYS_ENDDATE_DIFF
        grp = bureau_data_df[["SK_ID_CURR", "DAYS_ENDDATE_DIFF"]].groupby(by = ["SK_ID_CURR"])["DAYS_ENDDATE_DIFF"].mean().reset_index().rename(index = str, columns = {"DAYS_ENDDATE_DIFF": "AVG_ENDDATE_FUTURE"})
        bureau_stats_df = bureau_stats_df.merge(grp, on = ["SK_ID_CURR"], how = "left")
        del grp 
        bureau_data_df.drop(["CREDIT_ENDDATE_BINARY"], axis = 1, inplace = True)
        gc.collect()

        ## Ratio of total debt to total credit for each customer
        bureau_data_df["AMT_CREDIT_SUM_DEBT"] = bureau_data_df["AMT_CREDIT_SUM_DEBT"].fillna(0)
        bureau_data_df["AMT_CREDIT_SUM"] = bureau_data_df["AMT_CREDIT_SUM"].fillna(0)

        grp1 = bureau_data_df[["SK_ID_CURR", "AMT_CREDIT_SUM_DEBT"]].groupby(by = ["SK_ID_CURR"])["AMT_CREDIT_SUM_DEBT"].sum().reset_index().rename(index = str, columns = { "AMT_CREDIT_SUM_DEBT": "TOTAL_CUSTOMER_DEBT"})
        grp2 = bureau_data_df[["SK_ID_CURR", "AMT_CREDIT_SUM"]].groupby(by = ["SK_ID_CURR"])["AMT_CREDIT_SUM"].sum().reset_index().rename(index = str, columns = { "AMT_CREDIT_SUM": "TOTAL_CUSTOMER_CREDIT"})

        bureau_stats_df = bureau_stats_df.merge(grp1, on = ["SK_ID_CURR"], how = "left")
        bureau_stats_df = bureau_stats_df.merge(grp2, on = ["SK_ID_CURR"], how = "left")
        del grp2
        gc.collect()

        bureau_stats_df["DEBT_CREDIT_RATIO"] = bureau_stats_df["TOTAL_CUSTOMER_DEBT"] / bureau_stats_df["TOTAL_CUSTOMER_CREDIT"]

        ## Total amount of active credit
        bureau_data_df["ACTIVE_AMT_CREDIT_SUM"] = bureau_data_df["AMT_CREDIT_SUM"] * (bureau_data_df["CREDIT_ACTIVE"].apply(lambda x: int(x != "Closed"))).astype(np.int8)
        grp1 = bureau_data_df[["SK_ID_CURR", "ACTIVE_AMT_CREDIT_SUM"]].groupby(by = ["SK_ID_CURR"])["ACTIVE_AMT_CREDIT_SUM"].sum().reset_index().rename(index = str, columns = { "ACTIVE_AMT_CREDIT_SUM": "TOTAL_ACTIVE_CUSTOMER_CREDIT"})
        bureau_stats_df = bureau_stats_df.merge(grp1, on = ["SK_ID_CURR"], how = "left")

        ## Total amount of active annuity
        bureau_data_df["AMT_ANNUITY"] = bureau_data_df["AMT_ANNUITY"].fillna(0)
        bureau_data_df["ACTIVE_AMT_ANNUITY"] = bureau_data_df["AMT_ANNUITY"] * (bureau_data_df["CREDIT_ACTIVE"].apply(lambda x: int(x != "Closed"))).astype(np.int8)
        grp1 = bureau_data_df[["SK_ID_CURR", "ACTIVE_AMT_ANNUITY"]].groupby(by = ["SK_ID_CURR"])["ACTIVE_AMT_ANNUITY"].sum().reset_index().rename(index = str, columns = { "ACTIVE_AMT_ANNUITY": "TOTAL_ACTIVE_AMT_ANNUITY"})
        bureau_stats_df = bureau_stats_df.merge(grp1, on = ["SK_ID_CURR"], how = "left")
        
        ## Fraction of total debt overdue for each customer
        bureau_data_df["AMT_CREDIT_SUM_OVERDUE"] = bureau_data_df["AMT_CREDIT_SUM_OVERDUE"].fillna(0)

        grp2 = bureau_data_df[["SK_ID_CURR", "AMT_CREDIT_SUM_OVERDUE"]].groupby(by = ["SK_ID_CURR"])["AMT_CREDIT_SUM_OVERDUE"].sum().reset_index().rename(index = str, columns = { "AMT_CREDIT_SUM_OVERDUE": "TOTAL_CUSTOMER_OVERDUE"})

        bureau_stats_df = bureau_stats_df.merge(grp2, on = ["SK_ID_CURR"], how = "left")
        del grp1, grp2
        gc.collect()

        bureau_stats_df["OVERDUE_DEBT_RATIO"] = bureau_stats_df["TOTAL_CUSTOMER_OVERDUE"] / bureau_stats_df["TOTAL_CUSTOMER_DEBT"]
        
        ## Average number of loans prolonged
        bureau_data_df["CNT_CREDIT_PROLONG"] = bureau_data_df["CNT_CREDIT_PROLONG"].fillna(0)
        grp = bureau_data_df[["SK_ID_CURR", "CNT_CREDIT_PROLONG"]].groupby(by = ["SK_ID_CURR"])["CNT_CREDIT_PROLONG"].mean().reset_index().rename(index = str, columns = { "CNT_CREDIT_PROLONG": "AVG_CREDITDAYS_PROLONGED"})
        bureau_stats_df = bureau_stats_df.merge(grp, on = ["SK_ID_CURR"], how = "left")
                
        # Add prefix to columns
        bureau_stats_df.columns = ["bureau_" + c if c != "SK_ID_CURR" else c for c in bureau_stats_df.columns.tolist()]

        print("    Pre-processing 'bureau.csv'... done in", round(time.time() - st, 3), "secs")

        return bureau_stats_df

    def transform_previous_applications_data(self, previous_application_data_df, installments_payments_data_df):
        st = time.time()
        print("    Pre-processing 'previous_application.csv'...")

        # Replace missing values flags by NaNs
        previous_application_data_df["DAYS_LAST_DUE"].replace(365243, np.nan, inplace = True)
        previous_application_data_df["DAYS_LAST_DUE_1ST_VERSION"].replace(365243, np.nan, inplace = True)
        
        # Count number of missing values by row
        previous_application_data_df["missing_values_count"] = previous_application_data_df.isnull().sum(axis = 1)

        previous_application_data_df["APP_CREDIT_PERC"] = previous_application_data_df["AMT_APPLICATION"] / previous_application_data_df["AMT_CREDIT"]
        previous_application_data_df["APP_CREDIT_DIFF"] = previous_application_data_df["AMT_APPLICATION"] - previous_application_data_df["AMT_CREDIT"]
        previous_application_data_df["APP_CREDIT_DIFF_eq_0"] = (previous_application_data_df["APP_CREDIT_DIFF"] == 0).astype(np.int8)

        # Impute missing values
        previous_application_data_df["NAME_TYPE_SUITE"].fillna("missing", inplace = True)
        previous_application_data_df["AMT_DOWN_PAYMENT"].fillna(0, inplace = True)

        # Get the number of annuities for each previous loan
        tmp = installments_payments_data_df.groupby(["SK_ID_CURR", "SK_ID_PREV"]).size().reset_index()
        tmp.columns = ["SK_ID_CURR", "SK_ID_PREV", "number_of_annuities"]
        previous_application_data_df = previous_application_data_df.merge(tmp, how = "left", on = ["SK_ID_CURR", "SK_ID_PREV"])
        previous_application_data_df["number_of_annuities"].fillna(0, inplace = True)
        previous_application_data_df["credit_length"] = previous_application_data_df["DAYS_LAST_DUE"] - previous_application_data_df["DAYS_FIRST_DUE"]
        previous_application_data_df["credit_length_months"] = (previous_application_data_df["credit_length"] / 30) + 1
        previous_application_data_df["credit_length_months_gt_10000"] = (previous_application_data_df["credit_length_months"] > 10000).astype(np.int8)
        previous_application_data_df["AMT_CREDIT_/_AMT_ANNUITY"] = previous_application_data_df["AMT_CREDIT"] / previous_application_data_df["AMT_ANNUITY"]
        previous_application_data_df["number_of_annuities_-_AMT_CREDIT_/_AMT_ANNUITY"] = previous_application_data_df["number_of_annuities"] - previous_application_data_df["AMT_CREDIT_/_AMT_ANNUITY"]

        previous_application_data_df["subscribed_to_insurance"] = (previous_application_data_df["AMT_CREDIT"] > previous_application_data_df["AMT_GOODS_PRICE"]).astype(np.int8)
        previous_application_data_df["insurance_amount"] = previous_application_data_df["AMT_CREDIT"] - previous_application_data_df["AMT_GOODS_PRICE"]
        previous_application_data_df["insurance_percentage_goods_price"] = (previous_application_data_df["insurance_amount"] / previous_application_data_df["AMT_GOODS_PRICE"]) * 100
        previous_application_data_df["insurance_percentage_total_amount"] = (previous_application_data_df["insurance_amount"] / previous_application_data_df["AMT_CREDIT"]) * 100

        # Look for rejected loans
        previous_application_reject_reason_df = previous_application_data_df.groupby("SK_ID_CURR")["CODE_REJECT_REASON"].value_counts(normalize = False)
        previous_application_reject_reason_df = previous_application_reject_reason_df.unstack("CODE_REJECT_REASON")
        previous_application_reject_reason_df.columns = ["CODE_REJECT_REASON_" + c for c in previous_application_reject_reason_df.columns]
        previous_application_reject_reason_df.fillna(0, inplace = True)

        previous_application_contract_status_df = previous_application_data_df.groupby("SK_ID_CURR")["NAME_CONTRACT_STATUS"].value_counts(normalize = False)
        previous_application_contract_status_df = previous_application_contract_status_df.unstack("NAME_CONTRACT_STATUS")
        previous_application_contract_status_df.columns = ["NAME_CONTRACT_STATUS_" + c for c in previous_application_contract_status_df.columns]
        previous_application_contract_status_df.fillna(0, inplace = True)
        previous_application_contract_status_df["NAME_CONTRACT_STATUS_Refused_gt_0"] = (previous_application_contract_status_df["NAME_CONTRACT_STATUS_Refused"] > 0).astype(np.int8)

        # Look for changes in installments calendar
        previous_application_data_df["days_last_due_diff_version"] = previous_application_data_df["DAYS_LAST_DUE"] - previous_application_data_df["DAYS_LAST_DUE_1ST_VERSION"]
        tmp = installments_payments_data_df[["SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_VERSION"]].groupby(["SK_ID_CURR", "SK_ID_PREV"]).max().reset_index()
        tmp.columns = ["SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_VERSION_max"]
        previous_application_data_df = previous_application_data_df.merge(tmp, how = "left", on = ["SK_ID_CURR", "SK_ID_PREV"])
        
        # Encode categorical features
        tmp_df = previous_application_data_df[["SK_ID_CURR", "PRODUCT_COMBINATION", "NAME_YIELD_GROUP", "NAME_CONTRACT_STATUS", "NAME_CASH_LOAN_PURPOSE"]]
        tmp_df = tmp_df.merge(self.y_train, how = "left", on = "SK_ID_CURR")

        name_yield_group_mapping_dict = tmp_df[["NAME_YIELD_GROUP", "target"]].groupby("NAME_YIELD_GROUP").agg(lambda x: x.sum() / x.shape[0]).to_dict()["target"]
        previous_application_data_df["NAME_YIELD_GROUP_TgtAvg"] = previous_application_data_df["NAME_YIELD_GROUP"].map(name_yield_group_mapping_dict)

        product_combination_mapping_dict = tmp_df[["PRODUCT_COMBINATION", "target"]].groupby("PRODUCT_COMBINATION").agg(lambda x: x.sum() / x.shape[0]).to_dict()["target"]
        previous_application_data_df["PRODUCT_COMBINATION_TgtAvg"] = previous_application_data_df["PRODUCT_COMBINATION"].map(product_combination_mapping_dict)

        name_contract_status_mapping_dict = tmp_df[["NAME_CONTRACT_STATUS", "target"]].groupby("NAME_CONTRACT_STATUS").agg(lambda x: x.sum() / x.shape[0]).to_dict()["target"]
        previous_application_data_df["NAME_CONTRACT_STATUS_TgtAvg"] = previous_application_data_df["NAME_CONTRACT_STATUS"].map(product_combination_mapping_dict)

        name_cash_loan_purpose_mapping_dict = tmp_df[["NAME_CASH_LOAN_PURPOSE", "target"]].groupby("NAME_CASH_LOAN_PURPOSE").agg(lambda x: x.sum() / x.shape[0]).to_dict()["target"]
        previous_application_data_df["NAME_CASH_LOAN_PURPOSE_TgtAvg"] = previous_application_data_df["NAME_CASH_LOAN_PURPOSE"].map(name_cash_loan_purpose_mapping_dict)

        previous_application_data_df = self._previous_application_cfe.transform(previous_application_data_df)

        # Remove useless features
        useless_features_lst = ["CHANNEL_TYPE_AP+ (Cash loan)", "CHANNEL_TYPE_Car dealer", "CHANNEL_TYPE_Channel of corporate sales", "CHANNEL_TYPE_Contact center", 
                                "CHANNEL_TYPE_Credit and cash offices", "CHANNEL_TYPE_Regional / Local", "NAME_CASH_LOAN_PURPOSE_Building a house or an annex", "NAME_CASH_LOAN_PURPOSE_Business development", 
                                "NAME_CASH_LOAN_PURPOSE_Buying a garage", "NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land", "NAME_CASH_LOAN_PURPOSE_Buying a home", "NAME_CASH_LOAN_PURPOSE_Buying a new car", 
                                "NAME_CASH_LOAN_PURPOSE_Buying a used car", "NAME_CASH_LOAN_PURPOSE_Car repairs", "NAME_CASH_LOAN_PURPOSE_Education", "NAME_CASH_LOAN_PURPOSE_Everyday expenses", 
                                "NAME_CASH_LOAN_PURPOSE_Furniture", "NAME_CASH_LOAN_PURPOSE_Gasification / water supply", "NAME_CASH_LOAN_PURPOSE_Journey", "NAME_CASH_LOAN_PURPOSE_Money for a third person", 
                                "NAME_CASH_LOAN_PURPOSE_Payments on other loans", "NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment", "NAME_CASH_LOAN_PURPOSE_Refusal to name the goal", "NAME_CASH_LOAN_PURPOSE_Urgent needs", 
                                "NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday", "NAME_CLIENT_TYPE_XNA", "NAME_CONTRACT_TYPE_Cash loans", "NAME_CONTRACT_TYPE_Consumer loans", 
                                "NAME_CONTRACT_TYPE_Revolving loans", "NAME_GOODS_CATEGORY_Additional Service", "NAME_GOODS_CATEGORY_Auto Accessories", "NAME_GOODS_CATEGORY_Clothing and Accessories", 
                                "NAME_GOODS_CATEGORY_Construction Materials", "NAME_GOODS_CATEGORY_Consumer Electronics", "NAME_GOODS_CATEGORY_Direct Sales", "NAME_GOODS_CATEGORY_Education", 
                                "NAME_GOODS_CATEGORY_Fitness", "NAME_GOODS_CATEGORY_Gardening", "NAME_GOODS_CATEGORY_Homewares", "NAME_GOODS_CATEGORY_Insurance", 
                                "NAME_GOODS_CATEGORY_Medical Supplies", "NAME_GOODS_CATEGORY_Medicine", "NAME_GOODS_CATEGORY_Office Appliances", "NAME_GOODS_CATEGORY_Other", 
                                "NAME_GOODS_CATEGORY_Photo / Cinema Equipment", "NAME_GOODS_CATEGORY_Sport and Leisure", "NAME_GOODS_CATEGORY_Tourism", "NAME_GOODS_CATEGORY_Weapon", 
                                "NAME_PAYMENT_TYPE_Cash through the bank", "NAME_PAYMENT_TYPE_Cashless from the account of the employer", "NAME_PAYMENT_TYPE_Non-cash from your account", 
                                "NAME_PORTFOLIO_Cars", "NAME_SELLER_INDUSTRY_Auto technology", "NAME_SELLER_INDUSTRY_Consumer electronics", "NAME_SELLER_INDUSTRY_Jewelry", 
                                "NAME_SELLER_INDUSTRY_MLM partners", "NAME_SELLER_INDUSTRY_Tourism", "NAME_TYPE_SUITE_Group of people", "NAME_TYPE_SUITE_Spouse, partner", 
                                "PRODUCT_COMBINATION_Card Street", "PRODUCT_COMBINATION_Card X-Sell", "PRODUCT_COMBINATION_Cash Street: high", "PRODUCT_COMBINATION_Cash Street: low", 
                                "PRODUCT_COMBINATION_Cash Street: middle", "PRODUCT_COMBINATION_Cash X-Sell: high", "PRODUCT_COMBINATION_Cash X-Sell: low", "PRODUCT_COMBINATION_Cash X-Sell: middle", 
                                "PRODUCT_COMBINATION_POS household with interest", "PRODUCT_COMBINATION_POS household without interest", "PRODUCT_COMBINATION_POS industry with interest", "PRODUCT_COMBINATION_POS industry without interest", 
                                "PRODUCT_COMBINATION_POS mobile with interest", "PRODUCT_COMBINATION_POS mobile without interest", "PRODUCT_COMBINATION_POS other with interest", "PRODUCT_COMBINATION_POS others without interest"]
        previous_application_data_df.drop(useless_features_lst, axis = 1, inplace = True)

        # Compute average values and counts
        previous_application_stats_df = self._flatten_columns_names(previous_application_data_df.select_dtypes(include = np.number).drop("SK_ID_PREV", axis = 1).groupby("SK_ID_CURR").agg(["mean", "std", "min", "max", "sum", "nunique"]))
        previous_application_stats_df["number_of_previous_applications"] = previous_application_data_df[["SK_ID_CURR", "SK_ID_PREV"]].groupby("SK_ID_CURR").count()["SK_ID_PREV"]

        previous_application_stats_df = pd.concat([previous_application_stats_df, previous_application_reject_reason_df, previous_application_contract_status_df], axis = 1)

        # Ratio of refused previous applications by total number of previous applications
        previous_application_stats_df["refused_previous_applications_ratio"] = previous_application_stats_df["NAME_CONTRACT_STATUS_Refused"] / previous_application_stats_df["number_of_previous_applications"]

        # Insurance ratio
        previous_application_stats_df["NFLAG_INSURED_ON_APPROVAL_ratio"] = previous_application_stats_df["NFLAG_INSURED_ON_APPROVAL_sum"] / previous_application_stats_df["number_of_previous_applications"]

        # Look for trends
        subset_data1_df = previous_application_data_df[["SK_ID_CURR", "DAYS_DECISION", "NAME_YIELD_GROUP_TgtAvg"]].sort_values(["SK_ID_CURR", "DAYS_DECISION"], ascending = [True, True]).groupby("SK_ID_CURR")
        previous_application_stats_df["NAME_YIELD_GROUP_trend"] = subset_data1_df["NAME_YIELD_GROUP_TgtAvg"].apply(lambda x: self.create_trend_feature(x))

        # Add prefix to columns
        previous_application_stats_df.columns = ["previous_application_" + c if c != "SK_ID_CURR" else c for c in previous_application_stats_df.columns.tolist()]

        print("    Pre-processing 'previous_application.csv'... done in", round(time.time() - st, 3), "secs")

        return previous_application_stats_df

    def transform_credit_card_balance_data(self, credit_card_balance_data_df):
        st = time.time()
        print("    Pre-processing 'credit_card_balance.csv'...")

        # Replace outliers with NaNs
        credit_card_balance_data_df["AMT_DRAWINGS_ATM_CURRENT"].loc[credit_card_balance_data_df["AMT_DRAWINGS_ATM_CURRENT"] < 0] = np.nan
        credit_card_balance_data_df["AMT_DRAWINGS_CURRENT"].loc[credit_card_balance_data_df["AMT_DRAWINGS_CURRENT"] < 0] = np.nan

        # Count number of missing values by row
        credit_card_balance_data_df["missing_values_count"] = credit_card_balance_data_df.isnull().sum(axis = 1)

        # Encode categorical features
        credit_card_balance_data_df = self._credit_card_balance_cfe.transform(credit_card_balance_data_df)

        # Compute average values and counts
        card_balance_stats_df = self._flatten_columns_names(credit_card_balance_data_df.select_dtypes(include = np.number).drop("SK_ID_PREV", axis = 1).groupby("SK_ID_CURR").agg(["mean", "std", "min", "max", "sum", "nunique"]))
        card_balance_stats_df["nb_ccb"] = credit_card_balance_data_df[["SK_ID_CURR", "SK_ID_PREV"]].groupby("SK_ID_CURR").count()["SK_ID_PREV"]
        #card_balance_stats_df.drop("SK_ID_PREV", axis = 1, inplace = True)

        # Customer risk profile
        ## Number of loans per customer
        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR"])["SK_ID_PREV"].nunique().reset_index().rename(index = str, columns = {"SK_ID_PREV": "NO_LOANS"})
        card_balance_stats_df = card_balance_stats_df.merge(grp, on = ["SK_ID_CURR"], how = "left")
        del grp 
        gc.collect()

        ## Rate at which loan is paid back by customer
        ### No of Installments paid per Loan per Customer 
        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR", "SK_ID_PREV"])["CNT_INSTALMENT_MATURE_CUM"].max().reset_index().rename(index = str, columns = {"CNT_INSTALMENT_MATURE_CUM": "NO_INSTALMENTS"})
        grp1 = grp.groupby(by = ["SK_ID_CURR"])["NO_INSTALMENTS"].sum().reset_index().rename(index = str, columns = {"NO_INSTALMENTS": "TOTAL_INSTALMENTS"})
        card_balance_stats_df = card_balance_stats_df.merge(grp1, on = ["SK_ID_CURR"], how = "left")

        ### Average Number of installments paid per loan 
        card_balance_stats_df["INSTALLMENTS_PER_LOAN"] = (card_balance_stats_df["TOTAL_INSTALMENTS"] / card_balance_stats_df["NO_LOANS"]).astype("uint32")

        ## How much did the customer load a credit line
        credit_card_balance_data_df["AMT_CREDIT_LIMIT_ACTUAL1"] = credit_card_balance_data_df["AMT_CREDIT_LIMIT_ACTUAL"]
        
        ### Calculate the ratio of Amount Balance to Credit Limit - CREDIT LOAD OF CUSTOMER 
        ### This is done for each Credit limit value per loan per Customer 
        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR", "SK_ID_PREV", "AMT_CREDIT_LIMIT_ACTUAL"]).apply(lambda x: x["AMT_BALANCE"].max() / x["AMT_CREDIT_LIMIT_ACTUAL1"].max()).reset_index().rename(index = str, columns = {0: "CREDIT_LOAD1"})
        credit_card_balance_data_df.drop("AMT_CREDIT_LIMIT_ACTUAL1", axis = 1, inplace = True)

        ### We now calculate the mean Credit load of All Loan transactions of Customer 
        grp1 = grp.groupby(by = ["SK_ID_CURR"])["CREDIT_LOAD1"].mean().reset_index().rename(index = str, columns = {"CREDIT_LOAD1": "CREDIT_LOAD"})
        card_balance_stats_df = card_balance_stats_df.merge(grp1, on = ["SK_ID_CURR"], how = "left")
        del grp, grp1
        gc.collect()
        
        ## How many times did the customer miss the minimum payment
        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR", "SK_ID_PREV"]).apply(lambda x: (x["SK_DPD"] != 0).astype(np.int8).sum()).reset_index().rename(index = str, columns = {0: "NO_DPD"})
        grp1 = grp.groupby(by = ["SK_ID_CURR"])["NO_DPD"].mean().reset_index().rename(index = str, columns = {"NO_DPD" : "DPD_COUNT"})

        card_balance_stats_df = card_balance_stats_df.merge(grp1, on = ["SK_ID_CURR"], how = "left")
        del grp, grp1 
        gc.collect()

        ## What is the average number of days did customer go past due date
        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR"])["SK_DPD"].mean().reset_index().rename(index = str, columns = {"SK_DPD": "AVG_DPD"})
        card_balance_stats_df = card_balance_stats_df.merge(grp, on = ["SK_ID_CURR"], how = "left")
        del grp 
        gc.collect()

        ## What fraction of minimum payments were missed
        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR"]).apply(lambda x: (100 * (x["AMT_INST_MIN_REGULARITY"] < x["AMT_PAYMENT_CURRENT"]).astype(np.int8).sum()) / x["AMT_INST_MIN_REGULARITY"].shape[0]).reset_index().rename(index = str, columns = { 0 : "PERCENTAGE_MISSED_PAYMENTS"})
        card_balance_stats_df = card_balance_stats_df.merge(grp, on = ["SK_ID_CURR"], how = "left")
        del grp 
        gc.collect()

        # Customer behavior patterns
        ## Cash withdrawals VS overall spending ratio
        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR"])["AMT_DRAWINGS_ATM_CURRENT"].sum().reset_index().rename(index = str, columns = {"AMT_DRAWINGS_ATM_CURRENT" : "DRAWINGS_ATM"})
        card_balance_stats_df = card_balance_stats_df.merge(grp, on = ["SK_ID_CURR"], how = "left")

        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR"])["AMT_DRAWINGS_CURRENT"].sum().reset_index().rename(index = str, columns = {"AMT_DRAWINGS_CURRENT" : "DRAWINGS_TOTAL"})
        card_balance_stats_df = card_balance_stats_df.merge(grp, on = ["SK_ID_CURR"], how = "left")

        card_balance_stats_df["CASH_CARD_RATIO"] = (card_balance_stats_df["DRAWINGS_ATM"] / card_balance_stats_df["DRAWINGS_TOTAL"]) * 100
        
        ## Avg number of drawings per customer; Total drawings / Number of drawings
        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR"])["AMT_DRAWINGS_CURRENT"].sum().reset_index().rename(index = str, columns = {"AMT_DRAWINGS_CURRENT" : "TOTAL_DRAWINGS"})
        card_balance_stats_df = card_balance_stats_df.merge(grp, on = ["SK_ID_CURR"], how = "left")
        del grp
        gc.collect()

        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR"])["CNT_DRAWINGS_CURRENT"].sum().reset_index().rename(index = str, columns = {"CNT_DRAWINGS_CURRENT" : "NO_DRAWINGS"})
        card_balance_stats_df = card_balance_stats_df.merge(grp, on = ["SK_ID_CURR"], how = "left")
        del grp
        gc.collect()

        card_balance_stats_df["DRAWINGS_RATIO"] = (card_balance_stats_df["TOTAL_DRAWINGS"] / card_balance_stats_df["NO_DRAWINGS"]) * 100
        
        # Add prefix to columns
        card_balance_stats_df.columns = ["credit_card_balance_" + c if c != "SK_ID_CURR" else c for c in card_balance_stats_df.columns.tolist()]

        print("    Pre-processing 'credit_card_balance.csv'... done in", round(time.time() - st, 3), "secs")

        return card_balance_stats_df

    def transform_pos_cash_worker(self, data):
        # Unpack data
        data_subset_df = data[0]
        window = data[1]
        processing_type = data[2]

        if window is None:
            if processing_type == 5:
                feature_name = "normalized_nb_overdue"
                result_sr = data_subset_df["SK_DPD"].apply(lambda x: (x > 0).sum() / x.shape[0])
            elif processing_type == 6:
                feature_name = "normalized_nb_overdue_with_tolerance"
                result_sr = data_subset_df["SK_DPD_DEF"].apply(lambda x: (x > 0).sum() / x.shape[0])
            elif processing_type == 7:
                feature_name = "normalized_nb_days_overdue"
                result_sr = data_subset_df["SK_DPD"].apply(lambda x: x.values[x.values > 0].sum() / x.shape[0])
            elif processing_type == 8:
                feature_name = "normalized_nb_days_overdue_with_tolerance"
                result_sr = data_subset_df["SK_DPD_DEF"].apply(lambda x: x.values[x.values > 0].sum() / x.shape[0])
            elif processing_type == 9:
                feature_name = "overdue_apparition_in_installments"
                result_sr = data_subset_df["SK_DPD"].apply(lambda x: (x.values > 0).argmax(axis = 0) / x.shape[0])
            elif processing_type == 10:
                feature_name = "overdue_apparition_in_installments_with_tolerance"
                result_sr = data_subset_df["SK_DPD_DEF"].apply(lambda x: (x.values > 0).argmax(axis = 0) / x.shape[0])
            elif type(processing_type) == str:
                feature_name = processing_type
                result_sr = data_subset_df.apply(lambda x: self.create_trend_feature(x))

        elif window == 1:
            if processing_type == 1:
                feature_name = "last_installment_was_overdue"
                result_sr = (data_subset_df.head(1).reset_index(drop = True)["SK_DPD"] > 0).astype(np.int8)
            elif processing_type == 2:
                feature_name = "last_installment_was_overdue_with_tolerance"
                result_sr = (data_subset_df.head(1).reset_index(drop = True)["SK_DPD_DEF"] > 0).astype(np.int8)
            elif processing_type == 3:
                feature_name = "first_installment_was_overdue"
                result_sr = (data_subset_df.head(1).reset_index(drop = True)["SK_DPD"] > 0).astype(np.int8)
            elif processing_type == 4:
                feature_name = "first_installment_was_overdue_with_tolerance"
                result_sr = (data_subset_df.head(1).reset_index(drop = True)["SK_DPD_DEF"] > 0).astype(np.int8)

        else:
            if processing_type == 1:
                feature_name = "nb_overdue_" + str(window) + "_last_installments"
                result_sr = data_subset_df.head(window).groupby("SK_ID_CURR")["SK_DPD"].apply(lambda x: (x > 0).sum())
            elif processing_type == 2:
                feature_name = "nb_overdue_" + str(window) + "_last_installments_with_tolerance"
                result_sr = data_subset_df.head(window).groupby("SK_ID_CURR")["SK_DPD_DEF"].apply(lambda x: (x > 0).sum())
            elif processing_type == 3:
                feature_name = "nb_overdue_" + str(window) + "_first_installments"
                result_sr = data_subset_df.head(window).groupby("SK_ID_CURR")["SK_DPD"].apply(lambda x: (x > 0).sum())
            elif processing_type == 4:
                feature_name = "nb_overdue_" + str(window) + "_first_installments_with_tolerance"
                result_sr = data_subset_df.head(window).groupby("SK_ID_CURR")["SK_DPD_DEF"].apply(lambda x: (x > 0).sum())

        return (result_sr, feature_name)

    def transform_pos_cash_balance_data(self, pos_cash_balance_data_df):
        st = time.time()
        print("    Pre-processing 'pos_cash_balance.csv'...")

        # Count number of missing values by row
        pos_cash_balance_data_df["missing_values_count"] = pos_cash_balance_data_df.isnull().sum(axis = 1)

        pos_cash_balance_data_df["NUNIQUE_STATUS"] = pos_cash_balance_data_df[["SK_ID_CURR", "NAME_CONTRACT_STATUS"]].groupby("SK_ID_CURR").nunique()["NAME_CONTRACT_STATUS"]
        pos_cash_balance_data_df["NUNIQUE_STATUS2"] = pos_cash_balance_data_df[["SK_ID_CURR", "NAME_CONTRACT_STATUS"]].groupby("SK_ID_CURR").max()["NAME_CONTRACT_STATUS"]

        # Flag late payments
        pos_cash_balance_data_df["pos_cash_paid_late"] = (pos_cash_balance_data_df["SK_DPD"] > 0).astype(np.int8)
        pos_cash_balance_data_df["pos_cash_paid_late_with_tolerance"] = (pos_cash_balance_data_df["SK_DPD_DEF"] > 0).astype(np.int8)
        pos_cash_balance_data_df["low_debt_overdue"] = pos_cash_balance_data_df["SK_DPD"] - pos_cash_balance_data_df["SK_DPD_DEF"]
        pos_cash_balance_data_df["high_debt_ratio_overdue"] = pos_cash_balance_data_df["SK_DPD_DEF"] / pos_cash_balance_data_df["SK_DPD"]

        # Encode categorical features
        pos_cash_balance_data_df = self._pos_cash_balance_cfe.transform(pos_cash_balance_data_df)

        # Compute average values and counts
        pos_cash_balance_stats_df = self._flatten_columns_names(pos_cash_balance_data_df.select_dtypes(include = np.number).drop("SK_ID_PREV", axis = 1).groupby("SK_ID_CURR").agg(["mean", "std", "min", "max", "sum", "nunique"]))
        pos_cash_balance_stats_df["nb_pcb"] = pos_cash_balance_data_df[["SK_ID_CURR", "SK_ID_PREV"]].groupby("SK_ID_CURR").count()["SK_ID_PREV"]

        # Check if last and first installments were overdue
        subset_data1_df = pos_cash_balance_data_df[["SK_ID_CURR", "MONTHS_BALANCE", "SK_DPD"]].sort_values(["SK_ID_CURR", "MONTHS_BALANCE"], ascending = [True, False]).groupby("SK_ID_CURR")
        subset_data2_df = pos_cash_balance_data_df[["SK_ID_CURR", "MONTHS_BALANCE", "SK_DPD_DEF"]].sort_values(["SK_ID_CURR", "MONTHS_BALANCE"], ascending = [True, False]).groupby("SK_ID_CURR")
        subset_data3_df = pos_cash_balance_data_df[["SK_ID_CURR", "MONTHS_BALANCE", "SK_DPD"]].sort_values(["SK_ID_CURR", "MONTHS_BALANCE"], ascending = [True, True]).groupby("SK_ID_CURR")
        subset_data4_df = pos_cash_balance_data_df[["SK_ID_CURR", "MONTHS_BALANCE", "SK_DPD_DEF"]].sort_values(["SK_ID_CURR", "MONTHS_BALANCE"], ascending = [True, True]).groupby("SK_ID_CURR")

        chunk_data_lst = []
        # For last installments
        for i in [1, 5, 10, 20]:
            chunk_data_lst.append((subset_data1_df, i, 1))
            chunk_data_lst.append((subset_data2_df, i, 2))

        """
        # For first installments
        for i in [1, 5, 10, 20]:
            chunk_data_lst.append((subset_data3_df, i, 3))
            chunk_data_lst.append((subset_data4_df, i, 4))"""

        # Number of overdues for each loan, normalized by the loan's length
        chunk_data_lst.append((subset_data1_df, None, 5))
        chunk_data_lst.append((subset_data2_df, None, 6))
        
        # Cumulated number of overdue days normalized by loan's length
        chunk_data_lst.append((subset_data1_df, None, 7))
        chunk_data_lst.append((subset_data2_df, None, 8))

        # Look at which moment of the credit repayment the overdue appeared
        chunk_data_lst.append((subset_data3_df, None, 9))
        chunk_data_lst.append((subset_data4_df, None, 10))

        # Look for trends in overdue
        chunk_data_lst.append((subset_data3_df["SK_DPD"], None, "SK_DPD_trend"))
        
        pool = mp.Pool(self.n_jobs)
        results_lst = pool.map(self.transform_pos_cash_worker, chunk_data_lst)
        pool.close()
        pool.join()

        for result_sr, feature_name in results_lst:
            pos_cash_balance_stats_df[feature_name] = result_sr

        # Add prefix to columns
        pos_cash_balance_stats_df.columns = ["pos_cash_balance_" + c for c in pos_cash_balance_stats_df.columns.tolist()]

        print("    Pre-processing 'pos_cash_balance.csv'... done in", round(time.time() - st, 3), "secs")

        return pos_cash_balance_stats_df

    def transform_installments_payments_worker(self, data):
        # Unpack data
        data_subset_df = data[0]
        window = data[1]
        processing_type = data[2]
        
        if window is None:
            if processing_type == 5:
                feature_name = "normalized_nb_overdue"
                result_sr = data_subset_df["nb_overdue_days"].apply(lambda x: (x > 0).sum() / x.shape[0])
            elif processing_type == 6:
                feature_name = "normalized_nb_not_totally_paid"
                result_sr = data_subset_df["PAYMENT_PERC"].apply(lambda x: (x < 1).sum() / x.shape[0])
            elif processing_type == 7:
                feature_name = "normalized_nb_days_overdue"
                result_sr = data_subset_df["nb_overdue_days"].apply(lambda x: x.values[x.values > 0].sum() / x.shape[0])
            elif processing_type == 8:
                feature_name = "normalized_cumulated_not_totally_paid_amount"
                result_sr = data_subset_df["PAYMENT_PERC"].apply(lambda x: (1 - x.values[x.values < 1]).sum() / x.shape[0])
            elif processing_type == 9:
                feature_name = "overdue_apparition_in_installments"
                result_sr = data_subset_df["nb_overdue_days"].apply(lambda x: (x.values > 0).argmax(axis = 0) / x.shape[0])
            elif processing_type == 10:
                feature_name = "not_totally_paid_installment_apparition"
                result_sr = data_subset_df["PAYMENT_PERC"].apply(lambda x: (x.values < 1).argmax(axis = 0) / x.shape[0])
            elif type(processing_type) == str:
                feature_name = processing_type
                result_sr = data_subset_df.apply(lambda x: self.create_trend_feature(x))

        elif window == 1:
            if processing_type == 1:
                feature_name = "last_installment_was_overdue"
                result_sr = (data_subset_df.head(1).reset_index(drop = True)["nb_overdue_days"] > 0).astype(np.int8)
            elif processing_type == 2:
                feature_name = "last_installment_was_not_totally_paid"
                result_sr = (data_subset_df.head(1).reset_index(drop = True)["PAYMENT_PERC"] < 1).astype(np.int8)
            elif processing_type == 3:
                feature_name = "first_installment_was_overdue"
                result_sr = (data_subset_df.head(1).reset_index(drop = True)["nb_overdue_days"] > 0).astype(np.int8)
            elif processing_type == 4:
                feature_name = "first_installment_was_not_totally_paid"
                result_sr = (data_subset_df.head(1).reset_index(drop = True)["PAYMENT_PERC"] < 1).astype(np.int8)

        else:
            if processing_type == 1:
                feature_name = "nb_overdue_" + str(window) + "_last_installments"
                result_sr = data_subset_df.head(window).groupby("SK_ID_CURR")["nb_overdue_days"].apply(lambda x: (x > 0).sum())
            elif processing_type == 2:
                feature_name = "nb_not_totally_paid_" + str(window) + "_last_installments"
                result_sr = data_subset_df.head(window).groupby("SK_ID_CURR")["PAYMENT_PERC"].apply(lambda x: (x < 1).sum())
            elif processing_type == 3:
                feature_name = "nb_overdue_" + str(window) + "_first_installments"
                result_sr = data_subset_df.head(window).groupby("SK_ID_CURR")["nb_overdue_days"].apply(lambda x: (x > 0).sum())
            elif processing_type == 4:
                feature_name = "nb_not_totally_paid_" + str(window) + "_first_installments"
                result_sr = data_subset_df.head(window).groupby("SK_ID_CURR")["PAYMENT_PERC"].apply(lambda x: (x < 1).sum())

        return (result_sr, feature_name)

    def transform_installments_payments_data(self, installments_payments_data_df):
        st = time.time()
        print("    Pre-processing 'installments_payments.csv'...")

        # Count number of missing values by row
        installments_payments_data_df["missing_values_count"] = installments_payments_data_df.isnull().sum(axis = 1)

        installments_payments_data_df["PAYMENT_PERC"] = installments_payments_data_df["AMT_PAYMENT"] / installments_payments_data_df["AMT_INSTALMENT"]
        installments_payments_data_df["PAYMENT_PERC"] = installments_payments_data_df["PAYMENT_PERC"].replace([np.inf, -np.inf], np.nan).fillna(0)
        installments_payments_data_df["PAYMENT_DIFF"] = installments_payments_data_df["AMT_INSTALMENT"] - installments_payments_data_df["AMT_PAYMENT"]
        installments_payments_data_df["nb_overdue_days"] = installments_payments_data_df["DAYS_ENTRY_PAYMENT"] - installments_payments_data_df["DAYS_INSTALMENT"]
        installments_payments_data_df["is_installment_overdue"] = (installments_payments_data_df["nb_overdue_days"] > 0).astype(np.int8)
        installments_payments_data_df["no_installment_overdue"] = (installments_payments_data_df["nb_overdue_days"] == 0).astype(np.int8)
        installments_payments_data_df["is_payment_partial"] = (installments_payments_data_df["PAYMENT_DIFF"] > 0).astype(np.int8)

        # Groupby each customer and sort values of NUM_INSTALMENT_NUMBER in ascending order
        grp = installments_payments_data_df[["SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_NUMBER", "DAYS_ENTRY_PAYMENT", "PAYMENT_PERC", "PAYMENT_DIFF", "nb_overdue_days"]].sort_values(["SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_NUMBER"])

        ## Calculate Difference between each adjacent row
        grp["DAYS_ENTRY_PAYMENT_DIFF"] = grp.groupby(by = ["SK_ID_CURR", "SK_ID_PREV"])["DAYS_ENTRY_PAYMENT"].diff().fillna(0).astype(np.int32)
        grp["PAYMENT_PERC_DIFF"] = grp.groupby(by = ["SK_ID_CURR", "SK_ID_PREV"])["PAYMENT_PERC"].diff().fillna(0).astype(np.int32)
        grp["PAYMENT_DIFF_DIFF"] = grp.groupby(by = ["SK_ID_CURR", "SK_ID_PREV"])["PAYMENT_DIFF"].diff().fillna(0).astype(np.int32)
        grp["nb_overdue_days_DIFF"] = grp.groupby(by = ["SK_ID_CURR", "SK_ID_PREV"])["nb_overdue_days"].diff().fillna(0).astype(np.int32)
        
        grp.drop(["NUM_INSTALMENT_NUMBER", "DAYS_ENTRY_PAYMENT", "PAYMENT_PERC", "PAYMENT_DIFF", "nb_overdue_days"], axis = 1, inplace = True)
        grp = grp.sort_index()
        installments_payments_data_df["DAYS_ENTRY_PAYMENT_DIFF"] = grp["DAYS_ENTRY_PAYMENT_DIFF"]
        installments_payments_data_df["PAYMENT_PERC_DIFF"] = grp["PAYMENT_PERC_DIFF"]
        installments_payments_data_df["PAYMENT_DIFF_DIFF"] = grp["PAYMENT_DIFF_DIFF"]
        installments_payments_data_df["nb_overdue_days_DIFF"] = grp["nb_overdue_days_DIFF"]
        gc.collect()
        
        installments_payments_stats_df = self._flatten_columns_names(installments_payments_data_df.select_dtypes(include = np.number).drop("SK_ID_PREV", axis = 1).groupby("SK_ID_CURR").agg(["mean", "std", "min", "max", "sum", "nunique"]))
        installments_payments_stats_df["number_of_installments"] = installments_payments_data_df[["SK_ID_CURR", "SK_ID_PREV"]].groupby("SK_ID_CURR").count()["SK_ID_PREV"]

        installments_payments_stats_df["installment_overdue_ratio"] = installments_payments_stats_df["is_installment_overdue_sum"] / installments_payments_stats_df["number_of_installments"]

        # Check if last and first installments were overdue
        subset_data1_df = installments_payments_data_df[["SK_ID_CURR", "DAYS_INSTALMENT", "nb_overdue_days"]].sort_values(["SK_ID_CURR", "DAYS_INSTALMENT"], ascending = [True, False]).groupby("SK_ID_CURR")
        subset_data2_df = installments_payments_data_df[["SK_ID_CURR", "DAYS_INSTALMENT", "PAYMENT_PERC"]].sort_values(["SK_ID_CURR", "DAYS_INSTALMENT"], ascending = [True, False]).groupby("SK_ID_CURR")
        subset_data3_df = installments_payments_data_df[["SK_ID_CURR", "DAYS_INSTALMENT", "nb_overdue_days"]].sort_values(["SK_ID_CURR", "DAYS_INSTALMENT"], ascending = [True, True]).groupby("SK_ID_CURR")
        subset_data4_df = installments_payments_data_df[["SK_ID_CURR", "DAYS_INSTALMENT", "PAYMENT_PERC"]].sort_values(["SK_ID_CURR", "DAYS_INSTALMENT"], ascending = [True, True]).groupby("SK_ID_CURR")
        
        chunk_data_lst = []
        # For last installments
        for i in [1, 5, 10, 20]:
            chunk_data_lst.append((subset_data1_df, i, 1))
            chunk_data_lst.append((subset_data2_df, i, 2))

        """
        # For first installments
        for i in [1, 5, 10, 20]:
            chunk_data_lst.append((subset_data3_df, i, 3))
            chunk_data_lst.append((subset_data4_df, i, 4))"""

        # Number of overdues for each loan, normalized by the loan's length
        chunk_data_lst.append((subset_data1_df, None, 5))
        chunk_data_lst.append((subset_data2_df, None, 6))
        
        # Cumulated number of overdue days normalized by loan's length
        chunk_data_lst.append((subset_data1_df, None, 7))
        chunk_data_lst.append((subset_data2_df, None, 8))

        # Look at which moment of the credit repayment the overdue appeared
        chunk_data_lst.append((subset_data3_df, None, 9))
        chunk_data_lst.append((subset_data4_df, None, 10))

        # Look for trends in overdue
        chunk_data_lst.append((subset_data3_df["nb_overdue_days"], None, "nb_overdue_days_trend"))
        chunk_data_lst.append((subset_data4_df["PAYMENT_PERC"], None, "PAYMENT_PERC_trend"))
        
        pool = mp.Pool(self.n_jobs)
        results_lst = pool.map(self.transform_installments_payments_worker, chunk_data_lst)
        pool.close()
        pool.join()

        for result_sr, feature_name in results_lst:
            installments_payments_stats_df[feature_name] = result_sr

        # Add prefix to columns
        installments_payments_stats_df.columns = ["installments_payments_" + c for c in installments_payments_stats_df.columns.tolist()]

        print("    Pre-processing 'installments_payments.csv'... done in", round(time.time() - st, 3), "secs")

        return installments_payments_stats_df
    
    def transform(self, bureau_data_df, bureau_balance_data_df, credit_card_balance_data_df, installments_payments_data_df, pos_cash_balance_data_df, previous_application_data_df):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        bureau_data_df : pd.DataFrame
                Additional data frame.

        bureau_balance_data_df : pd.DataFrame
                Additional data frame.

        credit_card_balance_data_df : pd.DataFrame
                Additional data frame.

        installments_payments_data_df : pd.DataFrame
                Additional data frame.

        pos_cash_balance_data_df : pd.DataFrame
                Additional data frame.

        previous_application_data_df : pd.DataFrame
                Additional data frame.
                
        Returns
        -------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.
        """
                    
        print("Additional files preprocessing data...")    
        
        # Processing 'bureau.csv'
        bureau_stats_df = self.transform_bureau_data(bureau_data_df, bureau_balance_data_df)

        # Doing some housekeeping
        ## Remove constant features
        tmp = bureau_stats_df.nunique()
        constant_features_lst = tmp.loc[tmp == 1].index.tolist()
        bureau_stats_df.drop(constant_features_lst, axis = 1, inplace = True)

        ## Remove features with more than 80% of missing values
        tmp = bureau_stats_df.isnull().sum()
        too_many_nas_features_lst = tmp.loc[tmp > 0.8 * bureau_stats_df.shape[0]].index.tolist()
        bureau_stats_df.drop(too_many_nas_features_lst, axis = 1, inplace = True)

        ## Remove useless features
        useless_features_lst = ["bureau_CNT_CREDIT_PROLONG_nunique", "bureau_bureau_balance_MONTHS_BALANCE_max_max", "bureau_bureau_balance_STATUS_nunique_nunique", "bureau_bureau_balance_STATUS_2_nunique",
                                "bureau_CREDIT_ACTIVE_Active_nunique", "bureau_CREDIT_CURRENCY_OrdinalEncoder_min", "bureau_CREDIT_TYPE_Mortgage_min",
                                "bureau_CREDIT_DAY_OVERDUE_min", "bureau_CNT_CREDIT_PROLONG_min", "bureau_CNT_CREDIT_PROLONG_max", "bureau_AMT_CREDIT_SUM_OVERDUE_min", 
                                "bureau_bureau_balance_STATUS_3_sum", "bureau_bureau_balance_STATUS_3_nunique", "bureau_bureau_balance_STATUS_4_sum", "bureau_bureau_balance_STATUS_4_nunique", 
                                "bureau_bureau_balance_STATUS_5_sum", "bureau_bureau_balance_STATUS_5_nunique", "bureau_CREDIT_ACTIVE_Bad debt_mean", "bureau_CREDIT_ACTIVE_Bad debt_std", 
                                "bureau_CREDIT_ACTIVE_Bad debt_min", "bureau_CREDIT_ACTIVE_Bad debt_max", "bureau_CREDIT_ACTIVE_Bad debt_sum", "bureau_CREDIT_ACTIVE_Bad debt_nunique", 
                                "bureau_CREDIT_ACTIVE_Closed_nunique", "bureau_CREDIT_ACTIVE_Sold_min", "bureau_CREDIT_ACTIVE_Sold_max", "bureau_CREDIT_ACTIVE_Sold_sum", 
                                "bureau_CREDIT_ACTIVE_Sold_nunique", "bureau_CREDIT_CURRENCY_OrdinalEncoder_mean", "bureau_CREDIT_CURRENCY_OrdinalEncoder_std", "bureau_CREDIT_CURRENCY_OrdinalEncoder_max", 
                                "bureau_CREDIT_CURRENCY_OrdinalEncoder_sum", "bureau_CREDIT_CURRENCY_OrdinalEncoder_nunique", "bureau_CREDIT_TYPE_Car loan_mean", "bureau_CREDIT_TYPE_Car loan_std", 
                                "bureau_CREDIT_TYPE_Car loan_min", "bureau_CREDIT_TYPE_Car loan_max", "bureau_CREDIT_TYPE_Car loan_sum", "bureau_CREDIT_TYPE_Car loan_nunique", 
                                "bureau_CREDIT_TYPE_Consumer credit_mean", "bureau_CREDIT_TYPE_Consumer credit_std", "bureau_CREDIT_TYPE_Consumer credit_min", "bureau_CREDIT_TYPE_Consumer credit_max", 
                                "bureau_CREDIT_TYPE_Consumer credit_sum", "bureau_CREDIT_TYPE_Consumer credit_nunique", "bureau_CREDIT_TYPE_Credit card_mean", "bureau_CREDIT_TYPE_Credit card_std", 
                                "bureau_CREDIT_TYPE_Credit card_min", "bureau_CREDIT_TYPE_Credit card_max", "bureau_CREDIT_TYPE_Credit card_sum", "bureau_CREDIT_TYPE_Credit card_nunique", "bureau_CREDIT_TYPE_OTHER_min"]

        bureau_stats_df.drop(useless_features_lst, axis = 1, inplace = True)

        ## Remove duplicated features
        bureau_stats_df["bureau_nb_records"] = bureau_stats_df["bureau_bureau_count"]
        duplicated_features_lst = ["bureau_AVG_CREDITDAYS_PROLONGED", "bureau_TOTAL_CUSTOMER_CREDIT", "bureau_TOTAL_CUSTOMER_DEBT", "bureau_TOTAL_CUSTOMER_OVERDUE", "bureau_CREDIT_ENDDATE_PERCENTAGE",
                                   "bureau_AVG_ENDDATE_FUTURE", "bureau_bureau_count", "bureau_nb_bureau_records", "bureau_bureau_count", "bureau_BUREAU_LOAN_COUNT", "bureau_nb_bureau_records", 
                                   "bureau_BUREAU_LOAN_COUNT"]

        bureau_stats_df.drop(duplicated_features_lst, axis = 1, inplace = True)

        # Processing 'previous_application.csv'
        previous_application_stats_df = self.transform_previous_applications_data(previous_application_data_df, installments_payments_data_df)

        # Doing some housekeeping
        ## Remove constant features
        tmp = previous_application_stats_df.nunique()
        constant_features_lst = tmp.loc[tmp == 1].index.tolist()
        previous_application_stats_df.drop(constant_features_lst, axis = 1, inplace = True)

        # Processing 'credit_card_balance.csv'
        card_balance_stats_df = self.transform_credit_card_balance_data(credit_card_balance_data_df)
        
        # Processing 'pos_cash_balance.csv'
        pos_cash_balance_stats_df = self.transform_pos_cash_balance_data(pos_cash_balance_data_df)

        # Processing 'installments_payments.csv'
        installments_payments_stats_df = self.transform_installments_payments_data(installments_payments_data_df)

        # Merging all datasets into one                
        final_dataset_df = previous_application_stats_df.reset_index().merge(bureau_stats_df, how = "left", on = "SK_ID_CURR")
        final_dataset_df = final_dataset_df.merge(card_balance_stats_df, how = "left", on = "SK_ID_CURR")
        final_dataset_df = final_dataset_df.merge(pos_cash_balance_stats_df.reset_index(), how = "left", on = "SK_ID_CURR")
        final_dataset_df = final_dataset_df.merge(installments_payments_stats_df.reset_index(), how = "left", on = "SK_ID_CURR")
        print("Additional files preprocessing data... done")

        print("*** Total additional features:", final_dataset_df.shape[1], "***")

        return final_dataset_df

    def fit_transform(self, target_df, bureau_data_df, bureau_balance_data_df, credit_card_balance_data_df, installments_payments_data_df, pos_cash_balance_data_df, previous_application_data_df):
        """
        Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        bureau_data_df : pd.DataFrame
                Additional data frame.

        bureau_balance_data_df : pd.DataFrame
                Additional data frame.

        credit_card_balance_data_df : pd.DataFrame
                Additional data frame.

        installments_payments_data_df : pd.DataFrame
                Additional data frame.

        pos_cash_balance_data_df : pd.DataFrame
                Additional data frame.

        previous_application_data_df : pd.DataFrame
                Additional data frame.
                
        Returns
        -------
        final_dataset_df : pd.DataFrame
                Transformed data.
        """
        
        self.y_train = target_df

        return self.fit(bureau_data_df, bureau_balance_data_df, credit_card_balance_data_df, installments_payments_data_df, pos_cash_balance_data_df, previous_application_data_df).transform(bureau_data_df, bureau_balance_data_df, credit_card_balance_data_df, installments_payments_data_df, pos_cash_balance_data_df, previous_application_data_df)