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

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer

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
        self._previous_application_categ_feats_lst = ["NAME_CONTRACT_TYPE", "WEEKDAY_APPR_PROCESS_START", "FLAG_LAST_APPL_PER_CONTRACT", "NAME_CASH_LOAN_PURPOSE", "NAME_CONTRACT_STATUS", "NAME_PAYMENT_TYPE", 
                                                      "CODE_REJECT_REASON", "NAME_TYPE_SUITE", "NAME_CLIENT_TYPE", "NAME_GOODS_CATEGORY", "NAME_PORTFOLIO", "NAME_PRODUCT_TYPE", "CHANNEL_TYPE", 
                                                      "NAME_SELLER_INDUSTRY", "NAME_YIELD_GROUP", "PRODUCT_COMBINATION"]
        self._previous_application_encoders_lst = [LabelBinarizer(), LabelBinarizer(), OrdinalEncoder(), LabelBinarizer(), LabelBinarizer(), LabelBinarizer(), 
                                                   LabelBinarizer(), LabelBinarizer(), LabelBinarizer(), LabelBinarizer(), LabelBinarizer(), LabelBinarizer(), LabelBinarizer(), 
                                                   LabelBinarizer(), LabelBinarizer(), LabelBinarizer()]

        self._bureau_categ_feats_lst = ["CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE"]
        self._bureau_encoders_lst = [LabelBinarizer(), OrdinalEncoder(), GroupingEncoder(LabelBinarizer(), 4)]

        self._credit_card_balance_categ_feats_lst = credit_card_balance_data_df.select_dtypes(["object"]).columns.tolist()
        self._credit_card_balance_encoders_lst = [OrdinalEncoder() for _ in self._credit_card_balance_categ_feats_lst]

        self._pos_cash_balance_categ_feats_lst = pos_cash_balance_data_df.select_dtypes(["object"]).columns.tolist()
        self._pos_cash_balance_encoders_lst = [OrdinalEncoder() for _ in self._pos_cash_balance_categ_feats_lst]

        self._previous_application_cfe = CategoricalFeaturesEncoder(self._previous_application_categ_feats_lst, self._previous_application_encoders_lst)
        self._bureau_cfe = CategoricalFeaturesEncoder(self._bureau_categ_feats_lst, self._bureau_encoders_lst)
        self._credit_card_balance_cfe = CategoricalFeaturesEncoder(self._credit_card_balance_categ_feats_lst, self._credit_card_balance_encoders_lst)
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

        # Processing 'bureau_balance.csv'
        
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

        # Processing 'bureau.csv'
        print("    Pre-processing 'bureau.csv'...")

        # Computing credit profile for each customer
        ## Number of past loans per customer
        grp = bureau_data_df[["SK_ID_CURR", "DAYS_CREDIT"]].groupby(by = ["SK_ID_CURR"])["DAYS_CREDIT"].count().reset_index().rename(index = str, columns = {"DAYS_CREDIT": "BUREAU_LOAN_COUNT"})
        bureau_data_df = bureau_data_df.merge(grp, on = ["SK_ID_CURR"], how = "left")

        ## Number of types of past loans per customer
        grp = bureau_data_df[["SK_ID_CURR", "CREDIT_TYPE"]].groupby(by = ["SK_ID_CURR"])["CREDIT_TYPE"].nunique().reset_index().rename(index = str, columns = {"CREDIT_TYPE": "BUREAU_LOAN_TYPES"})
        bureau_data_df = bureau_data_df.merge(grp, on = ["SK_ID_CURR"], how = "left")

        ## Average number of past loans per type of loan per customer
        ### Average Number of Loans per Loan Type
        bureau_data_df["AVERAGE_LOAN_TYPE"] = bureau_data_df["BUREAU_LOAN_COUNT"] / bureau_data_df["BUREAU_LOAN_TYPES"]        
        gc.collect()

        ## Percentage of loans that are active per customer
        ### Create a new dummy column for whether CREDIT is ACTIVE OR CLOSED 
        bureau_data_df["CREDIT_ACTIVE_BINARY"] = (bureau_data_df["CREDIT_ACTIVE"].apply(lambda x: int(x != "Closed"))).astype(np.int8)
        
        ### Calculate mean number of loans that are ACTIVE per CUSTOMER 
        grp = bureau_data_df.groupby(by = ["SK_ID_CURR"])["CREDIT_ACTIVE_BINARY"].mean().reset_index().rename(index = str, columns = {"CREDIT_ACTIVE_BINARY": "ACTIVE_LOANS_PERCENTAGE"})
        bureau_data_df = bureau_data_df.merge(grp, on = ["SK_ID_CURR"], how = "left")
        bureau_data_df.drop("CREDIT_ACTIVE_BINARY", axis = 1, inplace = True)
        gc.collect()

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

        ## Percentage of loans per customer where end data for credit is past
        bureau_data_df["CREDIT_ENDDATE_BINARY"] = (bureau_data_df["DAYS_CREDIT_ENDDATE"].apply(lambda x: int(x >= 0))).astype(np.int8)
        grp = bureau_data_df.groupby(by = ["SK_ID_CURR"])["CREDIT_ENDDATE_BINARY"].mean().reset_index().rename(index = str, columns = {"CREDIT_ENDDATE_BINARY": "CREDIT_ENDDATE_PERCENTAGE"})
        bureau_data_df = bureau_data_df.merge(grp, on = ["SK_ID_CURR"], how = "left")
        gc.collect()

        ## Credit duration
        bureau_data_df["credit_duration"] = bureau_data_df["DAYS_CREDIT_ENDDATE"] - bureau_data_df["DAYS_CREDIT"]

        ## Number of days credit ends after initial end date
        bureau_data_df["nb_days_after_enddate"] = bureau_data_df["DAYS_ENDDATE_FACT"] - bureau_data_df["DAYS_CREDIT_ENDDATE"]

        ## Credit usage
        bureau_data_df["credit_usage"] = bureau_data_df["AMT_CREDIT_SUM_DEBT"] / bureau_data_df["AMT_CREDIT_SUM_LIMIT"]

        # Potential future delinquencies
        ## Average number of days in which credit expires in the future        
        ### We take only positive values of ENDDATE since we are looking at Bureau Credit VALID IN FUTURE 
        ### as of the date of the customer"s loan application with Home Credit 
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

        ### Calculate Average of DAYS_ENDDATE_DIFF
        grp = bureau_data_df[["SK_ID_CURR", "DAYS_ENDDATE_DIFF"]].groupby(by = ["SK_ID_CURR"])["DAYS_ENDDATE_DIFF"].mean().reset_index().rename(index = str, columns = {"DAYS_ENDDATE_DIFF": "AVG_ENDDATE_FUTURE"})
        bureau_data_df = bureau_data_df.merge(grp, on = ["SK_ID_CURR"], how = "left")
        del grp 
        bureau_data_df.drop(["CREDIT_ENDDATE_BINARY"], axis = 1, inplace = True)
        gc.collect()

        ## Ratio of total debt to total credit for each customer
        bureau_data_df["AMT_CREDIT_SUM_DEBT"] = bureau_data_df["AMT_CREDIT_SUM_DEBT"].fillna(0)
        bureau_data_df["AMT_CREDIT_SUM"] = bureau_data_df["AMT_CREDIT_SUM"].fillna(0)

        grp1 = bureau_data_df[["SK_ID_CURR", "AMT_CREDIT_SUM_DEBT"]].groupby(by = ["SK_ID_CURR"])["AMT_CREDIT_SUM_DEBT"].sum().reset_index().rename(index = str, columns = { "AMT_CREDIT_SUM_DEBT": "TOTAL_CUSTOMER_DEBT"})
        grp2 = bureau_data_df[["SK_ID_CURR", "AMT_CREDIT_SUM"]].groupby(by = ["SK_ID_CURR"])["AMT_CREDIT_SUM"].sum().reset_index().rename(index = str, columns = { "AMT_CREDIT_SUM": "TOTAL_CUSTOMER_CREDIT"})

        bureau_data_df = bureau_data_df.merge(grp1, on = ["SK_ID_CURR"], how = "left")
        bureau_data_df = bureau_data_df.merge(grp2, on = ["SK_ID_CURR"], how = "left")
        del grp2
        gc.collect()

        bureau_data_df["DEBT_CREDIT_RATIO"] = bureau_data_df["TOTAL_CUSTOMER_DEBT"] / bureau_data_df["TOTAL_CUSTOMER_CREDIT"]

        ## Total amount of active credit
        bureau_data_df["ACTIVE_AMT_CREDIT_SUM"] = bureau_data_df["AMT_CREDIT_SUM"] * (bureau_data_df["CREDIT_ACTIVE"].apply(lambda x: int(x != "Closed"))).astype(np.int8)
        grp1 = bureau_data_df[["SK_ID_CURR", "ACTIVE_AMT_CREDIT_SUM"]].groupby(by = ["SK_ID_CURR"])["ACTIVE_AMT_CREDIT_SUM"].sum().reset_index().rename(index = str, columns = { "ACTIVE_AMT_CREDIT_SUM": "TOTAL_ACTIVE_CUSTOMER_CREDIT"})
        bureau_data_df = bureau_data_df.merge(grp1, on = ["SK_ID_CURR"], how = "left")

        ## Total amount of active annuity
        bureau_data_df["AMT_ANNUITY"] = bureau_data_df["AMT_ANNUITY"].fillna(0)
        bureau_data_df["ACTIVE_AMT_ANNUITY"] = bureau_data_df["AMT_ANNUITY"] * (bureau_data_df["CREDIT_ACTIVE"].apply(lambda x: int(x != "Closed"))).astype(np.int8)
        grp1 = bureau_data_df[["SK_ID_CURR", "ACTIVE_AMT_ANNUITY"]].groupby(by = ["SK_ID_CURR"])["ACTIVE_AMT_ANNUITY"].sum().reset_index().rename(index = str, columns = { "ACTIVE_AMT_ANNUITY": "TOTAL_ACTIVE_AMT_ANNUITY"})
        bureau_data_df = bureau_data_df.merge(grp1, on = ["SK_ID_CURR"], how = "left")
        
        ## Fraction of total debt overdue for each customer
        bureau_data_df["AMT_CREDIT_SUM_OVERDUE"] = bureau_data_df["AMT_CREDIT_SUM_OVERDUE"].fillna(0)

        grp2 = bureau_data_df[["SK_ID_CURR", "AMT_CREDIT_SUM_OVERDUE"]].groupby(by = ["SK_ID_CURR"])["AMT_CREDIT_SUM_OVERDUE"].sum().reset_index().rename(index = str, columns = { "AMT_CREDIT_SUM_OVERDUE": "TOTAL_CUSTOMER_OVERDUE"})

        bureau_data_df = bureau_data_df.merge(grp2, on = ["SK_ID_CURR"], how = "left")
        del grp1, grp2
        gc.collect()

        bureau_data_df["OVERDUE_DEBT_RATIO"] = bureau_data_df["TOTAL_CUSTOMER_OVERDUE"] / bureau_data_df["TOTAL_CUSTOMER_DEBT"]
        
        ## Average number of loans prolonged
        bureau_data_df["CNT_CREDIT_PROLONG"] = bureau_data_df["CNT_CREDIT_PROLONG"].fillna(0)
        grp = bureau_data_df[["SK_ID_CURR", "CNT_CREDIT_PROLONG"]].groupby(by = ["SK_ID_CURR"])["CNT_CREDIT_PROLONG"].mean().reset_index().rename(index = str, columns = { "CNT_CREDIT_PROLONG": "AVG_CREDITDAYS_PROLONGED"})
        bureau_data_df = bureau_data_df.merge(grp, on = ["SK_ID_CURR"], how = "left")

        # Encode categorical features
        bureau_data_df = self._bureau_cfe.transform(bureau_data_df)

        # Impute NAs
        bureau_data_df[["AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_MAX_OVERDUE"]].fillna(0, inplace = True)
                
        # Compute average values and counts
        bureau_stats_df = self._flatten_columns_names(bureau_data_df.select_dtypes(include = np.number).drop("SK_ID_BUREAU", axis = 1).groupby("SK_ID_CURR").agg(["mean", "std", "min", "max", "sum", "nunique"]))
        bureau_stats_df["bureau_count"] = bureau_data_df[["SK_ID_BUREAU", "SK_ID_CURR"]].groupby("SK_ID_CURR").count()["SK_ID_BUREAU"]
        bureau_stats_df["nb_bureau_records"] = bureau_data_df.groupby("SK_ID_CURR").size()
        """avg_buro["avg_nb_records_by_bureau"] = bureau_data_df.groupby(["SK_ID_CURR", "SK_ID_BUREAU"]).size().groupby("SK_ID_CURR").mean()
        avg_buro["max_nb_records_by_bureau"] = bureau_data_df.groupby(["SK_ID_CURR", "SK_ID_BUREAU"]).size().groupby("SK_ID_CURR").max()
        avg_buro["min_nb_records_by_bureau"] = bureau_data_df.groupby(["SK_ID_CURR", "SK_ID_BUREAU"]).size().groupby("SK_ID_CURR").min()
        avg_buro["AMT_CREDIT_MAX_OVERDUE_sum"] = bureau_data_df.groupby(["SK_ID_CURR", "SK_ID_BUREAU"])["AMT_CREDIT_MAX_OVERDUE"].mean().groupby("SK_ID_CURR").sum()
        avg_buro["AMT_CREDIT_MAX_OVERDUE_sum_gt_zero"] = (avg_buro["AMT_CREDIT_MAX_OVERDUE_sum"] > 0).astype(np.int8)
        avg_buro["days_past_due_gt_0"] = bureau_data_df.groupby(["SK_ID_CURR", "SK_ID_BUREAU"])["STATUS_OTHER"].max().groupby("SK_ID_CURR").sum()
        avg_buro["CNT_CREDIT_PROLONG_sum"] = bureau_data_df.groupby(["SK_ID_CURR", "SK_ID_BUREAU"])["CNT_CREDIT_PROLONG"].max().groupby("SK_ID_CURR").sum()
        avg_buro["CNT_CREDIT_PROLONG_sum_gt_zero"] = (avg_buro["CNT_CREDIT_PROLONG_sum"] > 0).astype(np.int8)
        avg_buro["AMT_CREDIT_SUM_sum"] = bureau_data_df.groupby(["SK_ID_CURR", "SK_ID_BUREAU"])["AMT_CREDIT_SUM"].mean().groupby("SK_ID_CURR").sum()
        avg_buro["DAYS_CREDIT_ENDDATE_avg"] = bureau_data_df.groupby(["SK_ID_CURR", "SK_ID_BUREAU"])["DAYS_CREDIT_ENDDATE"].mean().groupby("SK_ID_CURR").mean()
        avg_buro["nb_active_credit"] = bureau_data_df.groupby(["SK_ID_CURR", "SK_ID_BUREAU"])["CREDIT_ACTIVE_Active"].max().groupby("SK_ID_CURR").sum()   """     
        
        #bureau_stats_df.drop("SK_ID_BUREAU", axis = 1, inplace = True)

        # Add prefix to columns
        bureau_stats_df.columns = ["bureau_" + c for c in bureau_stats_df.columns.tolist()]

        # Processing 'previous_application.csv'
        print("    Pre-processing 'previous_application.csv'...")

        previous_application_data_df["APP_CREDIT_PERC"] = previous_application_data_df["AMT_APPLICATION"] / previous_application_data_df["AMT_CREDIT"]

        # Impute missing values
        previous_application_data_df["NAME_TYPE_SUITE"].fillna("missing", inplace = True)

        # Encode categorical features
        previous_application_data_df = self._previous_application_cfe.transform(previous_application_data_df)

        # Compute average values and counts
        previous_application_stats_df = self._flatten_columns_names(previous_application_data_df.select_dtypes(include = np.number).drop("SK_ID_PREV", axis = 1).groupby("SK_ID_CURR").agg(["mean", "std", "min", "max", "sum", "nunique"]))
        previous_application_stats_df["nb_app"] = previous_application_data_df[["SK_ID_CURR", "SK_ID_PREV"]].groupby("SK_ID_CURR").count()["SK_ID_PREV"]
        #previous_application_stats_df.drop("SK_ID_PREV", axis = 1, inplace = True)

        # Add prefix to columns
        previous_application_stats_df.columns = ["previous_application_" + c for c in previous_application_stats_df.columns.tolist()]

        # Processing 'credit_card_balance.csv'
        print("    Pre-processing 'credit_card_balance.csv'...")

        # Customer risk profile
        ## Number of loans per customer
        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR"])["SK_ID_PREV"].nunique().reset_index().rename(index = str, columns = {"SK_ID_PREV": "NO_LOANS"})
        credit_card_balance_data_df = credit_card_balance_data_df.merge(grp, on = ["SK_ID_CURR"], how = "left")
        del grp 
        gc.collect()

        ## Rate at which loan is paid back by customer
        ### No of Installments paid per Loan per Customer 
        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR", "SK_ID_PREV"])["CNT_INSTALMENT_MATURE_CUM"].max().reset_index().rename(index = str, columns = {"CNT_INSTALMENT_MATURE_CUM": "NO_INSTALMENTS"})
        grp1 = grp.groupby(by = ["SK_ID_CURR"])["NO_INSTALMENTS"].sum().reset_index().rename(index = str, columns = {"NO_INSTALMENTS": "TOTAL_INSTALMENTS"})
        credit_card_balance_data_df = credit_card_balance_data_df.merge(grp1, on = ["SK_ID_CURR"], how = "left")

        ### Average Number of installments paid per loan 
        credit_card_balance_data_df["INSTALLMENTS_PER_LOAN"] = (credit_card_balance_data_df["TOTAL_INSTALMENTS"] / credit_card_balance_data_df["NO_LOANS"]).astype("uint32")

        ## How much did the customer load a credit line
        credit_card_balance_data_df["AMT_CREDIT_LIMIT_ACTUAL1"] = credit_card_balance_data_df["AMT_CREDIT_LIMIT_ACTUAL"]
        
        ### Calculate the ratio of Amount Balance to Credit Limit - CREDIT LOAD OF CUSTOMER 
        ### This is done for each Credit limit value per loan per Customer 
        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR", "SK_ID_PREV", "AMT_CREDIT_LIMIT_ACTUAL"]).apply(lambda x: x["AMT_BALANCE"].max() / x["AMT_CREDIT_LIMIT_ACTUAL1"].max()).reset_index().rename(index = str, columns = {0: "CREDIT_LOAD1"})
        credit_card_balance_data_df.drop("AMT_CREDIT_LIMIT_ACTUAL1", axis = 1, inplace = True)

        ### We now calculate the mean Credit load of All Loan transactions of Customer 
        grp1 = grp.groupby(by = ["SK_ID_CURR"])["CREDIT_LOAD1"].mean().reset_index().rename(index = str, columns = {"CREDIT_LOAD1": "CREDIT_LOAD"})
        credit_card_balance_data_df = credit_card_balance_data_df.merge(grp1, on = ["SK_ID_CURR"], how = "left")
        del grp, grp1
        gc.collect()
        
        ## How many times did the customer miss the minimum payment
        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR", "SK_ID_PREV"]).apply(lambda x: (x["SK_DPD"] != 0).astype(np.int8).sum()).reset_index().rename(index = str, columns = {0: "NO_DPD"})
        grp1 = grp.groupby(by = ["SK_ID_CURR"])["NO_DPD"].mean().reset_index().rename(index = str, columns = {"NO_DPD" : "DPD_COUNT"})

        credit_card_balance_data_df = credit_card_balance_data_df.merge(grp1, on = ["SK_ID_CURR"], how = "left")
        del grp, grp1 
        gc.collect()

        ## What is the average number of days did customer go past due date
        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR"])["SK_DPD"].mean().reset_index().rename(index = str, columns = {"SK_DPD": "AVG_DPD"})
        credit_card_balance_data_df = credit_card_balance_data_df.merge(grp, on = ["SK_ID_CURR"], how = "left")
        del grp 
        gc.collect()

        ## What fraction of minimum payments were missed
        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR"]).apply(lambda x: (100 * (x["AMT_INST_MIN_REGULARITY"] < x["AMT_PAYMENT_CURRENT"]).astype(np.int8).sum()) / x["AMT_INST_MIN_REGULARITY"].shape[0]).reset_index().rename(index = str, columns = { 0 : "PERCENTAGE_MISSED_PAYMENTS"})
        credit_card_balance_data_df = credit_card_balance_data_df.merge(grp, on = ["SK_ID_CURR"], how = "left")
        del grp 
        gc.collect()

        # Customer behavior patterns
        ## Cash withdrawals VS overall spending ratio
        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR"])["AMT_DRAWINGS_ATM_CURRENT"].sum().reset_index().rename(index = str, columns = {"AMT_DRAWINGS_ATM_CURRENT" : "DRAWINGS_ATM"})
        credit_card_balance_data_df = credit_card_balance_data_df.merge(grp, on = ["SK_ID_CURR"], how = "left")

        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR"])["AMT_DRAWINGS_CURRENT"].sum().reset_index().rename(index = str, columns = {"AMT_DRAWINGS_CURRENT" : "DRAWINGS_TOTAL"})
        credit_card_balance_data_df = credit_card_balance_data_df.merge(grp, on = ["SK_ID_CURR"], how = "left")

        credit_card_balance_data_df["CASH_CARD_RATIO1"] = (credit_card_balance_data_df["DRAWINGS_ATM"] / credit_card_balance_data_df["DRAWINGS_TOTAL"]) * 100

        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR"])["CASH_CARD_RATIO1"].mean().reset_index().rename(index = str, columns = {"CASH_CARD_RATIO1" : "CASH_CARD_RATIO"})
        credit_card_balance_data_df = credit_card_balance_data_df.merge(grp, on = ["SK_ID_CURR"], how = "left")
        del grp
        gc.collect()

        ## Avg number of drawings per customer; Total drawings / Number of drawings
        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR"])["AMT_DRAWINGS_CURRENT"].sum().reset_index().rename(index = str, columns = {"AMT_DRAWINGS_CURRENT" : "TOTAL_DRAWINGS"})
        credit_card_balance_data_df = credit_card_balance_data_df.merge(grp, on = ["SK_ID_CURR"], how = "left")
        del grp
        gc.collect()

        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR"])["CNT_DRAWINGS_CURRENT"].sum().reset_index().rename(index = str, columns = {"CNT_DRAWINGS_CURRENT" : "NO_DRAWINGS"})
        credit_card_balance_data_df = credit_card_balance_data_df.merge(grp, on = ["SK_ID_CURR"], how = "left")
        del grp
        gc.collect()

        credit_card_balance_data_df["DRAWINGS_RATIO1"] = (credit_card_balance_data_df["TOTAL_DRAWINGS"] / credit_card_balance_data_df["NO_DRAWINGS"]) * 100
        
        grp = credit_card_balance_data_df.groupby(by = ["SK_ID_CURR"])["DRAWINGS_RATIO1"].mean().reset_index().rename(index = str, columns = {"DRAWINGS_RATIO1" : "DRAWINGS_RATIO"})
        credit_card_balance_data_df = credit_card_balance_data_df.merge(grp, on = ["SK_ID_CURR"], how = "left")
        del grp 
        gc.collect()

        credit_card_balance_data_df["NUNIQUE_STATUS"] = credit_card_balance_data_df[["SK_ID_CURR", "NAME_CONTRACT_STATUS"]].groupby("SK_ID_CURR").nunique()["NAME_CONTRACT_STATUS"]
        credit_card_balance_data_df["NUNIQUE_STATUS2"] = credit_card_balance_data_df[["SK_ID_CURR", "NAME_CONTRACT_STATUS"]].groupby("SK_ID_CURR").max()["NAME_CONTRACT_STATUS"]

        # Encode categorical features
        credit_card_balance_data_df = self._credit_card_balance_cfe.transform(credit_card_balance_data_df)

        # Compute average values and counts
        card_balance_stats_df = self._flatten_columns_names(credit_card_balance_data_df.select_dtypes(include = np.number).drop("SK_ID_PREV", axis = 1).groupby("SK_ID_CURR").agg(["mean", "std", "min", "max", "sum", "nunique"]))
        card_balance_stats_df["nb_ccb"] = credit_card_balance_data_df[["SK_ID_CURR", "SK_ID_PREV"]].groupby("SK_ID_CURR").count()["SK_ID_PREV"]
        #card_balance_stats_df.drop("SK_ID_PREV", axis = 1, inplace = True)

        # Add prefix to columns
        card_balance_stats_df.columns = ["credit_card_balance_" + c for c in card_balance_stats_df.columns.tolist()]
        
        # Processing 'pos_cash_balance.csv'
        print("    Pre-processing 'pos_cash_balance.csv'...")

        pos_cash_balance_data_df["NUNIQUE_STATUS"] = pos_cash_balance_data_df[["SK_ID_CURR", "NAME_CONTRACT_STATUS"]].groupby("SK_ID_CURR").nunique()["NAME_CONTRACT_STATUS"]
        pos_cash_balance_data_df["NUNIQUE_STATUS2"] = pos_cash_balance_data_df[["SK_ID_CURR", "NAME_CONTRACT_STATUS"]].groupby("SK_ID_CURR").max()["NAME_CONTRACT_STATUS"]

        # Encode categorical features
        pos_cash_balance_data_df = self._pos_cash_balance_cfe.transform(pos_cash_balance_data_df)

        # Compute average values and counts
        pos_cash_balance_stats_df = self._flatten_columns_names(pos_cash_balance_data_df.select_dtypes(include = np.number).drop("SK_ID_PREV", axis = 1).groupby("SK_ID_CURR").agg(["mean", "std", "min", "max", "sum", "nunique"]))
        pos_cash_balance_stats_df["nb_pcb"] = pos_cash_balance_data_df[["SK_ID_CURR", "SK_ID_PREV"]].groupby("SK_ID_CURR").count()["SK_ID_PREV"]
        #pos_cash_balance_stats_df.drop("SK_ID_PREV", axis = 1, inplace = True)

        # Add prefix to columns
        pos_cash_balance_stats_df.columns = ["pos_cash_balance_" + c for c in pos_cash_balance_stats_df.columns.tolist()]

        # Processing 'installments_payments.csv'
        print("    Pre-processing 'installments_payments.csv'...")

        installments_payments_data_df["PAYMENT_PERC"] = installments_payments_data_df["AMT_PAYMENT"] / installments_payments_data_df["AMT_INSTALMENT"]
        installments_payments_data_df["PAYMENT_DIFF"] = installments_payments_data_df["AMT_INSTALMENT"] - installments_payments_data_df["AMT_PAYMENT"]
        """
        installments_payments_data_df["DPD"] = (installments_payments_data_df["DAYS_ENTRY_PAYMENT"] - installments_payments_data_df["DAYS_INSTALMENT"]).apply(lambda x: x if x > 0 else 0)
        installments_payments_data_df["DBD"] = (installments_payments_data_df["DAYS_INSTALMENT"] - installments_payments_data_df["DAYS_ENTRY_PAYMENT"]).apply(lambda x: x if x > 0 else 0)
        """

        installments_payments_stats_df = self._flatten_columns_names(installments_payments_data_df.select_dtypes(include = np.number).drop("SK_ID_PREV", axis = 1).groupby("SK_ID_CURR").agg(["mean", "std", "min", "max", "sum", "nunique"]))
        #installments_payments_stats_df.drop("SK_ID_PREV", axis = 1, inplace = True)
        
        # Add prefix to columns
        installments_payments_stats_df.columns = ["installments_payments_" + c for c in installments_payments_stats_df.columns.tolist()]
                
        final_dataset_df = previous_application_stats_df.reset_index().merge(bureau_stats_df.reset_index(), how = "left", on = "SK_ID_CURR")
        final_dataset_df = final_dataset_df.merge(card_balance_stats_df.reset_index(), how = "left", on = "SK_ID_CURR")
        final_dataset_df = final_dataset_df.merge(pos_cash_balance_stats_df.reset_index(), how = "left", on = "SK_ID_CURR")
        final_dataset_df = final_dataset_df.merge(installments_payments_stats_df.reset_index(), how = "left", on = "SK_ID_CURR")
        print("Additional files preprocessing data... done")

        #return bureau_data_df, bureau_balance_data_df, credit_card_balance_data_df, installments_payments_data_df, pos_cash_balance_data_df, previous_application_data_df
        return final_dataset_df

    def fit_transform(self, bureau_data_df, bureau_balance_data_df, credit_card_balance_data_df, installments_payments_data_df, pos_cash_balance_data_df, previous_application_data_df):
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
        
        return self.fit(bureau_data_df, bureau_balance_data_df, credit_card_balance_data_df, installments_payments_data_df, pos_cash_balance_data_df, previous_application_data_df).transform(bureau_data_df, bureau_balance_data_df, credit_card_balance_data_df, installments_payments_data_df, pos_cash_balance_data_df, previous_application_data_df)