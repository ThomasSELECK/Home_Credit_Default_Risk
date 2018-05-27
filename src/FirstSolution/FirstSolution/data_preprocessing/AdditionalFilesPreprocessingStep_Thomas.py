#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the Home Credit Default Risk Challenge                   #
#                                                                             #
# This file contains the code needed for the second preprocessing step.       #
# Developped using Python 3.6.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-05-27                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import numpy as np
import pandas as pd
import time

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
        buro_grouped_size = bureau_balance_data_df.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].size()
        buro_grouped_max = bureau_balance_data_df.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].max()
        buro_grouped_min = bureau_balance_data_df.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].min()

        buro_counts = bureau_balance_data_df.groupby("SK_ID_BUREAU")["STATUS"].value_counts(normalize = False)
        buro_counts_unstacked = buro_counts.unstack("STATUS")
        buro_counts_unstacked.columns = ["STATUS_0", "STATUS_1", "STATUS_2", "STATUS_3", "STATUS_4", "STATUS_5", "STATUS_C", "STATUS_X"]
        buro_counts_unstacked["MONTHS_COUNT"] = buro_grouped_size
        buro_counts_unstacked["MONTHS_MIN"] = buro_grouped_min
        buro_counts_unstacked["MONTHS_MAX"] = buro_grouped_max

        # Add prefix to columns
        buro_counts_unstacked.columns = ["bureau_balance_" + c for c in buro_counts_unstacked.columns.tolist()]

        bureau_data_df = bureau_data_df.join(buro_counts_unstacked, how = "left", on = "SK_ID_BUREAU")

        # Processing 'bureau.csv'
        print("    Pre-processing 'bureau.csv'...")

        # Encode categorical features
        bureau_data_df = self._bureau_cfe.transform(bureau_data_df)

        # Impute NAs
        bureau_data_df[["AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_MAX_OVERDUE"]].fillna(0, inplace = True)
                
        # Compute average values and counts
        avg_buro = bureau_data_df.groupby("SK_ID_CURR").mean()
        avg_buro["bureau_count"] = bureau_data_df[["SK_ID_BUREAU", "SK_ID_CURR"]].groupby("SK_ID_CURR").count()["SK_ID_BUREAU"]
        avg_buro["nb_bureau_records"] = bureau_data_df.groupby("SK_ID_CURR").size()
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
        
        avg_buro.drop("SK_ID_BUREAU", axis = 1, inplace = True)

        # Add prefix to columns
        avg_buro.columns = ["bureau_" + c for c in avg_buro.columns.tolist()]

        # Processing 'previous_application.csv'
        print("    Pre-processing 'previous_application.csv'...")

        # Impute missing values
        previous_application_data_df["NAME_TYPE_SUITE"].fillna("missing", inplace = True)

        # Encode categorical features
        previous_application_data_df = self._previous_application_cfe.transform(previous_application_data_df)

        # Compute average values and counts
        avg_prev = previous_application_data_df.groupby("SK_ID_CURR").mean()
        cnt_prev = previous_application_data_df[["SK_ID_CURR", "SK_ID_PREV"]].groupby("SK_ID_CURR").count()
        avg_prev["nb_app"] = cnt_prev["SK_ID_PREV"]
        avg_prev.drop("SK_ID_PREV", axis = 1, inplace = True)

        # Add prefix to columns
        avg_prev.columns = ["previous_application_" + c for c in avg_prev.columns.tolist()]

        # Processing 'credit_card_balance.csv'
        print("    Pre-processing 'credit_card_balance.csv'...")

        nunique_status = credit_card_balance_data_df[["SK_ID_CURR", "NAME_CONTRACT_STATUS"]].groupby("SK_ID_CURR").nunique()
        nunique_status2 = credit_card_balance_data_df[["SK_ID_CURR", "NAME_CONTRACT_STATUS"]].groupby("SK_ID_CURR").max()
        credit_card_balance_data_df["NUNIQUE_STATUS"] = nunique_status["NAME_CONTRACT_STATUS"]
        credit_card_balance_data_df["NUNIQUE_STATUS2"] = nunique_status2["NAME_CONTRACT_STATUS"]

        # Encode categorical features
        credit_card_balance_data_df = self._credit_card_balance_cfe.transform(credit_card_balance_data_df)

        # Compute average values and counts
        avg_ccb = credit_card_balance_data_df.groupby("SK_ID_CURR").mean()
        cnt_ccb = credit_card_balance_data_df[["SK_ID_CURR", "SK_ID_PREV"]].groupby("SK_ID_CURR").count()
        avg_ccb["nb_ccb"] = cnt_ccb["SK_ID_PREV"]
        avg_ccb.drop("SK_ID_PREV", axis = 1, inplace = True)

        # Add prefix to columns
        avg_ccb.columns = ["credit_card_balance_" + c for c in avg_ccb.columns.tolist()]
        
        # Processing 'pos_cash_balance.csv'
        print("    Pre-processing 'pos_cash_balance.csv'...")

        nunique_status = pos_cash_balance_data_df[["SK_ID_CURR", "NAME_CONTRACT_STATUS"]].groupby("SK_ID_CURR").nunique()
        nunique_status2 = pos_cash_balance_data_df[["SK_ID_CURR", "NAME_CONTRACT_STATUS"]].groupby("SK_ID_CURR").max()
        pos_cash_balance_data_df["NUNIQUE_STATUS"] = nunique_status["NAME_CONTRACT_STATUS"]
        pos_cash_balance_data_df["NUNIQUE_STATUS2"] = nunique_status2["NAME_CONTRACT_STATUS"]

        # Encode categorical features
        pos_cash_balance_data_df = self._pos_cash_balance_cfe.transform(pos_cash_balance_data_df)

        # Compute average values and counts
        avg_pcb = pos_cash_balance_data_df.groupby("SK_ID_CURR").mean()
        cnt_pcb = pos_cash_balance_data_df[["SK_ID_CURR", "SK_ID_PREV"]].groupby("SK_ID_CURR").count()
        avg_pcb["nb_pcb"] = cnt_pcb["SK_ID_PREV"]
        avg_pcb.drop("SK_ID_PREV", axis = 1, inplace = True)

        # Add prefix to columns
        avg_pcb.columns = ["pos_cash_balance_" + c for c in avg_pcb.columns.tolist()]

        # Processing 'installments_payments.csv'
        print("    Pre-processing 'installments_payments.csv'...")

        avg_payments = installments_payments_data_df.groupby("SK_ID_CURR").mean()
        avg_payments2 = installments_payments_data_df.groupby("SK_ID_CURR").max()
        avg_payments3 = installments_payments_data_df.groupby("SK_ID_CURR").min()

        avg_payments.drop("SK_ID_PREV", axis = 1, inplace = True)
        avg_payments2.drop("SK_ID_PREV", axis = 1, inplace = True)
        avg_payments3.drop("SK_ID_PREV", axis = 1, inplace = True)

        # Add prefix to columns
        avg_payments.columns = ["installments_payments_" + c for c in avg_payments.columns.tolist()]
        avg_payments2.columns = ["installments_payments2_" + c for c in avg_payments2.columns.tolist()]
        avg_payments3.columns = ["installments_payments3_" + c for c in avg_payments3.columns.tolist()]
                
        final_dataset_df = avg_prev.reset_index().merge(avg_buro.reset_index(), how = "left", on = "SK_ID_CURR")
        final_dataset_df = final_dataset_df.merge(avg_ccb.reset_index(), how = "left", on = "SK_ID_CURR")
        final_dataset_df = final_dataset_df.merge(avg_pcb.reset_index(), how = "left", on = "SK_ID_CURR")
        final_dataset_df = final_dataset_df.merge(avg_payments.reset_index(), how = "left", on = "SK_ID_CURR")
        final_dataset_df = final_dataset_df.merge(avg_payments2.reset_index(), how = "left", on = "SK_ID_CURR")
        final_dataset_df = final_dataset_df.merge(avg_payments3.reset_index(), how = "left", on = "SK_ID_CURR")
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