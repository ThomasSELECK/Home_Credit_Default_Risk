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
# Date: 2017-12-09                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import numpy as np
import pandas as pd
import time

from sklearn.base import BaseEstimator, TransformerMixin

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

        # Get list of categorical columns for each dataset
        self._previous_application_categ_feats_lst = previous_application_data_df.select_dtypes(["object"]).columns.tolist()
        self._bureau_categ_feats_lst = bureau_data_df.select_dtypes(["object"]).columns.tolist()
        self._credit_card_balance_categ_feats_lst = credit_card_balance_data_df.select_dtypes(["object"]).columns.tolist()
        self._pos_cash_balance_categ_feats_lst = pos_cash_balance_data_df.select_dtypes(["object"]).columns.tolist()

        # Create encoders for encoding categorical variables
        self._previous_application_encoders_lst = [OrdinalEncoder() for _ in self._previous_application_categ_feats_lst]
        self._bureau_encoders_lst = [OrdinalEncoder() for _ in self._bureau_categ_feats_lst]
        self._credit_card_balance_encoders_lst = [OrdinalEncoder() for _ in self._credit_card_balance_categ_feats_lst]
        self._pos_cash_balance_encoders_lst = [OrdinalEncoder() for _ in self._pos_cash_balance_categ_feats_lst]

        self._previous_application_cfe = CategoricalFeaturesEncoder(self._previous_application_categ_feats_lst, self._previous_application_encoders_lst)
        self._bureau_cfe = CategoricalFeaturesEncoder(self._bureau_categ_feats_lst, self._bureau_encoders_lst)
        self._credit_card_balance_cfe = CategoricalFeaturesEncoder(self._credit_card_balance_categ_feats_lst, self._credit_card_balance_encoders_lst)
        self._pos_cash_balance_cfe = CategoricalFeaturesEncoder(self._pos_cash_balance_categ_feats_lst, self._pos_cash_balance_encoders_lst)

        # Processing 'previous_application.csv'

        # Encode categorical features
        self._previous_application_cfe.fit(previous_application_data_df)
        
        # Processing 'bureau.csv'

        # Encode categorical features
        bureau_data_df = self._bureau_cfe.fit(bureau_data_df)

        # Processing 'credit_card_balance_data_df.csv'

        # Encode categorical features
        credit_card_balance_data_df = self._credit_card_balance_cfe.fit(credit_card_balance_data_df)

        # Processing 'pos_cash_balance_data_df.csv'

        # Encode categorical features
        pos_cash_balance_data_df = self._pos_cash_balance_cfe.fit(pos_cash_balance_data_df)
        
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

        # Processing 'previous_application.csv'

        # Encode categorical features
        previous_application_data_df = self._previous_application_cfe.transform(previous_application_data_df)

        # Compute average values and counts
        avg_prev = previous_application_data_df.groupby("SK_ID_CURR").mean()
        cnt_prev = previous_application_data_df[["SK_ID_CURR", "SK_ID_PREV"]].groupby("SK_ID_CURR").count()
        avg_prev["nb_app"] = cnt_prev["SK_ID_PREV"]
        avg_prev.drop("SK_ID_PREV", axis = 1, inplace = True)

        # Processing 'bureau.csv'
        
        # Encode categorical features
        bureau_data_df = self._bureau_cfe.transform(bureau_data_df)
        
        # Compute average values and counts
        avg_buro = bureau_data_df.groupby("SK_ID_CURR").mean()
        avg_buro["buro_count"] = bureau_data_df[["SK_ID_BUREAU", "SK_ID_CURR"]].groupby("SK_ID_CURR").count()["SK_ID_BUREAU"]
        avg_buro.drop("SK_ID_BUREAU", axis = 1, inplace = True)

        # Processing 'credit_card_balance_data_df.csv'

        # Encode categorical features
        credit_card_balance_data_df = self._credit_card_balance_cfe.transform(credit_card_balance_data_df)

        # Compute average values and counts
        avg_ccb = credit_card_balance_data_df.groupby("SK_ID_CURR").mean()
        cnt_ccb = credit_card_balance_data_df[["SK_ID_CURR", "SK_ID_PREV"]].groupby("SK_ID_CURR").count()
        avg_ccb["nb_ccb"] = cnt_ccb["SK_ID_PREV"]
        avg_ccb.drop("SK_ID_PREV", axis = 1, inplace = True)
        
        # Processing 'pos_cash_balance_data_df.csv'

        # Encode categorical features
        pos_cash_balance_data_df = self._pos_cash_balance_cfe.transform(pos_cash_balance_data_df)

        # Compute average values and counts
        avg_pcb = pos_cash_balance_data_df.groupby("SK_ID_CURR").mean()
        cnt_pcb = pos_cash_balance_data_df[["SK_ID_CURR", "SK_ID_PREV"]].groupby("SK_ID_CURR").count()
        avg_pcb["nb_pcb"] = cnt_pcb["SK_ID_PREV"]
        avg_pcb.drop("SK_ID_PREV", axis = 1, inplace = True)
                
        final_dataset_df = avg_prev.reset_index().merge(avg_buro.reset_index(), how = "left", on = "SK_ID_CURR")
        final_dataset_df = final_dataset_df.merge(avg_ccb.reset_index(), how = "left", on = "SK_ID_CURR")
        final_dataset_df = final_dataset_df.merge(avg_pcb.reset_index(), how = "left", on = "SK_ID_CURR")
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