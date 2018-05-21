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

        # Merge additional data to main dataframe
        X = X.merge(self._final_dataset_df, how = "left", left_index = True, right_on = "SK_ID_CURR")
        
        print("Preprocessing data... done")

        return X