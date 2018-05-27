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