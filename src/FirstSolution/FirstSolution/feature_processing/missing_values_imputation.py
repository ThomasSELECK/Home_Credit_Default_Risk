#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# EMLEK - Efficient Machine LEarning toolKit                                  #
#                                                                             #
# This file contains the code needed to impute missing values occuring in the #
# data.                                                                       #
#                                                                             #
# Developped using Python 3.6.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-05-19                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import time
import warnings
import re
import string
from scipy.stats import boxcox

from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["MissingValuesImputer"
          ]

class MissingValuesImputer(BaseEstimator, TransformerMixin):
    """
    This class defines a Scikit-Learn transformer that imputes missing values on provided data.
    """

    def __init__(self, num_col_imputation = "mean", cat_col_imputation = "NA"):
        """
        This is the class' constructor.

        Parameters
        ----------
        num_col_imputation : string (either "mean" or "median") or float or int (default = "mean")
                Way of imputing missing values for numercial columns.

        cat_col_imputation : string (default = "NA")
                Way of imputing missing values for categorical columns. Replaces missing values
                with the value of this argument.
                                                
        Returns
        -------
        None
        """

        self.num_col_imputation = num_col_imputation
        self.cat_col_imputation = cat_col_imputation
        
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
        self: TextStatisticsGenerator object
                Return current object.
        """

        self._categorical_columns_lst = X.select_dtypes(["object"]).columns.tolist()
        self._numerical_columns_lst = list(set(X.columns.tolist()) - set(self._categorical_columns_lst))

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
                Transformed data.
        """

        # Remove infinite values
        X = X.replace([-np.inf, np.inf], np.nan)

        X[self._categorical_columns_lst] = X[self._categorical_columns_lst].fillna(self.cat_col_imputation)

        if self.num_col_imputation == "mean":
            columns_mean_sr = X[self._numerical_columns_lst].mean()
            X[self._numerical_columns_lst] = X[self._numerical_columns_lst].fillna(columns_mean_sr)
        elif self.num_col_imputation == "median":
            columns_median_sr = X[self._numerical_columns_lst].median()
            X[self._numerical_columns_lst] = X[self._numerical_columns_lst].fillna(columns_median_sr)
        else:
            X[self._numerical_columns_lst] = X[self._numerical_columns_lst].fillna(self.num_col_imputation)

        return X