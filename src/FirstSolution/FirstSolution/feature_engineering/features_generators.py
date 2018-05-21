#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# EMLEK - Efficient Machine LEarning toolKit                                  #
#                                                                             #
# This file contains the code needed for creating various types of features,  # 
# based either on statistics or interactions between varaibles. It is         #
# compatible with the Scikit-Learn framework.                                 #
#                                                                             #
# Developped using Python 3.6.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-04-28                                                            #
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
import gc

from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["TextStatisticsGenerator",
           "PairwiseNumericalInteractionsGenerator",
           "DateTimeFeaturesExtractor"
          ]

class TextStatisticsGenerator(BaseEstimator, TransformerMixin):
    """
    This class defines a Scikit-Learn transformer that implements a generator of statistics based on text data.
    """

    def __init__(self, columns_names_lst, concatenate_text = "no"):
        """
        This is the class' constructor.

        Parameters
        ----------
        columns_names_lst : list
                Names of the columns we want to transform.

        concatenate_text : string in {"yes", "no", "both"} (default = "no")
                - If "yes", concatenate the strings of all columns given in 'columns_names_lst' 
                  and then compute statistics for the concatenated text.
                - If "no", compute statistics for each column given in 'columns_names_lst'.
                - If "both", compute statistics for the concatenated text and for each column 
                  taken separately.
                                                
        Returns
        -------
        None
        """

        self.columns_names_lst = columns_names_lst
        self.concatenate_text = concatenate_text
        
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

        # If self.concatenate_text is "yes" or "both", concatenate the text from given columns
        if self.concatenate_text in ["yes", "both"]:
            for column in self.columns_names_lst:
                if not "concatenated_text" in X.columns.tolist():
                    X["concatenated_text"] = X[column]
                else:
                    X["concatenated_text"] = X["concatenated_text"].str.cat(X[column], sep = " ")

        # Add concatenated column to columns list
        if self.concatenate_text == "yes":
            self.columns_names_lst = ["concatenated_text"]
        elif self.concatenate_text == "both":
            self.columns_names_lst.append("concatenated_text")

        # Compute the statistics for each column and save them into the data frame.
        for column in self.columns_names_lst:
            X[column + "_nb_chars"] = X[column].str.len()
            X[column + "_nb_tokens"] = X[column].str.lower().str.split(" ").str.len()
            X[column + "_nb_words"] = X[column].str.count("(\s|^)[a-z]+(\s|$)")
            X[column + "_nb_numbers"] = X[column].str.count("(\s|^)[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?(\s|$)")
            X[column + "_nb_letters"] = X[column].str.count("[a-zA-Z]")
            X[column + "_nb_uppercase_letters"] = X[column].str.count("[A-Z]")
            X[column + "_nb_lowercase_letters"] = X[column].str.count("[a-z]")
            X[column + "_nb_digits"] = X[column].str.count("[0-9]")
            X[column + "_nb_punctuation_signs"] = X[column].str.count("[" + re.escape(string.punctuation) + "]")
            X[column + "_nb_whitespaces"] = X[column].str.count("\s")
            X[column + "_nb_special_chars"] = X[column + "_nb_chars"] - X[column + "_nb_whitespaces"] - X[column + "_nb_punctuation_signs"] - X[column + "_nb_digits"] - X[column + "_nb_letters"]            

        # Drop 'concatenated_text' feature
        if "concatenated_text" in X.columns.tolist():
            X.drop("concatenated_text", axis = 1, inplace = True)

        return X

class PairwiseNumericalInteractionsGenerator(BaseEstimator, TransformerMixin):
    """
    This class defines a Scikit-Learn transformer that implements a generator of interactions based on pairs of numerical features.
    """

    def __init__(self, columns_names_lst = None, drop_interactions_with_overflow = True):
        """
        This is the class' constructor.

        Parameters
        ----------
        columns_names_lst : list (default = None)
                Names of the columns we want to use to generate interactions. If some of the columns 
                indicated in this list are not numerical, they'll be ignored.
                If this argument is None, all numerical columns will be used. Don't forget that pairwise 
                interactions generation generates O((n * (n - 1)) / 2) features where n is the number of
                initial features. This can greatly increase the memory consumption of the dataset.

        drop_interactions_with_overflow : bool (default = True)
                Drop interactions that causes overflow (e.g. exponential of a large number) if True.
                                                
        Returns
        -------
        None
        """

        self.columns_names_lst = columns_names_lst
        self.drop_interactions_with_overflow = drop_interactions_with_overflow
        self.columns_with_overflow_lst = []
        
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

        # Get numeric columns from X
        numerical_columns_lst = X.select_dtypes(include = [np.number]).columns.tolist()
        
        # If there is no numerical columns in the data, raise exception
        if len(numerical_columns_lst) == 0:
            raise TypeError("No numerical columns detected in your data. Add numerical data before using this transformer.")

        # Check every feature in self.columns_names_lst is numerical
        if self.columns_names_lst is not None and len(list(set(self.columns_names_lst) - set(numerical_columns_lst))) > 0:
            warnings.warn("Columns: " + str(list(set(self.columns_names_lst) - set(numerical_columns_lst))) + " are not numeric. They'll be ignored.")

            self.columns_names_lst = list(set(self.columns_names_lst) & set(numerical_columns_lst))

        elif self.columns_names_lst is None:
            self.columns_names_lst = numerical_columns_lst

            # Remove binary features
            binary_features_lst = X[numerical_columns_lst].columns[X[numerical_columns_lst].apply(lambda x: x.nunique() == 2)].tolist()
            print("binary_features_lst:", len(binary_features_lst))

            self.columns_names_lst = list(set(self.columns_names_lst) - set(binary_features_lst))
            
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
        
        if len(self.columns_names_lst) > 1: # If we have at least two features
            for i in range(len(self.columns_names_lst)):
                print("Generating features for i =", i, "/", len(self.columns_names_lst))
                for j in range(i + 1, len(self.columns_names_lst)):
                    # Get features names
                    f1 = self.columns_names_lst[i]
                    f2 = self.columns_names_lst[j]

                    # Generate interactions
                    X[f1 + "_+_" + f2] = X[f1] + X[f2]
                    X[f1 + "_-_" + f2] = X[f1] - X[f2]
                    X[f1 + "_*_" + f2] = X[f1] * X[f2]
                    X[f1 + "_/_" + f2] = X[f1] / X[f2]
                    X[f1 + "_+_log(1_+_" + f2 + ")"] = X[f1] + np.log1p(X[f2])
                    X[f1 + "_-_log(1_+_" + f2 + ")"] = X[f1] - np.log1p(X[f2])
                    X[f1 + "_*_log(1_+_" + f2 + ")"] = X[f1] * np.log1p(X[f2])
                    X[f1 + "_+_exp(" + f2 + ")"] = X[f1] + np.exp(X[f2])
                    X[f1 + "_-_exp(" + f2 + ")"] = X[f1] - np.exp(X[f2])
                    X[f1 + "_*_exp(" + f2 + ")"] = X[f1] * np.exp(X[f2])
                    X[f1 + "_+_sqrt(" + f2 + ")"] = X[f1] + np.sqrt(X[f2])
                    X[f1 + "_-_sqrt(" + f2 + ")"] = X[f1] - np.sqrt(X[f2])
                    X[f1 + "_*_sqrt(" + f2 + ")"] = X[f1] * np.sqrt(X[f2])
                    X["log(1_+_" + f1 + ")_+_" + f2] = np.log1p(X[f1]) + X[f2]
                    X["log(1_+_" + f1 + ")_-_" + f2] = np.log1p(X[f1]) - X[f2]
                    X["log(1_+_" + f1 + ")_*_" + f2] = np.log1p(X[f1]) * X[f2]
                    X["exp(" + f1 + ")_+_" + f2] = np.exp(X[f1]) + X[f2]
                    X["exp(" + f1 + ")_-_" + f2] = np.exp(X[f1]) - X[f2]
                    X["exp(" + f1 + ")_*_" + f2] = np.exp(X[f1]) * X[f2]
                    X["sqrt(" + f1 + ")_+_" + f2] = np.sqrt(X[f1]) + X[f2]
                    X["sqrt(" + f1 + ")_-_" + f2] = np.sqrt(X[f1]) - X[f2]
                    X["sqrt(" + f1 + ")_*_" + f2] = np.sqrt(X[f1]) * X[f2]

                gc.collect()

            if self.drop_interactions_with_overflow:
                # Get columns with overflow
                if len(self.columns_with_overflow_lst) == 0: # If columns are not already set (think train / test)
                    self.columns_with_overflow_lst = X.columns[X.isin([np.inf, -np.inf]).any(axis = 0)].tolist()

                # Drop the columns with overflow
                X.drop(self.columns_with_overflow_lst, axis = 1, inplace = True)

        print("X.shape after interactions generation:", X.shape)

        return X

class DateTimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    """
    This class defines a Scikit-Learn transformer that extracts features from DateTime Pandas feature.
    """

    def __init__(self, columns_names_lst):
        """
        This is the class' constructor.

        Parameters
        ----------
        columns_names_lst : list
                Names of the columns we want to transform.
                                                                
        Returns
        -------
        None
        """

        self.columns_names_lst = columns_names_lst
        
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

        for column in self.columns_names_lst:
            # Extract year, month, day
            X[column + "_year"] = X[column].dt.year
            X[column + "_month"] = X[column].dt.month
            X[column + "_day"] = X[column].dt.day
            X[column + "_day_of_week"] = X[column].dt.dayofweek

            # Extract hour, minutes, seconds
            X[column + "_hour"] = X[column].dt.hour
            X[column + "_minutes"] = X[column].dt.minute
            X[column + "_seconds"] = X[column].dt.second

        # Get time difference in seconds between dates
        if len(self.columns_names_lst) > 1: # If we have at least two features
            for i in range(len(self.columns_names_lst)):
                for j in range(i + 1, len(self.columns_names_lst)):
                    # Get features names
                    f1 = self.columns_names_lst[i]
                    f2 = self.columns_names_lst[j]

                    # Compute absolute value of time difference
                    X[f1 + "_" + f2 + "_time_diff_secs"] = np.abs(X[f1] - X[f2]).dt.seconds

        return X