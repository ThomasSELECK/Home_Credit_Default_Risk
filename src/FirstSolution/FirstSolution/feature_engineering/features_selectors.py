#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# EMLEK - Efficient Machine LEarning toolKit                                  #
#                                                                             #
# This file contains the code needed for selecting the best subset of features#
# from a set of features. It is compatible with the Scikit-Learn framework.   #
#                                                                             #
# Developped using Python 3.6.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-05-01                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import numpy as np
import pandas as pd
import time
import warnings
import networkx as nx

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectPercentile
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler

from wrappers.lightgbm_wrapper import LGBMClassifier, LGBMRegressor

__all__ = ["VarianceFeatureSelector",
           "L1NormFeatureSelector",
           "LGBMFeatureSelector",
           "ConstantFeaturesRemover",
           "DuplicatedFeaturesRemover"
          ]

class VarianceFeatureSelector(BaseEstimator, TransformerMixin):
    """
    This class defines a Scikit-Learn transformer that removes features with low variance.
    """

    def __init__(self, threshold):
        """
        This is the class' constructor.

        Parameters
        ----------
        threshold : float
                Features with a variance lower than this threshold will be removed.
                                                
        Returns
        -------
        None
        """

        self.threshold = threshold
        self.selector = None
        
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

        self.selector = VarianceThreshold(self.threshold)
        self.selector.fit(X, y)

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

        X = pd.DataFrame(self.selector.transform(X), index = X.index, columns = X.columns[self.selector.get_support()])

        return X

class L1NormFeatureSelector(BaseEstimator, TransformerMixin):
    """
    This class defines a Scikit-Learn transformer that selects features based on most important
    coefficients from linear model with L1 regularization.
    """

    def __init__(self, l1_penalty = 0.01, problem_type = "regression"):
        """
        This is the class' constructor.

        Parameters
        ----------
        threshold : float (default = 0.01)
                This indicates the percentage of features to keep.

        problem_type : string, either "regression" or "classification" (default = "regression")
                Indicates which type of problem we want to solve (regression or classification)
                                                
        Returns
        -------
        None
        """

        self.l1_penalty = l1_penalty
        self.problem_type = problem_type

        self._removed_columns_lst = []
        self._model = None
        
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

        # Choose the right model depending on problem type
        if self.problem_type == "classification":
            self._model = LogisticRegression(C = self.l1_penalty, solver = "saga", penalty = "l1", n_jobs = -1, random_state = 0)
        elif self.problem_type == "regression":
            self._model = Lasso(alpha = self.l1_penalty, random_state = 0)

        # Fit the model
        self._model.fit(X, y)

        # Get the features that will be dropped based on model's coefficients
        if self.problem_type == "classification":
            coeffs = np.mean(np.abs(self._model.coef_), axis = 0)
            self._removed_columns_lst = X.columns[coeffs == 0].tolist()

        elif self.problem_type == "regression":
            self._removed_columns_lst = X.columns[np.abs(self._model.coef_) == 0].tolist()       
        
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

        X.drop(self._removed_columns_lst, axis = 1, inplace = True)

        return X

class LGBMFeatureSelector(BaseEstimator, TransformerMixin):
    """
    This class defines a Scikit-Learn transformer that selects features based on LightGBM
    feature importance.
    """

    def __init__(self, threshold, problem_type = "regression", lgbm_params = None, enable_cv = True):
        """
        This is the class' constructor.

        Parameters
        ----------
        threshold : float between 0 and 1
                This indicates the percentage of features to keep.

        problem_type : string, either "regression" or "classification" (default = "regression")
                Indicates which type of problem we want to solve (regression or classification)

        lgbm_params : dict (default = None)
                Hyperparameters for LightGBM

        enable_cv : bool (default = True)
                Flag indicating if LightGBM will use Cross Validation to find the best number of epochs.
                                                
        Returns
        -------
        None
        """

        self.threshold = threshold
        self.problem_type = problem_type
        if lgbm_params is None:
            self.lgbm_params = {
                "learning_rate": 0.05,
                "max_depth": 10,
                "num_leaves": 100,
                "verbosity": -1,
                "colsample_bytree": 0.8,
                "subsample": 0.8,
                "subsample_freq": 100
            }            
        else:
            self.lgbm_params = lgbm_params

        self.enable_cv = enable_cv
        self._removed_columns_lst = []
        self._model = None
        
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

        # Choose the right model depending on problem type
        if self.problem_type == "classification":
            # Add classification hyperparameters
            num_classes = np.unique(y).shape[0]
            if num_classes == 2:
                self.lgbm_params["application"] = "binary"
                self.lgbm_params["metric"] = "binary_logloss"
            elif num_classes > 2:
                self.lgbm_params["application"] = "multiclass"
                self.lgbm_params["num_class"] = num_classes
                self.lgbm_params["metric"] = "multi_logloss"

            if self.enable_cv:
                self._model = LGBMClassifier(self.lgbm_params, early_stopping_rounds = 50, random_state = 0, test_size = 0.15, verbose_eval = 100)
            else:
                self._model = LGBMClassifier(self.lgbm_params, early_stopping_rounds = 50, random_state = 0, test_size = 0.15, verbose_eval = 100, nrounds = 10000)

        elif self.problem_type == "regression":
            # Add regression hyperparameters
            self.lgbm_params["application"] = "regression"
            self.lgbm_params["metric"] = "mse"

            if self.enable_cv:
                self._model = LGBMRegressor(self.lgbm_params, early_stopping_rounds = 50, random_state = 0, test_size = 0.15, verbose_eval = 100)
            else:
                self._model = LGBMRegressor(self.lgbm_params, early_stopping_rounds = 50, random_state = 0, test_size = 0.15, verbose_eval = 100, nrounds = 10000)

        # Fit the model
        self._model.fit(X, y)

        # Compute model's feature importance
        feature_importance_df = self._model.get_features_importance()

        # Only keep the first features
        nb_features_to_keep = int(feature_importance_df.shape[0] * self.threshold)
        self._removed_columns_lst = feature_importance_df["feature"].tail(feature_importance_df.shape[0] - nb_features_to_keep).tolist()

        # As the feature importance function replaces spaces in feature names by underscores, reverse this transformation
        transformation_dict = {}
        for key in X.columns.tolist():
            transformation_dict[key.replace(" ", "_")] = key

        for col_idx in range(len(self._removed_columns_lst)):
            self._removed_columns_lst[col_idx] = transformation_dict[self._removed_columns_lst[col_idx]]

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

        X.drop(self._removed_columns_lst, axis = 1, inplace = True)

        return X

class ConstantFeaturesRemover(BaseEstimator, TransformerMixin):
    """
    This class defines a Scikit-Learn transformer that removes features that are constant (zero variance).
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

        self._constant_features_lst = None
        
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
        
        print("Removing constant features...")

        # Get number of unique values for each feature
        unique_values_sr = X.nunique()
        self._constant_features_lst = unique_values_sr.loc[unique_values_sr == 1].index.tolist()

        # Add features that only contains NaNs
        tmp = X.isnull().all()
        self._constant_features_lst += tmp.loc[tmp == True].index.tolist()

        if len(self._constant_features_lst) > 0:
            print("Constant features that will be removed:")
            for f in self._constant_features_lst:
                print("    -", f)
        else:
            print("    No constant features found!")

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
            
        # Remove features that only have one unique value
        X.drop(self._constant_features_lst, axis = 1, inplace = True)

        return X

class DuplicatedFeaturesRemover(BaseEstimator, TransformerMixin):
    """
    This class defines a Scikit-Learn transformer that removes features that are duplicated.
    """

    def __init__(self, ignored_features_lst = []):
        """
        This is the class' constructor.

        Parameters
        ----------
        ignored_features_lst : list (default = [])
                This list contains the name of all features that will be ignored by the detection of duplicates.
                                                
        Returns
        -------
        None
        """

        self.ignored_features_lst = ignored_features_lst
        self._duplicated_features_lst = []
        
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

        print("Removing duplicated features...")

        # Copy the object to avoid any modification
        X_copy = X.copy(deep = True)
        
        features_lst = list(set(X_copy.columns.tolist()) - set(self.ignored_features_lst))
        constant_features_lst = []
        G = nx.Graph()

        for i in range(len(features_lst)):
            for j in range(i + 1, len(features_lst)):
                if X_copy[features_lst[i]].nunique() == 1 or X_copy[features_lst[i]].isnull().all():
                    if features_lst[i] not in constant_features_lst:
                        constant_features_lst.append(features_lst[i])
                        print("    -", features_lst[i], "is constant. Please use features_selectors.ConstantFeaturesRemover to remove it!")
                    continue

                if X_copy[features_lst[j]].nunique() == 1 or X_copy[features_lst[j]].isnull().all():
                    if features_lst[j] not in constant_features_lst:
                        constant_features_lst.append(features_lst[j])
                        print("    -", features_lst[j], "is constant. Please use features_selectors.ConstantFeaturesRemover to remove it!")
                    continue

                try: # For mixed type columns (containing numbers and strings), Pandas crosstab can fail.
                    confusion_matrix_df = pd.crosstab(X_copy[features_lst[i]], X_copy[features_lst[j]], normalize = "index")
                except:
                    X_copy[features_lst[i]] = X_copy[features_lst[i]].astype(str)
                    X_copy[features_lst[j]] = X_copy[features_lst[j]].astype(str)

                    confusion_matrix_df = pd.crosstab(X_copy[features_lst[i]], X_copy[features_lst[j]], normalize = "index")
                    
                if confusion_matrix_df.shape[0] == confusion_matrix_df.shape[1] and np.count_nonzero(confusion_matrix_df) == confusion_matrix_df.shape[0] and features_lst[i] not in self._duplicated_features_lst:
                    G.add_edge(features_lst[i], features_lst[j])

        # Get all connected components
        connected_components_lst = list(nx.connected_components(G))

        # For each connected component, if it's a complete graph, then features given by the nodes are duplicated
        for connected_component in connected_components_lst:
            connected_component = list(connected_component)

            # Get degree of each node
            nodes_degrees_set = list(set(dict(G.degree(connected_component)).values()))

            # If the subgraph is complete
            if len(nodes_degrees_set) == 1 and nodes_degrees_set[0] == len(connected_component) - 1:
                print("    - Feature:", connected_component[0], "is duplicated, the duplicates are:", ", ".join(connected_component[1:]))
                self._duplicated_features_lst.extend(connected_component[1:])

        if len(self._duplicated_features_lst) > 0:
            print("\nDuplicated features that will be removed:")
            for f in self._duplicated_features_lst:
                print("    -", f)
        else:
            print("    No duplicated feature found!")

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
        
        # Remove features that only have one unique value
        X.drop(self._duplicated_features_lst, axis = 1, inplace = True)

        return X