#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# EMLEK - Efficient Machine LEarning toolKit                                  #
#                                                                             #
# This file is an advanced wrapper for the package LightGBM that is           #
# compatible with scikit-learn API.                                           #
#                                                                             #
# Credits for LightGBM: https://github.com/Microsoft/LightGBM                 #
#                                                                             #
# Developped using Python 3.6.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-04-01                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from sklearn.metrics import r2_score

__all__ = ["LGBMClassifier",
           "LGBMRegressor",
          ]

class AbstractLGBMWrapper(ABC, BaseEstimator):
    """
    The purpose of this class is to provide a wrapper for a LightGBM model, with cross-validation for finding the best number of rounds.
    """
    
    def __init__(self, params, early_stopping_rounds, custom_eval_function = None, maximize = True, nrounds = 10000, random_state = 0, test_size = 0.1, verbose_eval = 1, enable_cv = True):
        """
        Class' constructor

        Parameters
        ----------
        params : dictionary
                This contains the parameters of the LightGBM model.

        early_stopping_rounds : integer
                This indicates the number of rounds to keep before stopping training when the score doesn't increase. If negative, disable this functionality.

        verbose_eval : positive integer
                This indicates the frequency of scores printing. E.g. 50 means one score printed every 50 rounds.

        custom_eval_function : function
                This is a function LightGBM will use as loss function.

        maximize : boolean
                Indicates if the function customEvalFunction must be maximized or minimized. Not used when customEvalFunction is None.

        nrounds : integer
                Number of rounds for LightGBM training.

        random_state : zero or positive integer
                Seed used by LightGBM to ensure reproductibility.

        test_size : float between 0 and 1.
                This indicates the size of the test set.

        verbose_eval : bool or int, optional (default = 1)
                If True, the eval metric on the valid set is printed at each boosting stage. If int, the eval metric on the valid set is 
                printed at every verbose_eval boosting stage. The last boosting stage or the boosting stage found by using early_stopping_rounds 
                is also printed.

        enable_cv : bool (default = True)
                If True, the best number of rounds is found using Cross Validation.
                
        Returns
        -------
        None
        """
        
        # Class' attributes
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        self.custom_eval_function = custom_eval_function
        self.maximize = maximize
        self.nrounds = nrounds
        self.random_state = random_state
        self.test_size = test_size
        self.verbose_eval = verbose_eval
        self.enable_cv = enable_cv

        self.lgb_model = None
        self.model_name = "LightGBM"

    @staticmethod
    def R2(preds, dtrain):
        labels = dtrain.get_label()
        return "R2", r2_score(labels, preds) * 100, True # f(preds: array, dtrain: Dataset) -> name: string, value: array, is_higher_better: bool
        
    def fit(self, X, y):
        """
        This method trains the LightGBM model.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the training data.

        y : Pandas Series
                This is the target related to the training data.
                
        Returns
        -------
        None
        """
        
        print("LightGBM training...")                       
        if self.enable_cv:
            dtrain = lgb.Dataset(X, label = y)
            watchlist = [dtrain]

            print("    Cross-validating LightGBM with seed: " + str(self.random_state) + "...")
            # If we deal with a regression problem, disable stratified split
            if self.params["application"] == "regression":
                stratified_split = False
            else:
                stratified_split = True
                
            if self.early_stopping_rounds < 0:
                cv_output = lgb.cv(self.params, dtrain, num_boost_round = self.nrounds, feval = self.custom_eval_function, verbose_eval = self.verbose_eval, show_stdv = True, stratified = stratified_split)
            else:
                cv_output = lgb.cv(self.params, dtrain, num_boost_round = self.nrounds, feval = self.custom_eval_function, early_stopping_rounds = self.early_stopping_rounds, verbose_eval = self.verbose_eval, show_stdv = True, stratified = stratified_split)

            cv_output_df = pd.DataFrame(cv_output)
            self.nrounds = cv_output_df[cv_output_df.filter(regex = ".*-mean").columns.tolist()[0]].index[-1] + 1

            print("    Final training of LightGBM with seed: " + str(self.random_state) + " and num rounds = " + str(self.nrounds) + "...")
            if self.early_stopping_rounds < 0:
                self.lgb_model = lgb.train(self.params, dtrain, self.nrounds, watchlist, feval = self.custom_eval_function, verbose_eval = self.verbose_eval)
            else:
                self.lgb_model = lgb.train(self.params, dtrain, self.nrounds, watchlist, feval = self.custom_eval_function, early_stopping_rounds = self.early_stopping_rounds, verbose_eval = self.verbose_eval)
        else:
            X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size = self.test_size, random_state = self.random_state)
            
            dtrain = lgb.Dataset(X_train, label = y_train)
            dvalid = lgb.Dataset(X_eval, label = y_eval)
            watchlist = [dtrain, dvalid]

            if self.early_stopping_rounds < 0:
                self.lgb_model = lgb.train(self.params, dtrain, self.nrounds, watchlist, feval = self.custom_eval_function, verbose_eval = self.verbose_eval)
            else:
                self.lgb_model = lgb.train(self.params, dtrain, self.nrounds, watchlist, feval = self.custom_eval_function, early_stopping_rounds = self.early_stopping_rounds, verbose_eval = self.verbose_eval)

            self.nrounds = self.lgb_model.best_iteration

        return self

    @abstractmethod
    def predict(self, X):
        """
        This method makes predictions using the previously trained model.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the testing data we want to make predictions on.
                
        Returns
        -------
        predictions_npa : numpy array
                Numpy array containing predictions for each sample of the testing set.
        """

        """# Sanity checks
        if self.lgb_model is None:
            raise ValueError("You MUST train the LightGBM model using fit() before attempting to do predictions!")

        print("Predicting outcome for testing set...")
        predictions_npa = self.lgb_model.predict(X)

        return predictions_npa"""

        raise NotImplementedError("Not yet implemented!")
    
    def plot_features_importance(self, importance_type = "gain", max_num_features = None, ignore_zero = True):
        """
        This method plots model's features importance.

        Parameters
        ----------
        importance_type : string, optional (default = "gain")
                How the importance is calculated. If "split", result contains numbers of times the feature is used in a model. 
                If "gain", result contains total gains of splits which use the feature.
                
        max_num_features : int or None, optional (default = None)
                Max number of top features displayed on plot. If None or < 1, all features will be displayed.
                
        ignore_zero : bool, optional (default = True)
                Whether to ignore features with zero importance.
                
        Returns
        -------
        None
        """

        lgb.plot_importance(self.lgb_model, importance_type = importance_type, max_num_features = max_num_features, ignore_zero = ignore_zero)
        plt.show()

    def get_features_importance(self, importance_type = "gain"):
        """
        This method gets model's features importance.

        Parameters
        ----------
        importance_type : string, optional (default = "gain")
                How the importance is calculated. If "split", result contains numbers of times the feature is used in a model. 
                If "gain", result contains total gains of splits which use the feature.
                                
        Returns
        -------
        feature_importance_df : Pandas Data Frame
                Feature importance of each feature of the dataset.
        """

        importance = self.lgb_model.feature_importance(importance_type = "gain")
        features_names = self.lgb_model.feature_name()

        feature_importance_df = pd.DataFrame({"feature": features_names, "importance": importance}).sort_values(by = "importance", ascending = False).reset_index(drop = True)

        return feature_importance_df

class LGBMClassifier(AbstractLGBMWrapper, ClassifierMixin):
    """
    The purpose of this class is to provide a wrapper for a LightGBM classifier.
    """
    
    def __init__(self, params, early_stopping_rounds, custom_eval_function = None, maximize = True, nrounds = 10000, random_state = 0, test_size = 0.1, verbose_eval = 1, enable_cv = True):
        """
        Class' constructor

        Parameters
        ----------
        params : dictionary
                This contains the parameters of the LightGBM model.
                
        early_stopping_rounds : integer
                This indicates the number of rounds to keep before stopping training when the score doesn't increase. If negative, disable this functionality.

        verbose_eval : positive integer
                This indicates the frequency of scores printing. E.g. 50 means one score printed every 50 rounds.

        custom_eval_function : function
                This is a function LightGBM will use as loss function.

        maximize : boolean
                Indicates if the function customEvalFunction must be maximized or minimized. Not used when customEvalFunction is None.

        nrounds : integer
                Number of rounds for LightGBM training. If negative, the model will look for the best value using cross-validation.

        random_state : zero or positive integer
                Seed used by LightGBM to ensure reproductibility.

        test_size : float between 0 and 1.
                This indicates the size of the test set.

        verbose_eval : bool or int, optional (default = 1)
                If True, the eval metric on the valid set is printed at each boosting stage. If int, the eval metric on the valid set is 
                printed at every verbose_eval boosting stage. The last boosting stage or the boosting stage found by using early_stopping_rounds 
                is also printed.

        enable_cv : bool (default = True)
                If True, the best number of rounds is found using Cross Validation.
                
        Returns
        -------
        None
        """
        
        # Call to superclass
        super().__init__(params, early_stopping_rounds, custom_eval_function, maximize, nrounds, random_state, test_size, verbose_eval, enable_cv)
        
    def predict(self, X):
        """
        This method makes predictions using the previously trained model.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the testing data we want to make predictions on.
                
        Returns
        -------
        predictions_npa : numpy array
                Numpy array containing predictions for each sample of the testing set.
        """

        # Sanity checks
        if self.lgb_model is None:
            raise ValueError("You MUST train the LightGBM model using fit() before attempting to do predictions!")

        print("Predicting outcome for testing set...")
        predictions_npa = self.lgb_model.predict(X)
        if len(predictions_npa.shape) == 2 and predictions_npa.shape[1] > 1:
            predictions_npa = np.argmax(predictions_npa, axis = 1)

        return predictions_npa

    def predict_proba(self, X):
        """
        This method makes predictions using the previously trained model and return probabilities.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the testing data we want to make predictions on.
                
        Returns
        -------
        predictions_npa : numpy array
                Numpy array containing probabilities of each class for each sample of the testing set.
        """

        # Sanity checks
        if self.lgb_model is None:
            raise ValueError("You MUST train the LightGBM model using fit() before attempting to do predictions!")

        print("Predicting outcome for testing set...")
        predictions_npa = self.lgb_model.predict(X)

        return predictions_npa

class LGBMRegressor(AbstractLGBMWrapper, RegressorMixin):
    """
    The purpose of this class is to provide a wrapper for a LightGBM regressor.
    """
    
    def __init__(self, params, early_stopping_rounds, custom_eval_function = None, maximize = True, nrounds = 10000, random_state = 0, test_size = 0.1, verbose_eval = 1, enable_cv = True):
        """
        Class' constructor

        Parameters
        ----------
        params : dictionary
                This contains the parameters of the LightGBM model.
                
        early_stopping_rounds : integer
                This indicates the number of rounds to keep before stopping training when the score doesn't increase. If negative, disable this functionality.

        verbose_eval : positive integer
                This indicates the frequency of scores printing. E.g. 50 means one score printed every 50 rounds.

        custom_eval_function : function
                This is a function LightGBM will use as loss function.

        maximize : boolean
                Indicates if the function customEvalFunction must be maximized or minimized. Not used when customEvalFunction is None.

        nrounds : integer
                Number of rounds for LightGBM training. If negative, the model will look for the best value using cross-validation.

        random_state : zero or positive integer
                Seed used by LightGBM to ensure reproductibility.

        test_size : float between 0 and 1.
                This indicates the size of the test set.

        verbose_eval : bool or int, optional (default = 1)
                If True, the eval metric on the valid set is printed at each boosting stage. If int, the eval metric on the valid set is 
                printed at every verbose_eval boosting stage. The last boosting stage or the boosting stage found by using early_stopping_rounds 
                is also printed.

        enable_cv : bool (default = True)
                If True, the best number of rounds is found using Cross Validation.
                
        Returns
        -------
        None
        """
        
        # Call to superclass
        super().__init__(params, early_stopping_rounds, custom_eval_function, maximize, nrounds, random_state, test_size, verbose_eval, enable_cv)
        
    def predict(self, X):
        """
        This method makes predictions using the previously trained model.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the testing data we want to make predictions on.
                
        Returns
        -------
        predictions_npa : numpy array
                Numpy array containing predictions for each sample of the testing set.
        """

        # Sanity checks
        if self.lgb_model is None:
            raise ValueError("You MUST train the LightGBM model using fit() before attempting to do predictions!")

        print("Predicting outcome for testing set...")
        predictions_npa = self.lgb_model.predict(X)

        return predictions_npa