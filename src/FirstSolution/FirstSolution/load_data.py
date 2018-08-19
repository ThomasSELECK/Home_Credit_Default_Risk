#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the Home Credit Default Risk Challenge                   #
#                                                                             #
# This file provides everything needed to load the data.                      #
# Developped using Python 3.6.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-05-18                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import numpy as np
import pandas as pd
import pickle
import os
import gc
from sklearn.model_selection import train_test_split

def load_data(training_data_path_str, testing_data_path_str, bureau_data_path_str, bureau_balance_data_path_str, credit_card_balance_data_path_str, installments_payments_data_path_str, pos_cash_balance_data_path_str, previous_application_data_path_str, enable_validation, target_name_str, cache_dir_str):
    """
    This function is a wrapper for the loading of the data.

    Parameters
    ----------
    training_data_path_str : string
            This is the main training set. This represents static data for all applications. 
            One row represents one loan in our data sample.

    testing_data_path_str : string
            This is the main testing set. This represents static data for all applications. 
            One row represents one loan in our data sample.

    bureau_data_path_str : string
            All client's previous credits provided by other financial institutions that were 
            reported to Credit Bureau (for clients who have a loan in our sample).

    bureau_balance_data_path_str : string
            Monthly balances of previous credits in Credit Bureau.

    credit_card_balance_data_path_str : string
            Monthly balance snapshots of previous credit cards that the applicant has with 
            Home Credit.

    installments_payments_data_path_str : string
            Repayment history for the previously disbursed credits in Home Credit related 
            to the loans in our sample.

    pos_cash_balance_data_path_str : string
            Monthly balance snapshots of previous POS (point of sales) and cash loans that 
            the applicant had with Home Credit.

    previous_application_data_path_str : string
            All previous applications for Home Credit loans of clients who have loans in 
            our sample.

    enable_validation : bool
            If true, split the training set into training set and testing set.
    
    target_name_str : string
            Name of the target column

    cache_dir_str : string
            Path of the folder where cache data is / will be stored.

    Returns
    -------
    X_train : pd.DataFrame
            A pandas DataFrame containing the training set.

    X_test : pd.DataFrame
            A pandas DataFrame containing the testing set.
            
    y_train : pd.Series
            The target values for the training part.

    y_test : pd.Series
            The target values for the validation part. This is None when 'enable_validation' 
            is False.
    """

    # Load the data
    if not os.path.exists(cache_dir_str + "datasets_cache.pkl"):
        print("Loading the data...")
        main_features_dtypes_dict = {}

        # Add np.int8 cols
        for col in ["CNT_CHILDREN", "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL", 
                    "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY", "HOUR_APPR_PROCESS_START", "REG_REGION_NOT_LIVE_REGION", 
                    "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", 
                    "LIVE_CITY_NOT_WORK_CITY", "OWN_CAR_AGE", "CNT_CREDIT_PROLONG", "MONTHS_BALANCE", "CNT_DRAWINGS_CURRENT",
                    "HOUR_APPR_PROCESS_START", "NFLAG_LAST_APPL_IN_DAY"]:
            main_features_dtypes_dict[col] = np.int8

        # Add np.int16 cols
        for col in ["DAYS_BIRTH", "DAYS_ID_PUBLISH", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "SK_DPD", "SK_DPD_DEF", "DAYS_DECISION",
                    "NUM_INSTALMENT_NUMBER"]:
            main_features_dtypes_dict[col] = np.int16

        # Add np.int32 cols
        for col in ["SK_ID_CURR", "DAYS_EMPLOYED", "SK_ID_BUREAU", "DAYS_CREDIT_UPDATE", "SK_ID_BUREAU", "SK_ID_PREV", "AMT_CREDIT_LIMIT_ACTUAL",
                    "SELLERPLACE_AREA"]:
            main_features_dtypes_dict[col] = np.int32

        # Add np.float16 cols ; these features are integers, but as they contains NAs, they only can be casted to float
        for col in ["OWN_CAR_AGE", "CNT_FAM_MEMBERS", "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
                    "DAYS_LAST_PHONE_CHANGE", "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON", 
                    "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "CNT_DRAWINGS_ATM_CURRENT",
                    "CNT_DRAWINGS_OTHER_CURRENT", "CNT_DRAWINGS_POS_CURRENT", "CNT_INSTALMENT_MATURE_CUM", "CNT_PAYMENT", "NFLAG_INSURED_ON_APPROVAL",
                    "NUM_INSTALMENT_VERSION", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"]:
            main_features_dtypes_dict[col] = np.float16

        # Add np.float32 cols
        for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "REGION_POPULATION_RELATIVE", "DAYS_REGISTRATION",
                    "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG", 
                    "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG", "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG",
                    "NONLIVINGAREA_AVG", "APARTMENTS_MODE", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BUILD_MODE", "COMMONAREA_MODE", 
                    "ELEVATORS_MODE", "ENTRANCES_MODE", "FLOORSMAX_MODE", "FLOORSMIN_MODE", "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE",
                    "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAREA_MODE", "APARTMENTS_MEDI", "BASEMENTAREA_MEDI", "YEARS_BEGINEXPLUATATION_MEDI", "YEARS_BUILD_MEDI", 
                    "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI", "FLOORSMAX_MEDI", "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI", 
                    "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI", "TOTALAREA_MODE", "DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT", 
                    "AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE", "AMT_ANNUITY",
                    "AMT_BALANCE", "AMT_DRAWINGS_ATM_CURRENT", "AMT_DRAWINGS_CURRENT", "AMT_DRAWINGS_OTHER_CURRENT", "AMT_DRAWINGS_POS_CURRENT", 
                    "AMT_INST_MIN_REGULARITY", "AMT_PAYMENT_CURRENT", "AMT_PAYMENT_TOTAL_CURRENT", "AMT_RECEIVABLE_PRINCIPAL", "AMT_RECIVABLE", 
                    "AMT_TOTAL_RECEIVABLE", "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT", "AMT_GOODS_PRICE", "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", 
                    "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION", "AMT_INSTALMENT", "AMT_PAYMENT"]:
            main_features_dtypes_dict[col] = np.float32
          
        for i in range(2, 22):
            main_features_dtypes_dict["FLAG_DOCUMENT_" + str(i)] = np.int8

        print("    Loading:", training_data_path_str, "...")
        training_set_df = pd.read_csv(training_data_path_str, dtype = main_features_dtypes_dict)
        print("    Loading:", testing_data_path_str, "...")
        testing_set_df = pd.read_csv(testing_data_path_str, dtype = main_features_dtypes_dict)
        print("    Loading:", bureau_data_path_str, "...")
        bureau_data_df = pd.read_csv(bureau_data_path_str, dtype = main_features_dtypes_dict)
        print("    Loading:", bureau_balance_data_path_str, "...")
        bureau_balance_data_df = pd.read_csv(bureau_balance_data_path_str, dtype = main_features_dtypes_dict)
        print("    Loading:", credit_card_balance_data_path_str, "...")
        credit_card_balance_data_df = pd.read_csv(credit_card_balance_data_path_str, dtype = main_features_dtypes_dict)
        print("    Loading:", installments_payments_data_path_str, "...")
        installments_payments_data_df = pd.read_csv(installments_payments_data_path_str, dtype = main_features_dtypes_dict)
        print("    Loading:", pos_cash_balance_data_path_str, "...")
        pos_cash_balance_data_df = pd.read_csv(pos_cash_balance_data_path_str, dtype = main_features_dtypes_dict)
        print("    Loading:", previous_application_data_path_str, "...")
        previous_application_data_df = pd.read_csv(previous_application_data_path_str, dtype = main_features_dtypes_dict)

        # Join tables into one
        """print("    Merging tables 'Application' and 'Bureau'...")
        training_set_df = training_set_df.merge(bureau_data_df, how = "left", on = "SK_ID_CURR")
        testing_set_df = testing_set_df.merge(bureau_data_df, how = "left", on = "SK_ID_CURR")
        print("    Merging last tables with 'Bureau balance'...")
        training_set_df = training_set_df.merge(bureau_balance_data_df, how = "left", on = "SK_ID_BUREAU")
        testing_set_df = testing_set_df.merge(bureau_balance_data_df, how = "left", on = "SK_ID_BUREAU")"""

        # Don't do this as the memory usage will blow up!!!
        """X_train = X_train.merge(previous_application_data_df, how = "left", on = "SK_ID_CURR")
        X_train = X_train.merge(pos_cash_balance_data_df, how = "left", on = ["SK_ID_CURR", "SK_ID_PREV"])
        X_train = X_train.merge(installments_payments_data_df, how = "left", on = ["SK_ID_CURR", "SK_ID_PREV"])
        X_train = X_train.merge(credit_card_balance_data_df, how = "left", on = ["SK_ID_CURR", "SK_ID_PREV"])"""
        
        # Put ID as index
        print("    Put 'SK_ID_CURR' as index...")
        training_set_df.index = training_set_df["SK_ID_CURR"]
        training_set_df.drop("SK_ID_CURR", axis = 1, inplace = True)
        testing_set_df.index = testing_set_df["SK_ID_CURR"]
        testing_set_df.drop("SK_ID_CURR", axis = 1, inplace = True)

        # Extract the target
        print("    Extracting the target...")
        y_train = training_set_df[target_name_str]
        training_set_df.drop(target_name_str, axis = 1, inplace = True)

        # Saving data to cache
        print("    Saving data to cache...")
        data_lst = [training_set_df, testing_set_df, y_train, bureau_data_df, bureau_balance_data_df, credit_card_balance_data_df, installments_payments_data_df, pos_cash_balance_data_df, previous_application_data_df]

        with open(cache_dir_str + "datasets_cache.pkl", "wb") as f:
            pickle.dump(data_lst, f, protocol = 4)

    else: # If cached data exists, load from cache to reduce loading time
        print("Loading data from cache...")

        with open(cache_dir_str + "datasets_cache.pkl", "rb") as f:
            data_lst = pickle.load(f)
            training_set_df = data_lst[0]
            testing_set_df = data_lst[1]
            y_train = data_lst[2]
            bureau_data_df = data_lst[3]
            bureau_balance_data_df = data_lst[4]
            credit_card_balance_data_df = data_lst[5]
            installments_payments_data_df = data_lst[6]
            pos_cash_balance_data_df = data_lst[7]
            previous_application_data_df = data_lst[8]

            del data_lst
            gc.collect()

    # Generate a validation set if enable_validation is True
    if enable_validation:
        print("Generating validation set...")
        test_size_ratio = 0.2

        training_set_df[target_name_str] = y_train.values
        
        train_split, test_split = train_test_split(training_set_df, test_size = test_size_ratio, random_state = 42)
        X_train = pd.DataFrame(train_split, columns = training_set_df.columns)
        X_test = pd.DataFrame(test_split, columns = training_set_df.columns)

        # Resample testing set to have same repartition of 'NAME_CONTRACT_TYPE' than in the original testing set (99% Cash loans and 1% Revolving loans)
        cash_loans_df = X_test.loc[X_test["NAME_CONTRACT_TYPE"] == "Cash loans"]
        revolving_loans_df = X_test.loc[X_test["NAME_CONTRACT_TYPE"] == "Revolving loans"].sample(n = int(cash_loans_df.shape[0] / 99))
        X_test = pd.concat([cash_loans_df, revolving_loans_df], axis = 0)
        X_test.sort_index(inplace = True)
    
        # Extract truth / target
        y_train = X_train[target_name_str]
        X_train.drop(target_name_str, axis = 1, inplace = True)

        y_test = X_test[target_name_str]
        X_test.drop(target_name_str, axis = 1, inplace = True)
        
        print("Generating validation set... done")
    else:
        X_train = training_set_df
        X_test = testing_set_df
        y_test = None
        
    print("Loading data... done")

    return X_train, X_test, y_train, y_test, bureau_data_df, bureau_balance_data_df, credit_card_balance_data_df, installments_payments_data_df, pos_cash_balance_data_df, previous_application_data_df