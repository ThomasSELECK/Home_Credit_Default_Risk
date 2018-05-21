#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the Home Credit Default Risk Challenge                   #
#                                                                             #
# This file contains all files paths of the datasets.                         #
# Developped using Python 3.6.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-05-18                                                            #
# Version: 1.0.0                                                              #
###############################################################################

# File paths
TRAINING_DATA_str = "../../../data/raw/application_train.csv"
TESTING_DATA_str = "../../../data/raw/application_test.csv"
BUREAU_DATA_str = "../../../data/raw/bureau.csv"
BUREAU_BALANCE_DATA_str = "../../../data/raw/bureau_balance.csv"
CREDIT_CARD_BALANCE_DATA_str = "../../../data/raw/credit_card_balance.csv"
INSTALLMENTS_PAYMENTS_DATA_str = "../../../data/raw/installments_payments.csv"
POS_CASH_BALANCE_DATA_str = "../../../data/raw/POS_CASH_balance.csv"
PREVIOUS_APPLICATION_DATA_str = "../../../data/raw/previous_application.csv"

CACHE_DIR_str = "E:/home_credit_pipeline_cache/"
OUTPUT_DIR_str = "../../../output/"
PREDICTIONS_DIR_str = "../../../predictions/"