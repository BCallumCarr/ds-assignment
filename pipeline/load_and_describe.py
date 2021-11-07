#!/usr/bin/env python
# coding: utf-8

# author: Brad Carruthers
# author_mail: brad@securewealth.co.za

import logging
import pandas as pd
from datetime import date

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


def load_data():
    logging.info("Reading in activities.csv")
    df_acts = pd.read_csv("../data/activities.csv", sep=",")

    logging.info("Reading in users.csv")
    df_users = pd.read_csv("../data/users.csv", sep=",")

    return df_acts, df_users


def describe_data_and_clean(file):
    """
        Answers to questions 1, 2, 3, 5
    """

    df_acts, df_users = load_data()

    file.write(f"\n1a)")
    file.write(f"\n There are {df_acts.shape[1]-1} variables and one key in activities.csv")
    file.write(f"\n There are {df_users.shape[1]-1} variables and one key in users.csv")

    file.write(f"\n\n1b)")
    file.write(f"\n There are {df_acts.shape[0]} observations in activities.csv")
    file.write(f"\n There are {df_users.shape[0]} observations in users.csv")

    male_counts = df_users["Gender"].value_counts()["M"]
    female_counts = df_users["Gender"].value_counts()["F"]
    missing_gender = df_users["Gender"].value_counts()[" "]

    file.write("\n\n2a)")
    file.write(f"\n There are {male_counts} male observations in users.csv")
    file.write("\n\n2b)")
    file.write(f"\n There are {female_counts} female observations in users.csv")
    file.write("\n\n2C)")
    file.write(f"\n There are {missing_gender} missing gender observations in users.csv")

    logging.info("Imputing Male for missing Genders")

    bool_missing_gender = df_users.loc[:,"Gender"] == " "
    df_users.loc[bool_missing_gender,"Gender"] = "M"

    return df_acts, df_users


def merge_and_agg_stats(file, df_acts, df_users):

    df_acts_users = pd.merge(
        left=df_acts,
        right=df_users,
        left_on="UserId",
        right_on="UserId",
        how="left"
    )

    # convert date cols to pd datetime
    df_acts_users["Date"] = pd.to_datetime(df_acts_users["Date"], format="%Y-%m-%d")
    df_acts_users["DOB"] = pd.to_datetime(df_acts_users["DOB"], format="%Y-%m-%d")
    df_acts_users["RegDate"] = pd.to_datetime(df_acts_users["RegDate"], format="%Y-%m-%d")

    df_acts_users["DaysRegtoDate"] = df_acts_users["Date"] - df_acts_users["RegDate"]
    df_acts_users["DaysRegtoToday"] = pd.to_datetime(date.today(), format="%Y-%m-%d") - df_acts_users["Date"]

    mean_rev = round( df_acts_users["Revenue"].mean(), 2)
    median_rev = df_acts_users["Revenue"].median()

    file.write("\n\n3)")
    file.write(f"\n The average earnings for an individual is {mean_rev} but the middle earning is {median_rev}.")
    file.write(f"\n The average earnings for an individual is {mean_rev} but the middle earning is {median_rev}.")
    file.write(f"\n I.e. the average earnings across all activities for the period users are active is {mean_rev},")
    file.write(f"\n There are also probably some outliers in the data, since the mean is more sensitive to outliers.")

    return df_acts_users


def acts_one_week(file, df_acts_users):

    file.write("\n\n5)")
    # careful to use 6 days since registration day is day 0
    df_acts_one_week = df_acts_users.loc[df_acts_users.loc[:,"DaysRegtoDate"] <= pd.Timedelta(6,'D'),:]
    
    one_week_mean_rev = df_acts_one_week["Revenue"].mean()
    # df_acts_one_week["Revenue"].median()
    file.write(f"\n The average earnings for an individual in their first week is {one_week_mean_rev}.")

    return df_acts_one_week

 