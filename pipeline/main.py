#!/usr/bin/env python
# coding: utf-8

# author: Brad Carruthers
# author_mail: brad@securewealth.co.za

import os
import logging
import pandas as pd
from datetime import date

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


def load_data_and_describe(file):
    """
        Answers to questions 1, 2, 3, 5
    """
    logging.info("Reading in activities.csv")
    df_acts = pd.read_csv("../data/activities.csv", sep=",")

    logging.info("Reading in users.csv")
    df_users = pd.read_csv("../data/users.csv", sep=",")

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

    #####

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

    df_acts_users.groupby("UserId").agg("mean").reset_index()
    df_acts_users.groupby("UserId").agg("median").reset_index()

    mean_rev = round( df_acts_users["Revenue"].mean(), 2)
    median_rev = df_acts_users["Revenue"].median()

    file.write("\n\n3)")
    file.write(f"\n The average earnings for an individual is {mean_rev} but the most common earning is {median_rev}.")
    file.write(f"\n I.e. the average earnings across all activities for the period users are active is {mean_rev},")
    file.write(f"\n and the fact that the mean is greater than the median implies that the data in the population are skewed to the right")


    return df_acts, df_users



if __name__ == "__main__" :

    with open("answers.txt", 'r+') as f:
        f.seek(0)
        
        # if not os.path.exists('answers.txt'):
        #     f = open('/tmp/test', 'w'): pass
        

        df_acts, df_users = load_data_and_describe(f)

        f.truncate()

