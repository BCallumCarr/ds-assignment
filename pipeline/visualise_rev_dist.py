#!/usr/bin/env python
# coding: utf-8

# author: Brad Carruthers
# author_mail: brad@securewealth.co.za

import pandas as pd
import matplotlib.pyplot as plt

def visualise_rev_dist(df_acts_users):

    plt.hist(df_acts_users["Revenue"], bins=100)
    plt.savefig("answers/Q4_Dist_Revenue_by_User.png") #[2][0].figure
    plt.clf()
    
    plt.hist(df_acts_users["Revenue"], bins=100, log=True)
    plt.savefig("answers/Q4_Dist_Revenue_by_User_Logged.png")
    plt.clf()

    # histogram of days between reg and date
    plt.hist(df_acts_users["DaysRegtoDate"] / pd.Timedelta(1, unit='d'), bins=100)
    plt.savefig("answers/Extra_Days_Between_Registration_and_Activity.png")
    plt.clf()