#!/usr/bin/env python
# coding: utf-8

# author: Brad Carruthers
# author_mail: brad@securewealth.co.za

def largest_country_gender_diff(file, df_acts_one_week):

    df_country_diffs = df_acts_one_week.groupby(["Country", "Gender"]).agg("mean").reset_index()[["Gender", "Country", "Revenue"]]

    df_pivot = df_country_diffs.pivot(index="Country", columns="Gender").reset_index()
    df_pivot.columns = ["Country", "Week1RevenueFemale", "Week1RevenueMale"]
    df_pivot["Delta"] = df_pivot["Week1RevenueMale"] - df_pivot["Week1RevenueFemale"]

    country = df_pivot.loc[df_pivot["Delta"] == df_pivot["Delta"].max(),"Country"]
    file.write("\n\n7)")
    file.write(f"\n Country with biggest gender difference for revenue is {country.to_list()[0]}")

    return df_pivot