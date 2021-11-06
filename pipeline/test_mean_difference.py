#!/usr/bin/env python
# coding: utf-8

# author: Brad Carruthers
# author_mail: brad@securewealth.co.za

from statsmodels.stats.weightstats import ztest
from statsmodels.stats.weightstats import CompareMeans


def test_mean_difference(file, df_acts_one_week):

    # one week since registration
    one_week_gender_means = df_acts_one_week.groupby("Gender").agg("mean").drop("UserId", axis="columns")
    one_week_female_mean = round( one_week_gender_means.loc["F",:].tolist()[0], 2)
    one_week_male_mean = round( one_week_gender_means.loc["M",:].tolist()[0], 2)
    
    file.write(f"\n\n6)")
    file.write(f"\n The one week female revenue average is {one_week_female_mean}, while the male average is {one_week_male_mean}")

    males = df_acts_one_week.loc[df_acts_one_week["Gender"] == "M","Revenue"]
    females = df_acts_one_week.loc[df_acts_one_week["Gender"] == "F","Revenue"]

    diff_means_p_val = round( ztest(x1=males, x2=females)[1], 5)

    file.write(f"\n For test of null hypothesis that means are equal (assuming an equal variance), p-value is {diff_means_p_val},")
    file.write(" thus we reject the null hypothesis that the means are equal")

    # assume non-equal variance?