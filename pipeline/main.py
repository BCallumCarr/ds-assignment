#!/usr/bin/env python
# coding: utf-8

# author: Brad Carruthers
# author_mail: brad@securewealth.co.za

import os

from load_and_describe import describe_data_and_clean, merge_and_agg_stats, acts_one_week
from visualise_rev_dist import visualise_rev_dist
from test_mean_difference import test_mean_difference
from largest_country_gender_diff import largest_country_gender_diff
from vis_country_gender_1weekrev import vis_country_gender_1weekrev
from lin_reg_modelling import lin_reg_week1_rev, predict_given_features

if __name__ == "__main__" :

    # create file if doesn't exist
    if not os.path.exists('answers/answers.txt'):
        with open('answers/answers.txt', 'w'): pass

    with open("answers/answers.txt", 'r+') as f:
        f.seek(0)
        
        df_acts, df_users = describe_data_and_clean(f)

        df_acts_users = merge_and_agg_stats(f, df_acts, df_users)

        df_acts_one_week = acts_one_week(f, df_acts_users)

        visualise_rev_dist(df_acts_users)

        test_mean_difference(f, df_acts_one_week)

        df_pivot = largest_country_gender_diff(f, df_acts_one_week)

        vis_country_gender_1weekrev(df_pivot)

        X, y = lin_reg_week1_rev(f, df_acts_users, df_users)

        predict_given_features(f, X, y)

        # overwrite existing file
        f.truncate()

