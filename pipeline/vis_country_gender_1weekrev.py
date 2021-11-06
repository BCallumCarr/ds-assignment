#!/usr/bin/env python
# coding: utf-8

# author: Brad Carruthers
# author_mail: brad@securewealth.co.za

import numpy as np
import matplotlib.pyplot as plt

def vis_country_gender_1weekrev(df_pivot):

    # data to plot
    n_groups = df_pivot.shape[0]
    week1_rev_male = df_pivot["Week1RevenueMale"].tolist()
    week1_rev_female = df_pivot["Week1RevenueFemale"].tolist()

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, week1_rev_male, bar_width,
    alpha=opacity,
    color='b',
    label='Males')

    rects2 = plt.bar(index + bar_width, week1_rev_female, bar_width,
    alpha=opacity,
    color='lightblue',
    label='Females')

    plt.xlabel('Country')
    plt.ylabel('Average Week 1 Revenue')
    plt.title('Average Week 1 Revenue per Gender by Country')
    plt.xticks(index + bar_width, df_pivot["Country"].tolist())
    plt.legend()

    plt.tight_layout()
    plt.savefig("answers/Q8_Relationship_between_Country_Gender_Week1Revenue.png") #[2][0].figure
    plt.clf()