#!/usr/bin/env python
# coding: utf-8

# author: Brad Carruthers
# author_mail: brad@securewealth.co.za

def sklearn_lin_reg_p_val_calc(X, y):
        """
            replicate p values for sklearn, source: 
            stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
        X: pandas df of features
        y: pandas df of target
        """ 
        import numpy as np
        import pandas as pd
        from scipy import stats
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression().fit(X, y)

        params = np.append(model.intercept_,model.coef_)
        predictions = model.predict(X)

        newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
        MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))

        var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
        sd_b = np.sqrt(var_b)
        ts_b = params/ sd_b


        p_values =[2*(1-stats.t.cdf(np.abs(i),(newX.shape[0]-newX.shape[1]))) for i in ts_b]

        sd_b = np.round(sd_b,3)
        ts_b = np.round(ts_b,3)
        p_values = np.round(p_values,3)
        params = np.round(params,4)

        df_sklearn_p_vals = pd.DataFrame()
        vars_and_intcpt = ["Intercept"]
        vars_and_intcpt += X.columns.tolist()
        
        df_sklearn_p_vals["Variables"],df_sklearn_p_vals["Coefficients"],df_sklearn_p_vals["Standard Errors"],df_sklearn_p_vals["t values"],df_sklearn_p_vals["Probabilities"] = [vars_and_intcpt,params,sd_b,ts_b,p_values]

        return df_sklearn_p_vals


def lin_reg_week1_rev(file, df_acts_users, df_users):

    import logging
    import pandas as pd
    from datetime import date
    from functools import reduce

    logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

    # calculate total week 1 revenue per user
    agg_week1_rev = (
        df_acts_users.loc[df_acts_users.loc[:,"DaysRegtoDate"] <= pd.Timedelta(6,'D'),:]
        .groupby("UserId").agg("sum").reset_index()
    )

    agg_week1_rev.rename(columns={"Revenue": "TotalWeek1Revenue"}, inplace=True)

    # calculate total day 1 revenue per user
    agg_day1_rev = (
        df_acts_users.loc[df_acts_users.loc[:,"DaysRegtoDate"] == pd.Timedelta(0,'D'),:]
        .groupby("UserId").agg("sum").reset_index()
    )
    agg_day1_rev.rename(columns={"Revenue": "TotalDay1Revenue"}, inplace=True)

    # check if UserID column duplicated
    userid_duplicated = df_users["UserId"].duplicated().any()
    logging.info(f"Are there duplicates in df_users UserId column? {userid_duplicated}")

    # calculate age age from df_users
    df_users["DOB"] = pd.to_datetime(df_users["DOB"], format="%Y-%m-%d")
    df_users["AgeInDays"] = pd.to_datetime(date.today(), format="%Y-%m-%d") - df_users["DOB"]
    df_users["AgeInDays"] = df_users["AgeInDays"] / pd.Timedelta(1, unit='d')

    # create final linear regression dataframe
    dataframes = [agg_week1_rev, agg_day1_rev, df_users]

    df_lin_reg = reduce(lambda left, right: pd.merge(left, right,on="UserId", how="left"), dataframes)

    logging.info("There are some users who have revenue within a week, but no revenue on day 1.")

    # fill nans with 0 for 1 day revenue
    df_lin_reg["TotalDay1Revenue"].fillna(0, inplace=True)

    # specify target
    y = df_lin_reg["TotalWeek1Revenue"]

    # specify features
    X = df_lin_reg[["Gender", "AgeInDays", "Country", "TotalDay1Revenue"]]

    logging.info(f"Feature data types: \n\n {X.dtypes}")

    # get dummy variables for categorical independent variables
    X = pd.get_dummies(data=X, drop_first=True)

    # # statsmodels Ordinary Least Squares
    # import statsmodels.api as sm
    # from scipy import stats

    # X2 = sm.add_constant(X)
    # est = sm.OLS(y, X2)
    # est2 = est.fit()
    # print(est2.summary())

    coefs_and_p_vals = sklearn_lin_reg_p_val_calc(X, y)

    stat_sig_vars = coefs_and_p_vals.loc[coefs_and_p_vals["Probabilities"] < 0.05,"Variables"].to_list()

    file.write("\n\n9)")
    file.write(f"\nT he statistically significant variables at a 5% level are {stat_sig_vars}")

    return X, y


def predict_given_features(file, X, y):

    import pandas as pd
    from sklearn.linear_model import LinearRegression

    ### Predict for 40-year-old French, German and British women having generated GBP 20 on RegDate ###
    predict_X = pd.DataFrame(columns=X.columns)

    predict_X.loc[len(predict_X.index)] = [40*365, 20, 0, 1, 0, 0, 0] # french women
    predict_X.loc[len(predict_X.index)] = [40*365, 20, 0, 0, 0, 0, 0] # german women
    predict_X.loc[len(predict_X.index)] = [40*365, 20, 0, 0, 1, 0, 0] # british women

    model = LinearRegression().fit(X, y)
    
    french_women_1week_earn = round(model.predict(predict_X)[0], 2)
    german_women_1week_earn = round(model.predict(predict_X)[1], 2)
    british_women_1week_earn = round(model.predict(predict_X)[2], 2)

    file.write(f"\n\n10)")
    file.write(f"\n The average 40-year-old French woman, having generated GBP 20 on registration date, would expect to earn {french_women_1week_earn} in her first week")
    file.write(f"\n The average 40-year-old German woman, having generated GBP 20 on registration date, would expect to earn {german_women_1week_earn} in her first week")
    file.write(f"\n The average 40-year-old British woman, having generated GBP 20 on registration date, would expect to earn {british_women_1week_earn} in her first week")



