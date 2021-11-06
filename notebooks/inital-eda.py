
# In[1]:


#!/usr/bin/env
#
# author: Brad Carruthers
# author_mail: brad@securewealth.co.za

# In[2]:


import logging
import pandas as pd
from datetime import date

logging.basicConfig(level=logging.INFO, filename='notebook.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# ## Block A

# In[3]:


logging.info("Reading in activities.csv")
df_acts = pd.read_csv("../data/activities.csv", sep=",")
logging.info(f"There are {df_acts.shape[1]-1} variables and one key in activities.csv")
logging.info(f"There are {df_acts.shape[0]} observations in activities.csv")

# In[4]:


df_acts

# In[5]:


logging.info("Reading in users.csv")
df_users = pd.read_csv("../data/users.csv", sep=",")
logging.info(f"There are {df_users.shape[1]-1} variables and one key in users.csv")
logging.info(f"There are {df_users.shape[0]} observations in users.csv")

# In[6]:


df_users

# In[7]:


df_users["Gender"].unique()

# In[8]:


male_counts = df_users["Gender"].value_counts()["M"]
female_counts = df_users["Gender"].value_counts()["F"]
missing_gender = df_users["Gender"].value_counts()[" "]

# In[9]:


logging.info(f"There are {male_counts} male observations in users.csv")
logging.info(f"There are {female_counts} female observations in users.csv")
logging.info(f"There are {missing_gender} missing gender observations in users.csv")

# In[10]:


# impute Male for missing Genders
bool_missing_gender = df_users.loc[:,"Gender"] == " "

df_users.loc[bool_missing_gender,"Gender"] = "M"

# In[11]:


df_acts_users = pd.merge(
    left=df_acts,
    right=df_users,
    left_on="UserId",
    right_on="UserId",
    how="left"
)

# In[12]:


df_acts_users

# In[13]:


df_acts_users.dtypes

# In[14]:


# convert date cols to pd datetime
df_acts_users["Date"] = pd.to_datetime(df_acts_users["Date"], format="%Y-%m-%d")
df_acts_users["DOB"] = pd.to_datetime(df_acts_users["DOB"], format="%Y-%m-%d")
df_acts_users["RegDate"] = pd.to_datetime(df_acts_users["RegDate"], format="%Y-%m-%d")

# In[15]:


df_acts_users["DaysRegtoDate"] = df_acts_users["Date"] - df_acts_users["RegDate"]

# In[16]:


df_acts_users["DaysRegtoToday"] = pd.to_datetime(date.today(), format="%Y-%m-%d") - df_acts_users["Date"]

# In[17]:


df_acts_users.groupby("UserId").agg("mean").reset_index()

# In[18]:


df_acts_users.groupby("UserId").agg("median").reset_index()

# In[19]:


df_acts_users["Revenue"].mean()

# In[20]:


df_acts_users["Revenue"].median()

# In[21]:


df_acts_users["Revenue"].max()

# In[22]:


%matplotlib inline

# In[23]:


import matplotlib.pyplot as plt

# In[24]:


plt.hist(df_acts_users["Revenue"], bins=100)

# In[25]:


plt.hist(df_acts_users["Revenue"], bins=100, log=True)

# In[26]:


# histogram of days between reg and date
plt.hist(df_acts_users["DaysRegtoDate"] / pd.Timedelta(1, unit='d'), bins=100)

# In[27]:


# careful to use 6 days since registration day is day 0
df_acts_one_week = df_acts_users.loc[df_acts_users.loc[:,"DaysRegtoDate"] <= pd.Timedelta(6,'D'),:]

# In[28]:


df_acts_one_week["Revenue"].mean()

# In[29]:


df_acts_one_week["Revenue"].median()




# ## Block B

# In[30]:


# full time-period
df_acts_users.groupby("Gender").agg("mean").drop("UserId", axis="columns").reset_index()

# In[31]:


# one week since registration
df_acts_one_week.groupby("Gender").agg("mean").drop("UserId", axis="columns").reset_index()

# In[32]:


import numpy as np

# In[33]:


df_acts_one_week.groupby("Gender").agg(np.std, ddof=0).reset_index()[["Gender", "Revenue"]]

# In[34]:


df_acts_one_week["Gender"].value_counts()

# In[35]:


from statsmodels.stats.weightstats import ztest
from statsmodels.stats.weightstats import CompareMeans

# In[36]:


males = df_acts_one_week.loc[df_acts_one_week["Gender"] == "M","Revenue"]

# In[37]:


females = df_acts_one_week.loc[df_acts_one_week["Gender"] == "F","Revenue"]

# In[38]:


logging.info(f"For test of null hypothesis that means are equal (assuming an equal variance), p-value is {ztest(x1=males, x2=females)[1]}, reject null that means are equal")

# In[39]:


# assume standard deviations are different?

# In[40]:


df_country_diffs = df_acts_one_week.groupby(["Country", "Gender"]).agg("mean").reset_index()[["Gender", "Country", "Revenue"]]

# In[41]:


df_country_diffs

# In[42]:


df_pivot = df_country_diffs.pivot(index="Country", columns="Gender").reset_index()
df_pivot.columns = ["Country", "Week1RevenueFemale", "Week1RevenueMale"]

# In[43]:


df_pivot["Delta"] = df_pivot["Week1RevenueMale"] - df_pivot["Week1RevenueFemale"]

# In[44]:


df_pivot

# In[45]:


country = df_pivot.loc[df_pivot["Delta"] == df_pivot["Delta"].max(),"Country"]

# In[46]:


logging.info(f"Country with biggest gender difference for revenue is {np.array(country)[0]}")

# ### Relationship between country, gender and week 1 revenue

# In[47]:


import numpy as np
import matplotlib.pyplot as plt

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
plt.show()

# ### Linear Regression Model

# In[48]:


from sklearn.linear_model import LinearRegression

# In[49]:


# calculate total week 1 revenue per user
agg_week1_rev = (
    df_acts_users.loc[df_acts_users.loc[:,"DaysRegtoDate"] <= pd.Timedelta(6,'D'),:]
    .groupby("UserId").agg("sum").reset_index()
)

# In[50]:


agg_week1_rev.rename(columns={"Revenue": "TotalWeek1Revenue"}, inplace=True)

# In[51]:


# calculate total day 1 revenue per user
agg_day1_rev = (
    df_acts_users.loc[df_acts_users.loc[:,"DaysRegtoDate"] == pd.Timedelta(0,'D'),:]
    .groupby("UserId").agg("sum").reset_index()
)
agg_day1_rev.rename(columns={"Revenue": "TotalDay1Revenue"}, inplace=True)

# In[52]:


# check if UserID column duplicated
df_users["UserId"].duplicated().any()

# In[53]:


# create final linear regression dataframe
from functools import reduce

dataframes = [agg_week1_rev, agg_day1_rev, df_users]

# calculate age age from df_users
df_users["DOB"] = pd.to_datetime(df_users["DOB"], format="%Y-%m-%d")
df_users["AgeInDays"] = pd.to_datetime(date.today(), format="%Y-%m-%d") - df_users["DOB"]
df_users["AgeInDays"] = df_users["AgeInDays"] / pd.Timedelta(1, unit='d')

df_lin_reg = reduce(lambda left, right: pd.merge(left, right,on="UserId", how="left"), dataframes)

# In[54]:


# there are some users who have revenue within a week, but no revenue on day 1
df_acts_users.loc[df_acts_users["UserId"] == 3109386,:]

# In[55]:


df_lin_reg

# In[56]:


# fill nans with 0 for 1 day revenue
df_lin_reg["TotalDay1Revenue"].fillna(0, inplace=True)

# In[57]:


# specify target
y = df_lin_reg["TotalWeek1Revenue"]

# In[58]:


# specify features
X = df_lin_reg[["Gender", "AgeInDays", "Country", "TotalDay1Revenue"]]

# In[59]:


X.dtypes

# In[60]:


# get dummy variables for categorical independent variables
X = pd.get_dummies(data=X, drop_first=True)

# In[61]:


X

# In[62]:


# statsmodels Ordinary Least Squares
import statsmodels.api as sm
from scipy import stats

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

# In[63]:


def sklearn_lin_reg_p_val_calc(X, y):
    """
        replicate p values for sklearn, source: 
        stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
    X: pandas df of features
    y: pandas df of target
    """ 
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

# In[64]:


coefs_and_p_vals = sklearn_lin_reg_p_val_calc(X, y)

# In[65]:


stat_sig_vars = coefs_and_p_vals.loc[coefs_and_p_vals["Probabilities"] < 0.05,"Variables"].to_list()

# In[66]:


logging.info(f"The statistically significant variables at a 5% level are {stat_sig_vars}")

# ### Predict for 40-year-old French, German and British women having generated GBP 20 on RegDate

# In[67]:


predict_X = pd.DataFrame(columns=X.columns)

# In[68]:


predict_X

# In[69]:


predict_X.loc[len(predict_X.index)] = [40*365, 20, 0, 1, 0, 0, 0] # french women
predict_X.loc[len(predict_X.index)] = [40*365, 20, 0, 0, 0, 0, 0] # german women
predict_X.loc[len(predict_X.index)] = [40*365, 20, 0, 0, 1, 0, 0] # british women

# In[70]:


predict_X

# In[71]:


model = LinearRegression().fit(X, y)

# In[72]:


model.predict(predict_X)

# In[73]:


french_women_1week_earn = round(model.predict(predict_X)[0], 2)
german_women_1week_earn = round(model.predict(predict_X)[1], 2)
british_women_1week_earn = round(model.predict(predict_X)[2], 2)

# In[74]:


logging.info(f"The average 40-year-old French woman, having generated GBP 20 on registration date, would expect to earn {french_women_1week_earn} in her first week")
logging.info(f"The average 40-year-old German woman, having generated GBP 20 on registration date, would expect to earn {german_women_1week_earn} in her first week")
logging.info(f"The average 40-year-old British woman, having generated GBP 20 on registration date, would expect to earn {british_women_1week_earn} in her first week")

# In[ ]:



