import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Your data loading code
df = pd.read_csv("Dataset.csv", encoding="ISO-8859-1")
rev_goal_med_pred = pd.read_csv("final_pred.csv", encoding="ISO-8859-1") 
#outside_sales.reset_index(drop=True, inplace=True)
#outside_sales = df[df["Department"] == "Outside Sales"]
#rev_med = pd.read_csv("rev_med.csv", encoding="ISO-8859-1") 
data_sum_per_region_date = pd.read_csv("data_sum_per_region_date.csv", encoding="ISO-8859-1")
rev_goal_med = pd.read_csv("rev_goal_median.csv", encoding="ISO-8859-1") 

data_sum_per_region_date['YearMonth'] = pd.to_datetime(data_sum_per_region_date['YearMonth'])
data_sum_per_region_date['Month'] = data_sum_per_region_date['YearMonth'].dt.month
data_sum_per_region_date['Year'] = data_sum_per_region_date['YearMonth'].dt.year
data_sum_per_region_date.drop(columns=["YearMonth"], inplace=True)

#rev_med['YearMonth'] = pd.to_datetime(rev_med['YearMonth'])
#rev_med['Month'] = rev_med['YearMonth'].dt.month
#rev_med['Year'] = rev_med['YearMonth'].dt.year
#rev_med.drop(columns=["YearMonth"], inplace=True)

rev_goal_med['YearMonth'] = pd.to_datetime(rev_goal_med['YearMonth'])
rev_goal_med['Month'] = rev_goal_med['YearMonth'].dt.month
rev_goal_med['Year'] = rev_goal_med['YearMonth'].dt.year
rev_goal_med.drop(columns=["YearMonth"], inplace=True)

sum_data_per_month_with_rev_goal_median = pd.merge(data_sum_per_region_date, rev_goal_med, on=['Year', 'Month'], how='left', suffixes=('', '_median'))
sum_data_per_month_with_rev_goal_median['KPi Achieved'] = sum_data_per_month_with_rev_goal_median.pop('KPi Achieved')
sum_data_per_month_with_rev_goal_median.to_csv("sum_data_per_month_with_rev_goal_median.csv", index=False)

# Extracting relevant data
train_data = sum_data_per_month_with_rev_goal_median.loc[:29]
test_data = sum_data_per_month_with_rev_goal_median.loc[30:35]
rev_goals_data = sum_data_per_month_with_rev_goal_median["Revenue Goal"].loc[30:35]
pred_data = rev_goal_med_pred["0"]

# Make sure the indices match
test_data = test_data.reset_index(drop=True)
pred_data.index = test_data.index

# Assign values to "Revenue Predictions" column
test_data["Revenue Median Predictions"] = pred_data

# Comparing rev goals with rev pred element-wise using apply
test_data["KPI Pred Achieve"] = test_data.apply(lambda row: 'Yes' if row['Revenue Median Predictions'] > row['Revenue_Goal_Median'] else 'No', axis=1)

test_data_comparation = test_data
#----------------------- Sum of revenue goals and revenue pred-----------------

#---------------- processing --------------------














#test_data_sum.to_csv('test_data_sum.csv', index=False)
#test_data.to_csv("new_test_data.csv", index=False)