import pandas as pd

df = pd.read_csv("Dataset.csv", encoding="ISO-8859-1")

# Group by the 'department' column
grouped_df = df.groupby('Department')

# Create a dictionary to store department datasets
department_data = {}

# Iterate over groups and store them in the dictionary
for department_name, group in grouped_df:
    department_data[department_name] = group.copy()

# Accessing the dataset for the "Outside Sales" department
#outside_sales = department_data["Outside Sales"]
#outside_sales.drop(columns=["Region"], inplace=True)

# Group by the 'Department', 'Region', and 'YearMonth' columns
grouped_df = df.groupby(['Department', 'YearMonth'])

# Calculate the sum for the specified columns
columns_to_sum = ['Revenue', 'Revenue Goal', 'Margin', 'Margin Goal', 'Sales Quantity', 'Customers']
data_sum_per_region_date = grouped_df[columns_to_sum].sum().reset_index()
data_sum_per_region_date.to_csv("data_sum_per_region_date_all_dept.csv", index=False)
