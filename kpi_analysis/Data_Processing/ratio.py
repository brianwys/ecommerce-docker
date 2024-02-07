import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

df = pd.read_csv("department_store_dataset.csv", encoding="ISO-8859-1")


# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])
# Extract the month and year from the 'Date' column
df['YearMonth'] = df['Date'].dt.to_period("Q")
# Drop the original 'Date' column
df = df.drop(columns=['Date'])

#df = df.drop(columns=['Date'])
#a = df["Department"].unique()
#b= len(a)
#c = df["Region"].unique()
#d = len(c)

jenis_barang = {'name': ['EletrÃ´nicos', 'VestuÃ¡rio', 'AcessÃ³rios', 'Casa', 'Brinquedo', 'Esportes', 'Papelaria']}
jenis_barang_baru = ['Jawa', 'Sumatra', 'Kalimantan', 'Papua', 'Sulawesi', 'NTB', 'Bali']
df['Region'] = [jenis_barang_baru[i % len(jenis_barang_baru)] for i in range(len(df))]

# Sample DataFrame
data = {'Name': ['LetÃ\xadcia Nascimento', 'Ana Sousa', 'Gustavo Martins', 'Beatriz Santos',
                 'Camila Lima', 'Thiago Barbosa', 'LetÃ\xadcia Ribeiro', 'Enzo Nascimento',
                 'Guilherme Santos', 'VitÃ³ria Ribeiro', 'Julia AraÃºjo', 'Lucas Rodrigues',
                 'Thiago Carvalho', 'Mateus Barbosa', 'Camila Carvalho', 'Diego Cardoso',
                 'Jorge Santos', 'Raphael Silva', 'Caroline Reis']}

# Department names
department_names = ['Sales Operations', 'Inside Sales', 'Outside Sales', 'Account Management',
                    'Business Development', 'Sales Enablement', 'Channel Sales', 'Key Accounts',
                    'Sales Analytics', 'Customer Success', 'Sales Training', 'Strategic Partnerships',
                    'Sales Support', 'Retail Sales', 'E-commerce Sales', 'Enterprise Sales',
                    'International Sales', 'Sales Administration', 'Sales Engineering']


df['Department'] = [department_names[i % len(department_names)] for i in range(len(df))]

df['KPi Achieved'] = ['NO' if r < rg else 'YES' for r, rg in zip(df["Revenue"], df["Revenue Goal"])]
df.to_csv("Dataset.csv", index=False)
print(df)

# Assuming 'KPi Achieved' column has values 'YES' and 'NO', you can convert it to 1 and 0 for calculation
df['KPi Achieved'] = df['KPi Achieved'].map({'YES': 1, 'NO': 0})
# Group by 'Department' and calculate the ratio
department_ratios = df.groupby('Department')['KPi Achieved'].mean().reset_index()
# Add the calculated ratio as a new feature in the original DataFrame
df = pd.merge(df, department_ratios, on='Department', how='left', suffixes=('', '_ratio'))
# Rename the new column for clarity
df.rename(columns={'KPi Achieved_ratio': 'KPI_Achievement_Ratio'}, inplace=True)
# Print or visualize the result
print(df)

# Plotting the ratios
plt.figure(figsize=(10, 6))
plt.bar(department_ratios['Department'], department_ratios['KPi Achieved'], color='skyblue')
plt.xlabel('Department')
plt.ylabel('KPI Achievement Ratio')
plt.title('KPI Achievement Ratio by Department')
plt.xticks(rotation=45, ha='right')
plt.show()

