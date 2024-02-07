import pandas as pd
import numpy as np

df = pd.read_csv("Dataset Projects goals.csv", encoding="ISO-8859-1")

print(df.columns)

data = df["product_value_rate"]

average_value = data.mean()

df["product_value_rate"] = df["product_value_rate"].apply(lambda x: "yes" if x < average_value else "no")

#y = df.drop("product_value_rate", axis=1)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
a = df.iloc[:,2]
print(a)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
a1 = le.fit_transform(a)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Assuming X is your input data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 1])], remainder='passthrough')
X = ct.fit_transform(X)




