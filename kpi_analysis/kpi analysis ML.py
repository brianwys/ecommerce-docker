import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Read the CSV file
df = pd.read_csv("Dataset Projects goals.csv", encoding="ISO-8859-1")

# Display column names
print(df.columns)

# Convert "product_value_rate" to binary values based on the average
average_value = df["product_value_rate"].mean()
df["product_value_rate"] = np.where(df["product_value_rate"] < average_value, "yes", "no")

# Extract features (X) and target variable (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Extract column "a" as a separate variable
a = df.iloc[:, 2].values

# Print values of column "a"
print(a)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Label encode the target variable (y) and column "a"
le_y = LabelEncoder()
le_a = LabelEncoder()
y = le_y.fit_transform(y)
a = le_a.fit_transform(a)

# Use a ColumnTransformer to one-hot encode specific columns in X
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 1])], remainder='passthrough')
X = ct.fit_transform(X)

# If needed, transform X_train and X_test using the same ColumnTransformer
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)
