import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

# ------ data 2017 to 2022-------#
df = pd.read_csv("Dataset.csv", encoding="ISO-8859-1")

# Extract features (X) and target variable (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Extract the first 3 columns as a DataFrame
a = df.iloc[:, 0:3]

# Apply LabelEncoder to the target variable 'y'
le = LabelEncoder()
y = le.fit_transform(y)

# Apply LabelEncoder to each column in DataFrame 'a'
a = a.apply(LabelEncoder().fit_transform)

#Replace the original columns in df with the transformed columns in 'a'
df.iloc[:, 0:3] = a

#------- data 2019 to 2022------#
new_ds = df.loc[13870:,:]#ini dari 2019 to 2022
# Extract features (X) and target variable (y)
x1 = new_ds.iloc[:, :-1].values
y1 = new_ds.iloc[:, -1].values

# Extract the first 3 columns as a DataFrame
b = new_ds.iloc[:, 0:3]

# Apply LabelEncoder to the target variable 'y'
le = LabelEncoder()
y1 = le.fit_transform(y1)

# Apply LabelEncoder to each column in DataFrame 'a'
b = b.apply(LabelEncoder().fit_transform)

#Replace the original columns in df with the transformed columns in 'a'
new_ds.iloc[:, 0:3] = b


#--------- Data train 2018 to 2021, test 2022 --------------#

new_ds1 = df.loc[6935:34694,:]#ini dari 2019 to 2022
# Extract features (X) and target variable (y)
x2 = new_ds1.iloc[:, :-1].values
y2 = new_ds1.iloc[:, -1].values

test_ds = df.loc[34694:,:]
x3 = test_ds.iloc[:, :-1].values
y3 = test_ds.iloc[:, -1].values

# Extract the first 3 columns as a DataFrame
c = new_ds1.iloc[:, 0:3]
d = test_ds.iloc[:,0:3]
# Apply LabelEncoder to the target variable 'y'
le = LabelEncoder()
y2 = le.fit_transform(y2)
le = LabelEncoder()
y3 = le.fit_transform(y3)
# Apply LabelEncoder to each column in DataFrame 'a'
c = c.apply(LabelEncoder().fit_transform)
d = d.apply(LabelEncoder().fit_transform)

#Replace the original columns in df with the transformed columns in 'a'
new_ds1.iloc[:, 0:3] = c
test_ds.iloc[:,0:3] = d
# Assuming 'c' and 'd' are DataFrames with the same structure as the first three columns of x2 and x3

# Replace the values in the first three columns of x2 with the transformed values of the first three columns of c
x2[:, 0:3] = c.values

# Replace the values in the first three columns of x3 with the transformed values of the first three columns of d
x3[:, 0:3] = d.values


# Create a random forest classifier
rf_classifier = RandomForestClassifier()

# Train the model
rf_classifier.fit(x2, y2)

# Make predictions on the test set
y_pred = rf_classifier.predict(x3)

# Evaluate the model
accuracy = accuracy_score(y3, y_pred)
print(f"Accuracy: {accuracy}")


#--------- Random Forest for data 2019 to 2022 -------------#
# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.25, random_state=1)

# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
       
# Train the Random Forest Classification model on the Training set
classifier = RandomForestClassifier(n_estimators=9, criterion='entropy', random_state=1)
classifier.fit(X_train, y_train)

# Predicting a new result
new_data_point = np.array([[0, 6, 0, 765.82, 1479.18, 0.09, 0.18, 3,11]])
new_data_point_scaled = sc.transform(new_data_point)
new_prediction = classifier.predict(new_data_point_scaled)
print("Prediction for the new data point:", new_prediction)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix for 2019-2022:\n", cm)
print("Accuracy for 2019-2022: {:.2f} %".format(accuracy_score(y_test, y_pred) * 100))

# Cross-validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Cross-Validation Accuracy for 2019-2022: {:.2f} %".format(accuracies.mean() * 100))
print("Standard Deviation for 2019-2022: {:.2f} %".format(accuracies.std() * 100))

#data_counter = new_ds["KPi Achieved"].value_counts("YES")
#print(data_counter)

# Compute the F1 score for each class
f1 = f1_score(y_test, y_pred, average=None)
# Print the F1 scores for each class
print("F1 Scores for each class in 2019-2022 dataset:")
for i in range(len(f1)):
    print(f"Class {i+1}: {f1[i]}")
# Print the average F1 score
print(f"Average F1 Score: {f1_score(y_test, y_pred, average='weighted')}")



#-------- Random Forest for data 2017 - 2022---------#


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
       
# Train the Random Forest Classification model on the Training set
classifier = RandomForestClassifier(n_estimators=9, criterion='entropy', random_state=1)
classifier.fit(X_train, y_train)

# Predicting a new result
new_data_point = np.array([[0, 6, 0, 765.82, 1479.18, 0.09, 0.18, 3,11]])
new_data_point_scaled = sc.transform(new_data_point)
new_prediction = classifier.predict(new_data_point_scaled)
print("Prediction for the new data point:", new_prediction)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy: {:.2f} %".format(accuracy_score(y_test, y_pred) * 100))

# Cross-validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Cross-Validation Accuracy: {:.2f} %".format(accuracies.mean() * 100))
print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))

data_counter = df["KPi Achieved"].value_counts("YES")
print(data_counter)

# Compute the F1 score for each class
f1 = f1_score(y_test, y_pred, average=None)
# Print the F1 scores for each class
print("F1 Scores for each class:")
for i in range(len(f1)):
    print(f"Class {i+1}: {f1[i]}")
# Print the average F1 score
print(f"Average F1 Score: {f1_score(y_test, y_pred, average='weighted')}")


