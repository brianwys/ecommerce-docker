import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score


# Load the dataset
dataset = pd.read_csv('encoded_dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

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

data_counter = dataset["KPi Achieved"].value_counts("YES")
print(data_counter)

from sklearn.metrics import f1_score
# Compute the F1 score for each class
f1 = f1_score(y_test, y_pred, average=None)
# Print the F1 scores for each class
print("F1 Scores for each class:")
for i in range(len(f1)):
    print(f"Class {i+1}: {f1[i]}")
# Print the average F1 score
print(f"Average F1 Score: {f1_score(y_test, y_pred, average='weighted')}")

"""


####
import tkinter as tk
from tkinter import messagebox

# Create the tkinter window
root = tk.Tk()
root.title("Random Forest Classifier")

# Create input fields
var_1 = tk.IntVar()
var_2 = tk.IntVar()
var_3 = tk.IntVar()
var_4 = tk.IntVar()
var_5 = tk.DoubleVar()
var_6 = tk.DoubleVar()
var_7 = tk.DoubleVar()
var_8 = tk.IntVar()
var_9 = tk.IntVar()

label_1 = tk.Label(root, text="Feature 1")
label_2 = tk.Label(root, text="Feature 2")
label_3 = tk.Label(root, text="Feature 3")
label_4 = tk.Label(root, text="Feature 4")
label_5 = tk.Label(root, text="Feature 5")
label_6 = tk.Label(root, text="Feature 6")
label_7 = tk.Label(root, text="Feature 7")
label_8 = tk.Label(root, text="Feature 8")
label_9 = tk.Label(root, text="Feature 9")

entry_1 = tk.Entry(root, textvariable=var_1)
entry_2 = tk.Entry(root, textvariable=var_2)
entry_3 = tk.Entry(root, textvariable=var_3)
entry_4 = tk.Entry(root, textvariable=var_4)
entry_5 = tk.Entry(root, textvariable=var_5)
entry_6 = tk.Entry(root, textvariable=var_6)
entry_7 = tk.Entry(root, textvariable=var_7)
entry_8 = tk.Entry(root, textvariable=var_8)
entry_9 = tk.Entry(root, textvariable=var_9)

button = tk.Button(root, text="Predict", command=lambda: predict(var_1.get(), var_2.get(), var_3.get(), var_4.get(), var_5.get(), var_6.get(), var_7.get(), var_8.get(), var_9.get()))

# Position the labels and input fields
label_1.grid(row=0, column=0)
label_2.grid(row=1, column=0)
label_3.grid(row=2, column=0)
label_4.grid(row=3, column=0)
label_5.grid(row=4, column=0)
label_6.grid(row=5, column=0)
label_7.grid(row=6, column=0)
label_8.grid(row=7, column=0)
label_9.grid(row=8, column=0)

entry_1.grid(row=0, column=1)
entry_2.grid(row=1, column=1)
entry_3.grid(row=2, column=1)
entry_4.grid(row=3, column=1)
entry_5.grid(row=4, column=1)
entry_6.grid(row=5, column=1)
entry_7.grid(row=6, column=1)
entry_8.grid(row=7, column=1)
entry_9.grid(row=8, column=1)

button.grid(row=9, column=1)

# Function to predict
def predict(feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9):
    new_data_point = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]])
    new_data_point_scaled = sc.transform(new_data_point)
    new_prediction = classifier.predict(new_data_point_scaled)
    messagebox.showinfo("Prediction", "The predicted class is: " + str(new_prediction[0]))

# Run the tkinter loop
root.mainloop()
"""