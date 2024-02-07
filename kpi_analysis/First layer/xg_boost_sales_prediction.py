import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt # Importing pyplot from matplotlib

# Assuming 'encoded_dataset.csv' contains your data
dataset = pd.read_csv('encoded_dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

le = LabelEncoder()
y = le.fit_transform(y)
correlation_df = pd.DataFrame(X, columns=dataset.columns[:-1])
correlation_df['Target'] = y

# Calculate correlation matrix
correlation_matrix = correlation_df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

y_pred = classifier.predict(X_test)
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
