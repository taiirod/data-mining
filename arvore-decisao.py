import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Read dataset to pandas dataframe
dataset = pd.read_csv('cleaned_data.csv')

dataset.info()
dataset.head()
dataset.describe()

X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 4].values
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Create a decision tree classifier object
clf = DecisionTreeClassifier()

# Train the classifier using the training data
clf = clf.fit(X_train, y_train)

# Predict the labels of the test data
y_pred = clf.predict(X_test)

# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Configure the figure size
plt.figure(figsize=(20, 10))

# Plot the decision tree
tree.plot_tree(clf, filled=True)

# Save the figure as a high-definition image
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()
