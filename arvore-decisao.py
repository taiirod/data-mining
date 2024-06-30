import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Read dataset to pandas dataframe
data = pd.read_csv('cleaned_data.csv')

X = data.drop(columns=['is_canceled']).values
y = data['is_canceled'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
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
plt.show()
