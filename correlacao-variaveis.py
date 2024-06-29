from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read dataset to pandas dataframe
data = pd.read_csv('cleaned_data.csv')

data.info()

classes = data['is_canceled'].value_counts()
print(classes)
sns.countplot(x='is_canceled', data=data, hue='is_canceled')
plt.show()
