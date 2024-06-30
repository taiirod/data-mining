from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read dataset to pandas dataframe
data = pd.read_csv('cleaned_data.csv')

sns.countplot(x='is_canceled', data=data, hue='is_canceled')

# Configurar o tamanho da figura
plt.figure(figsize=(20, 10))

# Plotar o heatmap usando seaborn
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)

# Ajustar o layout
plt.title('Matriz de Correlação')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# Mostrar o plot
plt.show()
plt.savefig('correlacao.png')

