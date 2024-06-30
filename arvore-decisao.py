import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('cleaned_data.csv')

X = data.drop(columns=['is_canceled']).values
y = data['is_canceled'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
print("Formato de X_train:", X_train.shape)
print("Formato de y_train:", y_train.shape)
print("Formato de X_test:", X_test.shape)
print("Formato de y_test:", y_test.shape)

# Criar um objeto classificador de árvore de decisão
clf = DecisionTreeClassifier()

# Treinar o classificador usando os dados de treinamento
clf = clf.fit(X_train, y_train)

# Prever os rótulos dos dados de teste
y_pred = clf.predict(X_test)

# Calcular e imprimir a precisão
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)

# Configurar o tamanho da figura
plt.figure(figsize=(20, 10))

# Plotar a árvore de decisão
tree.plot_tree(clf, filled=True)

# Mostrar o gráfico
plt.show()
