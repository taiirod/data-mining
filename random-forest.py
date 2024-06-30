import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('cleaned_data.csv')

# Selecionar variáveis independentes (features) e a variável alvo
X = data.drop(columns=['is_canceled']).values
y = data['is_canceled'].values

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Criar um objeto classificador Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinar o classificador usando os dados de treinamento
clf.fit(X_train, y_train)

# Prever os rótulos dos dados de teste
y_pred = clf.predict(X_test)

# Calcular e imprimir a pontuação de precisão
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Imprimir o relatório de classificação
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# Calcular e imprimir a matriz de confusão
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
