import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o arquivo CSV
data = pd.read_csv('cleaned_data.csv')

# Explorar e pré-processar os dados
print(data.head())
print(data.info())

# Verifique se há valores nulos
print(data.isnull().sum())

# Tratar valores nulos (se necessário)
data = data.dropna()

# Selecionar variáveis independentes e dependentes

X = data.drop(columns=['is_canceled'])  # Substitua pelas suas colunas
y = data['is_canceled']  # Substitua pela sua coluna alvo

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo de regressão logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Fazer previsões e avaliar o modelo
y_pred = model.predict(X_test)

# Avaliar a precisão
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy}')

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
print('Matriz de Confusão:')
print(cm)

# Relatório de Classificação
cr = classification_report(y_test, y_pred)
print('Relatório de Classificação:')
print(cr)

# Visualizar a Matriz de Confusão
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()
