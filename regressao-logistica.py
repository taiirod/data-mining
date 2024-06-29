import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Read dataset to pandas dataframe
data = pd.read_csv('cleaned_data.csv')

data.info()

# Exemplo de preparação dos data (substitua com seus próprios data)
X = data[['feature1', 'feature2', ...]]  # Escolha as colunas relevantes como variáveis independentes
y = data['target_column']  # Especifique a coluna alvo

# Dividir os data em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o modelo de regressão logística
modelo = LogisticRegression()

# Treinar o modelo nos data de treino
modelo.fit(X_train, y_train)

# Fazer previsões nos data de teste
previsoes = modelo.predict(X_test)

# Avaliar a precisão do modelo
acuracia = accuracy_score(y_test, previsoes)
print(f'Acurácia do modelo: {acuracia}')

# Exibir relatório de classificação
print(classification_report(y_test, previsoes))