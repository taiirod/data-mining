import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Carregar o arquivo CSV
df = pd.read_csv('cleaned_data.csv')

# Converter DataFrame do pandas para matriz numpy
matriz_dados = df.to_numpy()

# Calcular a correlação de Pearson entre todas as colunas
num_colunas = matriz_dados.shape[1]
correlation_matrix = np.zeros((num_colunas, num_colunas))

for i in range(num_colunas):
    for j in range(num_colunas):
        # Calcular a correlação de Pearson entre as colunas i e j
        correlation_matrix[i, j], _ = pearsonr(matriz_dados[:, i], matriz_dados[:, j])

# Exibir a matriz de correlação
print("Matriz de correlação de Pearson:")
print(correlation_matrix)