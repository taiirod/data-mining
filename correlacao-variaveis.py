from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read dataset to pandas dataframe
data = pd.read_csv('cleaned_data.csv')

sns.countplot(x='is_canceled', data=data, hue='is_canceled')

# Excluir linhas onde o valor da coluna 'coluna_especifica' seja igual a 'valor_X'
nao_cancelado = 0  # substitua com o valor que deseja excluir

data = data[data['is_canceled'] != nao_cancelado]

sns.countplot(x='is_canceled', data=data, hue='is_canceled')
corr_matrix = data[[
    'agent',
    'company',
    'lead_time',
    'hotel',
    'meal',
    'reserved_room_type',
    'assigned_room_type',
    'deposit_type',
    'reservation_status',
    'customer_type',
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'adults',
    'children',
    'babies',
    'previous_cancellations',
    'previous_bookings_not_canceled',
    'booking_changes',
    'days_in_waiting_list',
    'total_of_special_requests'
]]

# Configurar o tamanho da figura
plt.figure(figsize=(12, 10))

# Plotar o heatmap usando seaborn
sns.heatmap(corr_matrix.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)

# Ajustar o layout
plt.title('Matriz de Correlação')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# Mostrar o plot
plt.show()

