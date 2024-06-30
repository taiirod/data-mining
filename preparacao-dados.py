import pandas as pd
from scipy.stats.mstats import winsorize
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import calendar

# Carregar os dados
data = pd.read_csv('hotel_bookings.csv')

# Drop de colunas não necessarias para o dominio da aplicação
columns_to_drop = [
    'arrival_date_year',
    'arrival_date_week_number',
    'meal',
    'country',
    'is_repeated_guest',
    'babies',
    'previous_bookings_not_canceled',
    'customer_type',
    'company',
    'days_in_waiting_list',
    'reservation_status',
    'reservation_status_date'
]

data.drop(columns_to_drop, axis=1, inplace=True)

# Tratar valores faltantes
missing_data_columns = [
    'agent',
]
for col in missing_data_columns:
    data[col] = data[col].fillna(data[col].mean())

# Label Encoding para variáveis categóricas
le = LabelEncoder()
le_columns = [
    'hotel',
    'arrival_date_month',
    'market_segment',
    'distribution_channel',
    'reserved_room_type',
    'assigned_room_type',
    'deposit_type',
]

for col in le_columns:
    data[col] = le.fit_transform(data[col])

# Histograma do tempo de antecedência da reserva
data['lead_time'].hist(bins=100)
plt.xlabel('Lead Time (in days)')
plt.ylabel('Frequency')
plt.title('Distribution of Lead Time')
plt.show()

# Histograma dos dias totais gastos no hotel
data['total_nights'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
data['total_nights'].hist(bins=100)
plt.xlabel('Total Nights Spent')
plt.ylabel('Frequency')
plt.title('Distribution of Total Nights Spent in Hotel')
plt.show()

# Histograma do total de pessoas (adultos + crianças)
data['total_people'] = data['adults'] + data['children']
data['total_people'].hist(bins=100)
plt.xlabel('Total People')
plt.ylabel('Frequency')
plt.title('Distribution of Total People in Reservations')
plt.show()

# Criando o histograma para mês de chegada
month_names = [calendar.month_name[i] for i in range(1, 13)]
plt.figure(figsize=(10, 6))
plt.hist(data['arrival_date_month'], bins=100, edgecolor='black')
plt.title('Histogram of Arrival Months')
plt.xlabel('Month')
plt.ylabel('Frequency')
plt.xticks(range(0, 12), month_names, rotation=45)
plt.grid(True)
plt.show()

# Gráfico de dispersão tempo de antecedência da reserva vs preço médio por quarto
data = data[data['adr'] > 0]
data['adr'] = winsorize(data['adr'], limits=[0.05, 0.05])
plt.scatter(data['lead_time'], data['adr'])
plt.xlabel('Lead Time')
plt.ylabel('Average Price per Room')
plt.title('Lead Time vs. Average Price per Room')
plt.show()
plt.show()

# Buscar por colunas que possuem outliers
outliers_count = {}
winso_columns = []
for column in data.columns:
    if data[column].dtype != 'object':  # Exclui colunas não numéricas
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        # Encontra outliers e conta o total
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        outliers_count[column] = outliers.shape[0]
        # Adiciona a coluna à lista se tiver outliers
        if outliers_count[column] > 0:
            winso_columns.append(column)

# Tratar outliers com Winsorização
for col in winso_columns:
    data[col] = winsorize(data[col], limits=[0.05, 0.05])

# Normalizar colunas
columns_to_normalize = [
    'hotel',
    'lead_time',
    'arrival_date_month',
    'arrival_date_day_of_month',
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'adults',
    'children',
    'market_segment',
    'distribution_channel',
    'previous_cancellations',
    'reserved_room_type',
    'assigned_room_type',
    'booking_changes',
    'deposit_type',
    'agent',
    'adr',
    'required_car_parking_spaces',
    'total_of_special_requests',
]

scaler = StandardScaler()
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

data = data.dropna(axis=0, how='all')
data = data.drop_duplicates()

data.to_csv('cleaned_data.csv', index=False)

print('Arquivo salvo.')
