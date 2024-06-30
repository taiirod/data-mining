import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

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
