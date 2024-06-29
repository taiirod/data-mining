import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Carregar os dados
data = pd.read_csv('hotel_bookings.csv')

# 1. Tratar valores faltantes
missing_data_columns = [
    'agent',
    'company'
]
for col in missing_data_columns:
    if data[col].dtype in ['float64', 'int64']:
        data[col] = data[col].fillna(data[col].median())

# Label Encoding para variáveis categóricas de baixa cardinalidade e ordem natural
le = LabelEncoder()
le_columns = [
    'hotel',
    'meal',
    'country',
    'reserved_room_type',
    'assigned_room_type',
    'deposit_type',
    'reservation_status',
    'customer_type',
    'distribution_channel',
    'arrival_date_month',
    'market_segment'
]

# Aplicar Label Encoding a cada coluna
for col in le_columns:
    data[col] = le.fit_transform(data[col])

# 2. Tratar outliers com Winsorização
winso_columns = [
    'lead_time',
]

for col in winso_columns:
    data[col] = winsorize(data[col], limits=[0.05, 0.05])

columns_to_normalize = [
    'hotel',
    'lead_time',
    'arrival_date_year',
    'arrival_date_month',
    'arrival_date_week_number',
    'arrival_date_day_of_month',
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'adults',
    'children',
    'babies',
    'meal',
    'country',
    'market_segment',
    'distribution_channel',
    'is_repeated_guest',
    'previous_cancellations',
    'previous_bookings_not_canceled',
    'reserved_room_type',
    'assigned_room_type',
    'booking_changes',
    'deposit_type',
    'agent',
    'company',
    'days_in_waiting_list',
    'customer_type',
    'adr',
    'required_car_parking_spaces',
    'total_of_special_requests',
    'reservation_status'
]

# Criar e aplicar o StandardScaler
scaler = StandardScaler()
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

data = data.dropna(axis=0, how='all')
data = data.drop_duplicates()

data = data.drop(columns=['reservation_status_date'])
data.to_csv('cleaned_data.csv', index=False)

print('Arquivo salvo.')
