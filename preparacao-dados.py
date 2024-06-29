import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Carregar os dados
data = pd.read_csv('hotel_bookings.csv')

# 1. Tratar valores faltantes
missing_data_columns = ['agent', 'company']
for col in missing_data_columns:
    data[col].fillna(data[col].median())

# Label Encoding para variáveis categóricas de baixa cardinalidade e ordem natural
le = LabelEncoder()
le_columns = ['hotel',
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
    'required_car_parking_spaces',
    'arrival_date_year',
    'arrival_date_month',
    'arrival_date_week_number',
    'meal',
    'country',
    'market_segment',
    'reserved_room_type',
    'assigned_room_type',
    'customer_type',
    'reservation_status'
]

for col in winso_columns:
    data[col] = winsorize(data[col], limits=[0.05, 0.05])

columns_to_normalize = [
    'lead_time',
    'required_car_parking_spaces',
    'arrival_date_year',
    'arrival_date_month',
    'arrival_date_week_number',
    'meal',
    'country',
    'market_segment',
    'reserved_room_type',
    'assigned_room_type',
    'customer_type',
    'reservation_status',
    'lead_time',
    'arrival_date_year',
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'adults',
    'children',
    'babies',
    'previous_cancellations',
    'previous_bookings_not_canceled',
    'booking_changes',
    'required_car_parking_spaces',
    'days_in_waiting_list',
    'adr',
    'total_of_special_requests'
]

# Criar e aplicar o StandardScaler
scaler = StandardScaler()
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

data = data.dropna(axis=0, how='all')
data = data.drop_duplicates()

data = data.drop(columns=['reservation_status_date'])
data.to_csv('cleaned_data.csv', index=False)

print('Arquivo salvo.')
