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

# 2. Tratar outliers com Winsorização
data['lead_time'] = winsorize(data['lead_time'], limits=[0.05, 0.05])

# 3. One-Hot Encoding para variáveis categóricas de alta cardinalidade
ohe = OneHotEncoder(drop='first', sparse_output=False)
# 3. One-Hot Encoding para variáveis categóricas de alta cardinalidade
ohe_columns = ['arrival_date_month',
               'country',
               'market_segment',
               'distribution_channel']

ohe_data = ohe.fit_transform(data[ohe_columns])

# Converter a matriz transformada em um DataFrame
ohe_df = pd.DataFrame(ohe_data, columns=ohe.get_feature_names_out(ohe_columns))

# Concatenar o DataFrame original com o DataFrame one-hot encoded e remover as colunas originais
data = pd.concat([data.drop(columns=ohe_columns), ohe_df], axis=1)

# 4. Label Encoding para variáveis categóricas de baixa cardinalidade e ordem natural
le = LabelEncoder()
le_columns = ['hotel',
              'meal',
              'reserved_room_type',
              'assigned_room_type',
              'deposit_type',
              'reservation_status',
              'customer_type'
              ]

# Aplicar Label Encoding a cada coluna
for col in le_columns:
    data[col] = le.fit_transform(data[col])

columns_to_normalize = [
    'lead_time',
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'adults',
    'children',
    'babies',
    'previous_cancellations',
    'previous_bookings_not_canceled',
    'booking_changes',
    'days_in_waiting_list',
    'adr',
    'required_car_parking_spaces',
    'total_of_special_requests'
]

# Criar e aplicar o StandardScaler
scaler = StandardScaler()
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

# Definir variáveis dependentes (X) e a variável alvo (y)
X = data.drop(columns=['is_canceled'])
y = data['is_canceled']

# Separar os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

data = data.dropna(axis=0, how='all')
data = data.drop_duplicates()

print('Tamanho do conjunto de treinamento:', X_train.shape)
print('Tamanho do conjunto de teste:', X_test.shape)

data = data.drop(columns=['reservation_status_date'])
data.to_csv('cleaned_data.csv', index=False)
