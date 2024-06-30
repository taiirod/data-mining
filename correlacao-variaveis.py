import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read dataset to pandas dataframe
data = pd.read_csv('cleaned_data.csv')

# Contagem de reservas canceladas vs. não canceladas
booking_status_counts = data['is_canceled'].value_counts()

# Cálculo das porcentagens
booking_status_percentages = (booking_status_counts / booking_status_counts.sum()) * 100
ax = booking_status_counts.plot(kind='bar', color=['green', 'red'])
plt.xlabel('Booking Status')
plt.ylabel('Number of Bookings')
plt.title('Canceled vs. Not Canceled Bookings')

# Adicionar rótulos de porcentagem acima das barras
for i in ax.patches:
    ax.annotate(f'{i.get_height()} ({i.get_height() / booking_status_counts.sum() * 100:.2f}%)',
                (i.get_x() + i.get_width() / 2, i.get_height()),
                ha='center', va='center', xytext=(0, 8), textcoords='offset points')

plt.show()

# Configurar o tamanho da figura
plt.figure(figsize=(20, 10))

# Plotar o heatmap usando seaborn
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)

# Ajustar o layout
plt.title('Matriz de Correlação')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# Mostrar o plot
plt.show()
