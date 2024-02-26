# Importar las librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
import warnings

# Montar Google Drive
drive.mount("/content/Drive")

# Evitar mostrar warnings
warnings.filterwarnings("ignore")

# Leer el archivo CSV
df_banco = pd.read_csv('/content/Drive/MyDrive/Inmersion/german_credit.csv')

# Función para procesar los datos
def procesar_datos():
  global df_banco
  # Eliminar duplicados si existen
  df_banco = df_banco.drop_duplicates() if df_banco.duplicated().any() else df_banco
  # Eliminar filas con valores nulos si existen
  df_banco = df_banco.dropna() if df_banco.isnull().values.any() else df_banco

  # Mapear valores de las columnas categóricas a numéricos
  a = {'no checking account': 4,
      '>= 200 DM / salary assignments for at least 1 year': 3,
      '0 <= ... < 200 DM': 2,
      '< 0 DM': 1
  }
  df_banco['account_check_status'] = df_banco['account_check_status'].map(a)

  # Repetir el proceso de mapeo para las demás columnas categóricas

procesar_datos()

# Función para crear variables adicionales
def feature_engineering():
  global df_banco
  # Crear nuevas variables basadas en las existentes
  dic_sexo = {2:1, 5:1, 1:0, 3:0, 4:0}
  dic_est_civil = {3:1, 5:1, 1:0, 2:0, 4:0}
  df_banco['sexo'] = df_banco['personal_status_sex'].map(dic_sexo)
  df_banco['estado_civil'] = df_banco['personal_status_sex'].map(dic_est_civil)
  df_banco['rango_edad'] = pd.cut(x=df_banco['age'], bins=[18, 30, 40, 50, 60, 70, 80], labels=[1, 2, 3, 4, 5, 6]).astype(int)
  df_banco['rango_plazos_credito'] = pd.cut(x=df_banco['duration_in_month'], bins=[1, 12, 24, 36, 48, 60, 72], labels=[1, 2, 3, 4, 5, 6]).astype(int)
  df_banco['rango_valor_credito'] = pd.cut(x=df_banco['credit_amount'], bins=[1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000], labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]).astype(int)
  df_banco = df_banco.drop(columns=['personal_status_sex', 'age', 'duration_in_month', 'credit_amount'])

feature_engineering()

# Configurar el estilo de Seaborn
sns.set(style="whitegrid")

# Crear el histograma utilizando Seaborn
plt.figure(figsize=(5, 4))
sns.countplot(data=df_banco, x='sexo')
plt.title('Histograma de Sexo')
plt.xlabel('Sexo')
plt.ylabel('Frecuencia')
plt.show()

# Función para realizar análisis exploratorio
def analisis_exploratorio():
  global df_banco
  histogramas = ['sexo', 'estado_civil', 'rango_plazos_credito', 'rango_edad', 'default']
  lista_histogramas = list(enumerate(histogramas))
  plt.figure(figsize=(30, 20))
  plt.title('Histogramas')
  for i in lista_histogramas:
    plt.subplot(3, 2, i[0]+1)
    sns.countplot(x=i[1], data=df_banco)
    plt.xlabel(i[1], fontsize=20)
    plt.ylabel('Total', fontsize=20)

analisis_exploratorio()

# Analizar los datos de las distribuciones e identificar valores atípicos
print(df_banco.describe())

# Crear un mapa de calor para analizar la correlación de las variables
plt.figure(figsize=(12, 8))
sns.heatmap(df_banco.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de correlación')
plt.show()

# Conclusiones de los histogramas
print("Conclusión del histograma de sexo:")
print("El histograma muestra que hay más clientes de sexo masculino que femenino en el dataset.")

