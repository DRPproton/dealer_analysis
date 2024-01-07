# %% [markdown]
# ### Loading libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
pd.set_option('display.max_columns', None)

# %% [markdown]
# ### Importing data

# %%
Q1_2021 = pd.read_csv('Q1_2021 (1).csv')
Q2_2021 = pd.read_csv('Q2_2021 (1).csv')

# %% [markdown]
# ### Data Shape Exploration

# %%
Q1_2021.head()

# %%
print(f'Amount of rows and columns first quarter => {Q1_2021.shape}')
print(f'Amount of rows and columns second quarter => {Q2_2021.shape}')

# %% [markdown]
# > Both datasets have the same amount of columns, and almost the same a mount os rows(orders)

# %% [markdown]
# ### Adding trimestre para hacer una sola base de datos

# %%
Q1_2021.insert(0, 'QPR', 'Q1')
Q2_2021.insert(0, 'QPR', 'Q2')

# %%
# Unir las dos DataFrames 
df = pd.concat([Q1_2021, Q2_2021], axis=0)

# %%
df.head()

# %%
df.tail()

# %% [markdown]
# ### Some basic EDA

# %%
df.describe(include='all').T

# %%
df.info()

# %%
df.isnull().sum()

# %% [markdown]
# ### Clean data and data types

# %% [markdown]
# #### Necesito cambiar los valores de las columnas de pago a floats para análisis    

# %% [markdown]
# - Función para limpiar valores numéricos 

# %%
def clean_num_values(df, columns):
    df_temp = df.copy()
    for col in columns:
        df_temp[col] = df_temp[col].str.replace('$', '').str.replace(',', '').astype(float)
        
    return df_temp

# %%
columns_to_clean = [col for col in df.columns if '$' in col]  # Getting columns that needs to be clean

# %%
columns_to_clean

# %%
df_num_clean = clean_num_values(df, columns_to_clean)

# %%
df_num_clean.info()

# %% [markdown]
# #### Convert datetime columns

# %%
df_num_clean.FechaApertura = pd.to_datetime(df_num_clean.FechaApertura, format='%d/%m/%Y %H:%M')
df_num_clean.FechaFactura = pd.to_datetime(df_num_clean.FechaFactura, format='%d/%m/%Y %H:%M')
df_num_clean.FechaEntrega = pd.to_datetime(df_num_clean.FechaEntrega, format='%d/%m/%Y %H:%M')

# %% [markdown]
# #### Convert IdRegistro to category

# %%
df_num_clean.IdRegistro = df_num_clean.IdRegistro.astype('category')

# %%
df_num_clean[df_num_clean.IdRegistro.duplicated()]

# %% [markdown]
# - Cada Id es único

# %% [markdown]
# ### Check VIN Column

# %%
df_num_clean.Vin.value_counts().sort_values(ascending=False)[:10]

# %%
vin_null_fill = df_num_clean[df_num_clean.Vin.isnull()]
df_num_clean[df_num_clean.Vin.isnull()].shape

# %%
vin_null_fill[['FechaFactura', 'IdRegistro', 'Client', 'Branch','Factura', 'Taller', 'TipoOrden','NumeroOT','Odometro', 'Vin']]

# %% [markdown]
# - 301 filas no tienen registrado ViN. 

# %% [markdown]
# > IdRegistro son valores únicos si no hay VIN significa que El VIN no se registro. Pondré un PlaceHolder para los valores nulos en la columna VIN

# %%
# VIN ex. 1HGRW1844KL905654
df_num_clean.Vin.fillna('SIN-REGISTRAR', inplace=True)

# %%
df_num_clean.isnull().sum()

# %% [markdown]
# ### Fill Null for Dates columns

# %%
df_num_clean.head()

# %%
# Use boolean indexing to update values in FechaApertura y FechaEntrega where it is null using FechaFactura as the replacement. 
df_num_clean['FechaApertura'] = df_num_clean.apply(lambda row: row['FechaFactura'] 
                                                    if pd.isnull(row['FechaApertura']) 
                                                    else row['FechaApertura'], axis=1
                                                )


# %%
df_num_clean['FechaEntrega'] = df_num_clean.apply(lambda row: row['FechaFactura'] 
                                                    if pd.isnull(row['FechaEntrega']) 
                                                    else row['FechaEntrega'], axis=1
                                                )

# %%
df_num_clean.isnull().sum()

# %% [markdown]
# > Null values en Color se sustutiara por 'No-Regristrado', la columna 'Dias' con al numero 0. La columna 'Modelo' se borrara por no tener ningun valor associado. 

# %%
# Fill null values for the Color column
df_num_clean.Color.fillna('NO-REGISTRADO',inplace=True)
# Fill null values for Dias column
df_num_clean.Dias.fillna(0, inplace=True)
# Dropping the column modelo. 
df_num_clean.drop(columns='Modelo', inplace=True)

# Converting Ano to categorical 
df_num_clean.Ano = df_num_clean.Ano.astype('category')

# %%
df_num_clean[['TipoOrden', 'TipoPago']].isnull().sum()

# %% [markdown]
# #### TipoOrden y TipoPago

# %%
df_num_clean[df_num_clean.isnull().any(axis=1)]['TipoOrden'].value_counts()

# %% [markdown]
# > Todos los valores nulos en la columna de TipoOrden son the Aseguradoras. Necesito saber como pagan 

# %%
df_num_clean.groupby('TipoOrden')['TipoPago'].unique()

# %%
df_num_clean.groupby(['TipoOrden','TipoPago'])['TipoPago'].count()

# %%
107/2765*100

# %% [markdown]
# > Menos del 4% de las veces las aseguradoras pagaron al contado asi que fill null con 'CREDITO' en la fila donde el TipoOrden sea ASEGURADORA y TipoPago sea null

# %%
df_num_clean['TipoPago'] = df_num_clean.apply(lambda row: 'CREDITO' if row['TipoOrden'] == 'ASEGURADORA' and pd.isnull(row['TipoPago']) else row['TipoPago'], axis=1)

# %% [markdown]
# > Finalmente quedan null values donde TipoOrden y TipoPago son nulls. Fill con 'OTRO' nueva categoria que se puede cambiar luego

# %%
df_num_clean[['TipoOrden', 'TipoPago']].isnull().sum()

# %%
df_num_clean.TipoOrden.fillna('OTRO', inplace=True)
df_num_clean.TipoPago.fillna('OTRO', inplace=True)

# %%
df_num_clean.isnull().sum().sum()

# %% [markdown]
# - No nul values, data listo para análisis

# %%
df_num_clean.to_csv('clean_data.csv', index=False)

# %%
data = pd.read_csv('clean_data.csv')

# %% [markdown]
# ## Analysis

# %%
# Function to get summary stats
def get_summary_stats_by_columns(df):
    column_name = df.columns
    new_df = pd.DataFrame(index=['Data type', 'Min', '25%', '50%', '75%','Max', 'Mean', 'Median', 'Mode', 'Unique Values Num', 'STD', 'Skewness', 'Kurtosis', 'Count'])
    for col in column_name:
        if pd.api.types.is_numeric_dtype(df[col]):
            new_df[col] = [df[col].dtype, df[col].min(), df[col].quantile(.25), df[col].quantile(.5), df[col].quantile(.75),df[col].max(), df[col].mean(), df[col].median(),
                                df[col].mode()[0], df[col].nunique(), df[col].std(), df[col].skew(), df[col].kurt() , df[col].count()]
    return new_df

# %%
get_summary_stats_by_columns(data[columns_to_clean])

# %%
data.hist(figsize=(20,20));

# %%
data.describe(exclude='number')

# %%
data.describe(exclude='number').columns

# %%
data.groupby('Taller')['Taller'].count().sort_values()

# %%
cols = ['Taller', 'TipoOrden', 'TipoPago', 'Marca']
for col in cols:
    sns.countplot(data[col])
    plt.show()

# %%
data.groupby('Ano')['Ano'].count()

# %%
data.Ano = data.Ano.replace(2029, 2019)
data.Ano = data.Ano.replace(2190, 2019)
data.Ano = data.Ano.replace(2121, 2021)
data.Ano = data.Ano.replace(2218, 2018)
data.Ano = data.Ano.replace(201, 2010)
data.Ano = data.Ano.replace(2, 2000)

# %% [markdown]
# ### Parte 1. Análisis de Datos

# %% [markdown]
# > ¿Cuántos y cuales VIN´s únicos que visitaron el taller durante el primer trimestre del año, también visitaron el taller al segundo trimestre

# %%
data.head(3)

# %%
q1_vin_num = data[data.QPR == 'Q1']['Vin'].unique()  # Numero único de Vin que visitaron el taller en Q1 (Primer trimestre)
print(f'Cantidad de Vins únicos en Q1 {len(q1_vin_num)}')

# %%
print('Vin que visitaron el taller en Q1:')
for num in q1_vin_num:
    print(f'Vin => {(num)}')

# %%
q2_vin_num = data[data.QPR == 'Q2']['Vin'].unique() # # Numero único de Vin que visitaron el taller en Q2 (Primer trimestre)

# %%
common_vins = set(q1_vin_num) & set(q2_vin_num)  # Numero de vins que estuvieron en Q1 y Q2

# %% [markdown]
# #### Sanity Check

# %%
if 'SIN-REGISTRAR' in q1_vin_num:
    idx = np.where(q1_vin_num == 'SIN-REGISTRAR')
    print(q1_vin_num[idx])

# %%
common_vins.remove('SIN-REGISTRAR') # Remover el valor del set

# %%
# Count Vin values in the set that do not meet the length criteria
incorrect_values = {value for value in common_vins if len(value) != 17}

# Count the number of incorrect values
count_incorrect_values = len(incorrect_values)

print(f"Números de Vin con largo incorrecto: {count_incorrect_values}")
print("Valores Incorrectos: ", incorrect_values)

# %%
print(f'Numero de Vins que visitaron Q1 y Q2 => {len(common_vins)}')

# %%
print(f"Vins que visitaron el taller en Q1 and Q2: {', '.join(common_vins)}")

# %% [markdown]
# > Aunque un total 10115 VIN´s únicos visitaron el taller durante el primer trimestre del año, también visitaron el taller al segundo trimestre. 
# 15 Vin fueron guardados incorrectamente. 

# %% [markdown]
# ### ¿Cuál es el porcentaje de órdenes de trabajo por Taller para cada Tipo de Orden?

# %%
data.head()

# %%
data.Taller.value_counts()

# %%
pd.crosstab(data.Taller, data.TipoOrden) # Total

# %%
pd.crosstab(data.Taller, data.TipoOrden, normalize='all', margins=True, margins_name = 'Total') * 100 # Porcentaje

# %%
pd.crosstab(data.Taller, data.TipoOrden).plot.barh()

# %% [markdown]
# ### ¿Qué sucursal tiene un mayor tiempo promedio en la resolución de servicios en cada uno de los meses?

# %%
data.head()

# %%
data['FechaFactura'] = pd.to_datetime(data['FechaFactura'])

# %%
data['Mes'] = data['FechaFactura'].dt.strftime('%B')

# %%
data.head()

# %%
data[data.Dias > 10000]
data.Dias = data.Dias.replace(32767.0, 18)  # Replace a error en la días. el coche entro el 26 enero y salio 16 febrero 3 semanas menos 3 fines de semana 18 días

# %%
resolución_promedio_por_mes = data.groupby(['Mes', 'Branch'])['Dias'].mean()

# %%
resolución_promedio_por_mes.loc[resolución_promedio_por_mes.groupby('Mes').idxmax()]

# %% [markdown]
# ### ¿Cuál es la sucursal con mayor Margen de Utilidad Bruta en cada uno de los meses?

# %%
data.head()

# %% [markdown]
# $$ \text{Margen de Utilidad Bruta} = \left( \frac{\text{Venta Total} - \text{Costo Total}}{\text{Venta Total}} \right) \times 100 

# %% [markdown]
# > Voy a usar la columna Margen para responder esta pregunta, con mas tiempo volvería a calcular si el  resultado es el mismo si uso la formula

# %%
margen_utilidad_por_mes = data.groupby(['Mes', 'Branch'])['Margen'].mean()

# %%
margen_utilidad_por_mes.loc[margen_utilidad_por_mes.groupby('Mes').idxmax()] * 100

# %% [markdown]
# ### Elabora un gráfico que permita analizar el Tiket Promedio Total

# %%
data.head()

# %% [markdown]
# > Nota: El ticket promedio (también conocido como «promedio de venta» o «promedio de compra») 
# - Cómo calcular el ticket promedio
#     - Para calcular el ticket promedio de una empresa o negocio, simplemente debes seguir estos pasos:
#         - Suma el total de las ventas realizadas durante un período determinado (por ejemplo, un día, una semana o un mes).
#         - Divide esa cantidad por el número total de transacciones realizadas durante ese mismo período.
#         - El resultado será el monto promedio que cada cliente gastó en cada transacción.

# %%
# Agrupar por fecha y calcular el ticket promedio (VentaTotal$)
ticket_promedio_diario = data.groupby(data['FechaFactura'].dt.date)['VentaTotal$'].mean().reset_index()


# %%
ticket_promedio_diario['FechaFactura'] = pd.to_datetime(ticket_promedio_diario['FechaFactura'])

# %%
plt.figure(figsize=(12, 6))
plt.plot(ticket_promedio_diario['FechaFactura'], ticket_promedio_diario['VentaTotal$'], marker='o');
plt.title('Ticket Promedio Total Diario')
plt.xlabel('Fecha')
plt.ylabel('Ticket Promedio ($)')
plt.grid(True)
plt.show()

# %% [markdown]
# ### Elabora un gráfico que permita analizar las ventas por Tipo de Orden

# %%
data.TipoOrden.value_counts()

# %%
data.TipoOrden.value_counts().plot.barh();

# %% [markdown]
# > Sucursal con mayor numero de ordenes por meses?

# %%
cantidad_tipo_orden_por_mes = data.groupby(['Mes', 'Branch'])['TipoOrden'].count()

# %%
cantidad_tipo_orden_por_mes.loc[cantidad_tipo_orden_por_mes.groupby('Mes').idxmax()]

# %%
cantidad_tipo_orden_por_mes.loc[cantidad_tipo_orden_por_mes.groupby('Mes').idxmax()].plot.barh()

# %% [markdown]
# ### Elabora un gráfico que permita analizar las ventas acumuladas por Estado y Sucursal

# %% [markdown]
# - Necesito cargar Directorio.csv

# %%
directorio = pd.read_csv('Directorio.csv',  encoding = 'ISO-8859-1')

# %%
directorio.head()

# %%
directorio.drop(columns='Unnamed: 2', inplace=True)

# %%
directorio.head()

# %%
directorio.Nombre = directorio.Nombre.str.replace(" ", "_")

# %%
data.head(3)

# %% [markdown]
# #### Merge los datasets

# %%
data_merged = pd.merge(data, directorio[['Branch', 'Estado']], on='Branch', how='left')

# %%
ventas_acumuladas_estado = data_merged.groupby('Estado')['VentaTotal$'].sum().sort_values(ascending=False)

# %%
sns.barplot(ventas_acumuladas_estado, palette="Spectral")
plt.title('Ventas Acumuladas por Estado')
plt.xlabel('Cantidad en millones de pesos')
plt.show()

# %%
# Agrupar por Estado y Sucursal, y sumar las ventas
ventas_acumuladas = data_merged.groupby(['Estado', 'Branch'])['VentaTotal$'].sum().reset_index()

# Preparar los datos para el gráfico
pivot_ventas = ventas_acumuladas.pivot(index='Branch', columns='Estado', values='VentaTotal$').fillna(0)

# Crear un gráfico de barras apiladas para las ventas acumuladas por Estado y Sucursal
pivot_ventas.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Ventas Acumuladas por Estado y Sucursal')
plt.xlabel('Sucursal')
plt.ylabel('Ventas Acumuladas ($)')
plt.legend(title='Estado')
plt.grid(True)

# Mostrar el gráfico
plt.show()

# %% [markdown]
# ### Elabora un gráfico que consideres sea de valor para la toma de decisiones y explicarlo brevemente.

# %%
data_merged.head(3)

# %% [markdown]
# ### Comparamos el margen de utilidad bruta con el volumen de ventas

# %%
analisis_sucursal = data_merged.groupby('Branch').agg({'Margen': 'mean', 'VentaTotal$': 'sum'}).reset_index()

# %%
plt.figure(figsize=(12, 8))
sns.scatterplot(x='VentaTotal$', y='Margen', data=analisis_sucursal, hue='Branch', legend='full');
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()

# %% [markdown]
# #### Este gráfico es valioso para la toma de decisiones ya que permite identificar:
# 
# - Sucursales altamente rentables: Puntos en la parte superior derecha, indicando altas ventas y alto margen de utilidad.
# - Sucursales con potencial de mejora: Puntos en la parte inferior derecha, con altas ventas pero margen de utilidad más bajo.
# - Sucursales con bajo rendimiento: Puntos en la parte inferior izquierda, indicando bajas ventas y bajo margen de utilidad.

# %%
analisis_sucursal.sort_values(['Margen', 'VentaTotal$'],ascending=False).reset_index(drop=True)

# %%



