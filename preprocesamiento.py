# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample

# %%
df = pd.read_csv('train.csv')
df 

# %%
# Eliminar las columnas 'Artist Name' y 'Track Name'
df = df.drop(['Track Name', 'Artist Name','mode'], axis=1)

# %%
#obtiene la frecuencia de cada genero y su media
frecuencia_genero = df['Class'].value_counts()

#imprime los resultados
print(frecuencia_genero)

# %%
# Calculando la media de 'Popularity' e 'instrumentalness' por cada clase
mean_popularity = df.groupby('Class')['Popularity'].mean()
mean_instrumentalness = df.groupby('Class')['instrumentalness'].mean()

#Calcular la moda de 'key' por cada clase
moda_key = df.groupby('Class')['key'].agg(lambda x:x.value_counts().index[0])

# %%
#Imprimiendo los resultados
print(mean_popularity)
print(mean_instrumentalness)

# %%
print(moda_key)

# %%
#Rellenar valores nulos de 'Popularity' y 'instrumentalness' con la media de la columna segun su clase
df['Popularity'] = df['Popularity'].fillna(df.groupby('Class')['Popularity'].transform('mean'))
df['instrumentalness'] = df['instrumentalness'].fillna(df.groupby('Class')['instrumentalness'].transform('mean'))

#Rellenar valores nulos de 'key' con la moda de la columna segun su clase
df['key'] = df['key'].fillna(df.groupby('Class')['key'].transform(lambda x:x.value_counts().index[0]))

# %%
#Revisar valores nulos
df_nulls = df.isnull().sum()
df_nulls

# %%
#media de las frecuencias
media = df['Class'].value_counts().mean()
media

# %%
# Separar el dataset por clases
clases = df['Class'].unique()
data_por_clase = {clase: df[df['Class'] == clase] for clase in clases}

# Aplicar submuestreo y sobremuestreo
data_balanced = pd.DataFrame(columns=df.columns)
for clase, datos_clase in data_por_clase.items():
    if len(datos_clase) > media:
        # Submuestreo
        datos_muestreados = resample(datos_clase, replace=False, n_samples=int(media), random_state=0)
    else:
        # Sobremuestreo
        datos_muestreados = resample(datos_clase, replace=True, n_samples=int(media), random_state=0)
    data_balanced = pd.concat([data_balanced, datos_muestreados])

# Verificar el nuevo balance de las clases
df = data_balanced
#resetea el index
df = df.reset_index(drop=True)
nuevo_balance_sklearn = df['Class'].value_counts()
nuevo_balance_sklearn, media

# %%
#guarda el nuevo df
df.to_csv('nuevos_generos_sin_transformar.csv', index=False)

# %%
# Estandarización de los datos
le = LabelEncoder()
valores_numericos = df.columns.drop(['key', 'time_signature', 'Class'])

scaler = StandardScaler()

df[valores_numericos] = scaler.fit_transform(df[valores_numericos])

df.head()

# %%
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)

encoded_columns = encoder.fit_transform(df[['key', 'time_signature']])

encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(['key', 'time_signature']))

df = df.drop(['key', 'time_signature'], axis=1)
df = pd.concat([df, encoded_df], axis=1)
encoded_df

# %%
#guardar csv en la carpeta data con el nombre generos.csv
df.to_csv('generos.csv', index=False)


