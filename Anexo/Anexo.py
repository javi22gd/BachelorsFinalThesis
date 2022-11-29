#!/usr/bin/env python
# coding: utf-8

# # Trabajo de Fin de Grado
# ## Machine Learning para el tratamiento de datos y la detección de exoplanetas mediante el método de tránsito
# 
# ##### Resumen
# Uno de los siguientes pasos en la exploración espacial es encontrar planetas más allá del sistema solar que, potencialmente, puedan albergar signos de vida extraterrestre. Estos planetas que orbitan otras estrellas son conocidos como exoplanetas. Las complejas técnicas utilizadas para su detección recaban una inmensa cantidad de datos que deben ser cuidadosamente tratados y adecuados para su posterior análisis en busca de estos mundos.
# 	Uno de estos métodos, el de tránsito, consiste en observar las estrellas en busca de disminuciones de luz provocadas por posibles exoplanetas transitando entre la estrella y el observador. Esta información queda plasmada en los datos recogidos por los telescopios, que deben ser procesados, tratados y analizados. Estas tareas se pueden llevar a cabo de forma masiva y automatizada mediante distintas técnicas de *machine learning*.
# 	En este proyecto se presentarán las principales técnicas y modelos de machine learning para el tratamiento y clasificación de datos, se emplearán algunas de ellas para adecuar conjuntos de datos de observaciones realizadas por la misión *Kepler* de la NASA y, finalmente, se construirá un modelo de predicción y se analizará su precisión a la hora de detectar exoplanetas.

# In[82]:


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from scipy import ndimage
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense


# ## Datos de entrenamiento

# In[2]:


kepler_train_data_file_path = "../TFG/datos/exoTrain.csv"
kepler_train_data = pd.read_csv(kepler_train_data_file_path)

pd.DataFrame(kepler_train_data)


# ## Datos de test

# In[3]:


kepler_test_data_file_path = "../TFG/datos/exoTest.csv"
kepler_test_data = pd.read_csv(kepler_test_data_file_path)

pd.DataFrame(kepler_test_data)


# ## Representación Gráfica de los datos sin tratar
# #### Diferencias entre estrellas sin exoplanetas y estrellas con exoplanetas
# Estrellas sin exoplanetas ('LABEL' = 1)

# In[4]:


for i in [0, 99, 499, 999, 1999, 2999, 3999, 4999]:
    #Se extraen los datos de las distintas mediciones de luz, eliminando la columna "LABEL"
    flujo = kepler_train_data[kepler_train_data.LABEL == 1].drop('LABEL', axis = 1).iloc[i,:]
    tiempo = np.arange(len(flujo)) * (36/60) #Variable "tiempo" en horas
    plt.figure(figsize=(15, 5)) #Tamaño del gráfico
    plt.title('Flujo del brillo de la estrella nº {} SIN exoplanetas'.format(i+1))
    plt.xlabel('Tiempo (horas)')
    plt.ylabel('Flujo en fotones por segundo (e-s-1)')
    plt.plot(tiempo, flujo)


# Estrellas con exoplanetas ('LABEL' = 2)

# In[5]:


for i in [0, 1, 2, 3, 4, 9, 19, 29]:
    #Se extraen los datos de las distintas mediciones de luz, eliminando la columna "LABEL"
    flujo = kepler_train_data[kepler_train_data.LABEL == 2].drop('LABEL', axis = 1).iloc[i,:]
    tiempo = np.arange(len(flujo)) * (36/60) #Variable "tiempo" en horas
    plt.figure(figsize=(15, 5)) #Tamaño del gráfico
    plt.title('Flujo del brillo de la estrella nº {} CON exoplanetas'.format(i+1))
    plt.xlabel('Tiempo (horas)')
    plt.ylabel('Flujo en fotones por segundo (e-s-1)')
    plt.plot(tiempo, flujo)


# # Tratamiento de los datos
# Como se puede observar, las gráficas de los flujos de luz de las estrellas se presentan de forma muy heterogénea en cuanto a forma y magnitud de los datos, por lo que es necesario adaptarlos a un estándar para que el entrenamiento del modelo no se vea influenciado por estas diferencias.
# 
# Se va a utilizar la estrella nº 10 CON exoplanetas a modo de ejemplo del tratamiento de datos y, posteriormente, se aplicará a todo el conjunto de datos.

# ## Abstracción del flujo
# Se van a aplicar diferentes métodos para que los datos de cada estrella reflejen únicamente las variaciones de luz absolutas con el objetivo de facilitar la identificación de patrones mediante las variaciones de flujo que sí son indicativas de la presencia de exoplanetas.
# 
# Esto es necesario ya que cada estrella tiene una intensidad diferente debido a la multitud de tamaños, formas de rotación, heterogeneidad de superficies, etc. que presentan. Es por esto que cada estrella tiene una forma diferente de brillar que puede variar de múltiples formas incluso en una misma estrella. El objetivo es eliminar estas variaciones para desvincular las variaciones de flujo provocadas por posibles exoplanetas de las variaciones derivadas de las fluctuaciones de la propia estrella.
# 
# Para ello, es necesario encontrar un flujo general de los datos que se pueda sustraer del flujo original para obtener su representación plana relativa.

# Primero, se va a utilizar la técnica de desenfoque gaussiano para "suavizar" los datos.

# ###### Esta es la estrella nº 10 CON exoplanetas sin ningún tipo de tratamiento.

# In[6]:


i = 9
#Se extraen los datos de las distintas mediciones de luz, eliminando la columna "LABEL"
flujo1 = kepler_train_data[kepler_train_data.LABEL == 2].drop('LABEL', axis = 1).iloc[i,:]
tiempo = np.arange(len(flujo1)) * (36/60) #Variable "tiempo" en horas
plt.figure(figsize=(15, 5)) #Tamaño del gráfico
#Etiquetas
plt.title('Flujo del brillo de la estrella nº {} CON exoplanetas'.format(i+1))
plt.xlabel('Tiempo (horas)')
plt.ylabel('Flujo en fotones por segundo (e-s-1)')
plt.plot(tiempo, flujo1)


# #### 1. Desenfoque gaussiano.
# Se utiliza este método para abstraer el flujo de fluctuaciones propias de la estrella

# In[7]:


flujo2 = ndimage.filters.gaussian_filter(flujo1, sigma = 10) #Se crea la variable 'flujo2' con el resultado de aplicar el desenfoque gaussiano con sigma = 10
tiempo = np.arange(len(flujo2)) * (36/60) #Variable "tiempo" en horas
plt.figure(figsize=(15, 5)) #Tamaño del gráfico
#Etiquetas
plt.title('Flujo "suavizado" de la estrella nº {} CON exoplanetas'.format(i+1))
plt.xlabel('Tiempo (horas)')
plt.ylabel('Flujo en fotones por segundo (e-s-1)')
plt.plot(tiempo, flujo2)


# #### 2. Desvinculación de la tendencia de fluctuaciones de la estrella.
# Para abstraer esta información, se restan el flujo de datos original y el flujo de datos resultante del desenfoque gaussiano. Este nuevo flujo reflejará las variaciones de luz independientes de la tendencia de flujo de la estrella para mostrar aquellas variaciones que son relevantes para la detección de exoplanetas

# In[8]:


flujo3 = flujo1 - flujo2
tiempo = np.arange(len(flujo3)) * (36/60) #Variable "tiempo" en horas
plt.figure(figsize=(15, 5)) #Tamaño del gráfico
#Etiquetas
plt.title('Flujo independiente de la estrella nº {} CON exoplanetas'.format(i+1))
plt.xlabel('Tiempo (horas)')
plt.ylabel('Flujo en fotones por segundo (e-s-1)')
plt.plot(tiempo, flujo3)


# #### 3. Normalización
# 
# El siguiente paso es normalizar el flujo de datos

# In[9]:


flujo3n = (flujo3 - np.mean(flujo3)) / (np.max(flujo3) - np.min(flujo3))
tiempo = np.arange(len(flujo3n)) * (36/60) #Variable "tiempo" en horas
plt.figure(figsize=(15, 5)) #Tamaño del gráfico
#Etiquetas
plt.title('Flujo independiente normalizado de la estrella nº {} CON exoplanetas'.format(i+1))
plt.xlabel('Tiempo (horas)')
plt.ylabel('Flujo normalizado')
plt.plot(tiempo, flujo3n)


# #### 4. Eliminar datos atípicos superiores
# Los exoplanetas provocan una disminución de la luz percibida de la estrella al transitar delante de ella, por lo que es necesario eliminar todos los datos atípicos superiores, ya que son los inferiores los que son relevantes.

# In[10]:


def eliminar_datos_atipicos_superiores(X, reducir = 0.01, amplitud=4):
#https://www.kaggle.com/muonneutrino/exoplanet-data-visualization-and-exploration
    longitud = len(X.iloc[0,:])
    eliminar = int(longitud*reducir)
    for i in X.index.values:
        valores = X.loc[i,:]
        valores_ordenados = valores.sort_values(ascending = False)
        for j in range(eliminar):
            idx = valores_ordenados.index[j]
            #print(idx)
            nuevo_valor = 0
            c = 0
            idx_num = int(idx[5:])
            for k in range(2*amplitud+1):
                idx2 = idx_num + k - amplitud
                if idx2 <1 or idx2 >= longitud or idx_num == idx2:
                    continue
                nuevo_valor += valores['FLUX.'+str(idx2)]
                
                c += 1
            nuevo_valor /= c
            if nuevo_valor < valores[idx]:
                X.at[i,idx] = nuevo_valor
        
            
    return X


# ## Aplicar este proceso a todo el *dataset* de entrenamiento y test

# #### Crear una función para aplicar el desenfoque Gaussiano, la abstracción de flujo y la normalización de los datos

# In[11]:


def abstraer_normalizar_flujo(X):
    flujo1 = X
    flujo2 = ndimage.filters.gaussian_filter(flujo1, sigma = 10) #Desenfoque Gaussiano
    flujo3 = flujo1 - flujo2 #Abstraccion de flujo
    flujo3n = (flujo3 - np.mean(flujo3)) / (np.max(flujo3) - np.min(flujo3)) #Normalizacion
    return flujo3n


# #### Aplicar los pasos anteriores a todos los datos tanto de entrenamiento como de test

# In[12]:


kepler_train_data.iloc[:,1:] = kepler_train_data.iloc[:,1:].apply(abstraer_normalizar_flujo,axis=1) #Aplicar pasos 1, 2 y 3 en datos de entrenamiento
kepler_test_data.iloc[:,1:] = kepler_test_data.iloc[:,1:].apply(abstraer_normalizar_flujo,axis=1) #Aplicar pasos 1, 2 y 3 en datos de test
kepler_train_data.iloc[:,1:] = eliminar_datos_atipicos_superiores(kepler_train_data.iloc[:,1:])#Aplicar paso 4 en datos de entrenamiento
kepler_test_data.iloc[:,1:] = eliminar_datos_atipicos_superiores(kepler_test_data.iloc[:,1:])#Aplicar paso 4 en datos de test


# ###### Analizar el resultado de este tratamiento con la estrella nº 10 con exoplanetas

# In[13]:


i = 9
#Se extraen los datos de las distintas mediciones de luz, eliminando la columna "LABEL"
flujo = kepler_train_data[kepler_train_data.LABEL == 2].drop('LABEL', axis = 1).iloc[i,:]
tiempo = np.arange(len(flujo)) * (36/60) #Variable "tiempo" en horas
plt.figure(figsize=(15, 5)) #Tamaño del gráfico
#Etiquetas
plt.title('Flujo independiente, normalizado y sin datos atipicos superiores de la estrella nº {} CON exoplanetas'.format(i+1))
plt.xlabel('Tiempo (horas)')
plt.ylabel('Flujo normalizado')
plt.plot(tiempo, flujo)


# ### Visualización de los datos
# Así quedan los datos después del primer tratamiento
# ###### Datos de entrenamiento

# In[14]:


pd.DataFrame(kepler_train_data)


# ###### Datos de test

# In[15]:


pd.DataFrame(kepler_test_data)


# # *PCA*
# Con los datos ya tratados, el siguiente paso es aplicar *PCA* para la reducción de dimensiones.

# #### Instanciar y aplicar *PCA* en el *dataset* de entrenamiento

# In[16]:


flujos = kepler_train_data.drop(['LABEL'], axis = 1) #Eliminar la columna etiqueta
pca = PCA(0.9) #Crear una instancia de PCA con las dimensiones necesarias para obtener 0.9 de varianza
pca.fit(flujos) #Ajustar el modelo con los datos (generar matriz de covarianza, autovectores, autovalores, etc.)
flujos_pca = pca.transform(flujos) #Aplicar la reducción de dimensiones

print("Filas y dimensiones finales:", flujos_pca.shape)
expl = pca.explained_variance_ratio_
print('Suma de la varianza: ', sum(expl))

#Visualizar el grafico de la varianza acumulada
plt.title('Varianza acumulada respecto a las dimensiones (PCA)')
plt.xlabel('Nº de dimensiones')
plt.ylabel('Varianza acumulada')
plt.plot(np.cumsum(pca.explained_variance_ratio_))


# El resultado de *PCA* muestra que, para obtener una varianza acumulada de 0,9, son necesarias 603 dimensiones. El valor de estas dimensiones para cada estrella se cargan en "*flujos_pca*" ordenadas de mayor a menos varianza -es decir, de mayor a menor relevancia para el modelo-.

# #### Crear el nuevo *dataset* de entrenamiento
# Nuevo *dataset* que contiene únicamente las 603 dimensiones resultantes de aplicar *PCA*.

# In[17]:


kepler_train_data_pca = pd.DataFrame(data = flujos_pca) #Crear dataset con los datos de 'flujos_pca' con los valores de las 603 dimensiones
kepler_train_data_pca


# #### Adaptar el nuevo *dataset* de entrenaminto
# Dar el mismo formato a este nuevo *dataset* para que se asemeje al original. Esto consiste en renombrar las columnas a '*FLUX.n*' y añadir la columna '*LABEL*' con los mismos valores del *dataset* original.

# In[18]:


label = [] #Crear la lista en la que se cargaran los valores de la columna 'LABEL'
for i in range(len(kepler_train_data_pca.columns)):
    kepler_train_data_pca=kepler_train_data_pca.rename({i: 'FLUX.{}'.format(i+1)}, axis = 1) #Renombrar la i-esima columna
    
for i in range(len(kepler_train_data)):
    label.append(int(kepler_train_data.iloc[i]['LABEL'])) #Añadir el valor i-esimo de la columna 'LABEL' a la lista
    
kepler_train_data_pca.insert(0, 'LABEL', label, True) #Insertar en el nuevo dataset de entrenamiento la columna 'LABEL' con sus valores correspondientes
kepler_train_data_pca


# Arriba se muestra el nuevo *dataset* de entrenamiento con las 603 dimensiones.

# ### Aplicar al *dataset* de test
# Se aplica la misma función de *PCA* creada con los datos de entrenamiento para seleccionar las mismas dimensiones de los datos de test. Es importante seleccionar las mismas para que el modelo final evalúe los datos de test con las mismas dimensiones que tienen los datos con los que ha sido entrenado.

# In[19]:


flujos_test = kepler_test_data.drop(['LABEL'], axis = 1) #Eliminar la columna etiqueta
flujos_test_pca = pca.transform(flujos_test) #Aplicar la reducción de dimensiones con la misma instancia de PCA creada con los datos de entrenamiento

kepler_test_data_pca = pd.DataFrame(data = flujos_test_pca) #Crear dataset con los datos de 'flujos_test_pca' con los valores de las 603 dimensiones

label = [] #Crear la lista en la que se cargaran los valores de la columna 'LABEL'
for i in range(len(kepler_test_data_pca.columns)):
    kepler_test_data_pca=kepler_test_data_pca.rename({i: 'FLUX.{}'.format(i+1)}, axis = 1) #Renombrar la i-esima columna
    
for i in range(len(kepler_test_data)):
    label.append(int(kepler_test_data.iloc[i]['LABEL'])) #Añadir el valor i-esimo de la columna 'LABEL' a la lista
    
kepler_test_data_pca.insert(0, 'LABEL', label, True) #Insertar en el nuevo dataset de test la columna 'LABEL' con sus valores correspondientes
kepler_test_data_pca


# Así queda el flujo de la estrella nº 10 CON exoplanetas -del *dataset* de entrenamiento- con las 603 dimensiones seleccionadas por *PCA*.

# In[20]:


i = 9
#Se extraen los datos de las distintas mediciones de luz, eliminando la columna "LABEL"
flujo = kepler_train_data_pca[kepler_train_data_pca.LABEL == 2].drop('LABEL', axis = 1).iloc[i,:]
tiempo = np.arange(len(flujo)) * (36/60) #Variable "tiempo" en horas
plt.figure(figsize=(15, 5)) #Tamaño del gráfico
plt.title('Flujo del brillo de la estrella nº {} CON exoplanetas'.format(i+1))
plt.xlabel('Tiempo (horas)')
plt.ylabel('Flujo en fotones por segundo (e-s-1)')
plt.plot(tiempo, flujo)


# # *Data augmentation*
# Para balancear el *dataset* de entrenamiento mediante este método, es necesario conocer las características de los datos que van a ser replicados y alterados.
# Estos datos son los correspondientes a las estrellas con exoplanetas confirmados -*LABEL* = 2-, que son 37 de las 5.087 totales -alrededor del 0,7%-. Por ello, es necesario obtener unos 15 nuevos registros por cada uno de los 37 existentes para lograr un mayor balanceamiento, cercano al 10%.
# 
# Estos son los flujos de las 37 estrellas con exoplanetas que deben ser replicados:

# In[21]:


for i in range(len(kepler_train_data_pca[kepler_train_data_pca.LABEL == 2])):
    #Se extraen los datos de las distintas mediciones de luz, eliminando la columna "LABEL"
    flujo = kepler_train_data_pca[kepler_train_data_pca.LABEL == 2].drop('LABEL', axis = 1).iloc[i,:]
    tiempo = np.arange(len(flujo)) * (36/60) #Variable "tiempo" en horas
    plt.figure(figsize=(15, 5)) #Tamaño del gráfico
    plt.title('Flujo tras aplicar PCA de la estrella nº {} CON exoplanetas'.format(i+1))
    plt.xlabel('Tiempo (horas)')
    plt.ylabel('Flujo normalizado')
    plt.plot(tiempo, flujo)
    plt.rcParams.update({'figure.max_open_warning': 0})


# El objetivo es analizar cómo replicar estos datos y alterarlos de tal forma que no se pierda la información esencial que hace que sean detectados como exoplanetas.
# 
# Tras aplicar *PCA*, las dimensiones se encuentran ordenadas según su relevancia para el modelo, por lo que alterar su orden implicaría esa pérdida de información, además de que las curvas de luz se preducen con ciertos patrones que, de verse alterados, modificarían por completo la naturaleza de los datos. Por lo tanto, alterar el orden de las columnas no se presenta como una opción.
# 
# Las dos mejores opciones, dados los datos, son:
# - Incremento y decremento de una magnitud de los datos
# - Aplicación de ruido a todos los datos

# ## Incremento y decremento de magnitudes
# 
# Las magnitudes que se van a aplicar a los datos son: 0.02, 0.04, 0.06, 0.08 y 0.1.
# 
# De esta forma, los datos de estrellas con exoplanetas resultantes serán 407 en total

# ##### Estos son los datos que se van a replicar con los incrementos y decrementos

# In[22]:


flujos_exo = kepler_train_data_pca[kepler_train_data_pca.LABEL == 2].drop(['LABEL'], axis = 1) #Eliminar la columna etiqueta y cargar los flujos de las estrellas con exoplanetas
flujos_exo


# ##### Datos finales con los incrementos y decrementos

# In[23]:


flujos_exo_incrdecr = pd.DataFrame(flujos_exo) #Crear nuevo DataFrame para todos los datos resultantes de este metodo
#Incrementos
flujos_exo_incrdecr = flujos_exo_incrdecr.append((flujos_exo + 0.02), ignore_index = True)
flujos_exo_incrdecr = flujos_exo_incrdecr.append((flujos_exo + 0.04), ignore_index = True)
flujos_exo_incrdecr = flujos_exo_incrdecr.append((flujos_exo + 0.06), ignore_index = True)
flujos_exo_incrdecr = flujos_exo_incrdecr.append((flujos_exo + 0.08), ignore_index = True)
flujos_exo_incrdecr = flujos_exo_incrdecr.append((flujos_exo + 0.1), ignore_index = True)
#Decremento
flujos_exo_incrdecr = flujos_exo_incrdecr.append((flujos_exo - 0.02), ignore_index = True)
flujos_exo_incrdecr = flujos_exo_incrdecr.append((flujos_exo - 0.04), ignore_index = True)
flujos_exo_incrdecr = flujos_exo_incrdecr.append((flujos_exo - 0.06), ignore_index = True)
flujos_exo_incrdecr = flujos_exo_incrdecr.append((flujos_exo - 0.08), ignore_index = True)
flujos_exo_incrdecr = flujos_exo_incrdecr.append((flujos_exo - 0.1), ignore_index = True)
flujos_exo_incrdecr


# ## Aplicación de ruido
# 
# A las 407 filas obtenidas se les van a aplicar 6 instancias de ruido con las varianzas siguientes, repectivamente: 0.000015, 0.00003, 0.000045, 0.00006, 0.00075 y 0.00009.
# 
# Este proceso dejará un total de 2849 filas equivalentes a estrellas con exoplanetas, todas ellas diferentes entre sí.

# In[24]:


#Generar las 6 instancias de ruido con las varianzas correspondientes
ruido1 = np.random.normal(0, 0.000015, [407, 603])
ruido2 = np.random.normal(0, 0.00003, [407, 603])
ruido3 = np.random.normal(0, 0.000045, [407, 603])
ruido4 = np.random.normal(0, 0.00006, [407, 603])
ruido5 = np.random.normal(0, 0.000075, [407, 603])
ruido6 = np.random.normal(0, 0.00009, [407, 603])

#Añadir las instancias de ruido a los datos
flujos_exo_ruido1 = flujos_exo_incrdecr + ruido1
flujos_exo_ruido2 = flujos_exo_incrdecr + ruido2
flujos_exo_ruido3 = flujos_exo_incrdecr + ruido3
flujos_exo_ruido4 = flujos_exo_incrdecr + ruido4
flujos_exo_ruido5 = flujos_exo_incrdecr + ruido5
flujos_exo_ruido6 = flujos_exo_incrdecr + ruido6

#Crear nuevo DataFrame para todos los datos resultantes de este metodo
flujos_exo_da = pd.DataFrame(flujos_exo_incrdecr)

#Añadir todas las instancias finales en un unico DataFrame
flujos_exo_da = flujos_exo_da.append(flujos_exo_ruido1, ignore_index = True)
flujos_exo_da = flujos_exo_da.append(flujos_exo_ruido2, ignore_index = True)
flujos_exo_da = flujos_exo_da.append(flujos_exo_ruido3, ignore_index = True)
flujos_exo_da = flujos_exo_da.append(flujos_exo_ruido4, ignore_index = True)
flujos_exo_da = flujos_exo_da.append(flujos_exo_ruido5, ignore_index = True)
flujos_exo_da = flujos_exo_da.append(flujos_exo_ruido6, ignore_index = True)
flujos_exo_da


# Arriba se muestran todos los datos aumentados, junto a los originales, de las 37 estrellas con exoplanetas iniciales.
# 
# Abajo se muestran los flujos de la estrella 10 con exoplanetas y algunas de sus modificaciones -ruido y/o incremento/decremento-. Como se puede apreciar, a simple vista no son muy diferentes entre sí, lo que es un buen indicativo de que la relatividad de los datos sigue presente pese a las modificaciones. Pero, por otro lado, los datos son lo suficientemente diferentes para entrenar el modelo evitando sobreajustes.

# In[25]:


i = 9
while i < (len(flujos_exo_da)-1):
#Se extraen los datos de las distintas mediciones de luz, eliminando la columna "LABEL"
    flujo = flujos_exo_da.iloc[i,:]
    tiempo = np.arange(len(flujo)) * (36/60) #Variable "tiempo" en horas
    plt.figure(figsize=(15, 5)) #Tamaño del gráfico
    plt.title('Flujo de la estrella nº 9 CON exoplanetas')
    plt.xlabel('Tiempo (horas)')
    plt.ylabel('Flujo normalizado')
    plt.plot(tiempo, flujo)
    plt.rcParams.update({'figure.max_open_warning': 0})
    i += 74
    
plt.show()


# # Generar dataset de entrenamiento
# Tras realizar el primer tratamiento de datos, *PCA* y las dos técnicas de *data augmentation*, el siguiente paso es generar el *dataset* de entrenamiento con todos los datos

# In[26]:


label = [2] * len(flujos_exo_da)
flujos_exo_da.insert(0, 'LABEL', label, True) #Insertar la columna 'LABEL' con todos los valores igual a '2'
flujos_exo_da


# In[27]:


kepler_train_data_final = pd.DataFrame(flujos_exo_da) #Instanciar nuevo DataFrame con los datos de las estrellas con exoplanetas
kepler_train_data_final = kepler_train_data_final.append(kepler_train_data_pca[kepler_train_data_pca.LABEL == 1].iloc[:,:], ignore_index = True) #Añadir los datos de las estrellas sin exoplanetas
kepler_train_data_final


# Arriba se muestra el *dataset* final con el que se entrenará el modelo. Los datos han sido tratados; sus dimensiones, reducidas; y su desbalanceamiento, corregido. Contiene un total de 7899 estrellas, de las cuales 2849 tienen exoplanetas -'*LABEL*' = '2'- y 5050 no tienen exoplanetas -'*LABEL*' = '1'-

# # Modelo de predicción
# 
# El modelo de predicción consistirá en una red neuronal artificial entrenada con los datos de entrenamiento finales -tras el primer tratamiento, reducción de dimensiones y *data augmentation*- y probada con los datos de test, que se están conformados por 565 no exoplanetas y 5 exoplanetas.

# In[71]:


#Cargar en variables distintas los datos de entrenamiento y test, separando las dimensiones de la clase
x_entreno = kepler_train_data_final.drop('LABEL', axis = 1)
y_entreno = kepler_train_data_final['LABEL']
x_test = kepler_test_data_pca.drop('LABEL', axis = 1)
y_test = kepler_test_data_pca['LABEL']


# ##### Construcción y entrenamiento de la red

# In[80]:


#Instanciar modelo secuencial: el output de cada capa creada es el input de la siguiente
modelo = Sequential()

#Capa neuronal oculta tipo dense -con todas las conexiones entre neuronas- con las dimensiones de entrada del dataset de entrenamiento y 100 dimensiones de salida
modelo.add(Dense(100, input_dim=x_entreno.shape[1]))

#Capa neuronal de salida con 100 dimensiones de entrada y 1 de salida -clasificacion binaria-, con la funcion de activacion 'sigmoid'
modelo.add(Dense(1, activation='sigmoid'))

#Compilar el modelo con la loss function "binary_crossentropy" ya que la clasificacion es binaria
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Entrenar el modelo que estos parametros y los datos de validacion de test
modelo.fit(x_entreno, y_entreno, batch_size=20, epochs=10, validation_data=(x_test, y_test))


# ##### Predicción y precisión

# In[81]:


prediccion = modelo.predict_classes(x_test) #Crear prediccion
prediccion = np.where(prediccion == 0, 2, prediccion) #Ajustar valores para que concuerden con los del dataset

#Crear la matriz de confusion
cm = confusion_matrix(y_test, prediccion)
cm = {'Real: SIN exoplanetas': cm[0], 'Real: CON exoplanetas': cm[1]}
cm = pd.DataFrame.from_dict(cm, orient = 'index', columns = ['Predicción: SIN exoplanetas', 'Predicción: CON exoplanetas'])

#Medir la precision del modelo
print("La precisión total del modelo es del: {}%".format(round(accuracy_score(y_test,prediccion), 4)*100))
cm


# El modelo ha logrado predecir 2 de los 5 exoplanetas, y 562 de los 565 no exoplanetas. La precisión total de las predicciones del modelo es del 98.95%
