#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 08:40:04 2022

@author: dayan
"""

# Importamos las librerias que nos permitiran hacerle el tratamiento a los datos
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

# importamos las librerias que nos permitiran Gráficar 
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')

# importamos las librerias necesarias para el Preprocesado y modelado de los datos 
# ==============================================================================
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score

# libreria para la Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')



## metodo que calcula los valores del dendograma en focado al modelo de children 
def plot_dendrogram(model, **kwargs):
    '''
    Esta función extrae la información de un modelo AgglomerativeClustering
    y representa su dendograma con la función dendogram de scipy.cluster.hierarchy
    '''
    
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    #  se grafica los resultado obtenido en un diagrama Plot
    dendrogram(linkage_matrix, **kwargs)

##DATOS =====================================================
    
    
 # generamos los datos que seran utilizados para la Simulación de dato
# ==============================================================================
#creamos la variable x , y almacenara los datos de simulacion 

X, y = make_blobs(
        #se generaran una cantidad de 200 datos 
        n_samples    = 200, 
        n_features   = 2, 
        centers      = 4, 
        cluster_std  = 0.60, 
        shuffle      = True, 
        random_state = 0
       )

fig, ax = plt.subplots(1, 1, figsize=(6, 3.84))
for i in np.unique(y):
    ax.scatter(
        x = X[y == i, 0],
        y = X[y == i, 1], 
        c = plt.rcParams['axes.prop_cycle'].by_key()['color'][i],
        marker    = 'o',
        edgecolor = 'black', 
        label= f"Grupo {i}"
    )
ax.set_title('Datos simulados')
ax.legend();

# Escalado de datos
# ==============================================================================
# estandarisamos los datos para un optimo funcionamiento del los modelos 
X_scaled = scale(X)


# Modelos
# creamos cada uno de los modelos 
# ==============================================================================

#modelo_hclust_complete es  la variable que guardara los parametros del modelo complete 
modelo_hclust_complete = AgglomerativeClustering(
                            affinity = 'euclidean',
                            linkage  = 'complete',
                            distance_threshold = 0,
                            n_clusters         = None
                        )
# ingresamos los datos estandarisados 
modelo_hclust_complete.fit(X=X_scaled)

#modelo_hclust_average es  la variable que guardara los parametros del modelo  average 

modelo_hclust_average = AgglomerativeClustering(
                            affinity = 'euclidean',
                            linkage  = 'average',
                            distance_threshold = 0,
                            n_clusters         = None
                      )
#ingresamos los datos estandarizados 
modelo_hclust_average.fit(X=X_scaled)


#modelo_hclust_ward es  la variable que guardara los parametros del modelo ward 

modelo_hclust_ward = AgglomerativeClustering(
                            affinity = 'euclidean',
                            linkage  = 'ward',
                            distance_threshold = 0,
                            n_clusters         = None
                     )
modelo_hclust_ward.fit(X=X_scaled)

AgglomerativeClustering(distance_threshold=0, n_clusters=None)

# Dendrogramas
#ahora graficamos el dendograma de cada uno de los modelos 
# ==============================================================================
fig, axs = plt.subplots(3, 1, figsize=(8, 8))
# graficamos el modelo avenge
plot_dendrogram(modelo_hclust_average, color_threshold=0, ax=axs[0])
#le asignamos un titulo al grafico
axs[0].set_title("Distancia euclídea, Linkage average")

#graficamos el modelo complete y le asignamos un nombre al grafico 
plot_dendrogram(modelo_hclust_complete, color_threshold=0, ax=axs[1])
axs[1].set_title("Distancia euclídea, Linkage complete")

#graficamos el diagrama ward y le asignamos el nombre al grafico 
plot_dendrogram(modelo_hclust_ward, color_threshold=0, ax=axs[2])
axs[2].set_title("Distancia euclídea, Linkage ward")
plt.tight_layout();




## numero de cluster optimo para el dendograma 


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
#altura de la linea que se trasara en el dendograa 
altura_corte = 6
#graficamos con el modelo de ward y con la altura de corte igual a 6
plot_dendrogram(modelo_hclust_ward, color_threshold=altura_corte, ax=ax)
ax.set_title("Distancia euclídea, Linkage ward")
ax.axhline(y=altura_corte, c = 'black', linestyle='--', label='altura corte')
ax.legend();



## otra fonma de encontrar el numero optimo de cluster 

# Método silhouette para identificar el número óptimo de clusters
# =============================================================================
# lineas de codigo que nos ayuda a identificar el numero mas optimo de clusterin 
# y en un grafico se mostrar el resultado 
range_n_clusters = range(2, 15)
valores_medios_silhouette = []

for n_clusters in range_n_clusters:
    modelo = AgglomerativeClustering(
                    affinity   = 'euclidean',
                    linkage    = 'ward',
                    n_clusters = n_clusters
             )

    cluster_labels = modelo.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    valores_medios_silhouette.append(silhouette_avg)
    
fig, ax = plt.subplots(1, 1, figsize=(6, 3.84))
ax.plot(range_n_clusters, valores_medios_silhouette, marker='o')
ax.set_title("Evolución de media de los índices silhouette")
ax.set_xlabel('Número clusters')
ax.set_ylabel('Media índices silhouette');




# Modelo a evaluar 
# ==============================================================================
# segun los datos obtenido por la grafica anterios el numero de clusterin mas optim es 4

# creamos nuevamente el  modelo ward ahora con el numero de cluster mas optimo 
modelo_hclust_ward = AgglomerativeClustering(
                            affinity = 'euclidean',
                            linkage  = 'ward',
                            n_clusters = 4
                     )
# ingresamos los datos estandarizados 
modelo_hclust_ward.fit(X=X_scaled)





AgglomerativeClustering(n_clusters=4)


