import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

encuestas = pd.read_excel('encuestas.xlsx')

columnas = encuestas.columns[1:].values.flatten().tolist()
print(columnas)
num_columnas = len(columnas)
vector = []


for index, fila in encuestas.iterrows():
    vec_columnas = []
    for col in columnas:
        vec_columnas.append(fila[col])
    vector.append(vec_columnas)

x = np.array(vector)



num_grupo = 1
oper = True


while oper:
    grupos = KMeans(n_clusters=num_grupo)
    grupos.fit(X=x)
    
    labels = grupos.labels_
    centroid = grupos.cluster_centers_
    desv = []
    for i in range(len(centroid)):
        puntos = x[labels==i]
        des = np.std(puntos, axis=0)
        desv.append(des)
    
    oper = False
    
    for i in desv:
        for j in range(num_columnas):
            if i[j] >= 1.2:
                oper = True
    
    num_grupo += 1
        

dic = {
    "Usuario":[],
    "Grupo Correspondiente":[]
}

for i in range(len(labels)):
    grupo = labels[i]
    dic["Usuario"].append(encuestas["id_usuario"][i])
    dic["Grupo Correspondiente"].append(labels[i] + 1)
    

dic_df = pd.DataFrame(dic)

dic_df.to_csv("GruposFormados.csv", index=False)

# Agrupa los usuarios por su grupo correspondiente
grupos = dic_df.groupby('Grupo Correspondiente')['Usuario'].apply(list).reset_index()

# Guarda los grupos en un nuevo archivo CSV
grupos.to_csv('Agrupaciones.csv', index=False)
