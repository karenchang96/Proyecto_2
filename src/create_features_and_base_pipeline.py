
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def create_model_features():
    # Leer el dataset
    dataset = pd.read_csv("../data/raw/data.csv")
    
    # Validamos si hay valores nulos
    print(dataset.isnull().mean())
    
    # Visualización de la distribución de la variable 'mora'
    sns.countplot(x='mora', data=dataset)
    plt.show()
    
    # Selección de variables numéricas
    num_features = [
        'edad', 'dias_lab', 'exp_sf', 'nivel_ahorro', 'ingreso', 
        'linea_sf', 'deuda_sf', 'score', 'clasif_sbs'
    ]
    
    # Tratamiento de variables categóricas
    cat_features = ['vivienda', 'zona', 'nivel_educ']
    
    # Tratamiento de variables numéricas
    num_imputer = SimpleImputer(strategy='mean')
    dataset[num_features] = num_imputer.fit_transform(dataset[num_features])
    
    # Codificación de variables categóricas
    label_encoder = LabelEncoder()
    for col in cat_features:
        dataset[col] = label_encoder.fit_transform(dataset[col])
    
    # Imprimir las primeras filas del dataset
    print(dataset.head())
    
    # Guardar el dataset preparado
    dataset.to_csv(
        r'C:\Users\karen\Documents\Master\Business Intelligence y analisis de datos\8to Trimestre\Product Development\Proyecto2\dataset_preparado.csv',
        index=False
    )
