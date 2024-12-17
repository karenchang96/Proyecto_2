import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import OneHotEncoder, CountFrequencyEncoder


def create_model_features():
    # Cargar dataset
    dataset = pd.read_csv("../data/raw/data.csv")
    # Definición de variables
    VARS_TO_DROP = ['mora']
    CONTINUE_VARS_TO_IMPUT = [
        'edad', 'dias_lab', 'exp_sf', 'nivel_ahorro',
        'ingreso', 'linea_sf', 'deuda_sf', 'score', 'clasif_sbs'
    ]
    CATEGORICAL_VARS_TO_IMPUT = ['vivienda', 'zona', 'nivel_educ']
    OHE_VAR_TO_ENCODE = ['vivienda', 'zona', 'nivel_educ']
    # Definir X y y correctamente
    x_features = dataset.drop(labels=VARS_TO_DROP, axis=1)  # Elimina columnas no deseadas
    y_target = dataset['mora']  # Selecciona la variable objetivo como una serie
    # División de los datos
    X_train, X_test, y_train, y_test = train_test_split(
        x_features, y_target, test_size=0.3, shuffle=True, random_state=2025
    )
    # Pipeline de creación de características y base
    create_features_and_base_pipeline = Pipeline([
        # Imputación de variables continuas
        (
            'continues_var_mean_imputation',
            MeanMedianImputer(
                imputation_method='mean',
                variables=CONTINUE_VARS_TO_IMPUT
            )
        ),
        # Imputación de variables categóricas
        (
            'categorical_var_freq_imputation',
            CategoricalImputer(
                imputation_method='frequent',
                variables=CATEGORICAL_VARS_TO_IMPUT
            )
        ),
        # Codificación de variables categóricas
        (
            'categorical_encoding_ohe',
            OneHotEncoder(variables=OHE_VAR_TO_ENCODE,drop_last=True)
        ),
        # Estandarización de variables
        (
            'feature_scaling',
            StandardScaler()
        )
    ])
