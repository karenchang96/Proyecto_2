import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
import joblib
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Configuración inicial
data_path = "../data/raw/data.csv"
artifacts_path = "../artifacts/model_pipeline.pkl"

# Cargar dataset
dataset = pd.read_csv(data_path)

# Definición de variables
VARS_TO_DROP = ['mora']
CONTINUE_VARS_TO_IMPUT = [
    'edad', 'dias_lab', 'exp_sf', 'nivel_ahorro',
    'ingreso', 'linea_sf', 'deuda_sf', 'score', 'clasif_sbs'
]
CATEGORICAL_VARS_TO_IMPUT = ['vivienda', 'zona', 'nivel_educ']
OHE_VAR_TO_ENCODE = ['vivienda', 'zona', 'nivel_educ']

# Preparación de datos
x_features = dataset.drop(labels=VARS_TO_DROP, axis=1)
y_target = dataset['mora']

X_train, X_test, y_train, y_test = train_test_split(
    x_features, y_target, test_size=0.3, shuffle=True, random_state=2025
)

# Pipeline base
base_pipeline = Pipeline([
    ('continues_var_mean_imputation',
     MeanMedianImputer(imputation_method='mean', variables=CONTINUE_VARS_TO_IMPUT)),
    ('categorical_var_freq_imputation',
     CategoricalImputer(imputation_method='frequent', variables=CATEGORICAL_VARS_TO_IMPUT)),
    ('categorical_encoding_ohe',
     OneHotEncoder(variables=OHE_VAR_TO_ENCODE, drop_last=True)),
    ('feature_scaling', StandardScaler())
])

# Aplicar el pipeline base para preprocesamiento
X_train_processed = base_pipeline.fit_transform(X_train)
X_test_processed = base_pipeline.transform(X_test)

# Modelos candidatos
models = {
    "RandomForest": RandomForestClassifier(random_state=2025),
    "LogisticRegression": LogisticRegression(random_state=2025, max_iter=1000)
}

# Evaluar modelos
model_scores = {}
for model_name, model in models.items():
    scores = cross_val_score(model, X_train_processed, y_train, cv=5, scoring='f1')
    model_scores[model_name] = np.mean(scores)

# Seleccionar el mejor modelo
best_model_name = max(model_scores, key=model_scores.get)
best_model = models[best_model_name]

print(f"El modelo ganador es: {best_model_name} con F1-Score promedio: {model_scores[best_model_name]:.4f}")

# Agregar el modelo ganador al pipeline
final_pipeline = Pipeline([
    ('preprocessing', base_pipeline),
    ('model', best_model)
])

# Entrenar pipeline final
final_pipeline.fit(X_train, y_train)

# Guardar el pipeline entrenado
joblib.dump(final_pipeline, artifacts_path)
print(f"Pipeline final guardado en {artifacts_path}")
