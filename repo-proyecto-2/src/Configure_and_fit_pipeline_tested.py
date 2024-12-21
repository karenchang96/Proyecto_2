# %%
# Importamos las librerias que nos seran de utilidad
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
from pathlib import Path

# %%

# Configuración de rutas
artifacts_path = './artifacts'
experiment_name = 'default_experiment'

# Crear carpeta de artefactos si no existe
Path(artifacts_path).mkdir(parents=True, exist_ok=True)

# %%

# Cargar datos desde archivo CSV
dataset = pd.read_csv("../data/raw/data.csv")
print(dataset.head())

# Separar características y objetivo
X = dataset.drop(columns=['mora'])
y = dataset['mora']

# %%

# Manejamos los valores faltantes solo en columnas numéricas
numeric_columns = X.select_dtypes(include=['number']).columns
imputer = SimpleImputer(strategy='mean')
X[numeric_columns] = imputer.fit_transform(X[numeric_columns])

# Identificamos las columnas categóricas
categorical_columns = X.select_dtypes(include=['object', 'category']).columns

# Transformamos las columnas categóricas en numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ], remainder='passthrough'
)

# Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar transformaciones
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# %%

# Modelos y configuraciones para probar
models = {
    "LogisticRegression": {
        "class": "sklearn.linear_model.LogisticRegression",
        "params": {"random_state": 42}
    },
    "RandomForestClassifier": {
        "class": "sklearn.ensemble.RandomForestClassifier",
        "params": {"n_estimators": 100, "random_state": 42}
    },
    "KNeighborsClassifier": {
        "class": "sklearn.neighbors.KNeighborsClassifier",
        "params": {"n_neighbors": 5}
    },
    "GradientBoostingClassifier": {
        "class": "sklearn.ensemble.GradientBoostingClassifier",
        "params": {"random_state": 42}
    },
    "SVC": {
        "class": "sklearn.svm.SVC",
        "params": {"kernel": "rbf", "C": 1.0, "random_state": 42}
    },
}

BEST_MODEL = None
best_score = -float('inf')
best_pipeline = None

# %%

# Registrar modelos y entrenar
mlflow.set_experiment(experiment_name)
mlflow.end_run()  # End any active run
with mlflow.start_run():
    for model_name, model_config in models.items():
        with mlflow.start_run(nested=True):
            model_class = model_config['class']
            model_params = model_config['params']

            # Instanciar el modelo dinámicamente
            module_name, class_name = model_class.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            ModelClass = getattr(module, class_name)
            model = ModelClass(**model_params)

            # Crear pipeline
            from sklearn.pipeline import Pipeline
            pipeline = Pipeline([
                ("model", model)
            ])

            # Entrenar el pipeline
            pipeline.fit(X_train, y_train)

            # Evaluar desempeño
            y_pred = pipeline.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            # Registrar en mlflow
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("accuracy", score)
            mlflow.sklearn.log_model(pipeline, f"model_{model_name}")

            # Seleccionar el mejor modelo
            if score > best_score:
                best_score = score
                BEST_MODEL = model_name
                best_pipeline = pickle.dumps(pipeline)

# Guardar el mejor pipeline en artefactos
best_pipeline_path = os.path.join('../artifacts', 'trained_pipeline.pkl')
Path('../artifacts').mkdir(parents=True, exist_ok=True)  # Asegurar que la carpeta exista
with open(best_pipeline_path, 'wb') as f:
    f.write(best_pipeline)

mlflow.log_artifact(best_pipeline_path)  # Registrar el archivo en mlflow

print(f"El mejor modelo fue {BEST_MODEL} con una precisión de {best_score:.4f}")
