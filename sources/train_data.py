# Importamos librerías 
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import scipy.stats as stats

#mlflow
import mlflow
from mlflow.tracking import MlflowClient

# Preprocesamiento
from sklearn.impute import KNNImputer 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder,MinMaxScaler
from sklearn.preprocessing import power_transform, PowerTransformer
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.compose import make_column_transformer

# Preparación del set de datos
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler 
from sklearn.model_selection import StratifiedKFold, KFold

# Entrenamiento de modelos de prueba
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans


# Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as imbpipeline


# Esquemas de entrenamiento
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Evaluación de modelos
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score 

from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

# Semillas
seed = np.random.seed(22)
rng = np.random.RandomState(1)

# Versiones de librerías
print("".join(f"{x[0]}:{x[1]}\n" for x in [
    ("Pandas",pd.__version__),
    ("Numpy",np.__version__),
    ("Matplotlib",matplotlib.__version__),
    ("SKlearn",sklearn.__version__),
    ("Seaborn",sns.__version__)
]))

#Datos para seguimiento en mlflow en la DB
mlflow.set_tracking_uri("postgresql+psycopg2://postgres:XXX@35.xxx.xxx.xx:xxxx/postgres")
mlflow.set_registry_uri("postgresql+psycopg2://postgres:XXX@35.xxx.xxx.xx:xxxx/postgres")

experiment_id = mlflow.get_experiment_by_name("water_experiment")
if experiment_id:
    experiment_id = experiment_id.experiment_id

#Si no existe el experimento lo creamos
if not experiment_id:
    experiment_id = mlflow.create_experiment("water_experiment")

client = MlflowClient()

#Cargamos el dataset

colnames = ['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity','Potability']
water_data = pd.read_csv("./water.csv",delimiter=',',names=colnames, header=None)

print("Cantidad de muestras(filas) del dataset:",water_data.shape[0])
print("Cantidad de features(columnas) del dataset:",water_data.shape[1] )
print(water_data.head())

# **Paso 1**: separación del dataset en datos y target
X, y = water_data.drop(['Potability'], axis=1), water_data.Potability

# **Paso 2**: definición de lista de features numéricas y categóricas
# Lista de variables numéricas
numeric_features = X.select_dtypes('number').columns.to_list()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rng, stratify=y, shuffle=True)
X_train_sin_transf = X_train.copy()
X_test_sin_transf = X_test.copy()
power_data_train = X_train[['Solids','Conductivity']]
power_data_test = X_test[['Solids','Conductivity']]

#Transformar las dos características por el método de yeo-johnson
power_data_pt  = PowerTransformer(method='yeo-johnson',standardize=False,copy=False)

power_data_train = power_data_pt.fit_transform(power_data_train)
power_data_test = power_data_pt.transform(power_data_test)

#En train
feature1=power_data_train[:,0].flatten()
X_train=X_train.assign(Solids=feature1)
feature2=power_data_train[:,1].flatten()
X_train=X_train.assign(Conductivity=feature2)

#En test
feature3=power_data_test[:,0].flatten()
X_test=X_test.assign(Solids=feature3)
feature4=power_data_test[:,1].flatten()
X_test=X_test.assign(Conductivity=feature4)

# **Paso 3**: transformación de columnas

#Se elige el método de KNN para imputar los valores faltantes y StandardScaler para normalizar a media 0 y varianza 1.


numeric_transformer = Pipeline(steps=[('imputer', KNNImputer()),
                                      ('scaler', StandardScaler())
                                      ])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])


#Fiteamos cada modelo como un run de mlfow
mlflow.sklearn.autolog()

### Modelo Random Forest
with mlflow.start_run(experiment_id = experiment_id) as run:

    model_rf = imbpipeline(steps=[('preprocessor', preprocessor),
                               ('balance', RandomOverSampler()),
                               ('classifier', RandomForestClassifier())])

    params_rf = {'preprocessor__num__imputer__n_neighbors': [3,5],
                  'classifier__n_estimators': [100, 500],
                  'classifier__max_depth': [ 6],#None
                  'classifier__max_features': [6, 9],
                  'classifier__min_samples_leaf': [3,9],
                  'classifier__random_state': [rng],
                  'classifier__criterion': ['gini','entropy']}

    grid_rf = GridSearchCV(model_rf, params_rf, scoring='precision', cv=KFold(5, random_state=rng, shuffle=True))

    grid_rf.fit(X_train, y_train) 
    grid_predictions_rf = grid_rf.predict(X_test)

    accuracy_rf=accuracy_score(y_test, grid_predictions_rf)
    precision_rf=precision_score(y_test, grid_predictions_rf)
    recall_rf=recall_score(y_test, grid_predictions_rf)
    #print ("Random Forest Accuracy: ",accuracy_rf)
    #print ("Random Forest Precision: ",precision_rf)
    #print ("Random Forest Recall: ",recall_rf)
    mlflow.log_metric("presicion_test", precision_rf)
    mlflow.sklearn.log_model(grid_rf,'water_model')

mlflow.end_run()

### Modelo Decision Tree
with mlflow.start_run(experiment_id = experiment_id) as run:


    model_dt = imbpipeline(steps=[('preprocessor', preprocessor),
                               ('balance', RandomOverSampler()),
                               ('classifier', DecisionTreeClassifier())])

    params_dt = {'preprocessor__num__imputer__n_neighbors': [3, 5],
                  'classifier__max_depth': [ 3, 6],
                  'classifier__max_features': [6, 9],
                  'classifier__min_samples_leaf': [ 3, 5],
                  'classifier__criterion': ['gini','entropy'],
                  'classifier__random_state': [rng]}

    grid_dt = GridSearchCV(model_dt, params_dt, scoring='precision',cv=KFold(5, random_state=rng, shuffle=True))

    # fitting del modelo por Randomized Search
    grid_dt.fit(X_train, y_train)
    # impresión de los mejores hiperparámetros luego del tunning
    #print(grid_dt.best_params_)
    grid_predictions_dt = grid_dt.predict(X_test)

    accuracy_dt=accuracy_score(y_test, grid_predictions_dt)
    precision_dt=precision_score(y_test, grid_predictions_dt)
    recall_dt=recall_score(y_test, grid_predictions_dt)
    #print ("Desiccion Tree Accuracy: ",accuracy_dt)
    #print ("Desiccion Tree Precision: ",precision_dt)
    #print ("Desiccion Tree Recall: ",recall_dt)

    mlflow.log_metric("presicion_test", precision_dt)
    mlflow.sklearn.log_model(grid_dt,'water_model')

mlflow.end_run()

####Modelo SVM
with mlflow.start_run(experiment_id = experiment_id) as run:
    model_svm = imbpipeline(steps=[('preprocessor', preprocessor),
                                ('balance', RandomOverSampler()),
                                ('classifier', SVC())])

    params_svm = {'preprocessor__num__imputer__n_neighbors': [3,5],
                 'classifier__C': [1, 50],
                  'classifier__kernel': ["rbf","linear"],
                  'classifier__class_weight': ['balanced',None],
                  #'classifier__tol': [1e-6,1e-5],
                  #'classifier__max_iter': [10000, 20000],
                  'classifier__gamma': ["scale","auto"],
                  'classifier__random_state': [rng]}

    grid_svm = GridSearchCV(model_svm, params_svm, scoring='precision', cv=KFold(5, random_state=rng, shuffle=True)) 

    grid_svm.fit(X_train, y_train)
    grid_predictions_svm = grid_svm.predict(X_test)

    accuracy_svm=accuracy_score(y_test, grid_predictions_svm)
    precision_svm=precision_score(y_test, grid_predictions_svm)
    recall_svm=recall_score(y_test, grid_predictions_svm)
    #print ("SVM Accuracy: ",accuracy_svm)
    #print ("SVM Precision: ",precision_svm)
    #print ("SVM Recall: ",recall_svm)
    mlflow.log_metric("presicion_test", precision_svm)
    mlflow.sklearn.log_model(grid_svm,'water_model')

mlflow.end_run()

#### Modelo Regresión logística
with mlflow.start_run(experiment_id = experiment_id) as run:
    model_lr = imbpipeline(steps=[('preprocessor', preprocessor),
                               ('balance', RandomOverSampler()),
                               ('classifier', LogisticRegression())])

    params_lr = {'preprocessor__num__imputer__n_neighbors': [3, 5],
                  'classifier__penalty': ['l2'],
                  'classifier__class_weight': ['balanced'],
                 'classifier__solver': ['lbfgs','sag','newton-cg'],
                  'classifier__C': [1, 4, 20],
                  'classifier__random_state': [rng]}
    grid_lr = GridSearchCV(model_lr, params_lr, scoring='precision', cv=KFold(5, random_state=rng, shuffle=True))

    # fitting del modelo por Randomized Search
    grid_lr.fit(X_train, y_train)
    grid_predictions_lr = grid_lr.predict(X_test)

    accuracy_lr=accuracy_score(y_test, grid_predictions_lr)
    precision_lr=precision_score(y_test, grid_predictions_lr)
    recall_lr=recall_score(y_test, grid_predictions_lr)
    #print ("Reg logistic. Accuracy: ",accuracy_lr)
    #print ("Reg logistic. Precision: ",precision_lr)
    #print ("Reg logistic. Recall: ",recall_lr)
    mlflow.log_metric("presicion_test", precision_lr)
    mlflow.sklearn.log_model(grid_lr,'water_model')

mlflow.end_run()

print("FIN del proceso de entrenamiento")
