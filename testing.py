import pandas as pd
from functions.define_models import Models
from functions.data_processing import processing_functions
from dotenv import load_dotenv
import os

load_dotenv()
model = Models()
func = processing_functions()
df = pd.read_csv(os.getenv("DATASET_PATH"))

#preprocess data
X_train, X_test, y_train, y_test = func.preprocess_data(df)

#define model of your choice from define_models.py
base_models = [model.baseModel_Logistic(), model.baseModel_KNN(), model.baseModel_SVM(), model.baseModel_RFC(), model.baseModel_XGB(), model.baseModel_AdaBoost(), model.baseModel_GaussianNB()]
metalearner_model = model.metaLearner_neural_network()

#make and evaluate model
output_model, params = func.make_model( 
    y_train, 
    X_train,
    base_models,
    metalearner_model
)

acc, prec, rec, f1, conf_matrix = func.evaluate_model(output_model, X_test, y_test)
print("Best Model Parameters: ", params)
print(conf_matrix)