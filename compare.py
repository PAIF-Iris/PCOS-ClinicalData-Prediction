# this script compares various base models and stacking ensemble models using different meta-learners.
import pandas as pd
from functions.define_models import Models
from functions.data_processing import processing_functions
from dotenv import load_dotenv
import os
from itertools import combinations

load_dotenv()
model = Models()
func = processing_functions()
df = pd.read_csv(os.getenv("DATASET_PATH"))

X_train, X_test, y_train, y_test = func.preprocess_data(df)

#define models to compare
base_models = [model.baseModel_Logistic(), model.baseModel_KNN(), model.baseModel_SVM(), model.baseModel_RFC(), model.baseModel_XGB(), model.baseModel_AdaBoost(), model.baseModel_GaussianNB()]
metalearner_models = [model.metaLearner_RFC(), model.metaLearner_LogisticRegression(), model.metaLearner_AdaBoost(), model.metaLearner_xgboost(), model.metaLearner_neural_network()]

#dataframe to store results
results = pd.DataFrame(columns=["Stacking (Y/N)", "base_model", "meta_learner", "Accuracy", "Precision", "Recall", "F1-Score", "confusion_matrix", "best_model_params"])

#iterate through base models
for base_model in base_models:
    best_model, best_param = func.make_model( 
        y_train, 
        X_train,
        base_model,
    )
    print("passed")
    acc, prec, rec, f1, conf_matrix = func.evaluate_model(best_model, X_test, y_test)
    results.loc[len(results)] = [0, base_model, None, acc, prec, rec, f1, conf_matrix, best_param]

#iterate through stacking models
for metalearner_model in metalearner_models:
    for n in range(2, len(base_models) + 1):
        for comb in combinations(base_models, n):
            best_model, best_param = func.make_model( 
                y_train, 
                X_train,
                comb,
                metalearner_model
            )
            acc, prec, rec, f1, conf_matrix = func.evaluate_model(best_model, X_test, y_test)
            results.loc[len(results)] = [1, comb, metalearner_model, acc, prec, rec, f1, conf_matrix, best_param]
            print("passed")

results.to_csv('output.csv')

