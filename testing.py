import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from functions.define_models import Models
from functions.data_processing import processing_functions

model = Models()
func = processing_functions()
df = pd.read_csv(file_path)

X_train, X_test, y_train, y_test = func.preprocess_data(df)

base_models = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('nb', GaussianNB()),
    ('xgb', XGBClassifier(eval_metric='logloss', random_state=42)),
    ('ada', AdaBoostClassifier(random_state=42))
]

metalearner_model = model.metaLearner_RFC()

model = func.make_model(
    X_train, 
    y_train, 
    #base_models,
    #metalearner_model
    RandomForestClassifier(random_state=42)
)

func.evaluate_model(model, X_test, y_test)