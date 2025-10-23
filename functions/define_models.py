from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

class Models:
    def __init__(self):
            pass
    def metaLearner_RFC (self):
        meta_learner = RandomForestClassifier(random_state=42)
        return meta_learner
   
    def metaLearner_LogisticRegression (self):
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        return meta_learner
  
    def metaLearner_AdaBoost (self):
        meta_learner = AdaBoostClassifier(random_state=42)
        return meta_learner
   
    def metaLearner_xgboost (self):
        meta_learner = XGBClassifier(eval_metric='logloss', random_state=42)
        return meta_learner

    def metaLearner_neural_network (self):
        meta_learner = MLPClassifier(random_state=42, max_iter=1000)
        return meta_learner
