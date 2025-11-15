from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

class Models:
    def __init__(self):
            pass
    
    # Meta Learner Models
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
    

    # Base Models
    def baseModel_RFC (self):
        base_model = RandomForestClassifier(random_state=42)
        return base_model

    def baseModel_GaussianNB (self):
        base_model = GaussianNB()
        return base_model
    
    def baseModel_XGB (self):
        base_model = XGBClassifier(eval_metric='logloss', random_state=42)
        return base_model
    
    def baseModel_AdaBoost (self):
        base_model = AdaBoostClassifier(random_state=42)
        return base_model
    
    def baseModel_Logistic (self):
        base_model = LogisticRegression(max_iter=1000, random_state=42)
        return base_model

    def baseModel_KNN (self):
        base_model = KNeighborsClassifier() 
        return base_model

    def baseModel_SVM (self):
        base_model = svm.SVC(random_state=42)
        return base_model
   
    def baseModel_neural_network (self):
        base_model = MLPClassifier(random_state=42, max_iter=1000)
        return base_model