from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from imblearn.combine import SMOTEENN
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV
import numpy as np

class processing_functions:

    #this function preprocesses the data
    def preprocess_data(self, df):
        # Drop unnecessary columns
        df = df.drop(columns=['Sl. No', 'Patient File No.'])

        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        df[df.columns] = imputer.fit_transform(df)    #uses imputer to fill empty points with the median of that column

        #define predictor and target
        X = df.drop(columns=['PCOS (Y/N)'], errors='ignore')
        y = df['PCOS (Y/N)']

        #split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        #scale data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        #fix data imbalance
        X_train, y_train = SMOTEENN(random_state=42).fit_resample(X_train, y_train)

        #selecting predictors
        base = RandomForestClassifier(n_estimators=100, random_state=42) #the base model is RFC, where decision trees are created with splitting random features, and RF is created through many DT buildt on randomly selected features and subset of data. The decrease in impurity (how mixed the classes are) for each splitting is credited to that feature, making it important.
        rfe = RFE(base, n_features_to_select=30) #this is the RFE model, which will recursively train the dataset using the base model, calculate the average of the importance for feature, and eliminate the least important until only 30 features are left.
        X_train = rfe.fit_transform(X_train, y_train)
        X_test = rfe.transform(X_test)

        return X_train, X_test, y_train, y_test
    
    #this function defines the hyperparameter search space for each model and performs Bayesian Optimization to find the best hyperparameters.
    def search_best_params(self, estimator, base_models=None, meta_learner=None):
        
        #function to get hyperparameter search space for each model
        def get_param_space(model, prefix=""):
            space = {}

            if isinstance(model, RandomForestClassifier):
                space[prefix + "n_estimators"] = Integer(50, 2000)
                space[prefix + "max_depth"] = Integer(3, 30)
                space[prefix + "min_samples_split"] = Integer(2, 20)
                space[prefix + "min_samples_leaf"] = Integer(1, 20)
                space[prefix + "max_features"] = Categorical(["auto", "sqrt", "log2"])

            elif isinstance(model, LogisticRegression):
                space[prefix + "C"] = Real(1e-4, 1000, prior="log-uniform")
                space[prefix + "penalty"] = Categorical(["l1", "l2", "elasticnet"])
                space[prefix + "solver"] = Categorical(["liblinear", "saga"])

            elif isinstance(model, KNeighborsClassifier):
                space[prefix + "n_neighbors"] = Integer(3, 50)
                space[prefix + "weights"] = Categorical(["uniform", "distance"])
                space[prefix + "metric"] = Categorical(["euclidean", "manhattan", "minkowski"])

            elif isinstance(model, svm.SVC):
                space[prefix + "C"] = Real(1e-3, 1000, prior="log-uniform")
                space[prefix + "kernel"] = Categorical(["linear", "rbf", "poly"])

            elif isinstance(model, XGBClassifier):
                space[prefix + "n_estimators"] = Integer(50, 500)
                space[prefix + "max_depth"] = Integer(3, 15)
                space[prefix + "learning_rate"] = Real(1e-2, 2.0, prior="log-uniform")
                space[prefix + "max_depth"] = Integer(3, 15)

            elif isinstance(model, AdaBoostClassifier):
                space[prefix + "n_estimators"] = Integer(50, 500)
                space[prefix + "learning_rate"] = Real(1e-2, 2.0)

            elif isinstance(model, GaussianNB):
                space[prefix + "var_smoothing"] = Real(1e-12, 1e-6, prior="log-uniform")

            elif isinstance(model, MLPClassifier):
                space[prefix + "hidden_layer_sizes"] = Integer(1, 200)
                space[prefix + "activation"] = Categorical(["relu", "tanh", "logistic"])
                space[prefix + "alpha"] = Real(1e-6, 1e-1, prior="log-uniform")
                space[prefix + "solver"] = Categorical(["adam", "lbfgs"])
                space[prefix + "learning_rate_init"] = Real(1e-5, 1e-2)

            return space

        #tuning a single model
        if meta_learner is None:
            model = base_models[0]
            param_space = get_param_space(model)

            if not param_space:
                print("No search space found for model. Running with defaults.")
                param_space = {}

            opt = BayesSearchCV(
                estimator=estimator,
                search_spaces=param_space,
                n_iter=20,
                cv=3,
                scoring="accuracy",
                random_state=42,
                verbose=0
            )

            return opt

        #tuning a stacked model, including all base models and the meta-learner
        param_space = {}

        #iterate through base models to get their search spaces
        for i, mdl in enumerate(base_models):
            prefix = f"m{i}__"
            param_space.update(get_param_space(mdl, prefix))

        #get meta-learner search space
        param_space.update(get_param_space(meta_learner, prefix="final_estimator__"))

        if not param_space:
            print("⚠ No hyperparameters defined for stacked model. Running defaults.")

        opt = BayesSearchCV(
            estimator=estimator,
            search_spaces=param_space,
            n_iter=20,
            cv=3,
            scoring="accuracy",
            random_state=42,
            verbose=0
        )

        return opt

    #this function creates and trains either a base model or a stacking model
    def make_model(self, y_train, X_train, base_model, meta_learner=None):
      
        #train a single model
        if meta_learner is None:
            opt = self.search_best_params(base_model, [base_model])
            model = opt.fit(X_train, y_train)

            best_params = opt.best_params_

            return model, best_params

        #train a stacking model
        else:
            named_estimators = [(f"m{i}", m) for i, m in enumerate(base_model)]
            stacked_model = StackingClassifier(estimators=named_estimators, final_estimator=meta_learner, passthrough=False, cv=5) #this model combines base model predictions with meta-learner. with passthrough=False, the meta-learner only sees base model predictions and not the raw dataset
            opt = self.search_best_params(stacked_model, base_model, meta_learner)
            stacked_model = opt.fit(X_train, y_train)   

            best_params = opt.best_params_
            
            return stacked_model, best_params

    #this function evaluates the best parameter model on test data
    def evaluate_model(self, best_model, X_test, y_test):
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        return acc, prec, rec, f1, conf_matrix