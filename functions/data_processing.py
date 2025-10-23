from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from imblearn.combine import SMOTEENN
from skopt import BayesSearchCV

class processing_functions:
    def preprocess_data(self, df):
        # Drop unnecessary columns
        df = df.drop(columns=['Sl. No', 'Patient File No.'])

        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        df[df.columns] = imputer.fit_transform(df)    #uses imputer to fill empty points with the median of that column

        #define predictor and target
        X = df.drop(columns=['PCOS (Y/N)'], errors='ignore')
        y = df['PCOS (Y/N)']

        #selecting predictors
        base = RandomForestClassifier(n_estimators=100, random_state=42) #the base model is RFC, where decision trees are created with splitting random features, and RF is created through many DT buildt on randomly selected features and subset of data. The decrease in impurity (how mixed the classes are) for each splitting is credited to that feature, making it important.
        rfe = RFE(base, n_features_to_select=30) #this is the RFE model, which will recursively train the dataset using the base model, calculate the average of the importance for feature, and eliminate the least important until only 30 features are left.
        X_selected = rfe.fit_transform(X, y) #fitting the data and transforming X so it now only contains 30 selected predictors

        #splitting data
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y, random_state=42)

        #fix data imbalance
        X_train, y_train = SMOTEENN(random_state=42).fit_resample(X_selected, y) #SMOTEENN uses KNN to create synthetic data for the minority class (PCOS = 1) and removes ambiguous data points (points whose majority neighbor is not its actual class).
        
        return X_train, X_test, y_train, y_test

    def make_model(self, X_train, y_train, base_model, meta_learner=None):
        if meta_learner is None:
            base_model.fit(X_train, y_train)
            return base_model

        else:
            stacked_model = StackingClassifier(estimators=base_model, final_estimator=meta_learner, passthrough=False, cv=5) #this model combines base model predictions with meta-learner. with passthrough=False, the meta-learner only sees base model predictions and not the raw dataset

            param_grid = {                                          #defines which hyperparameters to optimize and what ranges
            'rf__n_estimators': (50, 200),
            'rf__max_depth': (3, 15),
            'final_estimator__n_estimators': (50, 200)
            }

            opt = BayesSearchCV(                             #this model is a smart way of searching for the best parameter combination. It choses a set of parameter, fits the stacked model, perform cv, compute accuracy, update probabilistic model, repeat.
            estimator=stacked_model,
            search_spaces=param_grid,
            n_iter=20,                                          #number of parameter it tests
            cv=3,
            scoring='accuracy',
            random_state=42,
            verbose=0
            )

            opt.fit(X_train, y_train)        #this will trigger a cascade of model calling
            
            best_model = opt.best_estimator_
            best_params = opt.best_params_
            print("Best model:", best_model)
            print("Best Parameters:", best_params)
            
            return best_model


    def evaluate_model(self, best_model, X_test, y_test):
        y_pred = best_model.predict(X_test)   #making prediction using the best model with the best parameters

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f"\nStacked ML (RFE + RF meta-learner) Performance:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print("\nDetailed Report:\n", classification_report(y_test, y_pred))
        print(f"Confusion Matrix:\n{conf_matrix}")
