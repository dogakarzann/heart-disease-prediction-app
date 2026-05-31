import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, RandomForestRegressor
from sklearn.svm import SVC
from xgboost import XGBClassifier
import shap
import joblib
import warnings

warnings.filterwarnings('ignore')

# paths and settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "heart_disease_uci.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
RANDOM_STATE = 42
TEST_SIZE = 0.2

categorical_cols = ['dataset', 'cp', 'restecg', 'slope', 'thal', 'num']
bool_cols = ['sex', 'fbs', 'exang']
numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']

# function to fill missing categorical data
def impute_categorical(df, target_col, missing_cols):
    df_null = df[df[target_col].isnull()]
    df_not_null = df[df[target_col].notnull()]

    X = df_not_null.drop(target_col, axis=1)
    y = df_not_null[target_col]
    
    other_missing = [col for col in missing_cols if col != target_col]
    le = LabelEncoder()

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = le.fit_transform(X[col])

    if target_col in bool_cols:
        y = le.fit_transform(y)
        
    imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), add_indicator=True)

    for col in other_missing:
        if X[col].isnull().sum() > 0:
            missing_vals = X[col].values.reshape(-1, 1)
            filled = imputer.fit_transform(missing_vals)
            X[col] = filled[:, 0]
            
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    X = df_null.drop(target_col, axis=1)
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = le.fit_transform(X[col])

    for col in other_missing:
        if X[col].isnull().sum() > 0:
            missing_vals = X[col].values.reshape(-1, 1)
            filled = imputer.fit_transform(missing_vals)
            X[col] = filled[:, 0]
                
    if len(df_null) > 0: 
        df_null[target_col] = rf.predict(X)
        if target_col in bool_cols:
            df_null[target_col] = df_null[target_col].map({0: False, 1: True})
            
    return pd.concat([df_not_null, df_null])[target_col]


# function to fill missing numerical data
def impute_numerical(df, target_col, missing_cols):
    df_null = df[df[target_col].isnull()]
    df_not_null = df[df[target_col].notnull()]

    X = df_not_null.drop(target_col, axis=1)
    y = df_not_null[target_col]
    
    other_missing = [col for col in missing_cols if col != target_col]
    le = LabelEncoder()

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = le.fit_transform(X[col])
    
    imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), add_indicator=True)

    for col in other_missing:
        if X[col].isnull().sum() > 0:
            missing_vals = X[col].values.reshape(-1, 1)
            filled = imputer.fit_transform(missing_vals)
            X[col] = filled[:, 0]
            
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)

    X = df_null.drop(target_col, axis=1)
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = le.fit_transform(X[col])

    for col in other_missing:
        if X[col].isnull().sum() > 0:
            missing_vals = X[col].values.reshape(-1, 1)
            filled = imputer.fit_transform(missing_vals)
            X[col] = filled[:, 0]
                
    if len(df_null) > 0: 
        df_null[target_col] = rf.predict(X)

    return pd.concat([df_not_null, df_null])[target_col]


# load and clean the dataset
def load_data(path):
    print("Loading data...")
    df = pd.read_csv(path)

    # convert target to 1 if sick, else 0
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

    print("Filling missing values...")
    missing_cols = df.isnull().sum()[df.isnull().sum() > 0].index.tolist()
    for col in missing_cols:
        if col in categorical_cols:
            df[col] = impute_categorical(df, col, missing_cols)
        elif col in numeric_cols:
            df[col] = impute_numerical(df, col, missing_cols)

    print("Removing outliers...")
    for col in ['thalch', 'oldpeak', 'chol']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # drop rows where trestbps is 0
    df = df[df['trestbps'] != 0]
    return df

# split data and set up preprocessor
def prep_model(df):
    print("Splitting train and test data...")
    X = df.drop(['target', 'num', 'id', 'dataset'], axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    num_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

    # make sure categoricals are strings
    for col in cat_features:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False), cat_features)
        ])

    X_train_clean = preprocessor.fit_transform(X_train)
    X_test_clean = preprocessor.transform(X_test)

    # get column names for plotting later
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
    feature_names = num_features + list(cat_names)

    return X_train_clean, X_test_clean, y_train, y_test, preprocessor, feature_names

# train different algorithms
def train_algos(X_train, y_train):
    print("Training models...")
    models = {}
    
    # RF
    rf = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), 
                      {'n_estimators': [100, 200], 'max_depth': [10, 15, None], 'min_samples_split': [2, 5]}, 
                      cv=5, scoring='roc_auc', n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf.best_estimator_
    print(f"Random Forest score: {rf.best_score_:.4f}")

    # XGB
    xgb = GridSearchCV(XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'), 
                       {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5]}, 
                       cv=5, scoring='roc_auc', n_jobs=-1)
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb.best_estimator_
    print(f"XGBoost score: {xgb.best_score_:.4f}")

    # SVC
    svc = GridSearchCV(SVC(probability=True, random_state=RANDOM_STATE), 
                       {'C': [0.1, 1, 10], 'kernel': ['rbf']}, 
                       cv=5, scoring='roc_auc', n_jobs=-1)
    svc.fit(X_train, y_train)
    models['SVC'] = svc.best_estimator_
    print(f"SVC score: {svc.best_score_:.4f}")

    # Voting (Ensemble)
    voting = VotingClassifier(
        estimators=[('rf', models['Random Forest']), ('xgb', models['XGBoost']), ('svc', models['SVC'])],
        voting='soft'
    )
    voting.fit(X_train, y_train)
    models['Voting Ensemble'] = voting

    return models


def evaluate_all(models, X_test, y_test):
    print("Testing models on test set...")
    results = []
    roc_data = {}
    cm_data = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        })

        # ROC curve data for comparison charts
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc_score(y_test, y_pred_proba)}

        # Confusion matrix
        cm_data[name] = confusion_matrix(y_test, y_pred)

    # sort by score
    res_df = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False)
    best_name = res_df.iloc[0]['Model']
    best_model = models[best_name]

    print(f"Best model found: {best_name} (Score: {res_df.iloc[0]['ROC-AUC']:.4f})")

    all_results = {
        'metrics': res_df.to_dict('records'),
        'roc_data': roc_data,
        'confusion_matrices': cm_data
    }

    return best_model, res_df.iloc[0].to_dict(), all_results


# save the best model and tools
def save_files(preprocessor, best_model, feature_names, metrics, X_test, all_results):
    print("Setting up SHAP explainer...")
    
    base_model = best_model
    if isinstance(best_model, VotingClassifier):
        base_model = best_model.estimators_[0] 
    
    if isinstance(base_model, (RandomForestClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(base_model)
    else:
        explainer = shap.KernelExplainer(base_model.predict_proba, shap.sample(X_test, 100))

    print("Saving files to artifacts folder...")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"))
    joblib.dump(best_model, os.path.join(ARTIFACTS_DIR, "model.pkl"))
    joblib.dump(explainer, os.path.join(ARTIFACTS_DIR, "explainer.pkl"))
    joblib.dump(feature_names, os.path.join(ARTIFACTS_DIR, "feature_names.pkl"))
    joblib.dump(metrics, os.path.join(ARTIFACTS_DIR, "metrics.pkl"))
    joblib.dump(all_results, os.path.join(ARTIFACTS_DIR, "all_results.pkl"))

    print("Done! Everything is saved.")


def main():
    print("Starting ML Pipeline...")
    print("-" * 30)

    df = load_data(DATA_PATH)
    X_train_clean, X_test_clean, y_train, y_test, preprocessor, feature_names = prep_model(df)
    models = train_algos(X_train_clean, y_train)
    best_model, metrics, all_results = evaluate_all(models, X_test_clean, y_test)
    save_files(preprocessor, best_model, feature_names, metrics, X_test_clean, all_results)

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
