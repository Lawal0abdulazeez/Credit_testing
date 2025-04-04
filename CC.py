import os # For file operations
import time # For time operations
import numpy as np # For numerical operations
import pandas as pd # For data manipulation
import joblib # For saving and loading objects
import matplotlib.pyplot as plt # For plotting
import seaborn as sns # For plotting
from datetime import datetime # For time operations
from tabulate import tabulate # For displaying tables
from tqdm import tqdm # For progress bars

# -------------------------------
# Import scikit-learn and TensorFlow/Keras
# -------------------------------
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor,
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, LSTM, Reshape
from tensorflow.keras.optimizers import Adam

import warnings # For handling warnings
warnings.filterwarnings('ignore') # Ignore warnings

# -------------------------------
# Custom Transformer for Label Encoding
# -------------------------------
class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        self.columns = columns
        self.encoders = {}
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        return self
    def transform(self, X):
        X = pd.DataFrame(X)
        X_transformed = X.copy()
        for col in self.columns:
            if col in X_transformed.columns:
                le = self.encoders.get(col)
                X_transformed[col] = le.transform(X_transformed[col].astype(str))
        return X_transformed

# -------------------------------
# Utility Functions
# -------------------------------
def print_log(message: str, color_code: str = ""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{color_code}[{timestamp}] {message}\033[0m")

def get_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    categorical_pipeline = Pipeline(steps=[
        ('label_encoder', MultiColumnLabelEncoder(columns=categorical_features))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])
    return preprocessor

def tune_model(model, param_grid: dict, X_train, y_train, scoring: str, cv: int = 3, n_iter: int = 10):
    search = RandomizedSearchCV(model, param_grid, scoring=scoring, n_iter=n_iter,
                                cv=cv, verbose=1, n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_

def evaluate_regression(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {"mse": mean_squared_error(y_test, y_pred), "r2": r2_score(y_test, y_pred)}

def save_object(obj, filename: str):
    joblib.dump(obj, filename)
    print_log(f"Saved object to {filename}")

def load_object(filename: str):
    obj = joblib.load(filename)
    print_log(f"Loaded object from {filename}")
    return obj

def prepare_sample_input(sample_dict: dict, required_columns: list) -> pd.DataFrame:
    for col in required_columns:
        if col not in sample_dict:
            sample_dict[col] = np.nan
    return pd.DataFrame([sample_dict], columns=required_columns)

def predict_sample(model, preprocessor, sample_df: pd.DataFrame) -> dict:
    sample_transformed = preprocessor.transform(sample_df)
    # For regression models, simply return the predicted value.
    pred_value = model.predict(sample_transformed)[0]
    return {"Predicted Credit Score": pred_value}

# -------------------------------
# Custom Deep Learning Model Wrappers (Regression)
# -------------------------------
class ANNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=0, hidden_units=64, dropout_rate=0.2, learning_rate=0.001,
                 epochs=10, batch_size=32, verbose=0):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_ = None

    def build_model(self):
        model = Sequential([
            Dense(self.hidden_units, activation='relu', input_dim=self.input_dim),
            Dropout(self.dropout_rate),
            Dense(self.hidden_units // 2, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='mse', metrics=['mse'])
        return model

    def fit(self, X, y):
        self.input_dim = X.shape[1]
        self.model_ = self.build_model()
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, X):
        preds = self.model_.predict(X, batch_size=self.batch_size, verbose=self.verbose)
        return preds.flatten()

class CNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=0, filters=32, kernel_size=3, hidden_units=64, dropout_rate=0.2,
                 learning_rate=0.001, epochs=10, batch_size=32, verbose=0):
        self.input_dim = input_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_ = None

    def build_model(self):
        model = Sequential([
            Reshape((self.input_dim, 1), input_shape=(self.input_dim,)),
            Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu'),
            Flatten(),
            Dense(self.hidden_units, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='mse', metrics=['mse'])
        return model

    def fit(self, X, y):
        self.input_dim = X.shape[1]
        self.model_ = self.build_model()
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, X):
        preds = self.model_.predict(X, batch_size=self.batch_size, verbose=self.verbose)
        return preds.flatten()

class LSTMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=0, lstm_units=50, dropout_rate=0.2, hidden_units=64,
                 learning_rate=0.001, epochs=10, batch_size=32, verbose=0):
        self.input_dim = input_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_ = None

    def build_model(self):
        model = Sequential([
            Reshape((1, self.input_dim), input_shape=(self.input_dim,)),
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(self.hidden_units, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='mse', metrics=['mse'])
        return model

    def fit(self, X, y):
        self.input_dim = X.shape[1]
        self.model_ = self.build_model()
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, X):
        preds = self.model_.predict(X, batch_size=self.batch_size, verbose=self.verbose)
        return preds.flatten()

# -------------------------------
# Multi-Model Credit Scoring Pipeline Function
# -------------------------------
def run_credit_scoring_pipeline_multi(data_path: str, nrows: int,
                                      categorical_features: list, numeric_features: list,
                                      target_column: str, models_with_params: dict):
    print_log(f"Starting Multi-Model Credit Scoring Pipeline for {target_column}", "\033[96m")
    df = pd.read_csv(data_path, nrows=nrows)
    df[categorical_features] = df[categorical_features].fillna("Unknown")
    df = df.dropna(subset=[target_column])
    # Remove target and other irrelevant columns from features
    drop_columns = [target_column, "Loan Approval Decision", "Loan Amounts", "Interest Rate"]
    X = df.drop(columns=drop_columns, errors='ignore')
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    preprocessor = get_preprocessor(numeric_features, categorical_features)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    results = {}
    for model_name, (model_instance, param_grid) in models_with_params.items():
        print_log(f"Training and tuning {model_name}...", "\033[92m")
        best_model = tune_model(model_instance, param_grid, X_train_transformed, y_train,
                                scoring='neg_mean_squared_error') if param_grid else model_instance.fit(X_train_transformed, y_train)
        metrics = evaluate_regression(best_model, X_test_transformed, y_test)
        results[model_name] = {"model": best_model, "mse": metrics["mse"], "r2": metrics["r2"]}
        print_log(f"{model_name} MSE: {metrics['mse']:.4f}, R2: {metrics['r2']:.4f}", "\033[94m")
    
    # -------------------------
    # Print a summary table of results
    results_table = []
    for model_name, metrics in results.items():
        results_table.append([model_name, f"{metrics['mse']:.4f}", f"{metrics['r2']:.4f}"])
    print_log("Summary of Model Performance:", "\033[96m")
    print(tabulate(results_table, headers=["Model", "MSE", "R2"], tablefmt="pretty"))
    

    best_model_name = min(results, key=lambda k: results[k]["mse"])
    best_model = results[best_model_name]["model"]
    print_log(f"Best Model for {target_column}: {best_model_name} with MSE: {results[best_model_name]['mse']:.4f}", "\033[95m")
    save_object(best_model, f"{best_model.__class__.__name__}_{target_column}.pkl")
    save_object(preprocessor, f"preprocessor_{target_column}.pkl")
    sample_input = X_test.iloc[[0]].copy()
    prediction = predict_sample(best_model, preprocessor, sample_input)
    print_log(f"Sample Prediction: {prediction}", "\033[94m")
    return best_model, preprocessor, list(X.columns)

# -------------------------------
# Main Execution for Credit Scoring
# -------------------------------
if __name__ == "__main__":
    project_folder = os.path.dirname(os.getcwd())
    data_file_path = os.path.join(project_folder, 'Data', 'cleaned-data', 'CleanedData.csv')
    print_log(f"Data file path: {data_file_path}", "\033[95m")
    
    # Define the categorical features as before.
    categorical_features = [
        "Gender", "Location", "Risk Ratings", "Marital Status",
        "Risk Status", "Working Sector", "Loan Types"
    ]
    # For credit scoring, remove "Credit Score" from the feature list since it is the target.
    numeric_features_cs = [
        "Age", "Delinquency Frequency", "Monthly Income (NGN)",
        "Debt-to-Income Ratio", "Number of Open Accounts"
    ]
    nrows = 1000  # Adjust as needed

    # Define regression models for predicting Credit Score
    credit_scoring_models = {
        "RandomForest": (
            RandomForestRegressor(random_state=42),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", "log2"]
            }
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5, 7]
            }
        ),
        "DecisionTree": (
            DecisionTreeRegressor(random_state=42),
            {
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }
        ),
        "LinearRegression": (
            LinearRegression(),
            {}  # No hyperparameters to tune
        ),
        "Ridge": (
            Ridge(),
            {
                "alpha": [0.1, 1, 10]
            }
        ),
        "Lasso": (
            Lasso(),
            {
                "alpha": [0.001, 0.01, 0.1, 1]
            }
        ),
        "SVR": (
            SVR(),
            {
                "C": [0.1, 1, 10],
                "epsilon": [0.1, 0.2, 0.5],
                "kernel": ["linear", "rbf"]
            }
        ),
        "KNeighbors": (
            KNeighborsRegressor(),
            {
                "n_neighbors": [3, 5, 7]
            }
        ),
        "AdaBoost": (
            AdaBoostRegressor(random_state=42),
            {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 1]
            }
        ),
        "ExtraTrees": (
            ExtraTreesRegressor(random_state=42),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20]
            }
        ),
        "ANN": (
            ANNRegressor(input_dim=0, hidden_units=64, dropout_rate=0.2, learning_rate=0.001,
                         epochs=10, batch_size=32, verbose=0),
            {
                "hidden_units": [64, 128],
                "dropout_rate": [0.2, 0.3],
                "learning_rate": [0.001, 0.01],
                "epochs": [10, 20],
                "batch_size": [32, 64]
            }
        ),
        "CNN": (
            CNNRegressor(input_dim=0, filters=32, kernel_size=3, hidden_units=64, dropout_rate=0.2,
                         learning_rate=0.001, epochs=10, batch_size=32, verbose=0),
            {
                "filters": [32, 64],
                "kernel_size": [3, 5],
                "hidden_units": [64, 128],
                "dropout_rate": [0.2, 0.3],
                "learning_rate": [0.001, 0.01],
                "epochs": [10, 20],
                "batch_size": [32, 64]
            }
        ),
        "LSTM": (
            LSTMRegressor(input_dim=0, lstm_units=50, dropout_rate=0.2, hidden_units=64,
                          learning_rate=0.001, epochs=10, batch_size=32, verbose=0),
            {
                "lstm_units": [50, 100],
                "dropout_rate": [0.2, 0.3],
                "hidden_units": [64, 128],
                "learning_rate": [0.001, 0.01],
                "epochs": [10, 20],
                "batch_size": [32, 64]
            }
        )
    }
    target_credit = "Credit Score"
    credit_model, credit_preprocessor, credit_features = run_credit_scoring_pipeline_multi(
        data_file_path, nrows, categorical_features, numeric_features_cs, target_credit, credit_scoring_models
    )
    
    # New sample prediction for credit scoring
    new_sample = {
        "Age": 30,
        "Gender": "Male",
        "Risk Ratings": "Medium",
        "Location": "Lagos",
        "Delinquency Frequency": 3,
        "Marital Status": "Single",
        "Monthly Income (NGN)": 500000,
        "Debt-to-Income Ratio": 0.5,
        "Working Sector": "Finance",
        "Loan Types": "Personal",
        "Number of Open Accounts": 10,
        "Risk Status": "Medium"
    }
    
    sample_input_df_credit = prepare_sample_input(new_sample, credit_features)
    credit_prediction = predict_sample(credit_model, credit_preprocessor, sample_input_df_credit)
    print_log(f"New sample credit score prediction: {credit_prediction}", "\033[94m")
