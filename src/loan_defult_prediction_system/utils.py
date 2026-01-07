import os
import sys
from src.loan_defult_prediction_system.exception import CustomException
from src.loan_defult_prediction_system.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from dataclasses import dataclass
from dotenv import load_dotenv
import pymysql
import pickle

def read_train_data(sample_size: int = None):
    """
    Read training data from CSV file
    Args:
        sample_size: If provided, randomly sample N rows for testing (default: None = load all data)
    Returns:
        DataFrame with loan data
    """
    logging.info("Reading train database")
    try:
        # Load from full dataset location
        file_path = "artifacts/data_ingestion/train.csv"
        
        # Fallback to artifacts/train.csv if main path doesn't exist
        if not os.path.exists(file_path):
            file_path = os.path.join("artifacts", "train.csv")
            logging.warning(f"Primary path not found, using fallback: {file_path}")
        
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded from: {file_path} - Total rows: {len(df)}")
        
        # Apply sampling if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            logging.info(f"Sampled {sample_size} rows for testing")
        
        print(f"Data loaded from: {file_path}")
        print(f"Dataset size: {len(df)} rows")
        print(df.head())
        return df
    except Exception as ex:
        raise CustomException(ex, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param, scoring='accuracy'):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            logging.info(f"Training model: {model_name}")

            # Calculate total combinations
            total_combinations = 1
            for p in para.values():
                total_combinations *= len(p)
            
            logging.info(f"Model: {model_name}, Total parameter combinations: {total_combinations}")

            # Use RandomizedSearchCV for large search spaces (more efficient for GPU)
            if total_combinations > 50:
                logging.info(f"Using RandomizedSearchCV for {model_name}")
                n_iter = min(30, total_combinations)  # Try 30 best combinations
                rs = RandomizedSearchCV(
                    model, para, 
                    cv=5,                    # 5-fold CV for better validation
                    n_iter=n_iter, 
                    n_jobs=1,                # Sequential for GPU models
                    verbose=2,               # Show detailed progress
                    scoring=scoring,
                    random_state=42
                )
                rs.fit(X_train, y_train)
                model.set_params(**rs.best_params_)
                logging.info(f"Best params for {model_name}: {rs.best_params_}")
                logging.info(f"Best CV score: {rs.best_score_:.4f}")
            else:
                logging.info(f"Using GridSearchCV for {model_name}")
                gs = GridSearchCV(
                    model, para, 
                    cv=5,                    # 5-fold CV
                    n_jobs=1, 
                    verbose=2, 
                    scoring=scoring
                )
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
                logging.info(f"Best params for {model_name}: {gs.best_params_}")
                logging.info(f"Best CV score: {gs.best_score_:.4f}")

            # Train the model with best params
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate based on the selected metric
            if scoring == 'accuracy':
                test_model_score = accuracy_score(y_test, y_test_pred)
            elif scoring == 'f1':
                test_model_score = f1_score(y_test, y_test_pred, pos_label=0)  # F1 for defaulters
            elif scoring == 'roc_auc':
                # Use probabilities for ROC-AUC
                if hasattr(model, "predict_proba"):
                    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
                    test_model_score = roc_auc_score(y_test, y_test_pred_proba)
                else:
                    test_model_score = roc_auc_score(y_test, y_test_pred)
            else:
                test_model_score = accuracy_score(y_test, y_test_pred)

            logging.info(f"Model: {model_name}, Metric: {scoring}, Test Score: {test_model_score:.4f}")

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)