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

def read_train_data():
    logging.info("Reading train database")
    try:
        # FIX: Path corrected to match DataIngestionConfig
        file_path = os.path.join("artifacts", "train.csv")
        
        # Fallback check
        if not os.path.exists(file_path):
             # Try alternate path if first one fails
             file_path = "artifacts/data_ingestion/train.csv"
        
        df = pd.read_csv(file_path)
        print(f"Data loaded from: {file_path}")
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

            # FIX: n_jobs=2 ensures Colab doesn't crash due to RAM overload
            if total_combinations <= 20:
                logging.info(f"Using GridSearchCV for {model_name}")
                gs = GridSearchCV(model, para, cv=3, n_jobs=2, verbose=1, scoring=scoring)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
            else:
                logging.info(f"Using RandomizedSearchCV for {model_name}")
                n_iter = min(10, total_combinations)
                rs = RandomizedSearchCV(model, para, cv=3, n_iter=n_iter, n_jobs=2, verbose=1, scoring=scoring)
                rs.fit(X_train, y_train)
                model.set_params(**rs.best_params_)

            # Train the model with best params
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate based on the selected metric
            if scoring == 'accuracy':
                test_model_score = accuracy_score(y_test, y_test_pred)
            elif scoring == 'f1':
                test_model_score = f1_score(y_test, y_test_pred)
            elif scoring == 'roc_auc':
                # ROC AUC needs probabilities ideally, but using score for now to be safe
                test_model_score = roc_auc_score(y_test, y_test_pred)
            else:
                test_model_score = accuracy_score(y_test, y_test_pred)

            logging.info(f"Model: {model_name}, Metric: {scoring}, Test Score: {test_model_score}")

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