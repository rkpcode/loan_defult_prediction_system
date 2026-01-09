import os
import sys
import numpy as np
import json
from dataclasses import dataclass

from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from src.loan_defult_prediction_system.exception import CustomException
from src.loan_defult_prediction_system.logger import logging
from src.loan_defult_prediction_system.utils import evaluate_models, save_object

@dataclass
class XGBoostTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "xgboost_model.pkl")
    model_metadata_file_path = os.path.join("artifacts", "xgboost_model_metadata.json")

class XGBoostTrainer:
    def __init__(self):
        self.model_trainer_config = XGBoostTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # --- GPU-OPTIMIZED MODELS w/ CPU FALLBACK ---
            import subprocess
            try:
                subprocess.check_output('nvidia-smi')
                is_gpu_available = True
                logging.info("GPU detected via nvidia-smi. Configuring XGBoost for GPU.")
            except Exception:
                is_gpu_available = False
                logging.info("GPU NOT detected. Configuring XGBoost for CPU.")

            # Calculate actual class weights for better imbalance handling
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight('balanced', 
                                                 classes=np.unique(y_train), 
                                                 y=y_train)
            scale_pos_weight = class_weights[0] / class_weights[1]
            
            logging.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

            # Define parameters based on device
            xgb_params = {
                'objective': 'binary:logistic',
                'scale_pos_weight': scale_pos_weight,
                'n_jobs': -1,
                'random_state': 42
            }
            
            if is_gpu_available:
                xgb_params['tree_method'] = 'hist'
                xgb_params['device'] = 'cuda'
            else:
                 xgb_params['tree_method'] = 'hist' 

            models = {
                "XGBClassifier": XGBClassifier(**xgb_params),
            }
            
            # --- COMPREHENSIVE HYPERPARAMETER TUNING ---
            params = {
                "XGBClassifier": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'n_estimators': [100, 300, 500, 1000],
                    'max_depth': [3, 5, 7, 10],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'gamma': [0, 0.1, 0.2],
                    'min_child_weight': [1, 3, 5],
                    'reg_alpha': [0, 0.01, 0.1, 1],
                    'reg_lambda': [1, 1.5, 2, 5]
                }
            }

            logging.info("Starting XGBoost Training with Hyperparameter Tuning...")
            
            # Use ROC AUC for grid search selection (competition evaluation metric)
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params, scoring='roc_auc')

            ## Get best model score and name
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.5:
                raise CustomException("No satisfactory model found (Score too low)")
            
            logging.info(f"Best Model Found: {best_model_name} with ROC AUC: {best_model_score:.4f}")

            # --- SAVE MODEL ---
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # --- CRITICAL: CUSTOM THRESHOLD LOGIC (The "Secret Sauce") ---
            # Hum 0.5 threshold use nahi karenge. Based on our analysis, 0.1 is optimal.
            OPTIMAL_THRESHOLD = 0.1
            
            # 1. Get Probability of Default (Class 0)
            # Assumption: Class 0 is Default, Class 1 is Paid.
            # predict_proba returns [Prob(0), Prob(1)]
            y_prob_default = best_model.predict_proba(X_test)[:, 0]
            
            # 2. Apply Custom Threshold
            # If Prob(Default) > 0.1, Predict Default (0), else Paid (1)
            custom_predictions = np.where(y_prob_default > OPTIMAL_THRESHOLD, 0, 1)

            # 3. Save Threshold Metadata (For API usage)
            metadata = {
                "best_model": best_model_name,
                "threshold": OPTIMAL_THRESHOLD,
                "description": "Custom threshold 0.1 selected to maximize Recall for Defaulters."
            }
            with open(self.model_trainer_config.model_metadata_file_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logging.info(f"Metadata saved at {self.model_trainer_config.model_metadata_file_path}")

            # 4. Log Correct Metrics based on Custom Threshold
            cm = confusion_matrix(y_test, custom_predictions)
            recall = recall_score(y_test, custom_predictions, pos_label=0) # Focus on Default Recall
            f1_custom = f1_score(y_test, custom_predictions, pos_label=0)
            
            # Calculate ROC AUC using predicted probabilities (primary evaluation metric)
            # Using Prob(0) vs y_test (where 0=Default) is tricky as roc_auc expects y_score increasing with positive label.
            # If we pass y_test (0/1) and y_prob_default (Prob(0)), we should probably invert one.
            # However, to be consistent with generic report, we'll assume the metric calculated by evaluate_models is the truth.
            
            roc_auc = best_model_score # Use the score from evaluation which was correct (Prob(1))

            print(f"\n================ FINAL REPORT ================")
            print(f"Best Model: {best_model_name}")
            print(f"ROC AUC Score: {roc_auc:.4f}")
            print(f"Optimal Threshold Used: {OPTIMAL_THRESHOLD}")
            print(f"Confusion Matrix:\n{cm}")
            print(f"Recall (Defaults Caught): {recall:.4f}")
            print(f"F1 Score (Class 0): {f1_custom:.4f}")
            print(f"==============================================")
            
            logging.info(f"ROC AUC Score: {roc_auc:.4f}")
            logging.info(f"Final Recall at threshold {OPTIMAL_THRESHOLD}: {recall}")

            return roc_auc

        except Exception as e:
            raise CustomException(e, sys)
