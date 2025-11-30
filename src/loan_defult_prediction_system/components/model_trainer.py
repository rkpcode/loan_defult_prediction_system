import os
import sys
import numpy as np
import json
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.loan_defult_prediction_system.exception import CustomException
from src.loan_defult_prediction_system.logger import logging
from src.loan_defult_prediction_system.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    model_metadata_file_path = os.path.join("artifacts", "model_metadata.json")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # --- GPU-OPTIMIZED MODELS (Enhanced for Accuracy & Class Imbalance) ---
            # Calculate actual class weights for better imbalance handling
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight('balanced', 
                                                 classes=np.unique(y_train), 
                                                 y=y_train)
            scale_pos_weight = class_weights[0] / class_weights[1]
            
            logging.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
            
            models = {
                "Random Forest": RandomForestClassifier(
                    class_weight='balanced',
                    n_jobs=-1,
                    random_state=42,
                    max_features='sqrt'
                ),
                "XGBClassifier": XGBClassifier(
                    tree_method='hist',          # Histogram-based (GPU auto-detected with device param)
                    device='cuda:0',             # GPU device (XGBoost 2.0+)
                    scale_pos_weight=scale_pos_weight,  # Dynamic class weight
                    n_jobs=-1,
                    random_state=42
                ),
                "CatBoost Classifier": CatBoostClassifier(
                    task_type='GPU',             # Explicit GPU usage
                    devices='0',                 # GPU device
                    auto_class_weights='Balanced',
                    verbose=0,
                    random_state=42,
                    allow_writing_files=False
                ),
            }
            
           # --- REAL FAST PARAMS (For 30 min run) ---
            params = {
                "Random Forest": {
                    'n_estimators': [100],        # Sirf 100 check karo
                    'max_depth': [10, 20],        # Deep trees slow hote hain
                    'min_samples_split': [5],     # Ek value kaafi hai
                    'max_features': ['sqrt']      # Log2 hata diya
                },
                "XGBClassifier": {
                    'learning_rate': [0.1],       # Standard rate
                    'n_estimators': [100, 200],   # 300 hata diya
                    'max_depth': [7],          
                    'subsample': [0.8],           # Fixed
                    'colsample_bytree': [0.8],    # Fixed
                    'gamma': [0]                  # Fixed
                },
                "CatBoost Classifier": {
                    'depth': [6],
                    'learning_rate': [0.1],
                    'iterations': [200],
                    'border_count': [32]
               }
            }

            logging.info("Starting Model Training with Hyperparameter Tuning...")
            
            # Use F1 for grid search selection to balance precision/recall during tuning
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params, scoring='f1')

            ## Get best model score and name
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.5:
                raise CustomException("No satisfactory model found (Score too low)")
            
            logging.info(f"Best Model Found: {best_model_name} with Base F1: {best_model_score}")

            # --- SAVE MODEL ---
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # --- CRITICAL: CUSTOM THRESHOLD LOGIC (The "Secret Sauce") ---
            # Hum 0.5 threshold use nahi karenge. Based on our analysis, 0.1 is optimal.
            OPTIMAL_THRESHOLD = 0.25
            
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
                "description": "Custom threshold 0.25 selected to maximize Recall for Defaulters."
            }
            with open(self.model_trainer_config.model_metadata_file_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logging.info(f"Metadata saved at {self.model_trainer_config.model_metadata_file_path}")

            # 4. Log Correct Metrics based on Custom Threshold
            cm = confusion_matrix(y_test, custom_predictions)
            recall = recall_score(y_test, custom_predictions, pos_label=0) # Focus on Default Recall
            f1_custom = f1_score(y_test, custom_predictions, pos_label=0)

            print(f"\n================ FINAL REPORT ================")
            print(f"Best Model: {best_model_name}")
            print(f"Optimal Threshold Used: {OPTIMAL_THRESHOLD}")
            print(f"Confusion Matrix:\n{cm}")
            print(f"Recall (Defaults Caught): {recall:.4f}")
            print(f"F1 Score (Class 0): {f1_custom:.4f}")
            print(f"==============================================")
            
            logging.info(f"Final Recall at threshold {OPTIMAL_THRESHOLD}: {recall}")

            return recall

        except Exception as e:
            raise CustomException(e, sys)