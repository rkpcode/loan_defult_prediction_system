import sys
import os
import pandas as pd
from src.loan_defult_prediction_system.exception import CustomException
from src.loan_defult_prediction_system.utils import load_object

import json

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            metadata_path = os.path.join("artifacts", "model_metadata.json")

            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Load Metadata for Threshold
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            threshold = metadata.get("threshold", 0.25) # Default to 0.1 if not found
            
            print(f"Loaded Threshold: {threshold}")
            print("After Loading")

            # --- FEATURE ENGINEERING (Must match Training) ---
            # 1. Income to Loan Ratio
            features['income_to_loan_ratio'] = features['annual_income'] / features['loan_amount']
            
            # 2. Total Debt (Approximation)
            features['total_debt'] = features['debt_to_income_ratio'] * features['annual_income']
            
            data_scaled = preprocessor.transform(features)
            
            # Use predict_proba for custom threshold
            # Class 0 is Default, Class 1 is Paid Back
            # We want to catch Defaults (Class 0)
            probs = model.predict_proba(data_scaled)
            
            # Assuming classes are [0, 1], probs[:, 0] is probability of Default
            prob_default = probs[:, 0]
            
            # Apply Custom Threshold
            preds = [0 if p > threshold else 1 for p in prob_default]
            
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
        annual_income: float,
        debt_to_income_ratio: float,
        credit_score: float,
        loan_amount: float,
        interest_rate: float,
        gender: str,
        marital_status: str,
        education_level: str,
        employment_status: str,
        loan_purpose: str,
        grade_subgrade: str):

        self.annual_income = annual_income
        self.debt_to_income_ratio = debt_to_income_ratio
        self.credit_score = credit_score
        self.loan_amount = loan_amount
        self.interest_rate = interest_rate
        self.gender = gender
        self.marital_status = marital_status
        self.education_level = education_level
        self.employment_status = employment_status
        self.loan_purpose = loan_purpose
        self.grade_subgrade = grade_subgrade

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "annual_income": [self.annual_income],
                "debt_to_income_ratio": [self.debt_to_income_ratio],
                "credit_score": [self.credit_score],
                "loan_amount": [self.loan_amount],
                "interest_rate": [self.interest_rate],
                "gender": [self.gender],
                "marital_status": [self.marital_status],
                "education_level": [self.education_level],
                "employment_status": [self.employment_status],
                "loan_purpose": [self.loan_purpose],
                "grade_subgrade": [self.grade_subgrade],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
