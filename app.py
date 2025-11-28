from flask import Flask, request, render_template, send_file
import os
import numpy as np
import pandas as pd
from src.loan_defult_prediction_system.pipelines.prediction_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            annual_income=float(request.form.get('annual_income')),
            debt_to_income_ratio=float(request.form.get('debt_to_income_ratio')),
            credit_score=float(request.form.get('credit_score')),
            loan_amount=float(request.form.get('loan_amount')),
            interest_rate=float(request.form.get('interest_rate')),
            gender=request.form.get('gender'),
            marital_status=request.form.get('marital_status'),
            education_level=request.form.get('education_level'),
            employment_status=request.form.get('employment_status'),
            loan_purpose=request.form.get('loan_purpose'),
            grade_subgrade=request.form.get('grade_subgrade')
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        
        # Convert prediction to readable format
        # 0 = Default, 1 = Paid Back
        prediction_label = "Loan Default" if results[0] == 0 else "Loan Paid Back"
        
        return render_template('home.html', results=prediction_label)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            
            # Predict for the uploaded CSV
            predict_pipeline = PredictPipeline()
            
            # Ensure the CSV has the required columns
            # In a real scenario, we might need to map columns or handle missing ones
            # For now, assuming the CSV structure matches the training data input features
            
            # We might need to preprocess or select specific columns if the CSV has extra data
            # Assuming the CSV has the raw features expected by the pipeline
            
            try:
                # We need to handle potential column mismatches or ID columns
                # For simplicity, let's try passing the dataframe directly if it matches
                # However, the pipeline expects specific columns. 
                # Let's assume the user uploads a CSV with the same columns as the manual input
                
                # If the CSV has an 'id' or 'loan_paid_back' column, we might want to drop it or ignore it
                # The preprocessor handles the transformation, so we just need to pass the dataframe
                
                results = predict_pipeline.predict(df)
                
                df['loan_paid_back'] = results
                # 1 = Paid Back (Give Loan: YES), 0 = Default (Give Loan: NO)
                df['Give_Loan'] = df['loan_paid_back'].apply(lambda x: "YES" if x == 1 else "NO")
                
                # Save the results to a CSV file for download
                output_file = os.path.join('artifacts', 'prediction_results.csv')
                os.makedirs('artifacts', exist_ok=True)
                df.to_csv(output_file, index=False)

                # Show first few rows
                return render_template('upload.html', 
                                     prediction_text="File uploaded and processed successfully!", 
                                     tables=[df.head().to_html(classes='table table-striped', header="true")],
                                     show_download=True)
            except Exception as e:
                return render_template('upload.html', prediction_text=f"Error during prediction: {str(e)}")
                
    return render_template('upload.html')

@app.route('/download')
def download_file():
    path = os.path.join('artifacts', 'prediction_results.csv')
    return send_file(path, as_attachment=True, download_name='prediction_results.csv')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
