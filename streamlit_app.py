import streamlit as st
import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
from src.loan_defult_prediction_system.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.loan_defult_prediction_system.utils import load_object

# Set Page Config
st.set_page_config(
    page_title="AI Risk Guard | Loan Default System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Mode & Finance Aesthetic)
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #464b59;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        color: white;
        overflow-wrap: break-word;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #262730;
    }
    /* Titles */
    h1, h2, h3 {
        color: #00ADB5 !important; 
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* Buttons */
    .stButton > button {
        background-color: #00ADB5;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #007f85;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Application Title
st.title("🛡️ AI-Powered Credit Risk Guard")
st.markdown("### *Predictive Intelligence for Reducing Non-Performing Assets (NPAs)*")
st.markdown("---")

# Sidebar - Loan Application Form
st.sidebar.header("📝 New Loan Application")
st.sidebar.markdown("Enter applicant details below:")

def user_input_features():
    col1, col2 = st.sidebar.columns(2)
    with col1:
        annual_income = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0, step=1000.0)
        credit_score = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=700.0)
        loan_amount = st.number_input("Loan Amount ($)", min_value=0.0, value=15000.0, step=500.0)
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.5, step=0.1)
    
    with col2:
        debt_to_income_ratio = st.number_input("DTI Ratio", min_value=0.0, value=15.0, step=0.1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        education_level = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
    
    employment_status = st.sidebar.selectbox("Employment Status", ["Employed", "Unemployed", "Self-Employed", "Retired"])
    loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Personal", "Education", "Home", "Auto", "Debt Consolidation"])
    grade_subgrade = st.sidebar.text_input("Grade/Subgrade (e.g., B2)", value="B2")
    
    data = {
        'annual_income': annual_income,
        'debt_to_income_ratio': debt_to_income_ratio,
        'credit_score': credit_score,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'gender': gender,
        'marital_status': marital_status,
        'education_level': education_level,
        'employment_status': employment_status,
        'loan_purpose': loan_purpose,
        'grade_subgrade': grade_subgrade
    }
    return data

input_data = user_input_features()

# Risk Threshold Slider
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚖️ Risk Strategy")
risk_threshold = st.sidebar.slider("Probability Threshold", 0.0, 1.0, 0.25, 0.05, help="Adjust limit for labelling a loan as 'Risky'. Lower = Safer (Recall Maximization).")

# Prediction Logic
if st.sidebar.button("🚀 Analyze Risk Profile"):
    with st.spinner("Processing Financial Data..."):
        try:
            # Prepare data
            custom_data = CustomData(
                annual_income=input_data['annual_income'],
                debt_to_income_ratio=input_data['debt_to_income_ratio'],
                credit_score=input_data['credit_score'],
                loan_amount=input_data['loan_amount'],
                interest_rate=input_data['interest_rate'],
                gender=input_data['gender'],
                marital_status=input_data['marital_status'],
                education_level=input_data['education_level'],
                employment_status=input_data['employment_status'],
                loan_purpose=input_data['loan_purpose'],
                grade_subgrade=input_data['grade_subgrade']
            )
            
            pred_df = custom_data.get_data_as_data_frame()
            
            # Predict
            predict_pipeline = PredictPipeline()
            # We access the raw model processing to get probabilities directly if needed, but the current pipeline returns probabilities
            results = predict_pipeline.predict(pred_df)
            result = results[0]
            
            # Extract Metrics
            prob_default = result['prob_default'] # Raw probability (0-1)
            prob_paid_back = result['prob_paid_back']
            
            # Custom Threshold Logic
            is_risky = prob_default > risk_threshold
            
            # Dynamic Risk Status based on Slider
            if is_risky:
                risk_label = "HIGH RISK (Red)"
                risk_color = "inverse"
                approval_status = "🛑 REJECTED"
                msg_type = st.error
                bar_color = ":red"
            else:
                # If safe, we can still have a 'warning' zone if close to threshold or just Green
                if prob_default > (risk_threshold * 0.8): # Warn if close to threshold
                    risk_label = "REVIEW (Yellow)"
                    risk_color = "off"
                    approval_status = "⚠️ CAUTION"
                    msg_type = st.warning
                    bar_color = ":orange"
                else:
                    risk_label = "SAFE (Green)"
                    risk_color = "normal"
                    approval_status = "✅ APPROVED"
                    msg_type = st.success
                    bar_color = ":green"
            
            # Display Results in a Professional Dashboard Layout
            
            # Row 1: Key Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(label="Default Probability", value=f"{prob_default*100:.2f}%", delta_color="inverse")
            
            with col2:
                st.metric(label="Risk Category", value=risk_label, delta=approval_status, delta_color=risk_color)
            
            # Safety Margin Calculation
            safety_margin = risk_threshold - prob_default
            safety_label = "Safety Margin" if safety_margin > 0 else "Risk Deficit"
            safety_color = "normal" if safety_margin > 0 else "inverse"
            
            with col3:
                 st.metric(label=safety_label, value=f"{safety_margin*100:.1f}%", help="Difference between your Threshold and the Applicant's Risk. Positive is Safe.")

            # Row 2: Visual Gauge substitute (Progress Bar with Custom Colors)
            st.markdown("### 📊 Risk Assessment Meter")
            
            msg_type(f"**Status: {approval_status}** | Probability: {prob_default*100:.2f}% | Threshold: {risk_threshold:.2f}")
            st.progress(prob_default, text=f"Risk Level {prob_default*100:.1f}%")
            
            # Explanation (Feature Importance)
            st.markdown("---")
            st.subheader("🔍 Decision Transparency (Why this prediction?)")
            try:
                model_path = os.path.join("artifacts", "xgboost_model.pkl")
                model = load_object(model_path)

                # Get feature importances from the model
                if hasattr(model, 'feature_importances_'):
                    # Get top features
                    feature_importance = model.feature_importances_
                    
                    # Create feature names (matching the preprocessing order)
                    numerical_features = ['Annual Income', 'DTI Ratio', 'Credit Score', 'Loan Amount', 
                                        'Interest Rate', 'Income/Loan Ratio', 'Total Debt']
                    
                    # For categorical features, we just show the category names (not all one-hot encoded versions)
                    categorical_features = ['Gender', 'Marital Status', 'Education', 'Employment', 
                                          'Loan Purpose', 'Grade/Subgrade']
                    
                    # Get top numerical feature importances (first 7 features are numerical after scaling)
                    num_importances = feature_importance[:7]
                    
                    # Create a dataframe for visualization
                    importance_df = pd.DataFrame({
                        'Feature': numerical_features,
                        'Importance': num_importances,
                        'Your Value': [
                            f"${input_data['annual_income']:,.0f}",
                            f"{input_data['debt_to_income_ratio']:.1f}%",
                            f"{input_data['credit_score']:.0f}",
                            f"${input_data['loan_amount']:,.0f}",
                            f"{input_data['interest_rate']:.1f}%",
                            f"{input_data['annual_income']/input_data['loan_amount']:.2f}",
                            f"${input_data['debt_to_income_ratio'] * input_data['annual_income']:,.0f}"
                        ]
                    }).sort_values('Importance', ascending=False)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color='#00ADB5')
                    ax.set_xlabel('Feature Importance', fontsize=12, color='white')
                    ax.set_title('Top Features Influencing This Prediction', fontsize=14, color='white', pad=20)
                    ax.tick_params(colors='white')
                    ax.spines['bottom'].set_color('white')
                    ax.spines['left'].set_color('white')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    fig.patch.set_facecolor('#0e1117')
                    ax.set_facecolor('#0e1117')
                    
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    
                    # Show feature values table
                    st.markdown("#### 📋 Your Application Details")
                    st.dataframe(
                        importance_df[['Feature', 'Your Value', 'Importance']].head(5),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.info("📊 **Feature Importance** shows which factors the model considers most important for this prediction. Higher bars = more influential features.")
                else:
                    st.warning("Model explanation not available for this model type.")

            except Exception as e:
                # Fallback for explainability if visualization fails
                st.warning(f"⚠️ Feature importance visualization unavailable.")
                st.info(f"💡 **Key Risk Factors:**\n- DTI Ratio: {input_data['debt_to_income_ratio']:.1f}%\n- Credit Score: {input_data['credit_score']:.0f}\n- Loan Amount: ${input_data['loan_amount']:,.0f}\n- Annual Income: ${input_data['annual_income']:,.0f}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Batch Processing Tab
st.markdown("---")
st.subheader("📂 Batch Processing")
uploaded_file = st.file_uploader("Upload Applicant CSV for Batch Scoring", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:", df.head())
        
        if st.button("Run Batch Prediction"):
            with st.spinner("Scoring Batch..."):
                predict_pipeline = PredictPipeline()
                results = predict_pipeline.predict(df)
                
                # Append results
                df['Default_Probability'] = [r['prob_default'] for r in results]
                # Apply current threshold
                df['Risk_Label'] = df['Default_Probability'].apply(lambda x: "HIGH RISK" if x > risk_threshold else "LOW RISK")
                
                st.dataframe(df)
                
                # Download CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Risk Report",
                    csv,
                    "risk_assessment_report.csv",
                    "text/csv",
                    key='download-csv'
                )
    except Exception as e:
        st.error(f"Error processing file: {e}")
