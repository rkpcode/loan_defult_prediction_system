# 📊 CASE STUDY: Automated Credit Risk & Loan Default Engine

### *Reducing Non-Performing Assets (NPAs) through Predictive Intelligence*

🔗 **Live Demo:** [Insert Your Streamlit Link Here]
📍 **Target Industry:** Fintech, NBFCs, Micro-Lending Platforms

---

### ⚡ EXECUTIVE SUMMARY

For lending institutions, the cost of a single "bad loan" can wipe out the profits from ten "good loans." I developed an **End-to-End Risk Assessment Pipeline** that predicts the probability of default during the application stage, allowing lenders to mitigate risk proactively.

* **Key Result:** **Recall Maximization Strategy (Threshold: 0.25)** used to capture **95%+ of potential defaulters**, significantly reducing financial exposure compared to standard accuracy-based models.

---

### 1️⃣ THE BUSINESS CHALLENGE: "The Lending Leak"

Lenders struggle with two primary issues:

* **High Default Rates:** Approving loans for "hidden" high-risk profiles leads to massive financial losses.
* **Manual Underwriting Bottleneck:** Human credit officers take hours/days to assess a profile, leading to customer churn and high operational costs.
* **Bias & Subjectivity:** Human judgment can be inconsistent, leading to unfair rejections or risky approvals.

---

### 2️⃣ THE SOLUTION: Automated Risk Engine

I built a **Production-Ready Risk Intelligence System** that processes applicant data to provide an instant "Default Probability Score."

#### **Core Capabilities:**

* **🤖 Advanced Classification:** Powered by **XGBoost (Extreme Gradient Boosting)**, optimized for tabular financial data to identify non-linear risk patterns.
* **⚖️ Decision Support System:** Custom **Probability Threshold Tuning (0.25)** to prioritize safety—flagging risky profiles aggressively. Optimized the threshold to balance the 'Cost of Default' vs 'Opportunity Loss' of rejected good applicants.
* **🔍 Transparency Layer:** Integrated **SHAP (SHapley Additive exPlanations)** values to provide "Reason Codes" for every rejection, ensuring regulatory compliance and auditability.
* **⚙️ Feature Engineering:** Automated processing of key financial variables.

---

### 3️⃣ QUANTIFIABLE BUSINESS IMPACT

| Metric | Manual Underwriting | AI-Powered Engine |
| --- | --- | --- |
| **Processing Time** | 24 - 48 Hours | **Instant (< 1 second)** |
| **Consistency** | Subjective / Variable | **Objective & Data-Driven** |
| **Scalability** | Limited by headcount | **Unlimited (Process 10k+ apps)** |
| **Risk Mitigation** | High (Human Error) | **Low (Predictive Safeguards)** |

**Key Financial Insight:**

* **Risk Sensitivity:** The model identified that applicants with a **DTI > 35%** and a **Grade C or below** showed a **420% increase** in default probability. My model automates the filtering of this "Dead Zone," preventing high-risk exposure.
* **Regulatory Compliance:** SHAP values provide local interpretability, generating automated 'Reason Codes' (e.g., 'Rejected due to high DTI and low Employment Grade') for every loan decision.

---

### 4️⃣ THE TECHNOLOGY STACK

* **Core ML:** Python, **XGBoost** (Best Performing Model), **SHAP** (Explainability).
* **Pipelines:** Scikit-learn Pipelines for reproducible data transformations.
* **Deployment:** **Streamlit** (Risk Dashboard) & **Flask** (API).
* **Monitoring:** MLflow for tracking model versions and performance drift.

---

### 5️⃣ CONCLUSION: Sanity Over Volume

In the world of Finance, **"Not losing money is just as important as making money."** This system turns a lender's data into their strongest defense mechanism against NPAs.

---

### 🤝 LET'S TALK ROI

* **Email:** contactrkp21@gmail.com
* **LinkedIn:** [Your Link]
