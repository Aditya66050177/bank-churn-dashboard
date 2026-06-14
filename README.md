# 🏦 Bank Customer Churn Analysis & Prediction

A professional end-to-end data analytics and predictive modeling portfolio project. This repository contains a Jupyter Notebook for Exploratory Data Analysis (EDA) and a premium interactive Streamlit Dashboard with an on-the-fly trained Random Forest Classifier to predict individual customer churn risk.

---

## 🗂️ Project Structure

```text
churn-analysis/
│
├── .python64/          ← Local 64-bit Python environment (redirection via D: junction)
├── data/
│   ├── bank_churn.csv                 ← Main customer dataset (10,000 records)
│   └── bank_churn_data_dictionary.csv  ← Metadata descriptions
├── notebooks/
│   └── analysis.ipynb  ← Exploratory Data Analysis (EDA) Jupyter Notebook
├── app.py              ← Premium Streamlit dashboard & Churn Calculator
├── setup_env.ps1       ← Environment setup script for 64-bit Windows environments
├── requirements.txt    ← Python package dependencies
└── README.md           ← Project Documentation
```

---

## 🚀 Quick Start & Installation

To run this project on a 64-bit Windows environment, a local 64-bit Python setup script has been provided to automatically configure a portable Python runtime and download all required packages (including heavy libraries like `pyarrow` and `scikit-learn`) without altering your global Python configuration.

### 1. Install Dependencies
Run the PowerShell setup script to initialize the 64-bit runtime:
```powershell
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```

### 2. Run the Streamlit Dashboard
Launch the dashboard using the local 64-bit Python interpreter:
```powershell
.python64\Scripts\streamlit.exe run app.py
```

---

## 🔍 Exploratory Data Analysis (EDA)
The analysis notebook (`notebooks/analysis.ipynb`) investigates the core driver patterns behind bank churn. The main findings include:
1. **Age Bracket Risk:** Customers aged 40-60 represent the highest churn risk, with churn rates exceeding **40%**.
2. **Engagement Decay:** Inactive members churn at double the rate of active members (**26.8% vs 14.2%**).
3. **Geographic Risk:** German clients exit at twice the rate of French or Spanish clients (**32.4% vs ~16%**).
4. **Product Clashing:** Customers possessing 3 or 4 products experience a sudden surge in churn (**>80%**), indicating product mismatch or pricing friction.

---

## 📊 Streamlit Dashboard Features

The custom dashboard in `app.py` features:
* **Interactive Filtering:** Sift data by Geography, Gender, Age Range, and Account Balance from the sidebar.
* **Executive Metrics:** View real-time Total Customers, Segment Churn Rate, Revenue at Risk, and Credit Score health.
* **Modern Visualization:** Fully interactive dark-theme charts powered by Plotly (Geo bar chart, Age bracket distributions, Product breakdowns, and Retained vs Churned Box plots).
* **Predictive Customer Sandbox:** Input customized profile parameters (Age, Credit Score, Balance, etc.) to get a real-time risk score calculated via a Random Forest model.
* **Data Exporter:** Generate and download a list of high-risk customers (`ChurnProbability >= 50%`) as a CSV for outreach.

---

## 📋 Actionable Business Strategies

Based on analytics insights, the recommended retention campaigns are:
1. **Targeted VIP Support for 40-60 Group:** Establish customized retirement and wealth planning packages, coupled with prioritized support.
2. **Re-engagement Offers:** Implement automated trigger emails for inactive members offering transaction fee relief or cashback rewards.
3. **Regional Germany Auditing:** Review German market banking regulations and evaluate local competitors to counter market share erosion.
4. **Product Package Restructuring:** Audit and restructure multi-product bundles to replace standalone product fees with integrated bundle rates.
