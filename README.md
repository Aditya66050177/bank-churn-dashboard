# Bank Customer Churn Analysis Dashboard
 
A Business Analyst & APM portfolio project analyzing customer churn patterns in retail banking — combining data analysis, ML-based risk prediction, and actionable business recommendations.
 
**Live App:** [bank-churn-dashboard.streamlit.app](https://bank-churn-dashboard-sgwst7xgjvzd84y7jkazc4.streamlit.app/)
 
---
 
## Business Problem
 
A retail bank is experiencing a **20.4% customer churn rate**, resulting in significant revenue loss through lost balances and reduced lifetime value. This project identifies *who* is churning, *why* they are churning, and *what the business should do* about it.
 
---
 
## Project Objectives
 
- Quantify churn rate and revenue impact across customer segments
- Identify high-risk cohorts by geography, age, product usage, and engagement
- Build a real-time churn risk calculator to flag individual at-risk customers
- Translate findings into actionable business recommendations
---
 
## Dashboard Structure
 
### Tab 1 — Executive Insights
High-level KPIs for leadership and stakeholders:
| Metric | Description |
|---|---|
| Total Customers | Count of customers in selected segment |
| Churn Rate (%) | % of customers who have left the bank |
| Revenue at Risk ($) | Total balance lost to churned customers |
| Avg Credit Score | Financial health indicator of the segment |
 
### Tab 2 — Segment Matrix
Deep-dive cohort analysis across demographics:
- Active vs. Inactive member churn comparison
- Segment breakdown by Geography × Gender
- Avg Balance, Avg Credit Score, and Churn Rate per cohort
### Tab 3 — Churn Risk Calculator
On-the-fly ML prediction using a **Random Forest model**:
- Input any customer profile and get an instant churn probability
- Risk categorization: Low (<30%) | Medium (30–60%) | High (>60%)
---
 
## Key Findings
 
- **Germany** has the highest churn rate (~32%) vs France and Spain (~16%)
- **Age group 40–60** churns significantly more than younger customers
- **Inactive members** are 2× more likely to churn than active ones
- Customers with **3–4 products** show unexpectedly high churn — suggesting product-fit issues
- High-balance customers represent disproportionate **revenue at risk**
---
 
## Business Recommendations
 
1. **Targeted retention campaigns** for the 40–60 age group with personalized offers
2. **Re-engagement program** for inactive members — triggered email/SMS within 30 days of inactivity
3. **Germany market review** — investigate local competitive landscape and pricing
4. **Product bundling audit** — review why multi-product customers churn more; assess onboarding quality
---
 
## Tech Stack
 
| Tool | Purpose |
|---|---|
| Python | Data analysis and ML modeling |
| Pandas | Data manipulation |
| Plotly | Interactive visualizations |
| Scikit-learn | Random Forest churn prediction model |
| Streamlit | Dashboard deployment |
 
---
 
## Project Structure
 
```
bank-churn-dashboard/
│
├── data/
│   └── bank_churn.csv          # Source dataset (Maven Analytics)
├── app.py                       # Main Streamlit dashboard
├── requirements.txt             # Python dependencies
└── README.md
```
 
---
 
## BA/APM Documentation
 
This project is supported by full business documentation:
 
- **BRD (Business Requirements Document)** — stakeholder requirements, AS-IS/TO-BE process, functional requirements
- **PRD (Product Requirements Document)** — feature specs, user stories, success metrics
*(Available on request)*
 
---
 
## Dataset
 
- **Source:** [Maven Analytics — Bank Customer Churn](https://mavenanalytics.io/data-playground/bank-customer-churn)
- **Size:** 10,000 customers
- **Features:** Credit Score, Geography, Age, Tenure, Balance, NumOfProducts, IsActiveMember, EstimatedSalary, Exited
---
 
## Author
 
**Aditya** — 3rd Year CSE Student | Aspiring APM & Business Analyst
- [GitHub](https://github.com/Aditya66050177)
- [LinkedIn](https://www.linkedin.com/in/aditya54/)
