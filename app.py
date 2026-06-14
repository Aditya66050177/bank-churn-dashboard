import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Set page config for premium look
st.set_page_config(
    page_title="Bank Customer Churn Portal",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern glassmorphic UI, responsive grids, and premium feel
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #21262d;
    }
    
    /* Card design */
    .metric-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 15px;
    }
    
    .metric-container:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
    }
    
    .metric-label {
        font-size: 14px;
        color: #8b949e;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 8px;
    }
    
    .metric-val {
        font-size: 36px;
        font-weight: 700;
        color: #f0f6fc;
        line-height: 1.2;
    }
    
    .metric-sub {
        font-size: 12px;
        color: #58a6ff;
        margin-top: 6px;
        font-weight: 400;
    }
    
    .title-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #0d9488 100%);
        border-radius: 20px;
        padding: 40px 30px;
        margin-bottom: 35px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        text-align: left;
        color: white;
    }
    
    .title-banner h1 {
        font-size: 42px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        color: white !important;
    }
    
    .title-banner p {
        font-size: 18px;
        opacity: 0.9;
        margin: 10px 0 0 0;
        font-weight: 300;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('data/bank_churn.csv')
    # Save a raw copy for modeling/predictions
    df_raw = df.copy()
    # Preprocess
    df['AgeGroup'] = pd.cut(
        df['Age'], 
        bins=[0, 30, 40, 50, 60, 100],
        labels=['<30', '30-40', '40-50', '50-60', '60+']
    )
    # Calculate Customer Lifetime Value (CLV)
    # CLV = (Balance * 3% net interest margin) * Tenure (years)
    df['CLV'] = (df['Balance'] * 0.03) * df['Tenure']
    return df, df_raw

df, df_raw = load_data()

# ----------------- MODEL TRAINING -----------------
@st.cache_resource
def train_model(data):
    X = data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1, errors='ignore')
    y = data['Exited']
    
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    cat_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    
    pipeline.fit(X, y)
    return pipeline

# Train classifier
model = train_model(df_raw)

# ----------------- SIDEBAR FILTERS -----------------
st.sidebar.image("https://img.icons8.com/isometric/100/bank.png", width=80)
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("Filter customer segment or run scenarios.")

# Filter by Geography
geographies = df['Geography'].unique().tolist()
selected_geo = st.sidebar.multiselect("Geography", geographies, default=geographies)

# Filter by Gender
genders = df['Gender'].unique().tolist()
selected_gender = st.sidebar.multiselect("Gender", genders, default=genders)

# Filter by Age Range
min_age = int(df['Age'].min())
max_age = int(df['Age'].max())
selected_age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))

# Filter by Balance Range
min_balance = float(df['Balance'].min())
max_balance = float(df['Balance'].max())
selected_balance_range = st.sidebar.slider(
    "Balance Range ($)", 
    min_balance, 
    max_balance, 
    (min_balance, max_balance),
    format="%d"
)

# Apply filters
filtered_df = df[
    (df['Geography'].isin(selected_geo)) &
    (df['Gender'].isin(selected_gender)) &
    (df['Age'] >= selected_age_range[0]) &
    (df['Age'] <= selected_age_range[1]) &
    (df['Balance'] >= selected_balance_range[0]) &
    (df['Balance'] <= selected_balance_range[1])
].copy()

# ----------------- HEADER BANNER -----------------
st.markdown("""
<div class="title-banner">
    <h1>🏦 Bank Customer Churn Portal</h1>
    <p>Advanced predictive analytics & customer retention dashboard for banking intelligence.</p>
</div>
""", unsafe_allow_html=True)

# ----------------- TABS SYSTEM -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Insights", 
    "🔍 Demographic & Product Analysis", 
    "🤖 Churn Risk Calculator",
    "📋 Business Recommendations"
])

# ----------------- TAB 1: EXECUTIVE INSIGHTS -----------------
with tab1:
    if len(filtered_df) == 0:
        st.warning("No data matches the selected filters. Please adjust the sidebar controls.")
    else:
        # KPIs Grid (2 Rows x 3 Columns)
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        
        total_customers = len(filtered_df)
        churn_count = len(filtered_df[filtered_df['Exited'] == 1])
        churn_rate = (churn_count / total_customers) * 100 if total_customers > 0 else 0
        revenue_at_risk = filtered_df[filtered_df['Exited'] == 1]['Balance'].sum()
        avg_credit_score = filtered_df['CreditScore'].mean()
        
        # Calculate CLV metrics
        retained_segment = filtered_df[filtered_df['Exited'] == 0]
        churned_segment = filtered_df[filtered_df['Exited'] == 1]
        
        avg_clv_retained = retained_segment['CLV'].mean() if len(retained_segment) > 0 else 0
        total_clv_lost = churned_segment['CLV'].sum()
        
        # Row 1: Segment Health Overview
        with row1_col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Total Customers</div>
                <div class="metric-val">{total_customers:,}</div>
                <div class="metric-sub">Active in selected segment</div>
            </div>
            """, unsafe_allow_html=True)
            
        with row1_col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Churn Rate</div>
                <div class="metric-val" style="color: #ff6b6b;">{churn_rate:.1f}%</div>
                <div class="metric-sub">{churn_count:,} customers exited</div>
            </div>
            """, unsafe_allow_html=True)
            
        with row1_col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Avg Credit Score</div>
                <div class="metric-val">{avg_credit_score:.0f}</div>
                <div class="metric-sub">Customer financial health</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Row 2: Financial Impact Metrics
        with row2_col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Revenue at Risk</div>
                <div class="metric-val" style="color: #ffd166;">${revenue_at_risk:,.0f}</div>
                <div class="metric-sub">Balance from churned segment</div>
            </div>
            """, unsafe_allow_html=True)
            
        with row2_col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Avg CLV (Retained)</div>
                <div class="metric-val" style="color: #00b4d8;">${avg_clv_retained:,.0f}</div>
                <div class="metric-sub">Avg Lifetime Value of active base</div>
            </div>
            """, unsafe_allow_html=True)
            
        with row2_col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Total CLV Lost</div>
                <div class="metric-val" style="color: #f25c54;">${total_clv_lost:,.0f}</div>
                <div class="metric-sub">Lifetime Value lost to churn</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        st.write("")
        
        # Primary Executive Charts
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Country Churn Chart
            geo_churn = filtered_df.groupby('Geography', observed=False)['Exited'].mean().reset_index()
            geo_churn['Churn Rate (%)'] = geo_churn['Exited'] * 100
            
            fig_geo = px.bar(
                geo_churn,
                x='Geography',
                y='Churn Rate (%)',
                title='<b>Churn Rate by Country</b>',
                color='Churn Rate (%)',
                color_continuous_scale='Reds',
                labels={'Geography': 'Country', 'Churn Rate (%)': 'Churn Rate (%)'},
                template='plotly_dark'
            )
            fig_geo.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=50, b=40),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_geo, width="stretch")
            
        with col_right:
            # Age Group Churn Chart
            age_churn = filtered_df.groupby('AgeGroup', observed=False)['Exited'].mean().reset_index()
            age_churn['Churn Rate (%)'] = age_churn['Exited'] * 100
            
            fig_age = px.bar(
                age_churn,
                x='AgeGroup',
                y='Churn Rate (%)',
                title='<b>Churn Rate by Age Group</b>',
                color='Churn Rate (%)',
                color_continuous_scale='Reds',
                labels={'AgeGroup': 'Age Bracket', 'Churn Rate (%)': 'Churn Rate (%)'},
                template='plotly_dark'
            )
            fig_age.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=50, b=40),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_age, width="stretch")

        # Bottom Row Charts
        col_b1, col_b2 = st.columns(2)
        
        with col_b1:
            # Product Churn
            prod_churn = filtered_df.groupby('NumOfProducts', observed=False)['Exited'].mean().reset_index()
            prod_churn['Churn Rate (%)'] = prod_churn['Exited'] * 100
            
            fig_prod = px.bar(
                prod_churn,
                x='NumOfProducts',
                y='Churn Rate (%)',
                title='<b>Churn Rate by Number of Products</b>',
                color='Churn Rate (%)',
                color_continuous_scale='Oranges',
                labels={'NumOfProducts': 'Number of Products Offered', 'Churn Rate (%)': 'Churn Rate (%)'},
                template='plotly_dark'
            )
            fig_prod.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=50, b=40),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_prod, width="stretch")
            
        with col_b2:
            # Balance Distribution
            filtered_df['Status'] = filtered_df['Exited'].map({0: 'Retained', 1: 'Churned'})
            fig_box = px.box(
                filtered_df,
                x='Status',
                y='Balance',
                title='<b>Balance Distribution: Retained vs Churned</b>',
                color='Status',
                color_discrete_map={'Retained': '#00b4d8', 'Churned': '#ef476f'},
                template='plotly_dark'
            )
            fig_box.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=50, b=40),
                showlegend=False
            )
            st.plotly_chart(fig_box, width="stretch")

# ----------------- TAB 2: DEMOGRAPHIC & PRODUCT ANALYSIS -----------------
with tab2:
    if len(filtered_df) == 0:
        st.warning("No data matches the selected filters.")
    else:
        st.subheader("Deep-Dive Insights")
        
        col_t2_1, col_t2_2 = st.columns(2)
        
        with col_t2_1:
            # Active Member Churn Chart
            active_churn = filtered_df.groupby('IsActiveMember', observed=False)['Exited'].mean().reset_index()
            active_churn['Status'] = active_churn['IsActiveMember'].map({0: 'Inactive Member', 1: 'Active Member'})
            active_churn['Churn Rate (%)'] = active_churn['Exited'] * 100
            
            fig_active = px.bar(
                active_churn,
                x='Status',
                y='Churn Rate (%)',
                title='<b>Churn Rate: Active vs Inactive Members</b>',
                color='Status',
                color_discrete_map={'Inactive Member': '#e76f51', 'Active Member': '#2a9d8f'},
                template='plotly_dark'
            )
            fig_active.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=50, b=40),
                showlegend=False
            )
            st.plotly_chart(fig_active, width="stretch")
            
        with col_t2_2:
            # Credit Score vs Balance scatter
            fig_scatter = px.scatter(
                filtered_df,
                x='CreditScore',
                y='Balance',
                color='Status',
                title='<b>Credit Score vs Balance Scatter Plot</b>',
                color_discrete_map={'Retained': '#00b4d8', 'Churned': '#ef476f'},
                opacity=0.6,
                hover_data=['Age', 'NumOfProducts'],
                template='plotly_dark'
            )
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig_scatter, width="stretch")
            
        # Segment Matrix table
        st.subheader("💡 Segment Matrix Summary")
        summary_table = filtered_df.groupby(['Geography', 'Gender'], observed=False).agg(
            Total_Customers=('CreditScore', 'count'),
            Average_Balance=('Balance', 'mean'),
            Average_CreditScore=('CreditScore', 'mean'),
            Churn_Rate_Pct=('Exited', lambda x: x.mean() * 100)
        ).reset_index()
        
        # Round values for nice output
        summary_table['Average_Balance'] = summary_table['Average_Balance'].map('${:,.2f}'.format)
        summary_table['Average_CreditScore'] = summary_table['Average_CreditScore'].map('{:.1f}'.format)
        summary_table['Churn_Rate_Pct'] = summary_table['Churn_Rate_Pct'].map('{:.2f}%'.format)
        summary_table.columns = ['Geography', 'Gender', 'Total Customers', 'Avg Balance', 'Avg Credit Score', 'Churn Rate (%)']
        
        st.dataframe(summary_table, width="stretch")

# ----------------- TAB 3: CHURN RISK CALCULATOR -----------------
with tab3:
    st.subheader("🔮 Predictive Customer Sandbox")
    st.markdown("Enter hypothetical customer traits to evaluate their probability of churning based on a Random Forest Classifier trained on historic data.")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        age_in = st.slider("Customer Age", 18, 100, 35)
        geo_in = st.selectbox("Geography", df_raw['Geography'].unique())
        gender_in = st.selectbox("Gender", df_raw['Gender'].unique())
        
    with col_input2:
        products_in = st.slider("Number of Products", 1, 4, 1)
        balance_in = st.number_input("Account Balance ($)", min_value=0.0, max_value=250000.0, value=50000.0, step=1000.0)
        salary_in = st.number_input("Estimated Annual Salary ($)", min_value=0.0, max_value=250000.0, value=85000.0, step=1000.0)
        
    with col_input3:
        credit_in = st.slider("Credit Score", 350, 850, 650)
        tenure_in = st.slider("Tenure (Years)", 0, 10, 5)
        active_in = st.radio("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        cc_in = st.radio("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    # Predict Button
    st.write("")
    if st.button("Calculate Churn Risk", type="primary"):
        # Make DataFrame
        input_data = pd.DataFrame([{
            'CreditScore': credit_in,
            'Geography': geo_in,
            'Gender': gender_in,
            'Age': age_in,
            'Tenure': tenure_in,
            'Balance': balance_in,
            'NumOfProducts': products_in,
            'HasCrCard': cc_in,
            'IsActiveMember': active_in,
            'EstimatedSalary': salary_in
        }])
        
        # Predict probability
        prob = model.predict_proba(input_data)[0][1] * 100
        
        st.divider()
        
        # Show outcome
        c_prob, c_status = st.columns(2)
        with c_prob:
            st.metric("Churn Probability", f"{prob:.1f}%")
            
        with c_status:
            if prob < 30:
                st.success("🟢 LOW RISK - The customer is highly likely to stay with the bank.")
            elif prob < 60:
                st.warning("🟡 MEDIUM RISK - The customer shows early warning signs. Targeted outreach recommended.")
            else:
                st.error("🔴 HIGH RISK - High likelihood of churn. Immediate retention action required!")
                
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "<b>Risk Score</b>", 'font': {'size': 20, 'color': '#ffffff'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#ffffff"},
                'bar': {'color': "#ff4b4b" if prob >= 60 else "#ffa500" if prob >= 30 else "#00cc96"},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 2,
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(0, 204, 150, 0.15)'},
                    {'range': [30, 60], 'color': 'rgba(255, 165, 0, 0.15)'},
                    {'range': [60, 100], 'color': 'rgba(255, 75, 75, 0.15)'}
                ],
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#ffffff"},
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_gauge, width="stretch")

# ----------------- TAB 4: BUSINESS RECOMMENDATIONS -----------------
with tab4:
    st.subheader("📋 Actionable Retention Playbook")
    
    st.markdown("""
    Based on the exploratory data analysis and our predictive modeling, these are the **four key high-impact customer retention strategies**:
    """)
    
    st.info("""
    ### 🎯 1. Target the 40-60 Age Bracket (High Value, High Risk)
    * **The Insight:** Customers aged 40 to 60 have a churn rate exceeding **40%**, representing the highest risk segment.
    * **Actionable Plan:** 
      - Launch tailored wealth management, retirement planning, or transition products.
      - Establish a "VIP Care Team" with dedicated relationship managers for older customers with high balances.
      - Offer loyalty points, family bundle benefits, or preferential loan/interest rates.
    """)
    
    st.warning("""
    ### 🔄 2. Re-Engage Inactive Members
    * **The Insight:** Inactive members churn at twice the rate of active members (**26.8% vs 14.2%**).
    * **Actionable Plan:**
      - Set up automated trigger-based activation email and SMS campaigns.
      - Offer temporary high-yield cash back, discount coupons, or zero transaction fees for restarting account activity.
      - Conduct brief customer experience surveys to identify why they became inactive.
    """)
    
    st.error("""
    ### 🇩🇪 3. Local Competitor Defense in Germany
    * **The Insight:** Germany accounts for the highest geographic churn rate (**32.4%** compared to France's 16.2% and Spain's 16.7%).
    * **Actionable Plan:**
      - Perform competitor mapping in the German region to evaluate if local competitors offer better terms.
      - Localize banking support and marketing campaigns for the German market.
      - Re-examine fee structures and local regulations impacting banking experience in Germany.
    """)
    
    st.success("""
    ### 📦 4. Review Multi-Product Bundles
    * **The Insight:** Customers with 3 or 4 products have churn rates above **80%**, while those with 1 or 2 products stay longer. This suggests high-product customers are experiencing product mismatch or feeling overcharged.
    * **Actionable Plan:**
      - Audit customer onboarding and service quality for multi-product users.
      - Offer customized packages (e.g. flat monthly fee instead of single fees for multiple products).
      - Establish active check-ins for customers holding 3+ products to resolve friction points.
    """)
    
    st.divider()
    
    # High-Risk Customer List Exporter
    st.subheader("📥 Export At-Risk Customers List")
    st.markdown("Filter to download details of customers in the current active segment who are predicted as highly likely to churn.")
    
    # Calculate risks for entire dataset
    if st.button("Generate At-Risk Customers List"):
        with st.spinner("Calculating individual probabilities..."):
            features_raw = df_raw.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1, errors='ignore')
            probs = model.predict_proba(features_raw)[:, 1]
            df_export = df_raw.copy()
            df_export['ChurnProbability'] = probs
            
            # Filter to high risk customers (>50% probability) and select columns
            high_risk_df = df_export[df_export['ChurnProbability'] >= 0.50].sort_values(by='ChurnProbability', ascending=False)
            high_risk_df = high_risk_df[['CustomerId', 'Surname', 'Geography', 'Gender', 'Age', 'Balance', 'NumOfProducts', 'ChurnProbability']]
            
            csv = high_risk_df.to_csv(index=False).encode('utf-8')
            
            st.success(f"Successfully identified {len(high_risk_df):,} customers at high risk of churn (>=50% probability).")
            
            st.download_button(
                label="📥 Download CSV for Customer Outreach",
                data=csv,
                file_name='high_risk_churn_customers.csv',
                mime='text/csv'
            )
            st.dataframe(high_risk_df.head(20), width="stretch")
