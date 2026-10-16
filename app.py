"""
=============================================================================
CustomerIQ - Live Interactive Web Application (Streamlit)
=============================================================================
Enterprise Customer Intelligence & AI Churn Prediction Engine
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="CustomerIQ | Enterprise AI Churn Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .kpi-card {
        background: #111827;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #1f2937;
        margin-bottom: 12px;
    }
    .kpi-title { font-size: 13px; color: #9ca3af; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
    .kpi-val { font-size: 26px; font-weight: 800; color: #ffffff; margin-top: 4px; }
    .kpi-sub { font-size: 12px; color: #38bdf8; font-weight: 500; margin-top: 4px; }
    .risk-high { color: #ef4444; font-weight: bold; }
    .risk-low { color: #10b981; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Helper function to load datasets & model
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "data", "customer_intelligence_final_with_predictions.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(base_dir, "data", "customer_intelligence_processed.csv")
    df = pd.read_csv(data_path)
    return df

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "models", "churn_prediction_pipeline.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

df = load_data()
model = load_model()

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/shopping-cart-loaded.png", width=64)
st.sidebar.title("CustomerIQ Engine")
st.sidebar.markdown("**Enterprise Retail & Quick-Commerce Intelligence**")
st.sidebar.divider()

st.sidebar.metric("Evaluated Customers", f"{len(df):,}")
st.sidebar.metric("Gross Merchandise Value", f"₹{df['Total_Spend'].sum()/10000000:.2f} Cr")
st.sidebar.metric("Baseline Churn Rate", f"{df['Is_Churned'].mean()*100:.1f}%")

st.sidebar.divider()
st.sidebar.info("💡 **Tip:** Use Tab 3 to simulate real-time AI Churn Risk for any customer profile!")

# Main Tabs
tabs = st.tabs([
    "📊 Executive Dashboard", 
    "👥 RFM Cohort Explorer", 
    "🤖 Live AI Churn Simulator", 
    "📈 Model Explainability & Proof"
])

# -----------------------------------------------------------------------------
# TAB 1: Executive Dashboard
# -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("🛍️ E-Commerce Commercial & Retention Overview")
    
    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #3b82f6;">
            <div class="kpi-title">Total Gross Revenue</div>
            <div class="kpi-val">₹{df['Total_Spend'].sum()/10000000:.2f} Cr</div>
            <div class="kpi-sub">Across 10,000 Orders</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #6366f1;">
            <div class="kpi-title">Total Customers</div>
            <div class="kpi-val">{len(df):,}</div>
            <div class="kpi-sub">12 Indian Metros</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #a855f7;">
            <div class="kpi-title">Avg Order Value (AOV)</div>
            <div class="kpi-val">₹{df['Avg_Order_Value'].mean():,.2f}</div>
            <div class="kpi-sub">Basket Size</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #f59e0b;">
            <div class="kpi-title">Platform Churn Rate</div>
            <div class="kpi-val" style="color:#f59e0b;">{df['Is_Churned'].mean()*100:.1f}%</div>
            <div class="kpi-sub">{df['Is_Churned'].sum():,} Churned Users</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        revenue_at_risk = df[df['Is_Churned'] == 1]['Total_Spend'].sum()
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #ef4444;">
            <div class="kpi-title">Revenue at High Risk</div>
            <div class="kpi-val" style="color:#ef4444;">₹{revenue_at_risk/10000000:.2f} Cr</div>
            <div class="kpi-sub">Retention Target</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Interactive Visuals Row
    col_chart1, col_chart2 = st.columns([1, 1.3])
    
    with col_chart1:
        st.markdown("##### 📦 Revenue by Favorite Category")
        cat_df = df.groupby('Favorite_Category')['Total_Spend'].sum().reset_index()
        fig_cat = px.pie(
            cat_df, names='Favorite_Category', values='Total_Spend',
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_cat.update_layout(margin=dict(t=20, b=20, l=10, r=10), template="plotly_dark")
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_chart2:
        st.markdown("##### 👥 Customer Cohorts (RFM Segments) & Spend")
        seg_df = df.groupby('Customer_Segment').agg(
            Count=('Customer_ID', 'count'),
            Revenue=('Total_Spend', 'sum')
        ).reset_index().sort_values(by='Revenue', ascending=True)

        fig_seg = px.bar(
            seg_df, y='Customer_Segment', x='Revenue', orientation='h',
            color='Customer_Segment',
            labels={'Revenue': 'Total Revenue (INR)', 'Customer_Segment': 'Cohort'},
            color_discrete_sequence=px.colors.sequential.Tealgrn
        )
        fig_seg.update_layout(showlegend=False, margin=dict(t=20, b=20, l=10, r=10), template="plotly_dark")
        st.plotly_chart(fig_seg, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: RFM Cohort Explorer
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("🔍 Deep-Dive Customer Segment & Filter Matrix")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        selected_seg = st.multiselect("Filter by Segment:", options=df['Customer_Segment'].unique(), default=df['Customer_Segment'].unique())
    with col_f2:
        selected_tier = st.multiselect("Filter by City Tier:", options=df['City_Tier'].unique(), default=df['City_Tier'].unique())
    with col_f3:
        risk_filter = st.selectbox("Filter by Churn Status:", ["All Customers", "Churned Only (1)", "Active Only (0)"])

    filtered_df = df[
        (df['Customer_Segment'].isin(selected_seg)) & 
        (df['City_Tier'].isin(selected_tier))
    ]
    if risk_filter == "Churned Only (1)":
        filtered_df = filtered_df[filtered_df['Is_Churned'] == 1]
    elif risk_filter == "Active Only (0)":
        filtered_df = filtered_df[filtered_df['Is_Churned'] == 0]

    st.markdown(f"**Showing {len(filtered_df):,} customers matching criteria:**")
    st.dataframe(
        filtered_df[[
            'Customer_ID', 'City', 'City_Tier', 'Customer_Segment', 
            'Total_Orders', 'Total_Spend', 'Recency_Days', 'Avg_Satisfaction_Score', 
            'Total_Complaints', 'Is_Churned'
        ]].head(100),
        use_container_width=True
    )

    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export Filtered Cohort CSV", data=csv_data, file_name="customeriq_filtered_cohort.csv", mime="text/csv")

# -----------------------------------------------------------------------------
# TAB 3: Live AI Churn Simulator
# -----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("🤖 Real-Time Customer Churn Risk Simulator")
    st.markdown("Adjust customer attributes below to test the **Machine Learning Pipeline** in real-time:")

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    
    with sim_col1:
        st.markdown("**👤 Demographics & Profile**")
        sim_age = st.slider("Customer Age", 18, 65, 32)
        sim_gender = st.selectbox("Gender", ["Female", "Male", "Non-Binary"])
        sim_tier = st.selectbox("City Tier", ["Tier-1", "Tier-2"])
        sim_tenure = st.slider("Tenure with Platform (Months)", 1, 36, 12)

    with sim_col2:
        st.markdown("**🛒 Shopping Behavior**")
        sim_orders = st.slider("Total Orders Lifetime", 1, 15, 4)
        sim_spend = st.number_input("Total Spend (INR)", value=18500.0, step=500.0)
        sim_aov = sim_spend / sim_orders
        sim_recency = st.slider("Recency (Days Since Last Order)", 1, 300, 45)
        sim_category = st.selectbox("Favorite Category", ["Electronics", "Fashion & Apparel", "Home & Kitchen", "Beauty & Personal Care", "Grocery & Essentials"])

    with sim_col3:
        st.markdown("**⚡ Service Experience & Quality**")
        sim_delays = st.slider("Total Delivery Delay Days", 0, 15, 1)
        sim_complaints = st.slider("Complaints Filed", 0, 5, 0)
        sim_satisfaction = st.slider("Average Satisfaction Score (1-5)", 1.0, 5.0, 4.2, step=0.1)
        sim_payment = st.selectbox("Preferred Payment", ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash on Delivery"])

    st.markdown("---")
    
    if st.button("🚀 Predict Customer Churn Risk", type="primary", use_container_width=True):
        if model is None:
            st.error("Model pipeline not loaded. Please ensure 03_train_ml_churn_model.py has been executed.")
        else:
            # Construct input sample dataframe
            input_data = pd.DataFrame([{
                'Customer_Age': sim_age,
                'Customer_Tenure_Months': sim_tenure,
                'Total_Orders': sim_orders,
                'Total_Spend': sim_spend,
                'Avg_Order_Value': sim_aov,
                'Avg_Discount_Percent': 15.0,
                'Total_Delivery_Delay_Days': sim_delays,
                'Avg_Delivery_Delay_Days': sim_delays / max(sim_orders, 1),
                'Total_Complaints': sim_complaints,
                'Avg_Satisfaction_Score': sim_satisfaction,
                'Recency_Days': sim_recency,
                'City_Tier': sim_tier,
                'Customer_Gender': sim_gender,
                'Favorite_Category': sim_category,
                'Preferred_Payment': sim_payment
            }])

            prob = model.predict_proba(input_data)[0, 1]
            risk_percent = prob * 100

            res_c1, res_c2 = st.columns([1, 1.5])
            
            with res_c1:
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_percent,
                    title={'text': "Churn Probability (%)", 'font': {'size': 20, 'color': "white"}},
                    number={'suffix': "%", 'font': {'color': "white"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': "white"},
                        'bar': {'color': "#ef4444" if risk_percent >= 50 else "#10b981"},
                        'steps': [
                            {'range': [0, 35], 'color': "rgba(16, 185, 129, 0.25)"},
                            {'range': [35, 70], 'color': "rgba(245, 158, 11, 0.25)"},
                            {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.35)"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig_gauge.update_layout(height=260, margin=dict(t=30, b=10, l=20, r=20), template="plotly_dark")
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_c2:
                if risk_percent >= 70:
                    st.error(f"🚨 **HIGH RISK CUSTOMER DETECTED ({risk_percent:.1f}%)**")
                    st.markdown("""
                    **Immediate Retention Action Plan:**
                    - 🎯 **Automated Trigger:** Dispatch a personalized ₹500 instant discount coupon via WhatsApp/SMS.
                    - 📞 **Concierge Support:** Route next query directly to Senior Support Executive.
                    - ⚡ **Priority Fulfillment:** Tag customer for zero-delay express shipping for the next 3 orders.
                    """)
                elif risk_percent >= 35:
                    st.warning(f"⚠️ **MEDIUM RISK / WATCHLIST ({risk_percent:.1f}%)**")
                    st.markdown("""
                    **Nudge Strategy:**
                    - 📬 Send customized product recommendations based on their favorite category.
                    - ⭐ Request feedback regarding their recent delivery experience.
                    """)
                else:
                    st.success(f"✅ **HEALTHY ACTIVE CUSTOMER ({risk_percent:.1f}%)**")
                    st.markdown("""
                    **Growth Strategy:**
                    - 🏆 Customer exhibits strong platform loyalty and healthy engagement.
                    - 💎 Eligible for VIP Rewards and early access to festival sales.
                    """)

# -----------------------------------------------------------------------------
# TAB 4: Model Explainability & Proof
# -----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("📈 Machine Learning Architecture & Benchmark Proof")
    st.markdown("Detailed verification evidence for technical interviewers and recruiters:")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("##### 🏆 Model Evaluation Metrics")
        metrics_summary = pd.DataFrame({
            "Classifier Model": ["Logistic Regression", "Random Forest", "Gradient Boosting (Production)"],
            "Accuracy": ["96.55%", "98.38%", "100.00%"],
            "Precision": ["96.99%", "100.00%", "100.00%"],
            "Recall": ["93.06%", "95.38%", "100.00%"],
            "ROC-AUC": ["0.9938", "0.9992", "1.0000"]
        })
        st.dataframe(metrics_summary, use_container_width=True)

    with col_m2:
        st.markdown("##### 🎯 Top Churn Predictive Drivers")
        st.info("""
        1. **Recency Days:** Inactive days is the strongest indicator of churn dropoff.
        2. **Average Satisfaction Score:** Ratings < 3.0 trigger exponential churn risk.
        3. **Total Delivery Delays:** Fulfillment latency directly hurts repeat purchase rate.
        4. **Total Complaints:** Unresolved complaints degrade lifetime value.
        """)

    st.markdown("---")
    img_col1, img_col2, img_col3 = st.columns(3)
    base_dir = os.path.dirname(__file__)
    
    p1 = os.path.join(base_dir, "outputs", "04_confusion_matrix.png")
    p2 = os.path.join(base_dir, "outputs", "05_roc_auc_curve.png")
    p3 = os.path.join(base_dir, "outputs", "06_feature_importance.png")
    
    if os.path.exists(p1):
        with img_col1:
            st.image(p1, caption="Confusion Matrix (Test Set)")
    if os.path.exists(p2):
        with img_col2:
            st.image(p2, caption="ROC-AUC Curve")
    if os.path.exists(p3):
        with img_col3:
            st.image(p3, caption="Feature Importance")
