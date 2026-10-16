"""
=============================================================================
CustomerIQ - Script 02: Exploratory Data Analysis & RFM Segmentation
=============================================================================
This script performs:
1. Transaction-level EDA & business metrics extraction.
2. Aggregation to Customer-level profile.
3. Industry-standard RFM (Recency, Frequency, Monetary) segmentation.
4. Ground-truth Churn labeling based on behavioral recency & service quality.
5. Saves clean processed dataset for Machine Learning (Day 2) and Power BI (Day 3).
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Windows console encoding safeguard
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Configure clean aesthetic for generated plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Segoe UI'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def run_analysis_and_rfm():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "..", "data")
    output_dir = os.path.join(base_dir, "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    raw_file = os.path.join(data_dir, "ecommerce_transactions_raw.csv")
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Raw data file not found at: {raw_file}")

    print("=" * 70)
    print("[CustomerIQ Day 1] Step 3 & 4: Data Analysis & RFM Segmentation")
    print("=" * 70)

    # 1. Load Data
    df = pd.read_csv(raw_file)
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])

    print(f"[+] Loaded {len(df):,} transactions.")
    print(f"[+] Date Range: {df['Order_Date'].min().strftime('%Y-%m-%d')} to {df['Order_Date'].max().strftime('%Y-%m-%d')}")
    print(f"[+] Total Gross Merchandise Value (GMV): INR {df['Order_Value_INR'].sum():,.2f}")
    print(f"[+] Average Order Value (AOV): INR {df['Order_Value_INR'].mean():,.2f}")

    # 2. Key Business Insights
    cat_sales = df.groupby('Product_Category')['Order_Value_INR'].agg(['sum', 'count', 'mean']).rename(
        columns={'sum': 'Total_Revenue', 'count': 'Order_Count', 'mean': 'Avg_Ticket_Size'}
    ).sort_values(by='Total_Revenue', ascending=False)
    
    print("\n--- Revenue & Orders by Category ---")
    for cat, row in cat_sales.iterrows():
        print(f" • {cat:<24}: INR {row['Total_Revenue']:>12,.2f} ({int(row['Order_Count']):>5} orders | Avg INR {row['Avg_Ticket_Size']:>7,.2f})")

    # City Tier Breakdown
    tier_sales = df.groupby('City_Tier')['Order_Value_INR'].agg(['sum', 'count', 'mean'])
    print("\n--- Revenue by City Tier ---")
    for tier, row in tier_sales.iterrows():
        print(f" • {tier:<10}: INR {row['sum']:>12,.2f} ({int(row['count']):>5} orders)")

    # 3. Customer-Level Aggregation for RFM
    # Reference Snapshot Date = 1 day after latest order in dataset
    snapshot_date = df['Order_Date'].max() + pd.Timedelta(days=1)
    print(f"\n[+] Snapshot Date for Recency calculation: {snapshot_date.strftime('%Y-%m-%d')}")

    # Mode helper function for categorical fields
    def get_mode(series):
        mode_vals = series.mode()
        return mode_vals[0] if not mode_vals.empty else series.iloc[0]

    customer_df = df.groupby('Customer_ID').agg(
        Customer_Age=('Customer_Age', 'first'),
        Customer_Gender=('Customer_Gender', 'first'),
        City=('City', 'first'),
        City_Tier=('City_Tier', 'first'),
        Customer_Tenure_Months=('Customer_Tenure_Months', 'first'),
        Total_Orders=('Transaction_ID', 'count'),
        Total_Spend=('Order_Value_INR', 'sum'),
        Avg_Order_Value=('Order_Value_INR', 'mean'),
        Last_Order_Date=('Order_Date', 'max'),
        Avg_Discount_Percent=('Discount_Percent', 'mean'),
        Total_Delivery_Delay_Days=('Delivery_Delay_Days', 'sum'),
        Avg_Delivery_Delay_Days=('Delivery_Delay_Days', 'mean'),
        Total_Complaints=('Complaints_Filed', 'sum'),
        Avg_Satisfaction_Score=('Satisfaction_Score', 'mean'),
        Favorite_Category=('Product_Category', get_mode),
        Preferred_Payment=('Payment_Method', get_mode)
    ).reset_index()

    # Recency in days
    customer_df['Recency_Days'] = (snapshot_date - customer_df['Last_Order_Date']).dt.days

    # 4. RFM Scoring (1 to 5)
    # Recency: Lower days is BETTER -> Score 5 is lowest days
    customer_df['R_Score'] = pd.qcut(customer_df['Recency_Days'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    
    # Frequency: Higher orders is BETTER (Use rank-based method to handle repeating ties safely)
    customer_df['F_Score'] = pd.qcut(customer_df['Total_Orders'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    # Monetary: Higher spend is BETTER
    customer_df['M_Score'] = pd.qcut(customer_df['Total_Spend'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Combined RFM Score
    customer_df['RFM_Group'] = (
        customer_df['R_Score'].astype(str) + 
        customer_df['F_Score'].astype(str) + 
        customer_df['M_Score'].astype(str)
    )

    # 5. Customer Segmentation Logic
    def segment_customer(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions (VIP)'
        elif r >= 3 and f >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 2:
            return 'Promising Newcomers'
        elif r <= 2 and f >= 3:
            return 'At-Risk (High Spenders Dropping)'
        elif r <= 2 and f <= 2 and m <= 2:
            return 'Lost / Inactive'
        else:
            return 'Potential Regulars'

    customer_df['Customer_Segment'] = customer_df.apply(segment_customer, axis=1)

    # 6. Churn Definition (Ground Truth for Day 2 Machine Learning)
    # Business Logic: Customer is considered Churned (1) if:
    # - No purchase in > 180 days OR
    # - Inactive for > 100 days with satisfaction <= 2.8 and complaints >= 1
    def evaluate_churn(row):
        if row['Recency_Days'] > 180:
            return 1
        elif row['Recency_Days'] > 90 and (row['Avg_Satisfaction_Score'] < 3.0 or row['Total_Complaints'] >= 2):
            return 1
        else:
            return 0

    customer_df['Is_Churned'] = customer_df.apply(evaluate_churn, axis=1)

    churn_rate = customer_df['Is_Churned'].mean() * 100
    print("\n--- Customer Intelligence Summary ---")
    print(f" • Total Unique Customers Evaluated: {len(customer_df):,}")
    print(f" • Overall Churn Rate: {churn_rate:.2f}% ({customer_df['Is_Churned'].sum():,} churned customers)")
    
    print("\n--- Customer Segments Distribution ---")
    seg_summary = customer_df.groupby('Customer_Segment').agg(
        Customer_Count=('Customer_ID', 'count'),
        Total_Revenue=('Total_Spend', 'sum'),
        Avg_Spend=('Total_Spend', 'mean'),
        Churn_Rate=('Is_Churned', lambda x: f"{x.mean() * 100:.1f}%")
    ).sort_values(by='Customer_Count', ascending=False)
    print(seg_summary.to_string())

    # 7. Generate Key EDA Charts (Saved to outputs/ folder for Power BI & GitHub)
    print("\n[+] Generating & Saving Visualizations to outputs/...")

    # Plot 1: Revenue & Orders by Category
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cat_sales['Total_Revenue'].plot(kind='barh', ax=axes[0], color='#2563eb')
    axes[0].set_title("Total Revenue by Product Category (INR)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Revenue (INR)")

    cat_sales['Order_Count'].plot(kind='barh', ax=axes[1], color='#10b981')
    axes[1].set_title("Total Orders by Category", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Number of Orders")
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "01_category_revenue_orders.png")
    plt.savefig(chart1_path, dpi=200)
    plt.close()

    # Plot 2: Customer Segments vs Churn Rate
    fig, ax = plt.subplots(figsize=(10, 5))
    seg_churn = customer_df.groupby('Customer_Segment')['Is_Churned'].mean().sort_values() * 100
    colors = ['#10b981' if val < 25 else '#f59e0b' if val < 60 else '#ef4444' for val in seg_churn.values]
    bars = ax.barh(seg_churn.index, seg_churn.values, color=colors)
    ax.set_title("Churn Rate % Across Customer Segments", fontsize=14, fontweight='bold')
    ax.set_xlabel("Churn Rate (%)")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va='center', fontweight='bold', fontsize=10)
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "02_segment_churn_risk.png")
    plt.savefig(chart2_path, dpi=200)
    plt.close()

    # Plot 3: Impact of Delivery Delays & Complaints on Satisfaction
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=customer_df,
        x='Total_Delivery_Delay_Days',
        y='Avg_Satisfaction_Score',
        hue='Is_Churned',
        palette={0: '#10b981', 1: '#ef4444'},
        alpha=0.6,
        s=45,
        ax=ax
    )
    ax.set_title("Delivery Delay Days vs Customer Satisfaction Score", fontsize=13, fontweight='bold')
    ax.set_xlabel("Total Delivery Delay Experienced (Days)")
    ax.set_ylabel("Average Satisfaction Score (1 - 5)")
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "03_delay_vs_satisfaction.png")
    plt.savefig(chart3_path, dpi=200)
    plt.close()

    # 8. Export Processed Data for Day 2 (ML) & Day 3 (Power BI)
    processed_file = os.path.join(data_dir, "customer_intelligence_processed.csv")
    customer_df.to_csv(processed_file, index=False)

    print("\n[SUCCESS] Day 1 Data Processing Complete!")
    print(f" • Processed Dataset Saved: {os.path.abspath(processed_file)}")
    print(f" • Rows: {len(customer_df):,} Unique Customers | Columns: {customer_df.shape[1]}")
    print(f" • Visualizations Saved in: {os.path.abspath(output_dir)}")
    print("=" * 70)

    return customer_df

if __name__ == "__main__":
    run_analysis_and_rfm()
