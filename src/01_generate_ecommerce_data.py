"""
=============================================================================
CustomerIQ - Script 01: Realistic E-Commerce Dataset Generator
=============================================================================
This script generates a realistic, production-style e-commerce dataset
modeling customer transactions, delivery service levels, customer satisfaction,
and churn behavior for 10,000+ orders across 2,500 distinct customers.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Ensure UTF-8 output safe for Windows console
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def generate_dataset(num_records=10000, num_customers=2500, random_seed=42):
    np.random.seed(random_seed)
    print("=" * 65)
    print("[Step 1] Initializing CustomerIQ Synthetic E-Commerce Generator")
    print(f"[+] Generating {num_records:,} transactions across {num_customers:,} unique customers...")
    print("=" * 65)

    # 1. Customer Profiles (Demographics & Baseline Attributes)
    customer_ids = [f"CUST-{10000 + i}" for i in range(num_customers)]
    
    cities_tier1 = ["Delhi NCR", "Mumbai", "Bengaluru", "Hyderabad", "Pune"]
    cities_tier2 = ["Kolkata", "Chennai", "Ahmedabad", "Jaipur", "Lucknow", "Indore", "Chandigarh"]
    all_cities = cities_tier1 + cities_tier2
    city_weights = [0.18, 0.16, 0.15, 0.12, 0.09, 0.06, 0.06, 0.05, 0.04, 0.03, 0.03, 0.03]

    customer_city_map = {cid: np.random.choice(all_cities, p=city_weights) for cid in customer_ids}
    customer_age_map = {cid: int(np.clip(np.random.normal(loc=32, scale=9), 18, 68)) for cid in customer_ids}
    customer_gender_map = {cid: np.random.choice(["Female", "Male", "Non-Binary"], p=[0.49, 0.49, 0.02]) for cid in customer_ids}
    customer_tenure_map = {cid: int(np.random.randint(1, 37)) for cid in customer_ids} # 1 to 36 months

    # 2. Product Categories & Price Distributions
    categories = [
        "Electronics", 
        "Fashion & Apparel", 
        "Home & Kitchen", 
        "Beauty & Personal Care", 
        "Grocery & Essentials"
    ]
    category_weights = [0.22, 0.28, 0.18, 0.17, 0.15]

    payment_methods = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash on Delivery"]
    payment_weights = [0.46, 0.24, 0.14, 0.06, 0.10]

    # Date range: 2025-01-01 to 2026-09-30
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2026, 9, 30)
    total_days = (end_date - start_date).days

    records = []
    
    # 3. Transaction Loop
    for i in range(1, num_records + 1):
        txn_id = f"TXN-{100000 + i}"
        cid = np.random.choice(customer_ids)
        age = customer_age_map[cid]
        gender = customer_gender_map[cid]
        city = customer_city_map[cid]
        tier = "Tier-1" if city in cities_tier1 else "Tier-2"
        tenure = customer_tenure_map[cid]

        # Order Timing
        random_day = np.random.randint(0, total_days)
        order_date = start_date + timedelta(days=random_day)

        category = np.random.choice(categories, p=category_weights)
        payment = np.random.choice(payment_methods, p=payment_weights)

        # Realistic Pricing by Category
        if category == "Electronics":
            order_val = float(np.random.uniform(1500, 28000))
        elif category == "Fashion & Apparel":
            order_val = float(np.random.uniform(499, 5500))
        elif category == "Home & Kitchen":
            order_val = float(np.random.uniform(399, 8500))
        elif category == "Beauty & Personal Care":
            order_val = float(np.random.uniform(249, 3200))
        else: # Grocery
            order_val = float(np.random.uniform(199, 2200))
        
        # Round order value
        order_val = round(order_val, 2)
        discount = np.random.choice([0, 5, 10, 15, 20, 25, 30], p=[0.25, 0.20, 0.25, 0.15, 0.08, 0.05, 0.02])

        # Service Quality & Experience Indicators
        delay_prob = 0.22 if tier == "Tier-2" else 0.15
        if np.random.rand() < delay_prob:
            delivery_delay_days = int(np.clip(np.random.geometric(p=0.45), 1, 8))
        else:
            delivery_delay_days = 0

        # Complaints correlation with delay
        if delivery_delay_days >= 3:
            complaint = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
        else:
            complaint = np.random.choice([0, 1], p=[0.92, 0.08])

        # Satisfaction Rating (1 to 5 stars)
        base_rating = 5.0
        if delivery_delay_days > 0:
            base_rating -= (0.5 * delivery_delay_days)
        if complaint > 0:
            base_rating -= (1.2 * complaint)
        satisfaction = int(round(float(np.clip(np.random.normal(loc=base_rating, scale=0.6), 1, 5))))

        records.append({
            "Transaction_ID": txn_id,
            "Customer_ID": cid,
            "Customer_Age": age,
            "Customer_Gender": gender,
            "City": city,
            "City_Tier": tier,
            "Customer_Tenure_Months": tenure,
            "Order_Date": order_date.strftime("%Y-%m-%d"),
            "Product_Category": category,
            "Payment_Method": payment,
            "Order_Value_INR": order_val,
            "Discount_Percent": discount,
            "Delivery_Delay_Days": delivery_delay_days,
            "Complaints_Filed": complaint,
            "Satisfaction_Score": satisfaction
        })

    df = pd.DataFrame(records)
    
    # 4. Save to CSV
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    raw_csv_path = os.path.join(output_dir, "ecommerce_transactions_raw.csv")
    df.to_csv(raw_csv_path, index=False)

    print("\n[SUCCESS] Generated Dataset Summary:")
    print(f" • File Saved At: {os.path.abspath(raw_csv_path)}")
    print(f" • Total Rows: {len(df):,}")
    print(f" • Total Columns: {df.shape[1]}")
    print(f" • Total Revenue Generated: INR {df['Order_Value_INR'].sum():,.2f}")
    print(f" • Unique Customers: {df['Customer_ID'].nunique():,}")
    print("=" * 65)

    return df

if __name__ == "__main__":
    generate_dataset()
