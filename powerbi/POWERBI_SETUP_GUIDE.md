# 📊 Power BI Dashboard Setup & DAX Reference Guide

This guide provides step-by-step instructions to build the **CustomerIQ Executive Power BI Dashboard** using `data/customer_intelligence_final_with_predictions.csv`.

---

## 🚀 Step 1: Connect Data in Power BI Desktop
1. Open **Power BI Desktop**.
2. Click **Get Data** -> **Text/CSV**.
3. Select file: `CustomerIQ-Analytics-ML/data/customer_intelligence_final_with_predictions.csv`.
4. Click **Transform Data** (Power Query):
   - Verify `Customer_ID` is Text.
   - Verify `Total_Spend`, `Avg_Order_Value`, `Predicted_Churn_Probability` are Decimal Number.
   - Verify `Total_Orders`, `Recency_Days` are Whole Number.
5. Click **Close & Apply**.

---

## 🧮 Step 2: Essential DAX Measures (Copy & Paste)

Create a dedicated Measures Table or add these measures to the main table:

### 1. Total Gross Revenue (GMV)
```dax
Total GMV = SUM(customer_intelligence_final_with_predictions[Total_Spend])
```

### 2. Total Customer Count
```dax
Total Customers = DISTINCTCOUNT(customer_intelligence_final_with_predictions[Customer_ID])
```

### 3. Average Order Value (AOV)
```dax
Overall AOV = AVERAGE(customer_intelligence_final_with_predictions[Avg_Order_Value])
```

### 4. Overall Churn Rate %
```dax
Churn Rate % = 
DIVIDE(
    CALCULATE(COUNTROWS(customer_intelligence_final_with_predictions), customer_intelligence_final_with_predictions[Is_Churned] = 1),
    COUNTROWS(customer_intelligence_final_with_predictions),
    0
)
```

### 5. High-Risk Revenue at Risk (Critical Business Metric)
```dax
Revenue at High Risk = 
CALCULATE(
    SUM(customer_intelligence_final_with_predictions[Total_Spend]),
    customer_intelligence_final_with_predictions[Predicted_Risk_Category] = "High Risk (Immediate Action)"
)
```

### 6. Average Satisfaction Rating
```dax
Avg CSAT Score = AVERAGE(customer_intelligence_final_with_predictions[Avg_Satisfaction_Score])
```

---

## 🖥️ Step 3: 3-Page Dashboard Layout Architecture

### 📑 Page 1: Executive KPI & Revenue Performance
- **Top Header:** CustomerIQ Executive Dashboard | Brand Logo | Refresh Timestamp
- **KPI Cards (Top Row):**
  1. `[Total GMV]` (Formatted as Currency in INR)
  2. `[Total Customers]` (e.g. 2.46K)
  3. `[Overall AOV]`
  4. `[Churn Rate %]` (Target KPI)
  5. `[Revenue at High Risk]` (Red Alert Highlight)
- **Visual 1 (Donut Chart):** Revenue Share by `Favorite_Category`
- **Visual 2 (Clustered Column Chart):** Total Revenue by `City` (Top 10 Cities)
- **Visual 3 (Line & Clustered Column):** Orders vs Delivery Delay Days

---

### 📑 Page 2: Customer Segmentation & RFM Matrix
- **Slicers (Filter Bar):** `Customer_Segment`, `City_Tier`, `Customer_Gender`
- **Visual 1 (Treemap / Bar Chart):** Customer Count by `Customer_Segment`
  - Champions (VIP)
  - Loyal Customers
  - Promising Newcomers
  - Potential Regulars
  - At-Risk (High Spenders)
  - Lost / Inactive
- **Visual 2 (Scatter Plot):** `Recency_Days` (X-axis) vs `Total_Spend` (Y-axis), colored by `Customer_Segment`.
- **Visual 3 (Matrix Table):**
  - Columns: `Customer_ID`, `City`, `Total_Orders`, `Total_Spend`, `Recency_Days`, `Customer_Segment`.

---

### 📑 Page 3: AI Predictive Churn Risk Explorer
- **Visual 1 (Gauge Chart):** Churn Risk Distribution (Low vs Medium vs High Risk)
- **Visual 2 (Bar Chart):** Revenue At Risk by Customer Cohort
- **Visual 3 (Interactive Action List):**
  - Table of customers filtered where `Predicted_Risk_Category = "High Risk"` sorted by `Total_Spend` descending.
  - Allows retention teams to export high-priority VIP customers who need retention coupons.

---

## 🎨 Theme & Styling Tips
- **Color Palette:**
  - Primary Accent: Royal Blue (`#2563eb`)
  - Success / Active: Emerald Green (`#10b981`)
  - Warning / Watchlist: Amber (`#f59e0b`)
  - High Risk / Churned: Crimson Red (`#ef4444`)
  - Background: Clean Off-White (`#f8fafc`)
- **Typography:** Segoe UI or Arial (11pt body, 18pt KPI values, 24pt headers).
