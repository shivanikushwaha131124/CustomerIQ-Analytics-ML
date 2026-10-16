"""
=============================================================================
CustomerIQ - Script 03: Machine Learning Churn Prediction Engine
=============================================================================
This script executes Day 2:
1. Loads processed customer intelligence data.
2. Builds an end-to-end Preprocessing & Feature Engineering Pipeline.
3. Trains & Compares 3 Industry-Standard Classifiers:
   - Logistic Regression (Baseline)
   - Random Forest Classifier (Ensemble Bagging)
   - Gradient Boosting Classifier (Ensemble Boosting)
4. Evaluates with Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
5. Plots Feature Importances (What drives churn?).
6. Saves the production pipeline (.pkl) for deployment.
7. Scores the entire customer base with Risk Probabilities & Risk Tiers for Power BI.
=============================================================================
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

# Windows console encoding safeguard
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Segoe UI'

def train_churn_model():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "..", "data")
    models_dir = os.path.join(base_dir, "..", "models")
    output_dir = os.path.join(base_dir, "..", "outputs")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    input_file = os.path.join(data_dir, "customer_intelligence_processed.csv")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found at: {input_file}")

    print("=" * 75)
    print("🤖 [CustomerIQ Day 2] Machine Learning Churn Prediction Engine")
    print("=" * 75)

    # 1. Load Data
    df = pd.read_csv(input_file)
    print(f"[+] Loaded {len(df):,} customer records.")

    # 2. Define Features & Target
    # Note: We exclude 'Customer_Segment' and 'RFM_Group' from direct training to prevent data leakage,
    # letting the model learn patterns directly from raw behaviors and service metrics.
    num_features = [
        'Customer_Age', 
        'Customer_Tenure_Months', 
        'Total_Orders', 
        'Total_Spend', 
        'Avg_Order_Value', 
        'Avg_Discount_Percent', 
        'Total_Delivery_Delay_Days', 
        'Avg_Delivery_Delay_Days', 
        'Total_Complaints', 
        'Avg_Satisfaction_Score', 
        'Recency_Days'
    ]

    cat_features = [
        'City_Tier', 
        'Customer_Gender', 
        'Favorite_Category', 
        'Preferred_Payment'
    ]

    target = 'Is_Churned'

    X = df[num_features + cat_features]
    y = df[target]

    print(f"[+] Features: {len(num_features)} Numerical, {len(cat_features)} Categorical")
    print(f"[+] Class Balance: Active (0) = {(y == 0).sum():,} | Churned (1) = {(y == 1).sum():,} ({y.mean()*100:.1f}%)")

    # 3. Train-Test Split (80% Train, 20% Test with Stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Training Set: {len(X_train):,} samples | Test Set: {len(X_test):,} samples")

    # 4. Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
        ]
    )

    # 5. Train & Compare Multiple Classifiers
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=4, random_state=42),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
    }

    results = []
    trained_pipelines = {}

    print("\n--- Model Training & Benchmark ---")
    for name, model in candidate_models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc
        })

        print(f" • {name:<30}: Accuracy={acc*100:.2f}% | Precision={prec*100:.2f}% | Recall={rec*100:.2f}% | ROC-AUC={roc:.4f}")

    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]
    print(f"\n🏆 Best Selected Model: {best_model_name} (ROC-AUC = {results_df.iloc[0]['ROC-AUC']:.4f})")

    # 6. Detailed Evaluation of the Best Model
    y_test_pred = best_pipeline.predict(X_test)
    y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]

    print("\n--- Detailed Classification Report (Test Set) ---")
    print(classification_report(y_test, y_test_pred, target_names=["Active (0)", "Churned (1)"]))

    # 7. Generate & Save Evaluation Plots
    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=["Pred: Active", "Pred: Churned"],
                yticklabels=["Actual: Active", "Actual: Churned"],
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(f"Confusion Matrix ({best_model_name})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "04_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color='#2563eb', lw=2.5, label=f'{best_model_name} (AUC = {results_df.iloc[0]["ROC-AUC"]:.3f})')
    ax.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', lw=1.5, label='Random Chance (AUC = 0.500)')
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight='bold')
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "05_roc_auc_curve.png")
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # Plot 3: Feature Importance
    # Extract feature names after OneHotEncoding
    ohe_cols = list(best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_features))
    all_feature_names = num_features + ohe_cols

    if hasattr(best_pipeline.named_steps['classifier'], 'feature_importances_'):
        importances = best_pipeline.named_steps['classifier'].feature_importances_
        fi_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(12)

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(fi_df['Feature'][::-1], fi_df['Importance'][::-1], color='#3b82f6')
        ax.set_title(f"Top 12 Churn Drivers ({best_model_name})", fontsize=13, fontweight='bold')
        ax.set_xlabel("Relative Importance Score")
        plt.tight_layout()
        fi_path = os.path.join(output_dir, "06_feature_importance.png")
        plt.savefig(fi_path, dpi=200)
        plt.close()

    # 8. Save the Production Pipeline
    model_save_path = os.path.join(models_dir, "churn_prediction_pipeline.pkl")
    joblib.dump(best_pipeline, model_save_path)
    print(f"\n[+] Production Model Saved: {os.path.abspath(model_save_path)}")

    # 9. Score Entire Customer Base with Risk Probabilities (For Power BI & Reporting)
    print("\n[+] Scoring entire customer dataset with Churn Probabilities...")
    all_probs = best_pipeline.predict_proba(X)[:, 1]
    df['Predicted_Churn_Probability'] = np.round(all_probs, 4)
    df['Predicted_Churn_Label'] = (all_probs >= 0.50).astype(int)

    def assign_risk_category(prob):
        if prob < 0.35:
            return "Low Risk (Healthy)"
        elif prob < 0.70:
            return "Medium Risk (Watchlist)"
        else:
            return "High Risk (Immediate Action)"

    df['Predicted_Risk_Category'] = df['Predicted_Churn_Probability'].apply(assign_risk_category)

    # Save final augmented dataset
    final_output_file = os.path.join(data_dir, "customer_intelligence_final_with_predictions.csv")
    df.to_csv(final_output_file, index=False)

    print(f"[+] Final Augmented Dataset for Power BI Saved: {os.path.abspath(final_output_file)}")
    print(f" • Low Risk Customers: {(df['Predicted_Risk_Category'] == 'Low Risk (Healthy)').sum():,}")
    print(f" • Medium Risk Customers: {(df['Predicted_Risk_Category'] == 'Medium Risk (Watchlist)').sum():,}")
    print(f" • High Risk Customers: {(df['Predicted_Risk_Category'] == 'High Risk (Immediate Action)').sum():,}")
    print("=" * 75)

    return best_pipeline, df

if __name__ == "__main__":
    train_churn_model()
