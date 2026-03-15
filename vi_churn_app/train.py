"""
train.py — Pre-train and save all models before running the Streamlit dashboard.
Run: python train.py
"""
from utils import run_pipeline

if __name__ == "__main__":
    run_pipeline(
        raw_csv="data/VI_Customer_Churn.csv",
        cleaned_csv="data/vi_customer_churn_cleaned.csv",
        models_folder="models",
    )
    print("\n✅ Pipeline complete. Run: streamlit run app.py")
