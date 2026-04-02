# credit_risk_model
### Overview
An end-to-end machine learning project analysing LendingClub loan data 
to determine whether interest rates reflect actual borrower default risk.

### Notebooks
- `01 data_ingestion_cleaning_eda.ipynb` — data cleaning, feature engineering and exploratory analysis
- `02 machine_learning_modeling.ipynb` — machine learning models and risk vs pricing analysis

### Key Findings
- XGBoost model achieved a ROC-AUC of 0.73 in predicting loan default
- High-risk borrowers in the top deciles are systematically undercharged
- Low-risk borrowers are overcharged relative to their predicted default probability
- LendingClub's pricing is driven primarily by grade label with limited within-grade risk adjustment

### Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (Logistic Regression)
- XGBoost
- Matplotlib, Seaborn

### Data
Dataset: LendingClub Loan Data 2007–2018, available on Kaggle.  
Not included in this repo due to file size.
