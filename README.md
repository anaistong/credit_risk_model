# Lending Club Credit Risk Analysis
## Are Loans Priced Correctly Based on Risk?

---

## Overview

This project applies machine learning to LendingClub's 
consumer loan data to answer a business question that 
sits at the heart of retail lending: **do interest rates 
actually reflect the risk of borrower default, or is 
there systematic mispricing in the market?**

The analysis follows the full data science pipeline — 
from raw messy data through cleaning, feature engineering, 
two machine learning models, and a commercial pricing 
analysis — with findings framed as actionable 
recommendations for a lender.

---

## Business Problem

Lenders set interest rates based on their assessment 
of borrower risk. If this pricing is accurate, 
high-risk borrowers pay more and low-risk borrowers 
pay less — and the lender's margin compensates 
for expected credit losses across the portfolio.

If pricing is inaccurate in either direction, 
there are real consequences:
- **Underpricing high-risk borrowers** means the lender 
  absorbs losses that the interest rate doesn't cover
- **Overpricing low-risk borrowers** means the best 
  customers refinance elsewhere, leaving the lender 
  with a progressively riskier book

This project uses a machine learning model to generate 
an independent estimate of default risk for each loan, 
then compares that predicted risk against the actual 
interest rate charged — to identify where the gaps are 
and how large they are.

---

## Dataset

**Source:** LendingClub Loan Data 2007–2018, 
available on [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

**Size:** 2.2 million accepted loans, 151 columns  
**Sample used:** 50,000 loans for prototyping  
**Target variable:** Binary default flag  
(1 = Charged Off or Default, 0 = Fully Paid)  
**Class balance:** ~80% repaid, ~20% defaulted

The dataset is not included in this repository 
due to file size. Download from Kaggle and place 
in the project root before running the notebooks.

---

## Project Structure
```
lending-club-analysis/
│
├── 01 data_ingestion_cleaning_eda.ipynb   
│   └── Data loading, column selection, cleaning,
│       feature engineering, exploratory analysis
│
├── 02 machine_learning_modeling.ipynb     
│   └── Model training, evaluation, risk vs 
│       pricing analysis, recommendations
│
├── cleaned_data.csv                       
│   └── Output of notebook 1, input to notebook 2
│       (not tracked in Git)
│
└── README.md
```

---

## Methodology

### 1. Data Cleaning
Starting from 151 raw columns, we retained 26 
that are available at loan origination and relevant 
to borrower risk. Administrative columns (IDs, URLs), 
post-origination columns (payment totals, recovery 
amounts), and columns with more than 80% missing 
values were all excluded.

Key cleaning steps included converting `emp_length` 
from string format ("10+ years") to numeric, 
extracting loan term as an integer, imputing sparse 
nulls with column medians, and standardising 
categorical formats.

### 2. Feature Engineering
Three features were engineered beyond the raw columns:

- **`fico_avg`** — midpoint of the FICO score band, 
  collapsing two correlated columns into one 
  continuous variable
- **`loan_to_income_ratio`** — loan amount as a 
  percentage of annual income, capturing relative 
  financial burden rather than absolute loan size
- **`default`** — binary target variable built from 
  loan status, treating Charged Off and Default 
  as 1, Fully Paid as 0

### 3. Modelling
Two models were trained on an 80/20 stratified 
train/test split:

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 80.25% | 0.7288 |
| XGBoost | 80.32% | 0.7315 |

Accuracy is not the primary metric here — a model 
predicting "repaid" for every loan would achieve 
~80% accuracy given the class balance. ROC-AUC 
measures how well the model ranks defaulters above 
non-defaulters and is the correct metric for 
an imbalanced classification problem.

XGBoost was selected as the primary model for the 
pricing analysis as it captures non-linear 
relationships and feature interactions that logistic 
regression cannot model.

### 4. Risk vs Pricing Analysis
Every loan was scored with its predicted default 
probability and grouped into ten equal-size deciles. 
Average interest rates were compared across deciles 
to assess whether pricing tracks predicted risk.

The pricing gap heatmap broke this down further 
by loan grade and risk decile simultaneously, 
identifying where within the portfolio mispricing 
is most concentrated.

---

## Key Findings

**1. High-risk borrowers are systematically undercharged.**  
In the upper risk deciles, predicted default probability 
rises steeply but interest rates flatten out. 
The lender is absorbing significantly more risk 
per decile without collecting proportionally 
higher returns.

**2. Low-risk borrowers are overcharged relative 
to their risk.**  
At the lowest risk deciles, rates start high 
relative to the predicted default probability. 
Safe borrowers are effectively subsidising 
riskier ones through excess margin collected 
from the low-risk segment.

**3. Pricing is driven by grade label, not 
granular risk.**  
The heatmap shows minimal rate variation within 
each grade band across risk deciles. Two Grade C 
borrowers with very different predicted default 
probabilities are charged near-identical rates. 
The grade label is doing almost all the 
pricing work.

---

## Recommendations

1. **Introduce within-grade risk-based pricing** — 
   rates should vary within each grade based on 
   additional signals beyond the grade label itself

2. **Reprice the highest-risk deciles upward** — 
   the top deciles carry default probabilities 
   that the current rate structure does not 
   adequately compensate for

3. **Lower rates for the safest borrowers** — 
   to improve retention of the most creditworthy 
   customers who are most likely to refinance 
   elsewhere

4. **Adopt a phased correction approach** — 
   lower low-risk rates first, then gradually 
   raise high-risk rates to avoid adverse selection

---

## Tech Stack

- **Python** — Pandas, NumPy
- **Machine Learning** — Scikit-learn, XGBoost
- **Visualisation** — Matplotlib, Seaborn
- **Environment** — Jupyter Notebook

---

## Limitations

- Analysis uses a 50,000 row sample of the 
  full 2.2 million loan dataset
- Model trained on 2007–2018 data spanning 
  multiple economic cycles including the 
  financial crisis — behaviour may differ 
  in other macro environments
- No macroeconomic features included 
  (unemployment rate, interest rate environment)
- Fair rate benchmark is model-derived and 
  dependent on model quality
