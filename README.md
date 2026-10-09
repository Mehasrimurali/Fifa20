# FIFA 20 Player Analytics and Machine Learning

## 1. Overview
A complete, end-to-end data science project on the FIFA 20 Career Mode player dataset. It performs data cleaning, exploratory data analysis, unsupervised clustering of players by skillset, answers three required analytical questions, and builds/tunes a machine-learning model that predicts a player's Overall rating from their raw skill attributes.

## 2. Objectives
- Clean and understand the FIFA 20 player dataset.
- Explore player attributes through univariate, bivariate, categorical and correlation analysis.
- Cluster players by their footballing skillset (not position or overall rating) using K-Means.
- Answer three specific questions: top countries by player count, how overall rating changes with age, and which offensive position is paid most.
- Build, compare, and tune multiple regression models, and interpret the best one.

## 3. Dataset
- **Source:** `players_20.csv`, provided as part of the PRCP-1004-Fifa20 capstone brief (originally scraped from SoFIFA).
- **Records:** 18,278 players.
- **Features:** 104 raw columns, covering identity/club/nationality info, overall/potential ratings, market value and wage, and ~60 detailed skill attributes (attacking, skill, movement, power, mentality, defending, goalkeeping).
- **Limitations:** Single-season snapshot (FIFA 20 only) — no multi-version (FIFA 15–20) history is present in the uploaded file, so version-over-version trend analysis was not possible. Wage/value fields are estimated by EA/SoFIFA and are noisier for lower-profile players.

## 4. Technologies
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook.

## 5. Project Workflow
Dataset → Cleaning (structural missing-value handling, type conversion, feature prep) → EDA (univariate/bivariate/categorical/correlation) → Feature Engineering (composite skill scores, age groups, potential growth) → Clustering (K-Means, k=4) → Machine Learning (5 regression models compared, Random Forest tuned) → Insights.

## 6. Key Findings
- Missing values are structural, not random: goalkeeping attributes are missing for outfield players and vice versa; no blind imputation was used.
- Top 10 countries by player count: England, Germany, Spain, France, Argentina, Brazil, Italy, Colombia, Japan, Netherlands.
- Players typically stop meaningfully improving in Overall rating around age 27–29, plateau through their early-to-mid thirties (reliable peak mean at age 31–32), and decline after roughly age 33–34.
- By mean wage, wingers (RW/LW) appear to earn more than strikers, but this is driven by a small number of superstars; by **median** wage, strikers actually earn more — a good example of why mean and median can tell opposite stories on skewed data.
- K-Means clustering (k=4) on 30 standardized skill attributes — without using position or Overall as inputs — recovers four groups that split along an attacking/defensive axis and a technical-quality tier, confirming that FIFA 20's skill ratings meaningfully encode real football roles.
- `movement_reactions` and `mentality_composure` are the strongest predictors of a player's Overall rating, ahead of any single technical attribute.

## 7. Machine Learning Results
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest (Tuned) | 0.872 | 1.144 | 0.9720 |
| Random Forest | 0.924 | 1.221 | 0.9681 |
| Gradient Boosting | 1.012 | 1.317 | 0.9629 |
| Decision Tree | 1.430 | 1.851 | 0.9267 |
| Linear Regression | 1.827 | 2.326 | 0.8841 |
| Ridge Regression | 1.827 | 2.326 | 0.8841 |

**Recommended production model:** Tuned Random Forest Regressor (`n_estimators=150, max_depth=20, min_samples_leaf=1, max_features='sqrt'`, found via `RandomizedSearchCV`), predicting Overall rating from raw skill/physical attributes with R² ≈ 0.972 on held-out test data.

## 8. Conclusion
This project demonstrates a full data science pipeline on real football data — from structural data cleaning through unsupervised clustering to a tuned, interpretable regression model — with every figure and conclusion traceable to the uploaded dataset.

## 9. Future Improvements
- Incorporate multiple FIFA versions (15–20) to study attribute trends and player development trajectories over time.
- Add SHAP-based interpretation for the tuned Random Forest for more granular, per-player explanations.
- Extend clustering to include goalkeepers with their own dedicated attribute set.
- Build a value/wage prediction model as a separate, explicitly-labelled financial forecasting task (kept out of this iteration to avoid leakage into the Overall-rating regression).

---

## Suggested Repository Structure
```
fifa20-player-analytics/
├── data/
│   └── players_20.csv
├── notebooks/
│   └── PRCP-1004-Fifa20-Analysis.ipynb
├── README.md
└── requirements.txt
```

## Resume-Ready Project Description
Built an end-to-end player analytics and machine learning pipeline on 18K+ FIFA 20 player records, combining structural data cleaning, exploratory analysis, unsupervised K-Means clustering of players by skillset, and a tuned Random Forest regression model (R² = 0.97) predicting player quality ratings from raw skill attributes.

## Resume Bullet Points
- Cleaned and analyzed an 18,278-row, 104-column football player dataset, correctly diagnosing structural (not random) missing-value patterns tied to goalkeeper vs. outfield attribute sets, avoiding invalid imputation.
- Engineered composite skill-score features and applied K-Means clustering (k selected via elbow/silhouette analysis) to segment players into 4 skillset-based archetypes using only raw attributes, validating results against real position data.
- Answered three data-driven business questions using rigorous statistical treatment of skewed financial data (mean vs. median wage comparison across positions).
- Compared 5 regression algorithms and tuned a Random Forest via RandomizedSearchCV, improving test RMSE from 1.22 to 1.14 and achieving R² = 0.972 in predicting player Overall rating from skill attributes.
- Interpreted model behavior via feature importance analysis, identifying player reactions and composure — not raw technical skill — as the strongest drivers of overall player quality.
