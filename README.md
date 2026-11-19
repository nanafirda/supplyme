# SupplyMe: *Predict. Anticipate. Convert.*

**Master 2 project, Quantitative Marketing & Applications, Aix-Marseille Université.**

SupplyMe is a predictive replenishment system for online grocery shopping. For each household and each product it usually buys, SupplyMe:

1. predicts **when** the next purchase is likely to happen;
2. decides whether to send a **reminder**, offer a **discount** or **do nothing**;
3. chooses the **best time** to contact the customer;
4. prepares a **pre-filled basket**;
5. estimates the **profitability** of the campaign.

For **new customers with no purchase history**, it offers a welcome basket and a welcome questionnaire rewarded with loyalty points.

---

## 1. Data

We use the group's common dataset (`SupplyMe_Final_Excel_and_Original_Data.zip`), built from the public **Instacart Market Basket Analysis** data:

| Content | Size |
|---|---|
| Real Instacart customers, randomly sampled (seed 20261118) | 3,000 customers |
| Orders | 48,306 |
| Order lines (products purchased) | 493,119 |
| Distinct products | 14,556 |
| Synthetic new accounts with welcome-questionnaire answers | 1,000 accounts |

**Data preparation:**
- **Missing values:** the only missing variable is `days_since_prior_order`. It is missing exactly for each customer's first order, so these are not errors.
- **Dates:** Instacart provides no calendar dates, only the gap between orders. We rebuild dates by summing these gaps. Each customer's last order is placed 0 to 13 days before the analysis date (31 October 2026).
- **Complementary data:** prices, gross margin and household size are not in Instacart. We take them from official sources (Section 7) and simulate them at the individual level.

---

## 2. The seven parts of the project

The full analysis is in `notebooks/SupplyMe_Full_Analysis.ipynb`. It is already executed.

| Part | Notebook section | Code |
|---|---|---|
| 1. Analyse household purchase histories | §7 | `data/` |
| 2. Identify consumption habits | §8 | `prediction/features.py` |
| 3. Predict the date of the next purchase | §9 to §13 | `prediction/models.py` |
| 4. Find the best time to contact the customer | §14 | `marketing/decision.py` |
| 5. Assess whether a discount is worthwhile | §15 | `marketing/economics.py` |
| 6. Build a personalised basket | §16 | `marketing/basket.py` |
| 7. Estimate campaign profitability | §17 | `marketing/decision.py` |
| New customers after their first purchase (cold start) | §12 bis | `prediction/cold_start.py` |
| New customers with no order (welcome basket, questionnaire, points) | §12 ter | `prediction/nouveaux_clients.py` |

---

## 3. Methodology

### 3.1 Time-based train / validation / test split

Shuffling purchases at random would let the model "see the future". We therefore split the data **in time**, using snapshots:

- at each snapshot date, the model only sees the **past**, and we observe what happens in the following 60 days;
- **12 training snapshots**, then a **60-day gap**, then **3 test snapshots** never used during training;
- the **last 3 training snapshots** are used as a **validation period** to choose the model.

Each (household, product) pair is described by **25 explanatory variables**, for example: days since the last purchase, the usual interval between purchases, its regularity, and the share of the household's orders that contain the product.

### 3.2 Two prediction tasks

| Task | Type | Target | Metric |
|---|---|---|---|
| A. Time until next purchase | Regression | number of days | MAE (mean absolute error, in days) |
| B. Purchase within 7 days | Classification | yes / no | AUC (area under the ROC curve) |

For each task, we compare a **simple baseline rule** with four models: a linear or logistic regression, Random Forest, XGBoost and LightGBM.

### 3.3 Incremental marketing decision

A message is only worth sending if it **causes** purchases that would not have happened otherwise. A discount given to a customer who would have bought anyway is a loss (a windfall effect).

- **Imminent need** (the product is due soon): a simple **reminder**.
- **Overdue habit** (no repurchase for 1.5 to 3 times the usual interval): a **discount** to win back the habit, valued over the next 90 days (customer lifetime value logic).
- **No action** when the expected incremental profit is negative.

The message is sent the day before the predicted need, one hour before the household's usual ordering time. Each message contains at most 12 products.

---

## 4. Results

All results below are measured on the **test period**, which the models never saw during training.

### 4.1 Task A: time until next purchase (regression)

| Model | MAE (days) | Improvement vs baseline |
|---|---|---|
| Baseline rule (usual interval minus days elapsed) | 17.45 | — |
| **LightGBM (selected)** | **12.11** | **−31%** |
| Random Forest | 12.19 | −30% |
| Linear regression | 12.25 | −30% |
| XGBoost | 12.25 | −30% |

**LightGBM** has the lowest error. On average, it predicts the next purchase date within about 12 days, compared with 17.5 days for the simple rule.

### 4.2 Task B: purchase within 7 days (classification)

| Model | AUC on validation (used for selection) | AUC on test |
|---|---|---|
| Baseline rule | — | 0.739 |
| **Logistic regression (selected)** | **0.767** | 0.789 |
| Random Forest | 0.759 | **0.796** |
| LightGBM | 0.741 | 0.779 |
| Tuned LightGBM | 0.740 | 0.782 |
| XGBoost | 0.736 | 0.777 |

**Why logistic regression and not Random Forest?** Random Forest has a slightly higher AUC **on the test period** (0.796 vs 0.789). However, the model is chosen on the **validation period**, before the test data are looked at. On validation, logistic regression is the best model (0.767 vs 0.759). Choosing the model that looks best on the test set would bias the reported performance upward, because the test would then be used twice: once to choose and once to evaluate.

The test gap between the two models is small (0.007 AUC). Logistic regression is also simpler and easier to interpret. All models clearly outperform the baseline rule (0.739).

### 4.3 Marketing campaign (analysis date: 31 October 2026)

| Strategy (same products for every strategy) | Households contacted | Incremental profit |
|---|---|---|
| No campaign | 0 | $0 |
| Reminder to everyone, no discount | 1,337 | $523 |
| 10% discount to everyone | 1,337 | $990 |
| **SupplyMe (targeted decision)** | **1,274** | **$1,538** |

The targeted strategy earns **55% more** than a discount for everyone, while contacting fewer households. In total, 4,264 product recommendations are sent to 1,274 households.

**Sensitivity analysis.** The reminder effect (5%) and the message cost ($0.05) are assumptions. When we vary the reminder effect from 0% to 10% and the price elasticity from 1.5 to 3.5, the profit stays positive in every scenario, between $296 and $2,832.

### 4.4 New customers

| Situation | Method | Result |
|---|---|---|
| No order yet | Welcome basket of best-sellers | Recovers 7.1% of the products in the real first order (Recall@10) |
| No order yet | Basket adapted to the questionnaire (simulated answers) | 5.1%, vs 4.9% with a placebo |
| After the first order | Cold-start model (LightGBM) vs simple rule | AUC 0.742 vs 0.697; timing error 13.9 vs 22.2 days |
| Effect of history length | Main model | AUC rises from 0.717 (1 to 3 orders) to 0.813 (21+ orders) |

**Interpretation:** for a very first order, best-sellers remain the best proposal. A single order is worth about **4 times** more than popularity (29% vs 7% of the second basket recovered). The questionnaire is therefore used to get to know the customer and to start the relationship through loyalty points: 10 points per department chosen, 50 bonus points for 3 departments, and 100 points = $1.

---

## 5. The application

The Streamlit application has two parts:

- **Business space** (7 tabs): Dashboard, Predictions, Models & AI, Households, Recommendations, Analytics, New customers.
- **Customer app**: a phone mock-up with a **35-second interactive demo** based on a real household (#40294), from the purchase history to the restocking order.

Every block is labelled **Real data**, **Simulation** or **Example**. The interface is in English by default; French is available from the side menu (arrow at the top left). The same menu gives access to a **Technical workshop** showing the Python calculations live.

---

## 6. Repository structure

```
supplyme/
├── notebooks/SupplyMe_Full_Analysis.ipynb   Full analysis, already executed (open in Google Colab)
├── main.py                 Entry point: python main.py run | app
├── config.py               Parameters, assumptions and sources
├── data/                   Data import, table merging, date reconstruction, complementary data
├── database/               SQL schema (tables and views) and database helpers
├── prediction/             Variables, model comparison, cold start, new customers
├── marketing/              Profit model, decisions, personalised basket
├── app/                    Streamlit application and interface
├── reports/                Summary and figures, regenerated at each run
├── tests/                  Automated tests (no time leakage, decisions, households)
├── supplyme.db.gz.part*    Compressed SQLite database used by the online application
└── requirements.txt
```

---

## 7. How to reproduce

**Read the analysis:** open the notebook in Google Colab. The results are already displayed.

**Re-run everything locally:**
```bash
pip install -r requirements.txt
# unzip the group dataset into data/base_groupe/
python main.py run      # about 6 minutes, reproduces all the results
python main.py app      # application at http://localhost:8501
python -m pytest -q     # tests
```

---

## 8. Sources for complementary data

- U.S. Bureau of Labor Statistics, *Average Price Data*, U.S. city average: product prices.
- NGA / FMS Solutions, *Independent Grocers Financial Survey*, fiscal year 2025: gross margin of 27.9%.
- U.S. Energy Information Administration, *Residential Energy Consumption Survey (RECS) 2024*: household size distribution.
- Bijmolt, Van Heerde & Pieters (2005), "New Empirical Generalizations on the Determinants of Price Elasticity", *Journal of Marketing Research*, 42(2): price elasticity of −2.62.
- Agrawal, Imielinski & Swami (1993): association rules.

---

## 9. Limitations

- Dates are reconstructed from gaps between orders, and Instacart caps these gaps at 30 days.
- Prices and household sizes are anchored on official sources but simulated for each household. Household size is derived from basket size, so it adds little new information to the model.
- The reminder effect (5%) and the message cost ($0.05) are assumptions, tested in the sensitivity analysis.
- Questionnaire answers and clicks in the group dataset are synthetic: the questionnaire part is a demonstration of the mechanism, not a measured effect.
