# Screen Time Behaviour Analysis

This data mining project investigates why some users have high daily video screen time compared with others. It has two goals: **prediction** (classify users as Moderate or High) and **explanation** (identify the factors driving high screen time).

## Problem Statement

The central research question is: *What are the main factors that increase daily screen time?* The analysis tests user background, watching behaviour, platform usage, device type, subscription type, content category, and internet connection.

## Dataset

The project uses 20,000 user records with 14 input attributes: Country, Age Group, Gender, Peak Viewing Hour, Content Category, Device Type, Subscription Type, Platform, Weekly Sessions, Average Session Duration, Number of Platforms Used, Internet Connection, Content Language, and Binge Watch Frequency.

The target variable is `Daily_Screen_Time_Hours`, binned into **Moderate** and **High**. No Low-screen-time users existed in this dataset.

## Exploratory Data Analysis

The target classes are imbalanced: **High 75.7%** and **Moderate 24.3%**. Plain accuracy can therefore be misleading because a model predicting High for every user could still appear reasonably accurate. Numeric features showed weak correlations overall, indicating low redundancy and minimal multicollinearity.

## Feature Selection with Mutual Information

Mutual Information (MI) was used because it can detect non-linear dependencies as well as linear ones.

| Feature | MI score |
|---|---:|
| Weekly Sessions | 0.186 |
| Country | 0.140 |
| Age Group | 0.101 |
| Other features | < 0.006 |

The six selected features were **Weekly Sessions, Country, Age Group, Platform, Content Language,** and **Number of Platforms**. This reduces noise and model complexity.

## Neural Network Model

A neural network was chosen to model complex, non-linear interactions in the 20,000-record dataset. The model uses 30% dropout and L2 regularisation to encourage generalisation. Data was split into 80% training and 20% validation-test data and trained for 10 epochs. Validation accuracy plateaued around epochs 8–10 and the train/validation gap stayed under 5%, indicating no clear overfitting.

## Evaluation

- Confusion-matrix accuracy: **96.2%**
- Classification-report accuracy: **84.85%**
- Precision: **0.8455**
- Recall: **0.8485**
- F1 score: **0.8467**
- Error rate: **15.15%**

The model performs better on High (precision **0.89**, recall **0.91**) than Moderate (precision **0.70**, recall **0.66**), as expected from the class imbalance.

> **Important:** The 96.2% confusion-matrix accuracy and 84.85% classification-report accuracy do not match. These results should be double-checked before being used in a final presentation.

## Cause Analysis

- **Weekly Sessions:** High users average 18.4 sessions/week versus 12.5 for Moderate users, making this the strongest behavioural signal.
- **Binge Watch Frequency:** Distributions are nearly identical, so it is not a meaningful driver.
- **Country:** The second-strongest factor, reflecting viewing habits across 44 countries.
- **Age Group:** Younger users tend to be more represented in the High category.

## Final Findings

The primary drivers of High screen time are **Weekly Sessions, Country,** and **Age Group**. Platform, Content Category, Device Type, Subscription Type, Internet Connection, Content Language, Peak Viewing Hour, and Binge Watch Frequency showed minimal influence in this analysis.

The key takeaway is that high screen time is driven mainly by **how often users open the app** and **who they are or where they are from**, rather than by what they watch or which device or platform they use.

## Limitations

- Weak MI scores for most features suggest limited predictive signal.
- Country is a coarse proxy that may hide culture, internet access, and economic conditions.

## Outcomes

The dataset shows that high screen time is observed across all countries and demographics. Typical users are adults with **17–18 weekly sessions** and **42–44 minutes per session**. They often use 2–4 platforms (e.g., Amazon Prime Video, Instagram, Netflix, Disney+, TikTok) and have diverse internet connections (Broadband, 4G LTE, 5G, ADSL). Content preferences vary widely (Action, Documentary, Horror, Sports, Animation, Drama).

For example:
- An adult male from Australia watches **Documentary** on **Amazon Prime Video** (Desktop, Basic plan), with 18 weekly sessions (~44 min each), using 3 platforms and 4G LTE.
- An adult male from Brazil watches **Sports** on **Instagram** (Tablet, Free plan), with 18 weekly sessions (~43 min each), using 3 platforms and 5G.

These profiles illustrate that **high screen time is driven more by usage frequency and regional/age factors than by content or device type** (as confirmed by the feature selection in this project).

## Repository Contents

- `ScreenTime_Analysis.ipynb` — complete analysis, visualisations, model training, evaluation, and cause analysis
- `Global_Video_Screen_Time_Analysis_20000_Rows.csv` — dataset used by the notebook

## How to Run

Open `ScreenTime_Analysis.ipynb` in Jupyter Notebook or VS Code and run the cells in order. The notebook uses pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and TensorFlow/Keras.
