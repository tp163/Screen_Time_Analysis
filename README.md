# Screen Time Behaviour Analysis

A data mining project that classifies daily screen-time behaviour as **Low**, **Moderate**, or **High** using a neural-network model.

## Objective

Predict a user's daily screen-time category from demographic, viewing, device, platform, and session-related attributes.

The target categories are:

- **Low:** 3 hours or less per day
- **Moderate:** more than 3 and up to 6 hours per day
- **High:** more than 6 hours per day

## Dataset

The project uses `Global_Video_Screen_Time_Analysis_20000_Rows.csv`, which contains 20,000 video screen-time records and 16 attributes. Example variables include country, age group, gender, viewing hour, content category, device type, subscription type, platform, weekly sessions, session duration, and internet connection.

## Workflow

The notebook follows this machine-learning workflow:

1. Exploratory data analysis and correlation analysis
2. Encoding categorical variables
3. Mutual-information feature selection to retain the six most informative features
4. Feature scaling with `StandardScaler`
5. Stratified 80/20 train-test split
6. Neural-network training with L2 regularisation and 30% dropout
7. Validation and overfitting checks
8. Evaluation using a confusion matrix, accuracy, precision, recall, F1 score, and error rate
9. Country-level summaries and profiles of screen-time behaviour

## Model

The classifier uses a feed-forward neural network with the following architecture:

`Input (6 selected features) → Dense (128) → Dense (64) → Dense (32) → Softmax output (3 classes)`

L2 regularisation and dropout are used to help reduce overfitting.

## Repository Contents

- `ScreenTime_Analysis.ipynb` — complete analysis, visualisations, model training, and evaluation
- `Global_Video_Screen_Time_Analysis_20000_Rows.csv` — dataset used by the notebook

## Requirements

Run the notebook with Python and install the packages it imports, including:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
```

## How to Run

1. Clone this repository.
2. Install the required packages.
3. Open `ScreenTime_Analysis.ipynb` in Jupyter Notebook or VS Code.
4. Run all cells in order.
