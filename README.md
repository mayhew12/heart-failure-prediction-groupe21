# Medical Decision Support Application: Predicting Heart Failure Risk

## Description
This project is an advanced clinical decision-support tool developed to accurately predict the risk of heart failure in patients using explainable machine learning.

## Reproducibility
To fully reproduce this project on your local machine, please follow these steps:

1. **Install dependencies:**
   `pip install -r requirements.txt`
2. **Train the model:**
   `python src/train_model.py`
3. **Run the web application:**
   `streamlit run app/app.py`

## Critical Questions & Analysis

### 1. Was the dataset balanced? How did you handle imbalance and what was the impact?
The original dataset was imbalanced, with approximately 68% of patients surviving and 32% deceased. To handle this, we applied SMOTE (Synthetic Minority Over-sampling Technique) exclusively to our training data after performing the train-test split. The impact of this technique was significant: it prevented our model from developing a bias toward predicting survival, ultimately improving our recall score for the minority class.

### 2. Which ML model performed best? Provide performance metrics.
After evaluating Random Forest, XGBoost, LightGBM, and Logistic Regression, **[Insert Winning Model Name Here]** performed the best.
* **ROC-AUC:** [Insert Score]
* **Accuracy:** [Insert Score]
* **Precision:** [Insert Score]
* **Recall:** [Insert Score]
* **F1-Score:** [Insert Score]

### 3. Which medical features most influenced predictions (SHAP results)?
Based on our SHAP summary plot analysis, the most influential features in predicting heart failure risk were **[Insert Top 3 Features from your SHAP plot, e.g., time, serum_creatinine, ejection_fraction]**. The waterfall plots in our application clearly demonstrate how abnormal levels in these specific features push individual patient risk scores higher.

### 4. Prompt Engineering Documentation
**Task Selected:** Developing the memory optimization function.
* **Prompt Used:** "Write a Python function for a pandas dataframe that iterates through columns and changes float64 to float32, and int64 to smaller integer types to optimize memory. Print the memory before and after."
* **Results:** The AI generated a highly effective function using `df.memory_usage()`, which successfully reduced the dataset's memory footprint by over 50%.
* **Effectiveness & Improvements:** The initial prompt was effective, but an improvement would be to specify handling edge cases, such as ignoring columns with "object" or "string" data types to prevent errors during the iteration.