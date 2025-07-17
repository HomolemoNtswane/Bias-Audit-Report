Bias Audit in IT Hiring - Combined Gender & Race Analysis
Overview

This application provides a fairness audit tool for analyzing potential biases in income prediction models, with a focus on combined gender and race groups. The tool evaluates model performance across different demographic groups and provides visualizations and metrics to assess fairness.

Features

- Dataset: Uses the UCI Adult dataset (census income data)
- Model: Logistic Regression classifier predicting income (>50K or ≤50K)
- Fairness Metrics:
  - Accuracy
  - Selection Rate
  - True Positive Rate
  - False Positive Rate
  - Demographic Parity Difference
  - Equalized Odds Difference
- Visualizations: Interactive bar charts showing metrics across groups
- Reports: Downloadable CSV reports of fairness metrics

How to Use

1. Install dependencies:
   ```bash
   pip install -r requirements.txt


2. Run the application:
   ```bash
   python app.py
  

3. Interface options:
   - Select "All" for either Gender or Race to view group-level fairness metrics
   - Select specific Gender and Race combinations to get individual predictions
   - Download fairness reports in CSV format

Key Functionality

- Combines gender and race attributes for intersectional fairness analysis
- Provides both individual predictions and group-level fairness metrics
- Visualizes fairness metrics across all demographic groups
- Explains key fairness concepts (Demographic Parity, Equalized Odds)

Technical Details

- Data Processing: 
  - Handles categorical variables with label encoding
  - Scales numerical features
  - Splits data into training/test sets (70/30)

- Model: 
  - Logistic Regression with 2000 max iterations
  - Evaluated on standard classification metrics

- Fairness Assessment:
  - Uses Fairlearn's MetricFrame for group metrics
  - Calculates demographic parity and equalized odds differences

Requirements

- Python 3.7+
- Packages listed in requirements.txt:
  - pandas, numpy
  - scikit-learn
  - fairlearn
  - gradio
  - plotly

