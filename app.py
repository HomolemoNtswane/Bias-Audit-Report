
import pandas as pd
import numpy as np
import gradio as gr
import tempfile
import os
from shutil import copyfile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (
    MetricFrame, selection_rate, true_positive_rate, false_positive_rate,
    demographic_parity_difference, equalized_odds_difference
)
import plotly.graph_objects as go

# Load dataset
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race",
        "sex", "capital_gain", "capital_loss", "hours_per_week",
        "native_country", "income"
    ]
    df = pd.read_csv(url, names=columns, sep=r",\s*", engine="python")
    df = df[~df.isin(["?"]).any(axis=1)].reset_index(drop=True)
    df["income"] = df["income"].apply(lambda x: 1 if x == ">50K" else 0)

    categorical = ["workclass", "education", "marital_status", "occupation",
                   "relationship", "race", "sex", "native_country"]
    for col in categorical:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

# Prepare data
df = load_data()
X = df.drop("income", axis=1)
y = df["income"]
sex_labels = df["sex"]
race_labels = df["race"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
sex_train, sex_test = train_test_split(sex_labels, test_size=0.3, random_state=42)
race_train, race_test = train_test_split(race_labels, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Helper to map labels
def gender_label(val):
    return "Female" if val == 0 else "Male"

race_map_labels = {0:"Amer-Indian-Eskimo",1:"Asian-Pac-Islander",2:"Black",3:"Other",4:"White"}

def race_label(val):
    return race_map_labels.get(val, "Other")

# Create combined group labels
combined_sensitive = sex_test.astype(str) + "-" + race_test.astype(str)
combined_sensitive = combined_sensitive.map(lambda x: f"{gender_label(int(x.split('-')[0]))}-{race_label(int(x.split('-')[1]))}")

# Fairness metrics for combined groups
def compute_combined_metric_frame():
    metrics = {
        "Accuracy": accuracy_score,
        "Selection Rate": selection_rate,
        "True Positive Rate": true_positive_rate,
        "False Positive Rate": false_positive_rate
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=combined_sensitive
    )
    return mf

# Plotly chart for combined groups
def create_combined_fairness_plot(df):
    fig = go.Figure()
    for metric in df.columns:
        fig.add_trace(go.Bar(
            x=df.index,
            y=df[metric],
            name=metric
        ))
    fig.update_layout(
        barmode='group',
        title="📊 Fairness Metrics by Gender-Race Groups",
        xaxis_title="Group (Gender-Race)",
        yaxis_title="Score",
        height=500,
        xaxis_tickangle=-45
    )
    return fig

# Gradio main function
def audit_result(gender: str, race: str):
    # Compute combined metric frame
    metric_frame = compute_combined_metric_frame()
    report_df = metric_frame.by_group.round(3)

    # Nicely formatted report table string
    report_str = report_df.reset_index().rename(columns={"index":"Group"}).to_string(index=False)

    # Save CSV with fixed name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="") as tmp:
        report_df.to_csv(tmp.name, index=True)
        csv_download_path = os.path.join(tempfile.gettempdir(), "gender_race_fairness_report.csv")
        copyfile(tmp.name, csv_download_path)

    # Fairness explanations for combined groups
    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=combined_sensitive)
    eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=combined_sensitive)

    dp_text = (
        f"🟣 **Demographic Parity Difference**: {dp_diff:.3f}\n"
        "Measures difference in selection rates between groups. Ideal value is 0."
    )

    eo_text = (
        f"🟢 **Equalized Odds Difference**: {eo_diff:.3f}\n"
        "Measures difference in error rates between groups. Closer to 0 is better."
    )

    fairness_chart = create_combined_fairness_plot(report_df)

    # If either input is "All" show the combined group report only
    if gender == "All" or race == "All":
        return (
            "ℹ️ Group-level fairness audit for combined Gender-Race groups. No individual prediction.",
            report_str,
            dp_text,
            eo_text,
            fairness_chart,
            csv_download_path
        )

    # Individual prediction
    sex_encoded = 1 if gender == "Male" else 0
    race_map = {"White": 4, "Black": 2, "Asian-Pac-Islander": 1, "Amer-Indian-Eskimo": 0, "Other": 3}
    race_encoded = race_map.get(race, 4)

    sample = np.mean(X, axis=0).copy()
    sample[X.columns.get_loc("sex")] = sex_encoded
    sample[X.columns.get_loc("race")] = race_encoded

    scaled = scaler.transform([sample])
    pred = model.predict(scaled)[0]
    prediction = "Predicted: >50K" if pred == 1 else "Predicted: ≤50K"

    return (
        prediction,
        report_str,
        dp_text,
        eo_text,
        fairness_chart,
        csv_download_path
    )

# Gradio UI
demo = gr.Interface(
    fn=audit_result,
    inputs=[
        gr.Dropdown(["All", "Male", "Female"], label="Select Gender"),
        gr.Dropdown(["All", "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"], label="Select Race")
    ],
    outputs=[
        gr.Textbox(label="Model Prediction"),
        gr.Textbox(label="📊 Combined Gender-Race Fairness Report (Readable Table)", lines=10),
        gr.Textbox(label="Demographic Parity Explanation"),
        gr.Textbox(label="Equalized Odds Explanation"),
        gr.Plot(label="📈 Visual Fairness Comparison (Gender-Race Groups)"),
        gr.File(label="📥 Download gender_race_fairness_report.csv")
    ],
    title="🔍 Bias Audit in IT Hiring — Combined Gender & Race",
    description="Select 'All' for Gender or Race to view group fairness across combined gender-race groups."
)

demo.launch()
