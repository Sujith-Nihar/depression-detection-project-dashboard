# Save this as app.py

import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

st.set_page_config(layout="wide", page_title="Depression Detection Dashboard")

# --- Title ---
st.title("📊 Depression Detection from Social Media Posts")
st.markdown("A comprehensive dashboard visualizing emotion trends and predictive model results for depression detection.")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", ["Project Overview", "Visualizations", "Model Performance"])

# --- Project Overview Section ---
if section == "Project Overview":
    st.header("📚 Project Overview")
    st.markdown("""
    **Depression Detection from Social Media Posts: Project Summary**

---

### 1. Mood Course (Mood Progression) Per Profile

We analyzed the mood progression for different patient profiles based on their social media activity. The summary and results of the mood progression for selected profiles are available at: [https://patient-medical-records.vercel.app/](https://patient-medical-records.vercel.app/)

The mood progression was derived by tracking changes in emotional states across a timeline, capturing trends and fluctuations in mental health over periods.

---

### 2. Emotion Scoring Using LLM

In the next step, we employed a Large Language Model (LLM) to evaluate each post for six fundamental emotions:

- Happiness
- Sadness
- Anger
- Fear
- Surprise
- Disgust

The LLM assigned emotion scores based on:

- **Image/Video** content analysis
- **Embedded Text** extracted from the media
- **Caption** accompanying the posts

Each post thus had a set of emotion scores across these six emotions, forming a multimodal emotional profile.

---

### 3. Visualization of Trends

We categorized the users into two groups:

- **Happy Users** (generally positive profiles)
- **Depressed Users** (profiles with signs of depression)

We visualized the trends in emotion scores for both groups over time:

- **Happy users** exhibited stable or increasing trends in happiness and lower scores in sadness and fear.
- **Depressed users** showed rising sadness, fear, and disgust levels while happiness declined.

These trends demonstrated the emotional divergence between happy and depressed users effectively.

---

### 4. Feature Extraction and Dataset Creation

From the LLM output and previous multimodal analysis (conducted using Gemini), we collected and cleaned the dataset.

Selected features:

- **Emotion Scores**: Happiness, Sadness, Anger, Fear, Surprise, Disgust.
- **PHQ-9 Features**: Standard clinical questionnaire features evaluating depression symptoms.
- **Image Features**: Saturation Value, Brightness Value.


This dataset was compiled for multiple user profiles and structured for machine learning.

---

### 5. Labeling of Posts

We defined a custom labeling function to identify whether a post was "Depressed" or "Not Depressed".

#### **Labeling Method**

```python
# Step 5: Define Labeling Function
def label_depressed_post(row):
    phq9_sum = row[phq9_columns].sum()

    cond1 = phq9_sum >= 3  # PHQ-9 Symptoms condition

    sadness = row['Sadness']
    fear = row['Fear']
    disgust = row['Disgust']
    happiness = row['Happiness']
    cond2 = ((sadness > 0.5) or (fear > 0.5) or (disgust > 0.5)) and (happiness < 0.3)

    brightness = row['Brightness Value']
    saturation = row['Saturation Value']
    cond3 = (brightness < 30) and (saturation < 30)

    conditions_met = sum([cond1, cond2, cond3])

    if conditions_met >= 2:
        return "Yes"
    else:
        return "No"
```
    
A post is labeled as "Depressed" if at least **two out of three conditions** are met:

1. PHQ-9 sum >= 3.
2. High Sadness/Fear/Disgust and low Happiness.
3. Low Brightness and Saturation values.

---

### 6. Machine Learning Model Training

- **Data Split**: We split the data into 80% training and 20% testing.
- **Features Used**: Emotion scores, PHQ-9 features, Saturation, Brightness.

#### Models Trained:

- **Logistic Regression**
- **Random Forest Classifier**

We trained the models to predict whether a post is a "Depressed" post or not.

---

### 7. Updated Model Performance (After Removing Hue Distribution)

#### **Random Forest Classifier**

- **Accuracy**: **98.06%**
- **Class 1 (Depressed) F1-Score**: **94.87%**
- **Macro Avg F1-Score**: **96.84%**

#### **Logistic Regression**

- **Accuracy**: **97.09%**
- **Class 1 (Depressed) F1-Score**: **92.68%**
- **Macro Avg F1-Score**: **95.43%**

**Key Findings**:

- Random Forest provided higher overall accuracy and F1-Score.
- Emotion scores and PHQ-9 features remained strong predictors.

---

### 8. Conclusion

Our approach combined multimodal emotional analysis, LLM-based emotion scoring, and clinical features to build a predictive model for depression detection from social media posts.

The visualization trends, combined with machine learning results, confirmed that emotional shifts and visual content characteristics can effectively signal mental health states.


---

    

    """)

# --- Visualization Section ---
elif section == "Visualizations":
    st.header("📈 Comparative Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("😃 Normal User")
        st.image("normal/pos_neg.png", caption="Positive vs Negative Trend", use_container_width=True)
        st.image("normal/emotion_trend.png", caption="Smoothed Emotion Trends", use_container_width=True)
        st.image("normal/radar.png", caption="Average Emotion Profile (Radar Chart)")

    with col2:
        st.subheader("😞 Depressed User")
        st.image("depressed/pos_neg.png", caption="Positive vs Negative Trend", use_container_width=True)
        st.image("depressed/emotion_trend.png", caption="Smoothed Emotion Trends", use_container_width=True)
        st.image("depressed/radar.png", caption="Average Emotion Profile (Radar Chart)")

    st.subheader("Detailed Comparison: Healthy vs Depressed Individuals")

    st.markdown("### 🔹 Cross-User Emotion Divergence")
    st.markdown("""
    - **Normal users** maintain emotional consistency with stable high happiness and nearly absent negative emotions.
    - **Depressed users** show a **temporal emotional shift** — happiness falls, and sadness + fear increase gradually.
    - The **positive-negative crossover** in depressed users (around 2022-09-09) is a red flag indicating mood reversal.
    """)

    st.markdown("### 🔹 Comparative Radar Chart Insights")
    st.markdown("""
    - Normal: Highly peaked toward **happiness (0.9+)**, negligible others.
    - Depressed: Flatter, with **sadness (~0.35)** and **fear (~0.2)** joining happiness (~0.5).
    - Indicates **diluted emotional positivity** and onset of negative emotions in depressed users.
    """)

    st.markdown("### 🔹 Summary Metrics")
    st.markdown("""
    | Emotion | Normal (avg) | Depressed (avg) |
    |---------|---------------|------------------|
    | Happiness | 0.90 | 0.50 |
    | Sadness | ~0.05 | 0.35 |
    | Fear | ~0.03 | 0.20 |
    | Disgust/Anger | ~0.01 | 0.05–0.08 |
    """)

elif section == "Model Performance":
    st.header("📝 Model Performance Summary")

    # --- Load Data ---
    st.subheader("🗂️ Data Overview")
    df = pd.read_excel('cleaned_merged_output.xlsx')
    st.table(df.head(10))

    # --- Feature Selection ---
    drop_columns = ['Media Name', 'Profile Name', 'Simple Description', 'Embedded Text', 'Caption', 'Important Note', 'Diagnosed Date', 'Media Type']
    drop_columns = [col for col in drop_columns if col in df.columns]

    X = df.drop(columns=drop_columns + ['Depressed post'], errors='ignore')
    X = X.select_dtypes(include=['number']).fillna(X.mean())

    y = df['Depressed post'].map({'Yes': 1, 'No': 0})

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model Training (Random Forest) ---
    rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # --- Model Training (Logistic Regression) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    logreg_model.fit(X_train_scaled, y_train)
    y_pred_logreg = logreg_model.predict(X_test_scaled)

    # --- Confusion Matrices ---
    st.subheader("Random Forest - Confusion Matrix")
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    fig_rf, ax_rf = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax_rf)
    st.pyplot(fig_rf, use_container_width=False)

    st.subheader("Logistic Regression - Confusion Matrix")
    cm_logreg = confusion_matrix(y_test, y_pred_logreg)
    fig_logreg, ax_logreg = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Purples', ax=ax_logreg)
    st.pyplot(fig_logreg, use_container_width=False)

    # --- Feature Importances ---
    st.subheader("Feature Importances (Random Forest)")
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig_feat, ax_feat = plt.subplots(figsize=(6, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis', ax=ax_feat)
    ax_feat.set_title('Feature Importances - Random Forest')
    st.pyplot(fig_feat)

    # --- ROC Curve ---
    st.subheader("ROC Curve Comparison")

    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)

    y_prob_logreg = logreg_model.predict_proba(X_test_scaled)[:, 1]
    fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_prob_logreg)
    auc_logreg = roc_auc_score(y_test, y_prob_logreg)

    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
    ax_roc.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
    ax_roc.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (AUC = {auc_logreg:.2f})')
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend(loc='lower right')
    st.pyplot(fig_roc)

    # --- Precision-Recall Curve ---
    st.subheader("Precision-Recall Curve Comparison")

    precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_prob_rf)
    ap_rf = average_precision_score(y_test, y_prob_rf)

    precision_logreg, recall_logreg, _ = precision_recall_curve(y_test, y_prob_logreg)
    ap_logreg = average_precision_score(y_test, y_prob_logreg)

    fig_pr, ax_pr = plt.subplots(figsize=(6, 4))
    ax_pr.plot(recall_rf, precision_rf, label=f'Random Forest (AP = {ap_rf:.2f})')
    ax_pr.plot(recall_logreg, precision_logreg, label=f'Logistic Regression (AP = {ap_logreg:.2f})')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.legend(loc='lower left')
    st.pyplot(fig_pr)

    # --- Correlation Heatmap ---
    st.subheader("Feature Correlation Matrix")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    corr = df.select_dtypes(include=['number']).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax_corr)
    ax_corr.set_title('Feature Correlation Matrix')
    st.pyplot(fig_corr)

    # --- Model Evaluation Metrics ---
    st.subheader("Model Performance Metrics")

    st.write(f"**Random Forest Accuracy:** {round(rf_model.score(X_test, y_test), 4)}")
    st.write(f"**Logistic Regression Accuracy:** {round(logreg_model.score(X_test_scaled, y_test), 4)}")

    st.markdown("**Random Forest - Classification Report**")
    rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
    st.table(pd.DataFrame(rf_report).transpose())

    st.markdown("**Logistic Regression - Classification Report**")
    logreg_report = classification_report(y_test, y_pred_logreg, output_dict=True)
    st.table(pd.DataFrame(logreg_report).transpose())

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Depression Detection Project | Built with ❤️ using Streamlit")