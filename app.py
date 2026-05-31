import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide", page_icon="📊")

@st.cache_data
def load_data():
    df = pd.read_csv('telco_churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    return df

@st.cache_resource
def load_models():
    rf = joblib.load('random_forest.pkl')
    gb = joblib.load('gradient_boosting.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    return rf, gb, scaler, features

df = load_data()
rf_model, gb_model, scaler, feature_names = load_models()

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "📁 Dataset Explorer",
    "🔍 EDA Dashboard",
    "🤖 Model Training",
    "🏆 Model Comparison",
    "🎯 Prediction"
])

if page == "🏠 Home":
    st.title("📊 Customer Churn Prediction Dashboard")
    st.markdown("### Complete ML Analysis — Telco Customer Churn Dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", "7,032")
    col2.metric("Features", "20")
    col3.metric("ML Models", "6")
    st.markdown("""
    ---
    ## 🎯 Business Problem
    Customer churn matlab customers ka service chhodna.
    Naye customers lana purane se **5x zyada costly** hota hai.
    ## 📌 Project Phases
    | Phase | Description |
    |-------|-------------|
    | Phase 1 | Problem Understanding |
    | Phase 2 | Dataset Analysis |
    | Phase 3 | Exploratory Data Analysis |
    | Phase 4 | Data Preprocessing |
    | Phase 5 | ML Model Training (6 Models) |
    | Phase 6 | Model Comparison |
    | Phase 7 | Hyperparameter Tuning |
    | Phase 8 | This Dashboard! |
    ## 🏆 Champion Model
    **Gradient Boosting** — ROC-AUC: 0.8410
    """)

elif page == "📁 Dataset Explorer":
    st.title("📁 Dataset Explorer")
    st.subheader("Raw Data")
    st.dataframe(df.head(100), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📐 Shape")
        st.write(f"Rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")
    with col2:
        st.subheader("❓ Missing Values")
        st.write(df.isnull().sum())
    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    st.subheader("🔢 Data Types")
    st.write(df.dtypes)

elif page == "🔍 EDA Dashboard":
    st.title("🔍 EDA Dashboard")
    tab1, tab2, tab3 = st.tabs(["Churn Distribution", "Feature Analysis", "Correlation"])
    with tab1:
        fig = px.histogram(df, x='Churn', color='Churn', title='Churn Distribution')
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        col = st.selectbox("Select Column", ['Contract', 'PaymentMethod', 'InternetService', 'SeniorCitizen'])
        fig = px.histogram(df, x=col, color='Churn', barmode='group', title=f'{col} vs Churn')
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        numeric_df = df.select_dtypes(include=np.number)
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, title='Correlation Heatmap', color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Model Training":
    st.title("🤖 Model Training")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    df_enc = pd.get_dummies(df.copy(), drop_first=True)
    df_enc['Churn'] = df_enc.get('Churn_Yes', df_enc.get('Churn', 0))
    if 'Churn_Yes' in df_enc.columns:
        df_enc.drop('Churn_Yes', axis=1, inplace=True)
    X = df_enc.drop('Churn', axis=1)
    y = df_enc['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_test_s = sc.transform(X_test)
    model_choice = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM", "Gradient Boosting"])
    if st.button("🚀 Train Model"):
        model_map = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(probability=True, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }
        with st.spinner(f"Training {model_choice}..."):
            m = model_map[model_choice]
            m.fit(X_train_s, y_train)
            y_pred = m.predict(X_test_s)
            y_prob = m.predict_proba(X_test_s)[:, 1]
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 4))
        col2.metric("Precision", round(precision_score(y_test, y_pred), 4))
        col3.metric("Recall", round(recall_score(y_test, y_pred), 4))
        col4.metric("F1 Score", round(f1_score(y_test, y_pred), 4))
        col5.metric("ROC-AUC", round(roc_auc_score(y_test, y_prob), 4))
        st.success(f"✅ {model_choice} training complete!")

elif page == "🏆 Model Comparison":
    st.title("🏆 Model Comparison")
    results = {
        "Logistic Regression": {"Accuracy":0.8038,"Precision":0.6476,"Recall":0.5749,"F1 Score":0.6091,"ROC-AUC":0.8357},
        "Decision Tree":       {"Accuracy":0.7164,"Precision":0.4661,"Recall":0.4599,"F1 Score":0.4630,"ROC-AUC":0.6343},
        "Random Forest":       {"Accuracy":0.7882,"Precision":0.6234,"Recall":0.5134,"F1 Score":0.5630,"ROC-AUC":0.8385},
        "KNN":                 {"Accuracy":0.7534,"Precision":0.5362,"Recall":0.5348,"F1 Score":0.5355,"ROC-AUC":0.7667},
        "SVM":                 {"Accuracy":0.7868,"Precision":0.6259,"Recall":0.4920,"F1 Score":0.5509,"ROC-AUC":0.7909},
        "Gradient Boosting":   {"Accuracy":0.7953,"Precision":0.6378,"Recall":0.5321,"F1 Score":0.5802,"ROC-AUC":0.8410},
    }
    results_df = pd.DataFrame(results).T
    st.subheader("📊 Metrics Leaderboard")
    st.dataframe(results_df.style.highlight_max(color='lightgreen'), use_container_width=True)
    fig = px.bar(results_df, y=results_df.index, x=['Accuracy','F1 Score','ROC-AUC'],
                barmode='group', title='Model Performance Comparison', orientation='h')
    st.plotly_chart(fig, use_container_width=True)
    st.success("🏆 Champion Model: Gradient Boosting (ROC-AUC: 0.8410)")

elif page == "🎯 Prediction":
    st.title("🎯 Customer Churn Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 0, 120, 65)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)
        senior = st.selectbox("Senior Citizen", [0, 1])
    with col2:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    with col3:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
    if st.button("🔮 Predict Churn"):
        input_dict = {f: 0 for f in feature_names}
        input_dict['tenure'] = tenure
        input_dict['MonthlyCharges'] = monthly_charges
        input_dict['TotalCharges'] = total_charges
        input_dict['SeniorCitizen'] = senior
        if 'Contract_One year' in input_dict and contract == "One year":
            input_dict['Contract_One year'] = 1
        if 'Contract_Two year' in input_dict and contract == "Two year":
            input_dict['Contract_Two year'] = 1
        if 'InternetService_Fiber optic' in input_dict and internet == "Fiber optic":
            input_dict['InternetService_Fiber optic'] = 1
        if 'InternetService_No' in input_dict and internet == "No":
            input_dict['InternetService_No'] = 1
        if 'PaymentMethod_Credit card (automatic)' in input_dict and payment == "Credit card (automatic)":
            input_dict['PaymentMethod_Credit card (automatic)'] = 1
        if 'PaymentMethod_Electronic check' in input_dict and payment == "Electronic check":
            input_dict['PaymentMethod_Electronic check'] = 1
        if 'PaymentMethod_Mailed check' in input_dict and payment == "Mailed check":
            input_dict['PaymentMethod_Mailed check'] = 1
        if 'PaperlessBilling_Yes' in input_dict and paperless == "Yes":
            input_dict['PaperlessBilling_Yes'] = 1
        if 'Partner_Yes' in input_dict and partner == "Yes":
            input_dict['Partner_Yes'] = 1
        if 'Dependents_Yes' in input_dict and dependents == "Yes":
            input_dict['Dependents_Yes'] = 1
        if 'PhoneService_Yes' in input_dict and phone == "Yes":
            input_dict['PhoneService_Yes'] = 1
        input_df = pd.DataFrame([input_dict])
        input_scaled = scaler.transform(input_df)
        prediction = gb_model.predict(input_scaled)[0]
        probability = gb_model.predict_proba(input_scaled)[0][1]
        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠️ **CHURN RISK DETECTED!**")
            st.error(f"Confidence: **{probability*100:.1f}%**")
        else:
            st.success(f"✅ **Customer Will Stay (Retained)**")
            st.success(f"Confidence: **{(1-probability)*100:.1f}%**")
        col1, col2 = st.columns(2)
        col1.metric("Churn Probability", f"{probability*100:.1f}%")
        col2.metric("Retention Probability", f"{(1-probability)*100:.1f}%")
