# 📊 Customer Churn Prediction Dashboard

🔗 **Live App:** https://multimodelappproject-auezkzbgnw3qwkct8kv6x9.streamlit.app

## 🎯 Business Problem
Telecom company ke customers churn kar rahe hain. Is project mein 6 ML models train kiye gaye hain jo predict karte hain kaunsa customer service chhodega.

## 📁 Dataset
- **Source:** Telco Customer Churn (Kaggle)
- **Records:** 7,032
- **Features:** 20

## 🤖 ML Models Used
| Model | ROC-AUC |
|-------|---------|
| Logistic Regression | 0.8357 |
| Decision Tree | 0.6343 |
| Random Forest | 0.8385 |
| KNN | 0.7667 |
| SVM | 0.7909 |
| **Gradient Boosting** ⭐ | **0.8410** |

## 🏆 Champion Model
**Gradient Boosting** — ROC-AUC: 0.8410

## 🛠️ Technologies
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Streamlit, Plotly
- GitHub, Streamlit Cloud

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📂 Project Structure
```
├── app.py              # Main Streamlit dashboard
├── requirements.txt    # Dependencies
├── telco_churn.csv     # Dataset
├── *.pkl               # Trained model files
└── README.md
```
