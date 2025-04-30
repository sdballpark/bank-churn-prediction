# 💼 Bank Churn Prediction: End-to-End ML Pipeline with CI/CD 🚀

This project showcases a **production-ready machine learning pipeline** to predict customer churn in the banking sector. It integrates modern data science practices, DevOps automation, and web deployment — making it a **recruiter-facing, enterprise-grade ML portfolio piece**.

---

## 🧠 Problem Statement

Customer churn is one of the most costly challenges in retail banking. Using historical customer and transaction data, we built a predictive model to identify customers most likely to leave the bank, enabling proactive retention strategies.

---

## 📊 Features & Highlights

- ✅ **EDA**: Outlier detection, class imbalance handling, correlation mapping
- ✅ **Data Preprocessing**: Label encoding, feature selection, train-test splitting
- ✅ **Modeling**: Trained with Random Forest, XGBoost, and Logistic Regression
- ✅ **Metrics**: Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC
- ✅ **Interpretability**: SHAP-based feature importance visualizations

---

## 🧱 Folder Structure

```bash
.
├── app/                      # Streamlit application
│   ├── streamlit_app.py
│   └── requirements.txt
├── data/
│   ├── raw/                  # Original Churn.csv
│   └── processed/            # Cleaned and transformed data (optional)
├── models/                   # Trained model artifact (model.pkl)
├── notebooks/                # EDA + Modeling notebooks
│   ├── 1_EDA.ipynb
│   └── 2_Modeling.ipynb
├── src/                      # Modular Python scripts
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── model_inference.py
├── .github/workflows/        # GitHub Actions pipeline
│   └── deploy.yml
├── terraform/                # Infrastructure as Code (optional)
├── .env                      # Local-only environment variables (excluded from repo)
├── Dockerfile                # For containerized deployment
└── README.md
```

---

## ⚙️ CI/CD with GitHub Actions

This project includes a **CI/CD pipeline** that:

- ✅ Validates model training code on push
- ✅ Automatically runs inference tests
- ✅ Deploys the Streamlit app
- ✅ Runs in headless mode for testing
- ✅ Blocks secrets using `.env` and `.gitignore`

```yaml
# .github/workflows/deploy.yml
- Checkout repo
- Setup Python 3.11
- Install dependencies
- Run training and inference scripts
- Smoke test Streamlit app
```

---

## 🌐 Streamlit App (Interactive)

Once deployed, the app:

- Accepts user-input customer data
- Outputs prediction (Churn / Not Churn)
- Displays SHAP-based explainability

**Launch locally:**

```bash
cd app
streamlit run streamlit_app.py
```

---

## 📦 Docker Deployment (Optional)

To containerize the app:

```bash
docker build -t churn-app .
docker run -p 8501:8501 churn-app
```

---

## 🧪 Future Enhancements

- ✅ Cross-validation + Hyperparameter tuning
- ✅ Streamlit Cloud deployment via GitHub hooks
- ✅ Unit tests on inference engine
- ✅ MLflow or DVC versioning
- ✅ Expand SHAP interpretability

---

## 🧑‍💻 Skills Showcased

| Category         | Technologies / Tools                                  |
|------------------|--------------------------------------------------------|
| ML & Modeling     | Scikit-learn, XGBoost, SHAP, Pandas, NumPy             |
| Visualization     | Seaborn, Matplotlib, Streamlit                        |
| Deployment        | Docker, GitHub Actions CI/CD, `.env` security         |
| DevOps & IaC      | Git, GitHub Workflows, Terraform (optional)          |
| Collaboration     | Modular codebase, clean documentation                 |

---

## 👤 Author

**Robert Bogan**  
Cybersecurity Architect & Machine Learning Engineer  
🔗 GitHub: [@sdballpark](https://github.com/sdballpark)  
🔗 LinkedIn: [linkedin.com/in/robert-l-bogan-jr](https://linkedin.com/in/robert-l-bogan-jr)

---

## ⭐️ Recruiter Notes

This project is intended as a **technical showcase** of real-world ML engineering skills:  
CI/CD integration, modeling best practices, and secure deployment — ideal for **ML Ops**, **Data Scientist**, and **ML Engineer** roles.

