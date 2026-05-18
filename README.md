# 🏥 Hospital Readmission Risk Predictor

A machine learning web app that predicts **30-day hospital readmission risk** for diabetic patients using clinical features — with SHAP-based explanations for every prediction.

---

## 📌 Project Overview

Hospital readmissions are costly and often preventable. This project uses a trained ML model to assess patient risk based on their medical history and clinical indicators, helping healthcare providers take proactive action.

---

## 🚀 Features

- 🎛️ Interactive sliders for patient input (age, diagnoses, medications, etc.)
- 🔴🟢 Real-time High / Low risk prediction
- 📊 SHAP waterfall chart explaining *why* the model made each decision
- ⚡ Fast inference using a pre-trained Random Forest / XGBoost model

---

## 🗂️ Project Structure

```
projectml/
├── app.py                  # Streamlit web app
├── diabetic_model.ipynb    # Model training notebook
├── hospital_d.ipynb        # EDA & data analysis notebook
├── model.pkl               # Trained ML model
├── features.pkl            # Feature list used during training
├── sample_input.csv        # Sample input data for testing
├── requirements.txt        # Python dependencies
└── .gitignore              # Files excluded from Git
```

> ⚠️ `diabetic_data.csv` (18 MB) is excluded from this repo due to size.  
> 📥 Download it from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)

---

## ⚙️ Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/your-username/hospital-readmission-predictor.git
cd hospital-readmission-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit app**
```bash
streamlit run app.py
```

---

## 🧠 Features Used by the Model

| Feature | Description |
|---|---|
| `AGE_NUM` | Patient age |
| `time_in_hospital` | Days admitted |
| `num_lab_procedures` | Number of lab tests |
| `num_medications` | Medications prescribed |
| `number_diagnoses` | Total diagnoses |
| `TOTAL_VISITS` | Outpatient + Emergency + Inpatient |
| `SEVERITY_INDEX` | Composite clinical severity score |
| `CHRONIC_FLAG` | 1 if diagnoses > 5 |
| `EMERGENCY_ADMISSION` | Whether admitted via emergency |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~71-74% |
| ROC-AUC | 0.72 |
| Precision | ~34-40% |

> Update these values after training your model.

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Scikit-learn** — ML model
- **SHAP** — Model explainability
- **Streamlit** — Web interface
- **Pandas / NumPy** — Data processing
- **Matplotlib** — Visualization

---

👤 Author

**Anuj Kumawat**  
[GitHub](https://github.com/anujkumawat20) • [LinkedIn](https://linkedin.com/in/anuj-kumawat-48351a293)
