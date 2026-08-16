# ESTIM AI – AI-Powered Construction Estimation System

> **Smarter Estimates. Better Planning. Faster Construction.**

ESTIM AI is an Artificial Intelligence and Machine Learning-based construction estimation system designed to estimate **material requirements and construction costs** for both **Building Construction** and **Road Construction** projects.

The system uses Machine Learning models to process project parameters and predict required materials, material cost, labour cost, and total construction cost.

---

## 🚀 Project Features

* 🏠 Building Construction Estimation
* 🛣️ Road Construction Estimation
* 🤖 Machine Learning-based prediction
* 🌳 Random Forest Regression
* 🌲 Decision Tree Regression
* 📊 Model comparison using R² Score
* 🧱 Material requirement prediction
* 💰 Material cost estimation
* 👷 Labour cost estimation
* 🏗️ Total construction cost estimation
* 🖥️ Interactive Streamlit web application
* 📄 Downloadable PDF estimation report

---

# 🧠 Machine Learning Models

The project trains and compares two Machine Learning regression models:

## 1. Random Forest Regressor

Random Forest uses multiple Decision Trees to generate more stable predictions.

## 2. Decision Tree Regressor

Decision Tree makes predictions by applying conditions based on the input parameters.

The models are compared using the **R² Score**, and the better-performing model is selected for prediction.

---

# 📊 Model Performance

## 🛣️ Road Construction

| Model             |   R² Score |
| ----------------- | ---------: |
| **Random Forest** | **0.9632** |
| Decision Tree     |     0.8623 |

🏆 **Selected Model: Random Forest**

---

## 🏠 Building Construction

| Model             |   R² Score |
| ----------------- | ---------: |
| **Random Forest** | **0.8385** |
| Decision Tree     |       0.70 |

🏆 **Selected Model: Random Forest**

---

# 🏠 Building Construction

The Building Construction model accepts parameters such as:

* Area Type
* Built-up Area
* Number of Floors
* Wall Thickness
* Roof Type
* Soil Type
* Paint Type
* Flooring
* Construction Quality

### Predicted Outputs

* Cement Required
* Sand Required
* Aggregate Required
* Steel Required
* Bricks Required
* Paint Required
* Total Material Cost
* Labour Cost
* Total Construction Cost

---

# 🛣️ Road Construction

The Road Construction model accepts parameters such as:

* Road Type
* Road Length
* Road Width
* Road Thickness
* Number of Lanes
* Soil Type
* Construction Quality

### Predicted Outputs

* Aggregate Required
* Sand Required
* Cement Required
* Bitumen Required
* Steel Required
* Material Cost
* Labour Cost
* Total Construction Cost

---

# ⚙️ System Workflow

```text
Construction Dataset
        ↓
Data Preprocessing
        ↓
One-Hot Encoding
        ↓
Feature and Target Selection
        ↓
Train-Test Split
        ↓
Random Forest + Decision Tree Training
        ↓
Model Evaluation using R² Score
        ↓
Best Model Selection
        ↓
User Project Parameters
        ↓
Material & Cost Prediction
        ↓
PDF Estimation Report
```

---

# 🛠️ Technologies Used

* **Python** – Core programming language
* **Pandas** – Data processing and manipulation
* **NumPy** – Numerical operations
* **Scikit-learn** – Machine Learning model development
* **Random Forest Regressor** – Final prediction model
* **Decision Tree Regressor** – Model comparison
* **Joblib** – Model saving and loading
* **Streamlit** – Interactive web application
* **ReportLab** – PDF report generation

---

# 📁 Project Structure

```text
AI-Construction-Estimation-System/
│
├── app.py
├── requirements.txt
├── README.md
│
├── ai_construction_estimation_records.csv
├── road_construction_dataset.csv
│
├── construction_rf_model.pkl
├── construction_dt_model.pkl
├── construction_feature_columns.pkl
│
├── road_rf_model.pkl
├── road_dt_model.pkl
├── road_feature_columns.pkl
├── road_y_scaler.pkl
│
├── project.ipynb
└── road.ipynb
```

---

# ▶️ Installation and Usage

### Clone the repository

```bash
git clone https://github.com/madhurawagh-tech/AI-Construction-Estimation-System.git
```

### Open the project folder

```bash
cd AI-Construction-Estimation-System
```

### Install required libraries

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
streamlit run app.py
```

---

# 📄 PDF Report

After entering the construction project parameters and generating a prediction, ESTIM AI can create a downloadable PDF report containing:

* Project information
* Input parameters
* Predicted material requirements
* Material cost
* Labour cost
* Total construction cost

---

# 🔮 Future Scope

Future versions of ESTIM AI can include:

* Larger and more real-world construction datasets
* Real-time material price updates
* Support for additional construction project types
* Improved Machine Learning models
* Government and private project estimation support
* Faster verification of manual estimations
* Detailed analytics and reporting
* Integration with construction planning systems

---

# 🎯 Conclusion

ESTIM AI demonstrates how **Artificial Intelligence and Machine Learning can support parametric construction estimation**.

By combining construction project parameters, data preprocessing, Machine Learning models, cost prediction, and automated reporting, the system provides a simple approach for estimating material requirements and construction costs.

### **From Manual Estimation to Intelligent Construction Planning**

---

## 👩‍💻 Developed By

**Madhura Dhatrak**
Diploma in Artificial Intelligence & Machine Learning
**RSM Polytechnic**

**Industry / Internship:** Codiant Solutions
