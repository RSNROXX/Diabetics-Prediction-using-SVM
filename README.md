# 🩺 Diabetes Prediction System using SVM

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange?style=flat&logo=scikit-learn)
![Seaborn](https://img.shields.io/badge/Visuals-Seaborn-9cf?style=flat&logo=pandas)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview
This project is a Machine Learning application designed to predict whether a patient is diabetic based on diagnostic measurements.
Using a **Support Vector Machine (SVM)** classifier with a linear kernel, the system analyzes health parameters (like Glucose, BMI, Age) to provide an accurate diagnosis.

The goal is to assist medical professionals by providing a quick, automated initial screening tool.

## 📊 Exploratory Data Analysis (EDA)
Understanding feature relationships is critical for model performance.
### Correlation Heatmap
The heatmap below displays the correlation between different health indicators.
*(Key Insight: Glucose levels and BMI show the strongest correlation with the diabetic outcome.)*

<img width="3000" height="2400" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/d9cb4f74-90f2-46fb-b768-2efb59875737" />

## ⚙️ Model Performance
The model was evaluated using a 20% test split.
* **Algorithm:** Support Vector Machine (SVM)
* **Kernel:** Linear (Chosen for its efficiency in high-dimensional spaces)

### Confusion Matrix
The confusion matrix highlights the model's ability to minimize False Negatives (cases where a diabetic person is incorrectly classified as healthy).

<img width="1800" height="1500" alt="confusion_matrix" src="https://github.com/user-attachments/assets/e82b889d-3884-4abd-8555-9a9d727811bd" />

### Accuracy Metrics
| Metric | Score | Description |
| :--- | :--- | :--- |
| **Training Accuracy** | **77.36%** | Model fit on training data |
| **Test Accuracy** | **76.62%** | Generalization to unseen data |

> **Analysis:** The minimal difference between training and test accuracy (<1%) indicates the model is **robust and not overfitting**.

## 🛠 Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (SVM, StandardScaler, Train-Test Split)
* **Visualization:** Seaborn, Matplotlib

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/diabetes-prediction-svm.git](https://github.com/yourusername/diabetes-prediction-svm.git)
   ```
2. **Install Dependencies:**
   ```bash
   pip install  - r requirements.txt
   ```
3. **Run the Notebook:**
   ```bash
   jupyter nootbook 'Diabetics Prediction .ipynb'
   ```
   
## 🧠 Usage Example
The system accepts manual input for real-time prediction. You can run this directly in the notebook:
   ```python
    Sample Input: (Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age)
    input_data = (5, 166, 72, 19, 175, 22.7, 0.6, 51)

    # Prediction Output
    # "The person is Diabetic!"
   ```
      
## 🔮 Future Scope
* Hyperparameter Tuning: Implementing GridSearchCV to find the optimal C value for the SVM.
* Model Comparison: Evaluating Random Forest and XGBoost to see if non-linear models yield higher accuracy.
* Deployment: Creating a web interface using Streamlit or Flask for easier user access.

---

Author: [Rishikesh Naware](https://www.linkedin.com/in/rishikesh-naware-9a18ab1a8/)
