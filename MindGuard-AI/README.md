# MindGuard-AI: Social Media Impact Predictor 🧠📱

## 📌 Project Overview
MindGuard-AI is a Machine Learning tool developed to analyze the relationship between social media habits and teen mental health. Utilizing a behavioral dataset of 1,200 entries, the model predicts the risk of depression based on digital lifestyle factors such as platform usage, sleep patterns, and screen time.

## 🚀 The Engineering Challenge: Class Imbalance
The primary challenge of this project was an **extreme class imbalance** (less than 1% of the dataset represented the "At Risk" category). A standard model achieved 97% accuracy but **0% recall**, failing to identify a single high-risk individual.

### 🛠️ How I Fixed It:
To move the model from a "silent failure" to a functional tool, I implemented:
1. **SMOTE (Synthetic Minority Over-sampling Technique):** To generate synthetic training data for the minority class.
2. **Stratified Splitting:** To ensure the tiny "At Risk" population was proportionally represented in both training and testing sets.
3. **Probability Thresholding:** I adjusted the decision threshold from **0.5 to 0.2**, prioritizing **Recall (Sensitivity)** over Precision to ensure high-risk cases are not missed in a wellness context.

## 📊 Results
* **Accuracy:** 94%
* **Recall (Risk Category):** 67% (Successfully identified 4/6 at-risk cases in the test set)
* **Macro Avg Recall:** 81%

## 🛠️ Tech Stack
- **Language:** Python 3.12
- **Libraries:** Pandas, Scikit-Learn, Imbalanced-Learn (SMOTE), Seaborn, Matplotlib
- **Model:** Random Forest Classifier with Balanced Class Weights

## 📈 Key Insights (Feature Importance)
The model identified the following as the strongest predictors of mental health risk:
1. **Sleep Hours:** The strongest biological predictor.
2. **Daily Social Media Usage:** Quantity of time spent.
3. **Platform Type:** Identifying specific impacts of TikTok vs. Instagram algorithms.

## 💻 How to Run
1. Clone the repository.
2. Install dependencies: `pip install pandas scikit-learn imbalanced-learn seaborn`
3. Run the main script: `python ml_project_03.py`

---
*Developed as part of a CSE Machine Learning exploration into Digital Wellness.*