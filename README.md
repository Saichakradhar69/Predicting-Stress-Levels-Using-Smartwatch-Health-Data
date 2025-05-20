# Predicting Stress Levels Using Smartwatch Health Data

### Group Members:
- **Sai Chakradhar Mattaparthi** – SXM230175  
- **Neel Maheshwari** – NXM230045

---

## 📌 Project Motivation

Wearable technology has transformed personal health monitoring by providing real-time insights into physical activity, heart rate, sleep, and stress. In this project, we analyzed smartwatch health data to build predictive models for estimating stress levels using physiological and behavioral metrics.

Our objective was to explore correlations between health signals and stress, then apply machine learning to predict whether an individual is experiencing **Low**, **Medium**, or **High** stress.

---

## 📂 Dataset Overview

- **Source**: [Kaggle – Smartwatch Health Data](https://www.kaggle.com/datasets/shubhambathwal/stress-level-detection)  
- **Rows**: 10,000 observations  
- **Columns**:  
  - User ID  
  - Heart Rate (BPM)  
  - Blood Oxygen Level (%)  
  - Step Count  
  - Sleep Duration (hours)  
  - Activity Level (categorical)  
  - Stress Level (target – converted to categorical using quantile binning)

---

## 🧭 Project Roadmap

### ✅ Phase 1: Setup and Preprocessing

- Imported essential libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, etc.)
- Cleaned the dataset:
  - Handled missing values
  - Standardized activity labels (`Highly_Active` → `Highly Active`, etc.)
  - Removed or capped outliers (e.g., Heart Rate > 200 BPM, SpO2 < 90%)
- Encoded categorical data (`LabelEncoder`)
- Converted `Stress Level` to **Low**, **Medium**, **High** using `pd.qcut`

---

### ✅ Phase 2: Exploratory Data Analysis (EDA)

- Visualized data distributions using histograms and boxplots
- Detected outliers (e.g., step count > 60,000)
- Observed class imbalance in `Stress Level`
- Noted weak correlations between individual features and stress level
- Concluded that feature interactions might be more valuable than individual predictors

---

### ✅ Phase 3: Modeling

- Problem framed as a **classification** task
- Used 80/20 Train/Test split
- Trained and compared the following models:
  - **Random Forest Classifier**
  - **Logistic Regression**
  - **Support Vector Machine (SVM)**

#### Evaluation Metrics:
- Accuracy
- Precision, Recall, F1-score for each class

#### Observations:
- Most models performed well only on the dominant class (`Medium`)
- Poor recall for `Low` and `High` stress levels indicated class imbalance issues

---

### ✅ Phase 4: Results & Interpretation

- **Random Forest Accuracy**: ~42%
- **Logistic Regression/SVM**: ~42%, heavily biased toward `Medium` class
- Feature importance from tree-based models highlighted:
  - Heart Rate and Sleep Duration as top contributors
- Discussed limitations:
  - Noisy data (e.g., “ERROR” entries in categorical fields)
  - Imbalanced class distribution
  - No temporal context (e.g., trends or sequences across time)

---

### ✅ Phase 5: Final Report (Jupyter Notebook)

- Report structured by project phases
- Each block includes explanations and visualizations
- Code is executable from top to bottom
- Final section includes markdown interpretation, results discussion, and references

---

## 💡 Phase 6: Pushing the Boundaries

To push beyond course basics, we explored additional classification models:

- **Gradient Boosting Classifier**  
- **HistGradient Boosting Classifier**

These models handled the dataset more flexibly and produced more balanced results than simpler models:

- **GradientBoosting Accuracy**: ~41.6%
- **HistGradientBoosting Accuracy**: ~36.4%

We also attempted to improve prediction through feature engineering (e.g., combining health metrics into a custom “Health Score”) and explored class imbalance implications.

---

## 📅 Timeline

| Day | Task                          |
|-----|-------------------------------|
| 1–2 | Preprocessing and cleaning    |
| 3   | Exploratory Data Analysis     |
| 4–5 | Model training and tuning     |
| 6   | Results interpretation        |
| 7   | Final notebook polish         |

---

## 📚 References

- Kaggle. (2023). *Smartwatch Health Data*. https://www.kaggle.com/datasets/shubhambathwal/stress-level-detection  
- Choi, Y., & Lee, J. (2021). *Wearable Device Data and Stress Detection: A Review*. Journal of Healthcare Informatics Research. https://doi.org/10.1007/s41666-021-00091-9  
- McKinney, W. (2010). *pandas: Data Structures for Statistical Computing in Python*. https://doi.org/10.25080/Majora-92bf1922-00a  
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html  
- Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment*. https://doi.org/10.1109/MCSE.2007.55  
- Waskom, M. (2021). *Seaborn: Statistical Data Visualization*. https://joss.theoj.org/papers/10.21105/joss.03021  
- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. https://doi.org/10.1214/aos/1013203451  
- OpenAI. (2024). *ChatGPT: Language Model Assistance*. https://openai.com/chatgpt

---

## 📌 License

This project is academic work under fair use. Data sources and libraries are credited. Code and results are the original work of the authors.
