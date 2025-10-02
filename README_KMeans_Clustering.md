# 🚀 K-Means Clustering Exercise – Titanic Dataset

This repository contains a practical exercise on **K-Means clustering** using the **Titanic dataset**.  
The goal is to cluster passengers into groups (survived vs. not survived) using unsupervised learning.

---

## 📌 Overview
- **Dataset**: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)  
- **Algorithm**: K-Means Clustering (Unsupervised Machine Learning)  
- **Libraries Used**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`  

---

## 📊 Steps Covered in the Notebook

### 🔹 1. Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
```

---

### 🔹 2. Load Dataset
```python
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
```

- **Train Data**: 891 records (with Age, Sex, Fare, Class, etc.)  
- **Test Data**: 418 records  

---

### 🔹 3. Data Exploration
- Checked **missing values** → (Age, Cabin, Embarked, Fare).  
- Filled missing **Age & Fare** with mean values.  
- Dropped irrelevant columns: `Name`, `Ticket`, `Cabin`, `Embarked`.  
- Encoded categorical feature `Sex` using **LabelEncoder**.  

---

### 🔹 4. Exploratory Data Analysis (EDA)
- Survival rate by **Pclass**, **Sex**, **SibSp**.  
- Histograms of **Age distribution** across survivors vs. non-survivors.  
- Visualized with **Seaborn FacetGrid**.  

---

### 🔹 5. Feature Selection
Final features used for clustering:  
- `PassengerId`  
- `Pclass`  
- `Sex`  
- `Age`  
- `SibSp`  
- `Parch`  
- `Fare`  

Target (only for evaluation): **Survived**  

---

### 🔹 6. K-Means Clustering
```python
X = np.array(train.drop(['Survived'], axis=1).astype(float))
y = np.array(train['Survived'])

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
```

---

### 🔹 7. Model Evaluation
```python
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float)).reshape(-1, len(X[i]))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
print(correct/len(X))
```

📌 **Result**: Accuracy ≈ **49%**  
*(Note: Since K-Means is unsupervised, clusters may not align perfectly with survival labels.)*  

---

## 🔑 Key Learning Outcomes
✅ How to **preprocess real-world data** (missing values, encoding).  
✅ Understanding **unsupervised learning** (clustering without labels).  
✅ Applying **K-Means** to Titanic passenger data.  
✅ Observing **limitations of clustering** vs. actual labels.  

---

## 📚 References
- [Scikit-learn KMeans Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)  
- [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)  
- [Seaborn FacetGrid](https://seaborn.pydata.org/generated/seaborn.FacetGrid.html)  

---

## 🔗 Explore My Other Repositories
- 📊 [R Programming Practicals](https://github.com/KaustubhSN12/R-Practice)  
- 📈 [Power BI – Salary, Gender & Family Trends](https://github.com/KaustubhSN12/Power-BI_salary-gender-family-trends)  
- 🖥️ [Software Engineering Basics](https://github.com/KaustubhSN12/Software-Engineering-Basics)  
- 🐍 [Python Practice Hub](https://github.com/KaustubhSN12/Python-Practice-Hub)  

---

## 📜 License
This project is licensed under the **MIT License** – free to use and share with credit.  

---

✨ *Star this repository if you found it useful for learning K-Means clustering!*  
