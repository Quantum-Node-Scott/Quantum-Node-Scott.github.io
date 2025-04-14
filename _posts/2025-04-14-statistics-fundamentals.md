---
layout: single
title: "#4 Mastering Statistics for AI Applications"
categories: Bootcamp
tag: [패스트캠퍼스, 패스트캠퍼스AI부트캠프, 업스테이지패스트캠퍼스, UpstageAILab, 국비지원, 패스트캠퍼스업스테이지에이아이랩, 패스트캠퍼스업스테이지부트캠프]
author_profile: false
---

# 📊 Mastering Statistics: A Beginner’s Guide to AI Applications

**Hello again**, welcome to my first AI blog post on statistics!  
This week, I learned foundational concepts in statistics with tutor **Oh Youngseok**, which are essential for analyzing and processing data in AI and machine learning projects.

![Statistics Fundamentals](/assets/images/statistics_fundamentals.jpg)

## 💡 Why Statistics in AI?

Statistics is the backbone of **data science** and **AI**.  
By applying statistical methods, we can:

- Analyze data distributions and patterns
- Estimate and test hypotheses
- Predict outcomes based on collected data

In this post, I’ll walk you through core statistical concepts that are vital for AI projects, paired with practical Python code examples.

---

## 🧱 1. Day 1: Basic Probability and Counting Methods

### 1.1 The Basics of Statistics

#### ✅ What is Statistics?
Statistics is the science of **collecting, analyzing, interpreting**, and **presenting data**. In AI, it helps us make informed decisions, identify patterns, and predict future trends.

#### 🔎 Learning Objectives:
- **Descriptive Statistics:** Summarizing and analyzing data
- **Inferential Statistics:** Making predictions and decisions based on data samples

### 1.2 Rules of Probability

#### 📌 Rule of Sum (Addition Rule)
When two events are mutually exclusive (cannot happen at the same time), the total number of outcomes is the sum of the individual events.

```python
# Example: Rolling a die, how many ways can we get an even or odd number?
even_cases = 3  # 2, 4, 6
odd_cases = 3   # 1, 3, 5
total_cases = even_cases + odd_cases
print("Sum Rule Total Cases:", total_cases)
```

#### 📌 Rule of Product (Multiplication Rule)
For independent events, the total number of outcomes when both events occur is the product of their individual outcomes.

```python
# Example: Choosing a flower vase and a rose from different options
vases = 3
roses = 6
total_combinations = vases * roses
print("Product Rule Total Combinations:", total_combinations)
```

---

## 🧩 2. Day 2: Permutation and Combination

### 2.1 Permutations: Arrangements Matter

#### 📌 What is Permutation?
A permutation is an arrangement of objects where the **order** matters.  
Example: The arrangement "AB" is different from "BA."

```python
import itertools

# Calculating permutation using Python
n, r = 4, 2
permutations = list(itertools.permutations(range(1, n+1), r))
print("Permutations:", permutations)
```

#### 📌 Formula:
- Total permutations of \(n\) objects taken \(r\) at a time:  
  \[
  P(n, r) = \frac{n!}{(n - r)!}
  \]

---

### 2.2 Combinations: Order Doesn't Matter

#### 📌 What is Combination?
A combination is a selection of objects where the **order does not matter**.  
Example: Choosing 3 out of 5 items is the same regardless of order.

```python
import itertools

# Calculating combination using Python
combinations = list(itertools.combinations(range(1, n+1), r))
print("Combinations:", combinations)
```

#### 📌 Formula:
- Total combinations of \(n\) objects taken \(r\) at a time:  
  \[
  C(n, r) = \frac{n!}{r!(n - r)!}
  \]

---

## 🧠 3. Day 3: Estimation and Hypothesis Testing

### 3.1 Estimation: Making Predictions

#### 📌 Point Estimation vs. Interval Estimation
- **Point Estimation:** Predicting a single value for a parameter.  
- **Interval Estimation:** Predicting a range (confidence interval) for a parameter.

```python
# Example: Estimating the mean using a sample
sample_mean = sum([22.5, 23.1, 24.8]) / 3
print("Sample Mean Estimate:", sample_mean)
```

#### 📌 Confidence Intervals
A **confidence interval** is a range that is likely to contain the true parameter with a specified probability (e.g., 95%).

```python
import scipy.stats as stats
import math

# Example: Calculating a 95% confidence interval for the mean
sample_mean = 50
sample_std = 10
n = 30  # sample size
z_score = stats.norm.ppf(0.975)  # for 95% confidence

margin_of_error = z_score * (sample_std / math.sqrt(n))
lower_bound = sample_mean - margin_of_error
upper_bound = sample_mean + margin_of_error

print(f"95% Confidence Interval: ({lower_bound}, {upper_bound})")
```

---

### 3.2 Hypothesis Testing: Validating Assumptions

#### 📌 Null Hypothesis (H₀) vs. Alternative Hypothesis (H₁)
- **Null Hypothesis (H₀):** Assumes no effect or no difference.
- **Alternative Hypothesis (H₁):** Assumes a significant effect or difference.

```python
# Example: Testing if the mean height is equal to 170 cm
H0 = 170  # null hypothesis
sample_data = [168, 172, 175, 169, 171]
sample_mean = sum(sample_data) / len(sample_data)

# t-test can be used for hypothesis testing
from scipy import stats
t_stat, p_value = stats.ttest_1samp(sample_data, H0)
print(f"t-statistic: {t_stat}, p-value: {p_value}")
```

> If the **p-value** is less than a significance level (e.g., 0.05), reject H₀.

---

## 🏗️ 4. Day 4: Logistic Regression for Binary Classification

### 4.1 Binary Classification Overview

#### 📌 What is Binary Classification?
Binary classification is the task of classifying data into one of two categories.  
Examples include classifying emails as **spam** or **not spam**.

```python
# Logistic Regression using Python (sklearn)
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X = data.data
y = (data.target == 0).astype(int)  # Convert to binary classification (setosa vs non-setosa)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X, y)

# Predict for a new data point
print("Predicted Class:", model.predict([X[0]]))
```

#### 📌 Sigmoid Function
The sigmoid function converts any value to a probability between 0 and 1:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

```python
import numpy as np

# Sigmoid function implementation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

z = 0.5  # Example value
print("Sigmoid(0.5):", sigmoid(z))
```

---

### 4.2 Loss Function: Binary Cross-Entropy

#### 📌 Binary Cross-Entropy (Log Loss)
The **binary cross-entropy** is used as the loss function for binary classification, representing how well the model's predictions match the actual labels.

```python
from sklearn.metrics import log_loss

# Example: Model's predicted probabilities and true labels
y_true = [1, 0, 1, 0, 1]
y_pred = [0.9, 0.1, 0.8, 0.3, 0.95]

# Calculate log loss
loss = log_loss(y_true, y_pred)
print("Log Loss:", loss)
```

---

## 🎯 Conclusion: Statistics for AI and Machine Learning

In this post, we covered the core concepts of **probability**, **permutation and combination**, **estimation and hypothesis testing**, and **binary classification**.  
With Python, we can apply these concepts effectively to **analyze data**, **make predictions**, and **optimize machine learning models**.

Stay tuned for more insights in upcoming posts!