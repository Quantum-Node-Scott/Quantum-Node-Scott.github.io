---
layout: single
title:  "#3 Mastering Python Fundamentals"
categories: Bootcamp
tag: [패스트캠퍼스, 패스트캠퍼스AI부트캠프, 업스테이지패스트캠퍼스, UpstageAILab, 국비지원, 패스트캠퍼스업스테이지에이아이랩, 패스트캠퍼스업스테이지부트캠프]
author_profile: false
---

# 🐍 Mastering Python Fundamentals: A Beginner’s Guide to AI Coding

**Hello again**, this is my third AI blog post.
This week, with tutor **Devman, I learnt PythonPython🔥

![Python Fundamentals](/assets/images/python_fundamentals.jpg)

## 💡 Why Python?
Python is not just for AI — it’s great for beginners too!  
You can use Python to:

- Build simple calculators  
- Organize files on your computer  
- Automate boring tasks (like renaming files)  
- Make your own web scraper  
- Or even create fun mini-games with turtle graphics!  

Python is the most popular language in AI and computer science.  
It’s beginner-friendly, readable, and powerful — perfect for learning how to code smart machines.

In this post, I’ll walk through Python basics in an easy way.  
Let’s explore how to write code that works — and makes sense.

---

## 🧱 1. Data Types: Python’s Building Blocks

Python lets you store different types of data easily:

```python
# Numbers
age = 25
pi = 3.14
complex_num = 2 + 3j

# Text and collections
name = "Scott"
fruits = ["apple", "banana"]
coordinates = (4.5, 7.2)
unique_ids = {101, 102, 103}
user = {"name": "Scott", "age": 32}
```

> **Tip:** For AI, `NumPy` arrays are better for math than normal Python lists.

---

## 🧩 2. Functions: Reusable Logic

Functions are blocks of code you can run again and again:

```python
def calculate(x, y, learning_rate=0.01):
    slope = 2 * x + 3 * y
    new_x = x - learning_rate * slope
    return new_x

print(calculate(5, 3))  # 4.61
```

- You can add **type hints**:  
  `def greet(name: str) -> str:`
- Or write quick **lambda functions**:  
  `square = lambda x: x**2`

---

## 🧠 3. Methods: Built-In Object Tools

Every object in Python can do things — we call these “methods.”

```python
text = "hello ai"
print(text.upper())           # HELLO AI
print(text.replace("ai", "world"))  # hello world

scores = [0.2, 0.5, 0.8]
scores.append(1.0)
scores.pop(0)  # removes 0.2
```

> In machine learning, we use `.fit()` to train models — it’s a method too!

---

## 🏗️ 4. Classes: Building Custom Structures

You can group related data and actions into a class:

```python
class SimpleModel:
    def __init__(self, layers, activation='relu'):
        self.layers = layers
        self.activation = activation

    def forward(self, inputs):
        print(f"Using {self.activation} activation")
        return sum(inputs) * len(self.layers)

model = SimpleModel([64, 32, 16])
print(model.forward([1, 2, 3]))  # 18
```

> **Advanced Tip:** You can use `class CNN(SimpleModel)` to build on it.

---

## 🔁 5. Loops: Repeat with Power

Loops help process data over and over:

```python
data = [0.1, 0.3, 0.6, 0.9]
normalized = []

for value in data:
    normalized.append(value / max(data))

print(normalized)  # [0.111, 0.333, 0.666, 1.0]
```

```python
metrics = {"accuracy": 0.92, "loss": 0.15}
for key, value in metrics.items():
    print(f"{key}: {value * 100:.1f}%")
```

> Use list comprehensions too:  
`squared = [x**2 for x in range(5)]`

---

## 📊 6. Pandas: Handle Table Data with Ease

```python
import pandas as pd

data = {
    "Temp": [22.5, 23.1, 24.8],
    "Humidity": [45, 62, 58],
    "Rainfall": [0.0, 0.2, 0.5]
}

df = pd.DataFrame(data)
dry_days = df[df["Rainfall"] == 0.0]
print(dry_days.mean())  # Temp: 22.5, Humidity: 45
```

> Pandas is great for reading CSVs, filtering data, and handling missing values.

---

## 🌐 7. Web Crawling with ChromeDriver

You can also use Python to **get data from websites**.  
Here’s a basic example using `selenium` and ChromeDriver:

```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# Set up the browser
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("https://example.com")

# Get the title of the page
print(driver.title)

# Get all paragraph texts
paragraphs = driver.find_elements(By.TAG_NAME, "p")
for p in paragraphs:
    print(p.text)

driver.quit()
```

> 📌 Tip: You need to install `selenium`:
> ```bash
> pip install selenium
> ```

---
## 📈 8. Matplotlib: Show Your Results Visually

```python
import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]
loss = [0.9, 0.6, 0.3, 0.2, 0.1]

plt.plot(epochs, loss, 'r--', label='Loss')
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
```

> Try `Seaborn` too if you want beautiful charts like heatmaps.

---

## 🎯 With Python, you can analyze any type for AI
