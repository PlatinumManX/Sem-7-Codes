import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('Salary_Data.csv')

# print(df.isnull().sum())

# print(df.head())

x = df["YearsExperience"]
y = df["Salary"]

n = len(x)

m = (n*(x*y).sum() - x.sum()*y.sum()) / (n*(x*x).sum() - (x.sum())**2)

c = (y.sum() - m*x.sum()) / n

print("Slope =", m)
print("Intercept =", c)

y_pred = m*x + c

plt.scatter(x, y)
plt.plot(x, y_pred, color="red")
plt.show()

# Using machine learning models

m = 0
c = 0

lr = 0.01

epochs = 1000

n = len(x)

for i in range(epochs):

    y_pred = m*x + c

    dm = (-2/n) * ((x*(y-y_pred)).sum())

    dc = (-2/n) * ((y-y_pred).sum())

    m = m - lr*dm

    c = c - lr*dc

print(m)
print(c)