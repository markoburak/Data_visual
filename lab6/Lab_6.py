import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

correct_number = 4 - 2

def analys_corr(corr):
    if corr == 1:
        print("A perfect positive relationship.")
    elif 0.8 <= corr < 1:
        print("A fairly strong positive relationship.")
    elif 0.6 <= corr < 0.8:
        print("A moderate positive relationship.")
    elif 0 < corr < 0.6:
        print("A weak positive relationship.")
    elif corr == 0:
        print("No relationship.")
    elif -0.6 < corr < 0:
        print("A weak negative relationship.")
    elif -0.8 < corr <= -0.6:
        print("A moderate negative relationship.")
    elif -1 < corr <= -0.8:
        print("A fairly strong negative relationship.")
    elif corr == -1:
        print("A perfect negative relationship.")
    else:
        print("Cannot analys coef corr!")

data = pd.read_csv("data.csv", delimiter=";")
data["X"] += correct_number
data["Y"] += correct_number
data.plot.scatter("X", "Y",title="Лінійна регресія")
corr = np.corrcoef(data["X"], data["Y"])[0][1]
print("Coef Correlation:", round(corr, 5))
analys_corr(corr)
m, b = np.polyfit(data["X"], data["Y"], 1)
plt.plot(data["X"], m*data["X"] + b, color="green")
plt.show()