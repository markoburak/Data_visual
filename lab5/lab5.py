# Needed packages
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

# Variant
correct_number = 4 - 2

def analys(data):
    # Data analys
    n = len(data.index)
    min = data["Values"].min()
    max = data["Values"].max()
    k = math.ceil(1 + 3.32 * math.log10(n))
    l = (max - min) / k
    print("Volume of data:", n)
    print("Min value:", min)
    print("Max value:", max)
    print("Amount of classes:", k)
    print("Interval:", round(l, 3))
    data_table = (data.groupby(pd.cut(data["Values"], np.linspace(min-0.01, max, num=k+1))).count())
    summa = data_table["Values"].sum()
    data_table["Values_%"] = (data_table["Values"] / summa * 100)
    data_table["Cumulative_Sum"] = data_table["Values"].cumsum()
    data_table["Cumulative_Sum_%"] = data_table["Values_%"].cumsum()
    print(data_table)

    # Graphic Visualization
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    data_table.plot(y="Values",ax=axes[0, 0], kind="bar", title="Amount of rating")
    plt.xticks(rotation='90')
    data_table.plot(y="Values_%", ax=axes[0, 1], kind="bar", title="Amount_% of rating")
    plt.xticks(rotation='90')
    data_table.plot(y="Cumulative_Sum", ax=axes[1, 0], kind="bar", title="Cumulative_Sum of rating")
    plt.xticks(rotation='90')
    data_table.plot(y="Cumulative_Sum_%",ax=axes[1, 1], kind="bar", title="Cumulative_Sum_% of rating")
    plt.xticks(rotation='90')
    plt.show()
    me = data_table.plot(y="Cumulative_Sum_%", kind="line")
    data_table.plot(y="Values_%", kind="bar",ax=me)
    me.grid()
    plt.show()
analys(pd.read_csv("data.csv"))