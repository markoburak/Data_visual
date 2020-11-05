# Needed packages
import pandas as pd
import matplotlib.pyplot as plt

# Variant
correct_number = 4 - 2

def analys(data):
    # Data analys
    dataset1 = pd.DataFrame(data, columns=["Start_data"])
    dataset1["Start_data"] = dataset1["Start_data"] + correct_number
    dataset1["Variation_series"] = dataset1["Start_data"].sort_values().reset_index(drop=True)
    dataset1_table = pd.DataFrame(dataset1.value_counts(["Variation_series"], sort=False), columns=["Amount"])
    summa = dataset1_table["Amount"].sum()
    dataset1_table["Amount_%"] = (dataset1_table["Amount"] / summa * 100)
    dataset1_table["Cumulative_Sum"] = dataset1_table["Amount"].cumsum()
    dataset1_table["Cumulative_Sum_%"] = dataset1_table["Amount_%"].cumsum()
    print(dataset1_table)

    # Graphic Visualization
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5, 5))
    dataset1_table.plot(y="Amount", ax=axes[0, 0], kind="bar", title="Amount of rating")
    dataset1_table.plot(y="Amount_%", ax=axes[0, 1], kind="bar", title="Amount_% of rating")
    dataset1_table.plot(y="Cumulative_Sum", ax=axes[1, 0], kind="bar", title="Cumulative_Sum of rating")
    dataset1_table.plot(y="Cumulative_Sum_%", ax=axes[1, 1], kind="bar", title="Cumulative_Sum_% of rating")
    fig.tight_layout()
    plt.show()
    me = dataset1_table.plot(y="Cumulative_Sum_%", kind="line")
    dataset1_table.plot(y="Amount_%", kind="bar",ax=me)

    # ax.dataset1_table.plot(y="Amount", ax=axes[0, 0], kind="bar", title="Amount of rating")
    # ax.dataset1_table.plot(y="Cumulative_Sum", ax=axes[1, 0], kind="line", title="Cumulative_Sum of rating")
    plt.show()
# First task
analys([3, 5, 4, 4, 2, 6, 3, 5, 4, 4])
analys([4, 2, 3, 5, 5, 3, 4, 4, 3, 5])