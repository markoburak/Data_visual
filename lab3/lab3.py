#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import plotnine as pn
import statsmodels.api as sm

#%%

anscombe = sns.load_dataset('anscombe')
anscombe.head()

#%%

print(f'{anscombe[anscombe.dataset.eq("I")]}')
print(f'{anscombe[anscombe.dataset.eq("II")]}')
print(f'{anscombe[anscombe.dataset.eq("III")]}')
print(f'{anscombe[anscombe.dataset.eq("IV")]}')


print('Середнє значення змінної x = 9.0')
print('Дисперсія змінної x = 10.0')
print('Середнє значення змінної y = 7.5')
print('Дисперсія змінної y = 3.75')
print('Пряма лінійної регресії y= 3+0.5x')

#%%

def plot_anscombe(dataset):
    sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=dataset,
           col_wrap=2, ci=None, palette="muted", height=4,
           scatter_kws={"s": 30, "alpha": 1})

plot_anscombe(anscombe)

print()
#%%

def correlation(dataset):
    print (f'Pearson correlation coefficient for {dataset} dataset')
    print(anscombe[(anscombe['dataset']==f'{dataset}')].corr(method='pearson').round(5))

#%%

correlation('I')
correlation('II')
correlation('III')
correlation('IV')

#%%

def linear_regression(data, name_of_dataset):
    y1 = list(data[(data['dataset']==name_of_dataset)]['y'])
    x = list(data[(data['dataset']==name_of_dataset)]['x'])
    x1 = [[] for i in range(len(x))]
    for i in range(len(x)):
        x1[i].append(x[i])

    model = LinearRegression().fit(x1,y1)
    model_y_predict = model.predict(x1)
    plt.scatter(x1, y1, color = 'gray')
    plt.plot(x1, model_y_predict, linewidth = 2.5, color = 'r')
    plt.title(f"Linear regression for {name_of_dataset}")

    return y1, model_y_predict


#%%

y1, model_y1_predict = linear_regression(anscombe, 'I')
residuals1 = y1 - model_y1_predict

#%%

y2, model_y2_predict = linear_regression(anscombe, 'II')
residuals2 = y2 - model_y2_predict
print(residuals2)

#%%

y3, model_y3_predict = linear_regression(anscombe, 'III')
residuals3 = y3 - model_y3_predict
print(residuals3)

#%%

anscombe.groupby("dataset").describe()

#%%

y4, model_y4_predict = linear_regression(anscombe, 'IV')
residuals4 = y4 - model_y4_predict
print(residuals4)

#%%

plt.hist(residuals1, bins=11)
plt.ylabel('Frequency')
plt.xlabel('I quartet residuals')
plt.show()

#%%

plt.hist(residuals2, bins=11)
plt.ylabel('Frequency')
plt.xlabel('II quartet residuals')
plt.show()

#%%

plt.hist(residuals3, bins=11)
plt.ylabel('Frequency')
plt.xlabel('III quartet residuals')
plt.show()

#%%

plt.hist(residuals4, bins=11)
plt.ylabel('Frequency')
plt.xlabel('IV quartet residuals')
plt.show()

#%% оцінка нормальності розподілу 1

sm.qqplot(residuals1, line='s')


#%% оцінка нормальності розподілу 2

sm.qqplot(residuals2, line='r')

#%% оцінка нормальності розподілу 3

sm.qqplot(residuals3, line='q')

#%% оцінка нормальності розподілу 4

sm.qqplot(residuals4, line='r')


#%%Оцінюємо варіативність залишків 1:

plt.scatter(model_y1_predict, residuals1, color='orange')
plt.ylabel('predicted')
plt.xlabel('residuals1')
plt.show()

#%%Оцінюємо варіативність залишків 2:

plt.scatter(model_y2_predict, residuals2, color='pink')
plt.ylabel('predicted')
plt.xlabel('residuals2')
plt.show()

#%%Оцінюємо варіативність залишків 3:

plt.scatter(model_y3_predict, residuals3, color='grey')
plt.ylabel('predicted')
plt.xlabel('residuals3')
plt.show()

#%%Оцінюємо варіативність залишків 4:

plt.scatter(model_y4_predict, residuals4, color='green')
plt.ylabel('predicted')
plt.xlabel('residuals4')
plt.show()

#%%

diamonds = pd.read_csv('diamonds.csv', delimiter=',', index_col=0 )

#%%

diamonds.head()

#%%

plt.scatter(diamonds['carat'], diamonds['price'], color = 'orange', s=10)

#%%

print(diamonds['carat'].corr(diamonds['price']).round(4))

#%%

pn.ggplot(diamonds, pn.aes(x='carat', y='price')) + \
    pn.geom_point(color='orange') + \
    pn.facet_wrap(['cut'])

#%%

pn.ggplot(diamonds, pn.aes(x='carat', y='price')) + \
    pn.geom_point(color='pink') + \
    pn.geom_smooth(method='lm', se=False) +\
    pn.facet_wrap(['cut'])

#%%

def linear(df, name_of_column):
    y_d_i = list(df[(df['cut']==name_of_column)]['price'])
    x = list(df[(df['cut']==name_of_column)]['carat'])
    x_d_i = [[] for i in range(len(x))]
    for i in range(len(x)):
        x_d_i[i].append(x[i])
    model_d_i = LinearRegression()
    model_d_i = model_d_i.fit(x_d_i, y_d_i)
    model_d_i_y_predict = model_d_i.predict(x_d_i)

    plt.scatter(x_d_i, y_d_i, color='grey')
    plt.plot(x_d_i, model_d_i_y_predict, linewidth=2, color='red')
    plt.axvline(x=1.0)
    plt.title(f'{name_of_column}')
    plt.grid(True)
    plt.show()

#%%Побудуйте моделі лінійної регресії lin.diamond.fair залежності ціни від ваги для обробки

linear(diamonds, 'Ideal')


#%%

linear(diamonds, 'Fair')

#%%
