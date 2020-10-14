import random

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

df = pd.read_csv('filmdeathcounts.csv',decimal=".",sep=',')



body_per_min = []

for i in range(len(df)):
    body_per_min.append(df.iloc[i,2]/(df.iloc[i,6]))
df['body_per_min'] = body_per_min



# hist count bodies
sums3 = df['Body_Count']
sums3.plot.hist(bins=[x for x in range(-21,875,42)])
# plt.hist(df["Body_Count"], bins=20)
plt.show()

#sort
body_sort = df.sort_values(by = ['Body_Count'],ascending=False)
body_per_min =  df.sort_values(by = ['body_per_min'],ascending=False)



# imdb simulation
rating = df['IMDB_Rating']
rating.plot.hist(bins=[i for i in range(0,11,1)])
plt.show()


#середнє значення для змінної IMDBrating

imdb_mean = df['IMDB_Rating'].mean()


# середньоквадратичне відхилення для змінної IMDBrating

imdb_sd = df['IMDB_Rating'].std()




#нормальний розподіл
random.seed(900)
imdb_simulation = np.random.normal(imdb_mean, imdb_sd, df.shape[0])
df['imdb_simulation'] = imdb_simulation
plt.hist(df['imdb_simulation'], bins=10)
plt.show()

#симуляція imdb_simulation
sm.qqplot(df['imdb_simulation'])
plt.show()

#справжній рейтинг IMDB_Rating:
sm.qqplot(df['IMDB_Rating'])
plt.show()