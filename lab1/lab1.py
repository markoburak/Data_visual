import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('flats.csv',decimal=",")


print(df)
# кількість квартир за кімнатами
sums = df.groupby('Кімнат').size().reset_index(name='Кількість')
sums.set_index('Кімнат', inplace=True)
ax = sums.plot(kind='bar')
ax.set_ylabel("Кількість")
plt.show()


# кількість загальних площ
sums2 = df.groupby('Загальна_площа').size().reset_index(name='Кількість')
x_new = [x for x in range(0,201,50)]
x_ax = sums2['Загальна_площа'].tolist()
y_ax = sums2['Кількість'].tolist()
plt.bar(x_ax,y_ax)
plt.xlabel('Загальна_площа')
plt.ylabel('Кількість')
plt.xticks(x_new)
plt.show()

# гістограма по загальній площі з кроком 25
sums3 = df['Загальна_площа']
ax = sums3.plot.hist(bins=[x for x in range(0,251,25)])
ax.set_ylabel("Кількість")
ax.set_xlabel('Загальна_площа')
plt.show()

# гістограма по загальній площі з кроком 50
sums4 = df['Загальна_площа']
ax = sums4.plot.hist(bins=[x for x in range(0,251,50)])
ax.set_ylabel("Кількість")
ax.set_xlabel('Загальна_площа')
plt.show()

# scatterо
x_ax = df['Загальна_площа']
y_ax = df['Ціна']
plt.scatter(x_ax,y_ax)
plt.xlabel('Загальна_площа')
plt.ylabel('Ціна')
plt.show()

# box
plt.figure(figsize=(15, 15))
box=sns.boxplot(y='Місто', x="Ціна",orient='h', data=df, linewidth=2)
plt.savefig('foo.png')
plt.show()