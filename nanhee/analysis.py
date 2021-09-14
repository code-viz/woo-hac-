import csv
import pandas as pd
from matplotlib import pyplot as plt
# index = range(0, 1000)
# columns = ['time', 'survival', 'treat', 'genetic', 'clinical']
# df = pd.DataFrame(index=index, columns=columns)

df1=pd.read_csv('Survival_time_event.csv', index_col=0)
df2=pd.read_csv('Treatment.csv', index_col=0)
df3=pd.read_csv('Clinical_Variables.csv', index_col=0)
df4=pd.read_csv('Genetic_alterations.csv', index_col=0)

df = pd.concat([df1,df2,df3,df4], axis=1)

print(df['time'].mean())
print(df['time'].max())
print(df['time'].min())

# row 생략 없이 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)
print(df.describe())

plt.hist(df['time'], bins=10)
plt.title('histogram of time table')
plt.show()