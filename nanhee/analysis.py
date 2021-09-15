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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# print(df.describe())


# (1) 유전자 변이 개수(한 사람의 유전자 변이 300개 다 합해서) 상황이 똑같은 환자 찾기
unique_genetic = df4.drop_duplicates()
# print('1. ', unique_genetic.shape) # 1000,300

#
df4['sum'] = df4.sum(axis=1)
# print('sum of rows', df4['sum'])

each_gsum = df4.sum(axis=0)
each_gsum=each_gsum.drop(index=['sum'])
print('sum of cols\n', each_gsum)
each_gsum.plot(kind='bar', )
plt.show()

# (1) 임상 정보 상황이 똑같은 환자 찾기
unique_clinical = df3.drop_duplicates()
# print('1. ', unique_clinical.shape) # 1000,10

# (7) 사망 시간 분류
# print(df['time'].mean())
# print(df['time'].max())
# print(df['time'].min())

# (3) Survival Time의 분포 그래프(x축-시간, y축-사람 수) 그려보기
# plt.hist(df['time'], bins=5)
# plt.title('histogram of time table - 5')
#
# plt.hist(df['time'], bins=10)
# plt.title('histogram of time table - 10')
#
# plt.hist(df['time'], bins=15)
# plt.title('histogram of time table - 15')
# plt.show()
