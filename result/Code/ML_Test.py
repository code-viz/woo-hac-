# !pip install xgboost
# !pip install lightgbm
# !pip install imblearn
# !pip install borutashap
# !pip install eli5

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import random
from IPython.display import display


plt.rcParams["figure.figsize"] = (200,10)


'''
###########################################

Load CSV Dataset to DataFrame

###########################################
'''


time_table = pd.read_csv('Survival_time_event.csv', index_col=0)
clinic_table = pd.read_csv('Clinical_Variables.csv', index_col=0)
genetic_table = pd.read_csv('Genetic_alterations.csv', index_col=0)
survival_treatment_table = pd.read_csv('Label.csv', index_col=0)


'''
###########################################

Correlating Numerical Features of Time Data

-  Dropped Outlier Value

###########################################
'''


print('outlier of time: ')
print(time_table.loc[time_table['time'] < 0, 'time'], end='\n\n')

time_table_outlier = time_table.copy()
time_table_outlier.loc[time_table_outlier['time'] < 0, 'time'] = abs(time_table_outlier.loc[time_table_outlier['time'] < 0, 'time'])
print(time_table_outlier.describe(), end='\n\n')


'''
###########################################

Correlating Numerical Features of Clinic Data

-  Dropped Outlier Value

###########################################
'''


clinic_table_outlier = clinic_table.copy()

# drop outlier
for col in clinic_table_outlier.columns:
    for outlier in range(10,13):
        clinic_table_outlier = clinic_table_outlier.replace(outlier, 9)

# visualize
for col in clinic_table_outlier.columns:
    print('#', col)
    print(clinic_table_outlier[col].value_counts())
    print('-'*20)
   

'''
###########################################

Correlating Numerical Features of Clinic Data

- Normalization

###########################################
'''


clinic_table_normalization = clinic_table_outlier.copy()

# normalization
for col in clinic_table_normalization.columns:
    clinic_table_normalization[col] = (clinic_table_normalization[col] + 1)/10.0
    
# visualize
for col in clinic_table_normalization.columns:
    print('#', col)
    print(clinic_table_normalization[col].value_counts())
    print('-'*20)    
    
    
'''
###########################################

Correlating Numerical Features of Genetic Data

- Normalization

###########################################
'''


genetic_table_normalization = genetic_table.copy()

# normalization
for col in genetic_table_normalization.columns:
    genetic_table_normalization[col] -= 0.5
    
    
print(genetic_table_normalization.head(10))


'''
###########################################

Machine Learning Test Function

###########################################
'''

def TestML(model,selected_genetic):
    train_acc = []
    test_acc = []
    for i in range(20):
        genetic_10 = genetic_table_normalization.copy()
        genetic_10 = genetic_10[selected_genetic]

        input_dataset = pd.concat([survival_treatment_table, time_table_outlier, clinic_table_normalization ,genetic_10], axis=1)
        input_dataset = input_dataset.drop(['event'], axis=1)

        all_index = np.arange(1000)
        train_data, test_data = input_dataset.iloc[all_index[:800],:], input_dataset.iloc[all_index[800:1000],:]
        X_train = train_data.drop(['newlabel'], axis=1)
        Y_train = train_data['newlabel']

        X_test = test_data.drop(['newlabel'], axis=1)
        Y_test = test_data['newlabel']

        model.fit(X_train, Y_train)
        train_acc.append(np.round(model.score(X_train, Y_train), 10))
        
        Y_pred = model.predict(X_test)
        test_acc.append(np.round(accuracy_score(Y_test, Y_pred), 10))
        
        
    print('## Train 정확도: ', np.round(np.max(train_acc), 3), np.round(np.min(train_acc), 3))
    print('## Test 정확도: ', np.round(np.max(test_acc),3), np.round(np.min(test_acc),3))
    print()
    
    return train_acc, test_acc


'''
###########################################

Sorting String Values of Genetic Names Function

###########################################
'''

def sort_genetic_table(genetic_list):
    np_rgl = np.array(genetic_list)
    np_rgl = np.char.replace(np_rgl, 'G', '')
    np_rgl = np.array(np_rgl, dtype='int')
    np_rgl = sorted(np_rgl)
    np_rgl = np.array(np_rgl, dtype='str')
    np_rgl = np.char.add(np.array(['G']*np_rgl.shape[0]),np_rgl)

    return np_rgl.tolist()


'''
###########################################

Selected Genetic Set Test

###########################################
'''


selected_genetic = ['G211', 'G179', 'G80', 'G27', 'G147', 'G139', 'G242', 'G264', 'G290', 'G130']

test_model1 = RandomForestClassifier(n_estimators=60)
test_model2 = lgb.LGBMClassifier(n_estimators=25,num_leaves=64,n_jobs=-1,boost_from_average=False)

print('## test_model1 (Train accuracy가 100인 경우의 모델)')
model1_train, model1_test = TestML(test_model1, selected_genetic)

print('## test_model2 (Train accuracy가 100이 아닌 경우의 모델)')
model2_train, model2_test = TestML(test_model2, selected_genetic)

print(model1_train)
print(model1_test)
print(model2_train)
print(model2_test)

#draw table
model_train_results = np.array([[np.round(np.max(model1_train), 3), np.round(np.min(model1_train), 3)],
                               [np.round(np.max(model1_test), 3),np.round(np.min(model1_test), 3)],
                               [np.round(np.max(model2_train), 3),np.round(np.min(model2_train), 3)],
                               [np.round(np.max(model2_test), 3),np.round(np.min(model2_test), 3)]])

display(pd.DataFrame(model_train_results, columns=['Max', 'Min'],
                     index=['RandomForest Train', 'RandomForest Test','LightGBM Train','LightGBM Test']))
print()
print()

'''
###########################################

Random Genetic Set Test - test_model1: RANDOM FOREST

###########################################
'''


genetic = ['G'+str(i) for i in range(1,301)]


over_genetic = []
random_genetic_list = []

for i in range(100):
    test_genetic = random.sample(genetic, 10)
    print('## 임의의 10개 유전자: ', test_genetic)
    random_genetic_list.extend(test_genetic)
    
    train_accuracy, test_accuracy = TestML(test_model1, test_genetic)

    if np.max(test_accuracy) >= 0.65 or np.min(test_accuracy) >= 0.565:
        # print('-----------------Over!-----------------', end='\n\n\n')
        if np.max(test_accuracy) >= 0.65 and np.min(test_accuracy) >= 0.565:
            whatover = 'max&min'
        elif np.max(test_accuracy) >= 0.65:
            whatover = 'max'
        elif np.min(test_accuracy) >= 0.56:
            whatover = 'min'
        over_genetic.append([test_genetic,whatover])

# draw table
print(over_genetic) 

if over_genetic:
    over_genetic_results = np.array(over_genetic)
    pd.set_option('display.max_colwidth', -1)
    display(pd.DataFrame(over_genetic_results, columns=['RandomSet', 'Test Result']))

# draw graph
fig, ax = plt.subplots()

kwargs = dict(alpha=0.3, bins=300)

sorted_random_genetic_list = sort_genetic_table(random_genetic_list)
ax.hist(sorted_random_genetic_list, **dict(alpha=0.3, bins=300), color='r', label='random genetic table')
#ax.hist(selected_genetic, **dict(alpha=1, bins=300), color='g', label='selected genetic table')
ax.set(title='Frequency Histogram of Random genetic table', ylabel='Frequency')
#fig.savefig('Frequency Histogram of Random genetic table - Acc_100.png', dpi = 200)
plt.show()


'''
###########################################

Random Genetic Set Test - test_model2: LIGHTGBM 

###########################################
'''


over_genetic = []
random_genetic_list = []

for i in range(100):
    test_genetic = random.sample(genetic, 10)
    print('## 임의의 10개 유전자: ', test_genetic)
    random_genetic_list.extend(test_genetic)
    
    train_accuracy, test_accuracy = TestML(test_model2, test_genetic)
    
    if np.max(test_accuracy) >= 0.635:
        # print('-----------------Over!-----------------', end='\n\n\n')
        over_genetic.append([test_genetic,np.max(test_accuracy)])

#draw table
print(over_genetic) 

if over_genetic:
    over_genetic_results = np.array(over_genetic)
    pd.set_option('display.max_colwidth', -1)
    display(pd.DataFrame(over_genetic_results, columns=['RandomSet', 'Test Result']))
    

# draw graph
fig, ax = plt.subplots()

kwargs = dict(alpha=0.3, bins=300)

sorted_random_genetic_list = sort_genetic_table(random_genetic_list)
ax.hist(sorted_random_genetic_list, **dict(alpha=0.3, bins=300), color='r', label='random genetic table')
#ax.hist(selected_genetic, **dict(alpha=1, bins=300), color='g', label='selected genetic table')
ax.set(title='Frequency Histogram of Random genetic table', ylabel='Frequency')
#fig.savefig('Frequency Histogram of Random genetic table - Acc_not_100.png', dpi = 200)
plt.show()

