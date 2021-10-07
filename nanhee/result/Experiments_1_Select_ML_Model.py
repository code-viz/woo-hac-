import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score
import lightgbm as lgb
# !pip install xgboost
# !pip install lightgbm
# !pip install imblearn
# !pip install borutashap
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

Dataset of Best Accuracy

###########################################
'''

input_dataset = pd.concat([survival_treatment_table, time_table_outlier, clinic_table_normalization ,genetic_table_normalization], axis=1)
input_dataset = input_dataset.drop(['event'], axis=1)


'''
###########################################

Model list

###########################################
'''

ensemble_models = [
    ('lrcv', LogisticRegression(max_iter = 10000)),
    ('ada', AdaBoostClassifier()),
    ('bc', BaggingClassifier()),
    ('etc',ExtraTreesClassifier()),
    ('gbc', GradientBoostingClassifier()),
    ('rfc', RandomForestClassifier(n_estimators=20)),
    ('knn', KNeighborsClassifier(n_neighbors = 4)),
    ('svc', SVC(probability=True)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('lgbm', LGBMClassifier()),
    ('dtc', DecisionTreeClassifier()),
    ('gnb',GaussianNB()),
]

models = [VotingClassifier(ensemble_models, voting='soft'),
          lgb.LGBMClassifier(n_estimators=30,num_leaves=64,n_jobs=-1,boost_from_average=False),
          LogisticRegression(max_iter = 10000), 
          SVC(), 
          KNeighborsClassifier(n_neighbors = 4), 
          GaussianNB(), 
          Perceptron(),
          SGDClassifier(), 
          DecisionTreeClassifier(), 
          RandomForestClassifier(n_estimators=60)]


'''
###########################################

Select Model of Best Accuracy

###########################################
'''
from sklearn.model_selection import StratifiedKFold


def training(model_list):
    best_model = []
    for model in model_list:
        model_name = str(model)[:str(model).find('(')]
        print('Model: ', model_name)
        print()
        features = input_dataset.drop(['newlabel'], axis=1)
        labels = input_dataset['newlabel']
        
        splits = [5, 10, 7]
        
        for s in splits:
            skfold = StratifiedKFold(n_splits=s)
            idx_iter=0
            cv_accuracy=[]
            cv_precision=[]
            cv_recall=[]
            cv_f1score=[]

            for i in range(10):
                features = features.sample(frac=1).reset_index(drop=True)
                labels = labels.sample(frac=1).reset_index(drop=True)

                for train_index, test_index in skfold.split(features,labels):
                    np.random.shuffle(train_index)
                    np.random.shuffle(test_index)

                    # split train and test set
                    X_train, X_test = features.iloc[train_index,:], features.iloc[test_index,:]
                    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

                    # train ans prediction
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)

                    idx_iter += 1

                    # 
                    accuracy = np.round(accuracy_score(y_test, pred), 4)
                    cv_accuracy.append(accuracy)

                    precision = np.round(precision_score(y_test, pred, average='weighted', zero_division=0), 4)
                    cv_precision.append(precision)

                    recall = np.round(recall_score(y_test, pred, average='weighted', zero_division=0), 4)
                    cv_recall.append(recall)

                    f1score = np.round(f1_score(y_test, pred, average='weighted', zero_division=0), 4)
                    cv_f1score.append(f1score)

                    #train_size = X_train.shape[0]
                    #test_size = X_test.shape[0]

                    #print('\n#{0} 교차 검증 정확도: {1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'.format(idx_iter, accuracy, train_size, test_size))
                    #print('#{0} 검증 세트 인덱스: MIN{1}, MAX{2}'.format(idx_iter, min(test_index), max(test_index)))

                    #print('학습 레이블 데이터 분포: \n', pd.Series(y_train).value_counts())
                    #print('검증 레이블 데이터 분포: \n', pd.Series(y_test).value_counts())

            print('## 교차 검증 총 횟수: ', len(cv_accuracy), '(분할개수:', s, ')')
            # print('## 교차 검증별 정확도: ', np.round(cv_accuracy, 4))
            print('## 평균 검증 정확도: ', np.round(np.mean(cv_accuracy), 5))
            print('## 평균 검증 F1 Score: ', np.round(np.mean(cv_f1score), 5))
            print('##')
            
            #save model name, split num, acc, pre, rec, f1
            best_model.append([model_name, s, 
                               np.round(np.mean(cv_accuracy), 5), 
                               np.round(np.mean(cv_precision), 5),
                               np.round(np.mean(cv_recall), 5),
                               np.round(np.mean(cv_f1score), 5)])
        print()
        print('-'*100)
        print()
        
training(models)
