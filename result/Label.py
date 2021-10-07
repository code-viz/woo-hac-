import pandas as pd
import numpy as np

clinic_table = pd.read_csv('Clinical_Variables.csv', index_col=0).to_numpy()[:,:]
genetic_table = pd.read_csv('Genetic_alterations.csv', index_col=0).to_numpy()[:,:]
survival_table = pd.read_csv('Survival_time_event.csv', index_col=0).to_numpy()[:,:]
treatment_table = pd.read_csv('Treatment.csv', index_col=0).to_numpy()[:,:]

treatment = 1
no_treatmet = 0
treatment_index = np.where(treatment==treatment_table)[0] # 치료를 받은 사람들의 Index (총 488명)
no_treatment_index = np.where(no_treatmet==treatment_table)[0] # 치료를 받지 않은 사람들의 Index (총 512명)
print('Treatment :', treatment_index.shape[0])
print('No Treatment :', no_treatment_index.shape[0])
print('Total :', treatment_index.shape[0] + no_treatment_index.shape[0])
print()

survive = 0
no_survive = 1
survive_index = np.where(survive==survival_table[:,1])[0] # 생존한 사람들의 Index (총 109명)
no_survive_index = np.where(no_survive==survival_table[:,1])[0] # 생존하지 못 한 사람들의 Index (총 891명)
print('Survive :', survive_index.shape[0])
print('No Survive :', no_survive_index.shape[0])
print('Total :', survive_index.shape[0] + no_survive_index.shape[0])
print()

treatment_survive_index = np.intersect1d(treatment_index, survive_index) # 치료 O, 생존 O (총 48명)
treatment_no_survive_index = np.intersect1d(treatment_index, no_survive_index) # 치료 O, 생존 X (총 440명)
no_treatment_survive_index = np.intersect1d(no_treatment_index, survive_index) # 치료 X, 생존 O (총 61명)
no_treatment_no_survive_index = np.intersect1d(no_treatment_index, no_survive_index) # 치료 X, 생존 X (총 451명)
print('Treatment & Survive :', treatment_survive_index.shape[0])
print('Treatment & No Survive :', treatment_no_survive_index.shape[0])
print('No Treatment & Survive :', no_treatment_survive_index.shape[0])
print('No Treatment & No Survive :', no_treatment_no_survive_index.shape[0])
print('Total :', treatment_survive_index.shape[0] + treatment_no_survive_index.shape[0] + no_treatment_survive_index.shape[0] + no_treatment_no_survive_index.shape[0])
print()


survival_treatment_table = pd.DataFrame(columns=['newlabel'], index=range(0,1000))
survival_treatment_table.iloc[treatment_survive_index,0] = [0] * treatment_survive_index.shape[0]
survival_treatment_table.iloc[treatment_no_survive_index,0] = [1] * treatment_no_survive_index.shape[0]
survival_treatment_table.iloc[no_treatment_survive_index,0] = [2] * no_treatment_survive_index.shape[0]
survival_treatment_table.iloc[no_treatment_no_survive_index,0] = [3] * no_treatment_no_survive_index.shape[0]

print(survival_treatment_table)
survival_treatment_table.to_csv('Label.csv')