import os
import pandas as pd
import torch

class HealthCare_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, test_ratio, is_train=True, selected_genetic=''):
        super(HealthCare_Dataset, self).__init__()
        self.is_train = is_train

        clinic_table = pd.read_csv(os.path.join(data_path, 'Clinical_Variables.csv'), index_col=0)
        genetic_table = pd.read_csv(os.path.join(data_path, 'Genetic_alterations.csv'), index_col=0)
        survival_table = pd.read_csv(os.path.join(data_path, 'Survival_time_event.csv'), index_col=0)
        treatment_table = pd.read_csv(os.path.join(data_path, 'Treatment.csv'), index_col=0)
        label_table = pd.read_csv(os.path.join(data_path, 'Label.csv'), index_col=0)

        if selected_genetic != '':
            selected_genetic = selected_genetic.split(',')
            genetic_table = genetic_table[selected_genetic]

        # Outlier Check
        survival_table.loc[survival_table['time'] < 0, 'time'] = abs(survival_table.loc[survival_table['time'] < 0, 'time'])
        for outlier in range(10, 13):
            clinic_table = clinic_table.replace(outlier, 9)

        # Train & Test Ratio
        data_size = len(clinic_table)
        self.train_size = int(data_size * (1 - test_ratio))
        self.test_size = int(data_size * test_ratio)

        if self.is_train:
            self.clinic_tensor = ((torch.Tensor(clinic_table.values)[:self.train_size,:] + 1) / 10)
            self.genetic_tensor = torch.Tensor(genetic_table.values)[:self.train_size,:] - 0.5
            self.survival_tensor = torch.Tensor(survival_table.values)[:self.train_size,:]
            self.treatment_tensor = torch.Tensor(treatment_table.values)[:self.train_size,:]
            self.label_tensor = torch.Tensor(label_table.values)[:self.train_size,:]
        else:
            self.clinic_tensor = ((torch.Tensor(clinic_table.values)[self.train_size:,:] + 1) / 10)
            self.genetic_tensor = torch.Tensor(genetic_table.values)[self.train_size:,:] - 0.5
            self.survival_tensor = torch.Tensor(survival_table.values)[self.train_size:,:]
            self.treatment_tensor = torch.Tensor(treatment_table.values)[self.train_size:,:]
            self.label_tensor = torch.Tensor(label_table.values)[self.train_size:,:]


    def __getitem__(self, index):
        return self.clinic_tensor[index], self.genetic_tensor[index], self.survival_tensor[index, 0], self.label_tensor[index]

    def __len__(self):
        if self.is_train:
            return self.train_size
        else:
            return self.test_size
