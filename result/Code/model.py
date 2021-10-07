import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, embedding_size, selected_genetic=''):
        super(Model, self).__init__()
        self.embedding_size = embedding_size
        self.selected_genetic = selected_genetic
        self.clinic_parameter = nn.Parameter(torch.ones(10, embedding_size))

        if selected_genetic == '':
            self.genetic_parameter = nn.Parameter(torch.ones(300, embedding_size))
            self.out_layer = nn.Linear(93101, 4)
        else:
            self.genetic_parameter = nn.Parameter(torch.ones(10, embedding_size))
            self.out_layer = nn.Linear(301, 4)
    
    def forward(self, clinic, genetic, survival_time):
        B = clinic.shape[0]
        clinic = clinic.view(B, 10, 1)
        clinic_embedded = torch.empty((B, 10, self.embedding_size))

        if self.selected_genetic == '':
            genetic = genetic.view(B, 300, 1)
            genetic_embedded = torch.empty((B, 300, self.embedding_size))
        else:
            genetic = genetic.view(B, 10, 1)
            genetic_embedded = torch.empty((B, 10, self.embedding_size))

        if torch.cuda.is_available():
            clinic_embedded = clinic_embedded.cuda()
            genetic_embedded = genetic_embedded.cuda()

        for i, (c, g) in enumerate(zip(clinic, genetic)):
            clinic_embedded[i] = c * self.clinic_parameter
            genetic_embedded[i] = g * self.genetic_parameter
        
        clinic_clinic = torch.bmm(clinic_embedded, clinic_embedded.permute(0, 2, 1)).view(B, -1)
        clinic_genetic = torch.bmm(clinic_embedded, genetic_embedded.permute(0, 2, 1)).view(B, -1)
        genetic_genetic = torch.bmm(genetic_embedded, genetic_embedded.permute(0, 2, 1)).view(B, -1)
        h = torch.cat((clinic_clinic, clinic_genetic, genetic_genetic, survival_time.unsqueeze(1)), dim=1)

        out = self.out_layer(h)
        return out