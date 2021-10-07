import torch
from util import accuracy

def test(test_loader, model, criterion):
    epoch_loss = 0
    epoch_total_acc = 0

    model.eval()
    for clinic, genetic, survival, label in test_loader:
        label = label.squeeze().long()
        
        if torch.cuda.is_available():
            clinic = clinic.cuda()
            genetic = genetic.cuda()
            survival = survival.cuda()
            label = label.cuda()
        
        out = model(clinic, genetic, survival)

        loss = criterion(out, label)
        total_acc = accuracy(out, label, topk=(1,))
        
        epoch_loss += loss.item()
        epoch_total_acc += total_acc[0].item()

    loss = epoch_loss/len(test_loader)
    total_acc = epoch_total_acc/len(test_loader)

    return loss, total_acc