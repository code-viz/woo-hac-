import torch
from util import accuracy

def train(train_loader, model, criterion, optimizer, is_train=True):
    epoch_loss = 0
    epoch_total_acc = 0

    if is_train:
        model.train()
    else:
        model.eval()

    for clinic, genetic, survival, label in train_loader:
        label = label.squeeze().long()

        if torch.cuda.is_available():
            clinic = clinic.cuda()
            genetic = genetic.cuda()
            survival = survival.cuda()
            label = label.cuda()
        
        out = model(clinic, genetic, survival)

        loss = criterion(out, label)
        total_acc = accuracy(out, label, topk=(1,))
        
        if is_train:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        epoch_total_acc += total_acc[0].item()
        
    loss = epoch_loss/len(train_loader)
    total_acc = epoch_total_acc/len(train_loader)

    return loss, total_acc