import argparse
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataloader import HealthCare_Dataset
from model import Model
from train import train
from test import test
from util import save_model

parser = argparse.ArgumentParser(description='HealthCare Competition')
parser.add_argument('--gpu', type=int, default=0,
                    help='gpu number')
parser.add_argument('--data_path', default='./',
                    help='data path')
parser.add_argument('--test_freq', type=int, default=1,
                    help='test frequency')
parser.add_argument('--save_freq', type=int, default=1000,
                    help='save frequency')
parser.add_argument('--save_folder', default='./checkpoint/',
                    help='save folder')
parser.add_argument('--test_ratio', type=float, default=0.2,
                    help='test set ratio')
parser.add_argument('--batch_size', type=int, default=800,
                    help='batch size')
parser.add_argument('--workers', type=int, default=0,
                    help='num of workers to use')
parser.add_argument('--embedding_size', type=int, default=1,
                    help='embedding size')
parser.add_argument("--lr", default=0.0001, type=float,
                    help="initial learning rate")
parser.add_argument('--start_epoch', type=int, default=0,
                    help='start epoch')
parser.add_argument('--epochs', type=int, default=1,
                    help='number of training epochs')
parser.add_argument('--test', default=False, action='store_true',
                    help='test mode')
parser.add_argument('--save_name', default='test',
                    help='save name')
parser.add_argument('--resume_model', default='',
                    help='resume model name')
parser.add_argument('--selected_genetic', default='',
                    help='Selected Top10 Genetic')
args = parser.parse_args()


def main():
    train_set = HealthCare_Dataset(args.data_path, args.test_ratio, is_train=True, selected_genetic=args.selected_genetic)
    test_set = HealthCare_Dataset(args.data_path, args.test_ratio, is_train=False, selected_genetic=args.selected_genetic)

    if args.test:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = Model(args.embedding_size, args.selected_genetic)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    if torch.cuda.is_available():
        print('GPU {} Setting.'.format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        print('CPU Setting.')
    print()

    if args.test:
        load_path = '{}/{}/Best_Total_Acc_Model.pth'.format(args.save_folder, args.save_name)
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        load_msg = model.load_state_dict(checkpoint['model'])
        print('model parameters matching :', load_msg)
        print()

        print('Epoch {} Model'.format(checkpoint['epoch']))
        print('##############################################################################')
        loss, total_acc = train(train_loader, model, criterion, optimizer, is_train=False)
        print('Train')
        print(f'Loss: {loss:.5f} | Total_Acc: {total_acc:.3f}')
        print()

        loss, total_acc = test(test_loader, model, criterion)
        print('Test')
        print(f'Loss: {loss:.5f} | Total_Acc: {total_acc:.3f}')
        print('##############################################################################')
        print()

        return

    writer = SummaryWriter('./tensorboard_logs/{}'.format(args.save_name))
    os.makedirs(args.save_folder, exist_ok=True)
    os.makedirs(os.path.join(args.save_folder, args.save_name), exist_ok=True)

    best_loss = 1000
    best_train_acc = 0
    best_test_acc = 0
    best_total_acc = 0

    if args.resume_model != '':
        load_path = '{}/{}/Last_Model.pth'.format(args.save_folder, args.resume_model)
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        load_msg = model.load_state_dict(checkpoint['model'])
        print('model parameters matching :', load_msg)
        print()
        args.start_epoch = checkpoint['epoch']
        print('Start from epoch {}.'.format(args.start_epoch))
        print()

        best_loss = checkpoint['best_loss']
        best_train_acc = checkpoint['best_train_acc']
        best_test_acc = checkpoint['best_test_acc']
        best_total_acc = checkpoint['best_total_acc']

        print(f'Best_Loss: {best_loss:.5f} | Best_Total_Acc: {best_total_acc:.2f}')
        print(f'Best_Train_Acc: {best_train_acc:.2f} | Best_Test_Acc: {best_test_acc:.2f}')
        print()

    print('{} Training Start.'.format(args.save_name))
    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        
        loss, total_acc = train(train_loader, model, criterion, optimizer, is_train=True)
        train_acc = total_acc

        writer.add_scalars('Loss', {'Train Loss':loss}, epoch)
        writer.add_scalars('Accuracy', {'Train Total Accuracy':total_acc}, epoch)
        print(f'Epoch {epoch+0:03} Train')
        print(f'Loss: {loss:.5f} | Total_Acc: {total_acc:.3f}')
        print()

        loss, total_acc = test(test_loader, model, criterion)
        test_acc = total_acc

        writer.add_scalars('Loss', {'Test Loss':loss}, epoch)
        writer.add_scalars('Accuracy', {'Test Total Accuracy':total_acc}, epoch)
        print('##############################################################################')
        print(f'Epoch {epoch+0:03} Test')
        print(f'Loss: {loss:.5f} | Total_Acc: {total_acc:.3f}')
        print('##############################################################################')
        print()

        if best_loss > loss:
            best_loss = loss
            save_model(args, epoch, model, optimizer, best_loss, best_train_acc, best_test_acc, best_total_acc, file_name='Best_Loss_Model.pth')
            
        if best_train_acc < train_acc:
            best_train_acc = train_acc
            save_model(args, epoch, model, optimizer, best_loss, best_train_acc, best_test_acc, best_total_acc, file_name='Best_Train_Acc_Model.pth')
            
        if best_test_acc < test_acc:
            best_test_acc = test_acc
            save_model(args, epoch, model, optimizer, best_loss, best_train_acc, best_test_acc, best_total_acc, file_name='Best_Test_Acc_Model.pth')
            
        if best_total_acc < (train_acc + test_acc) // 2:
            best_total_acc = (train_acc + test_acc) // 2
            save_model(args, epoch, model, optimizer, best_loss, best_train_acc, best_test_acc, best_total_acc, file_name='Best_Total_Acc_Model.pth')

    save_model(args, args.epochs, model, optimizer, best_loss, best_train_acc, best_test_acc, best_total_acc, file_name='Last_Model.pth')
    writer.close()
    print('{} Training End.'.format(args.save_name))
    print()

if __name__ == '__main__':
    main()
