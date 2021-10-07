import os
import numpy as np
import torch

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

        
def save_model(args, epoch, model, optimizer, best_loss, best_train_acc, best_test_acc, best_total_acc, file_name=''):
    print('==> Epoch {} : {} Checkpoint Saving...'.format(epoch, file_name))
    save_file = os.path.join(
        os.path.join(args.save_folder, args.save_name), file_name)
    state = {
        'args' : args,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'best_loss': best_loss,
        'best_train_acc': best_train_acc,
        'best_test_acc': best_test_acc,
        'best_total_acc': best_total_acc,
    }
    torch.save(state, save_file)
    del state
    print()