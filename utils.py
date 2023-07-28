import torch
import torch.nn.functional as F
import os
from datetime import datetime
import time

def log(filename, message, log_folder, write_time=False):
    with open(log_folder+filename+".txt", "a") as f:
        if write_time:
            f.write(str(datetime.now()))
            f.write("\n")
        f.write(str(message))
        f.write("\n")

def check_path(folder_path):
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    except Exception as e:
        print("bug may occur when multiple processes or multiple machines are used")

def train(device, train_loader, model, optimizer, scheduler=None, reg="l2", lamda=1e-5):
    correct = 0
    total = 0
    total_loss = 0
    # total_time = 0
    for batch_idx, batch in enumerate(train_loader):
        im, target, _ = batch
        # start_time = time.time()
        total+= im.shape[0]

        im, target = im.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(im)
        
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        ce_loss = F.cross_entropy(output, target)
        loss = ce_loss
        
        for weight in model.reg_params:
            if reg == "l1":
                temp = torch.sum(torch.abs(weight))
            elif reg == "l2":
                temp = torch.sqrt(torch.sum(torch.square(weight)))
            elif reg == "l12":
                temp = torch.sum(torch.sqrt(torch.sum(torch.square(weight)), dim=1))
            else:
                continue
            if temp == 0:
                continue
            loss += lamda*temp

        total_loss += loss
        loss.backward()
        optimizer.step()
        # total_time += time.time() - start_time
    if scheduler != None:
        scheduler.step()
    # with open("result.txt", "w") as f:
    #     f.write(str(total_time))
    total_loss = float(total_loss)
    accuracy = float(correct.item()/total)
    return total_loss, accuracy

def eval(model, val_loaders, device):
    class_correct = 0
    for batch_idx, batch in enumerate(val_loaders):
        im, target, _ = batch
        im, target = im.to(device), target.to(device)
        output = model(im)
        pred = output.data.max(1, keepdim=True)[1]
        class_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    return class_correct/len(val_loaders.dataset)