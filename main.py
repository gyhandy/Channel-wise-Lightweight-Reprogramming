import torch.optim as optim
import sys
from torchsummary import summary
import argparse

from dataset.dataloader_reader import *
from network.model import *
from utils import *
import pandas as pd

lr = 1e-3
lamda = 1e-5
epochs = 60
gamma = 0.1
step_size = epochs//3

df = pd.read_csv("stat.csv")
task_id_list = df["newid"].to_list()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, default='./result/')
    parser.add_argument('--weight', type=str, default='./weight/')
    parser.add_argument('--data', type=str, default="/lab/tmpig15b/u/skill_dataset/")
    parser.add_argument('--method', type=str, default="Ghost_reduced_v1")
    args = parser.parse_args()

    log_folder = args.result
    weight_folder = args.weight
    data_path = args.data
    check_path(log_folder)
    check_path(weight_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # task_id = int(args[1])
    model_type = args.method
    batch_size = 32

    for task_id in task_id_list:
        train_dataset, val_dataset, train_loader, val_loader = load_dataloader(task_id, data_path, batch_size)

        task_name = train_dataset.dataset_name
        train_num_class = train_dataset.num_classes
        log(f"{task_name}_{model_type}_lr{lr}_lambda{lamda}_gamma{gamma}_stepsize{step_size}", str(train_dataset.label_dict), log_folder)
        
        if model_type == "Ghost":
            model = Res_Ghost(train_num_class).to(device)
        elif model_type == "Linear":
            model = Res_Baseline(train_num_class).to(device)
        elif model_type == "Finetune":
            model = Res_Baseline(train_num_class, finetune=True).to(device)
        elif model_type == "Xception_Ghost":
            model = Xception_Ghost(train_num_class).to(device)
        elif model_type == "Ghost_a":
            model = Res_Ghost_A(train_num_class).to(device)
        elif model_type == "Resnet_Scratch":
            model = Res_Baseline(train_num_class, finetune=True, scratch=True).to(device)
        elif model_type == "Ghost_reduced_v1":
            model = Res_Ghost(train_num_class, model_type="reduced_v1").to(device)
        elif model_type == "Ghost_reduced_v2":
            model = Res_Ghost(train_num_class, model_type="reduced_v2").to(device)
        elif model_type == "Xception_Linear":
            model = Xception_Baseline(train_num_class).to(device)
        elif model_type == "li18":
            model = li18(train_num_class).to(device)
        elif model_type == "li50":
            model = li50(train_num_class).to(device)

        reg_params = model.reg_params
        noreg_params = model.noreg_params
        log(f"{task_name}_{model_type}_lr{lr}_lambda{lamda}_gamma{gamma}_stepsize{step_size}", f"reg params number: {len(reg_params)}, noreg params number: {len(noreg_params)}", log_folder)

        optimizer = optim.Adam(reg_params+noreg_params, lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma)

        accs = []
        best_eval = 0
        best_state_dict = None
        early_stop_index = 1
        for epoch in range(1, epochs+1):
            if model_type in ["Ghost", "Finetune", "Xception_Ghost", "Ghost_a", "Resnet_Scratch", "Ghost_reduced_v1", "Ghost_reduced_v2"]:
                model.train()
            elif model_type in ["Linear", "Xception_Linear", "li18", "li50"]:
                model.eval()
            log(f"{task_name}_{model_type}_lr{lr}_lambda{lamda}_gamma{gamma}_stepsize{step_size}", f"**********Start Training task:{task_name}, epoch:{epoch}**********", log_folder, True)
            loss, train_acc = train(device, train_loader, model, optimizer, scheduler=scheduler, reg="l2", lamda=lamda)
            log(f"{task_name}_{model_type}_lr{lr}_lambda{lamda}_gamma{gamma}_stepsize{step_size}", f'Train Epoch: {epoch} Train Loss: {loss} Acc_Train: {train_acc}', log_folder)
            model.eval()
            acc = eval(model, val_loader, device)
            accs.append(acc)
            if acc > best_eval:
                best_eval = acc
                best_state_dict = deepcopy(model.state_dict())
                log(f"{task_name}_{model_type}_lr{lr}_lambda{lamda}_gamma{gamma}_stepsize{step_size}", f"epoch:{epoch} accuracy is {acc}", log_folder)
                early_stop_index = epoch
            elif epoch - early_stop_index > 21:
                log(f"{task_name}_{model_type}_lr{lr}_lambda{lamda}_gamma{gamma}_stepsize{step_size}", f"more than 20 epochs without change in acc, stop here, epoch:{epoch}", log_folder)
                break
        log(f"{task_name}_{model_type}_lr{lr}_lambda{lamda}_gamma{gamma}_stepsize{step_size}", f"accs are {accs}", log_folder)
        log(f"{task_name}_{model_type}_lr{lr}_lambda{lamda}_gamma{gamma}_stepsize{step_size}", f"best acc is {best_eval} or {max(accs)}", log_folder)
        with open(log_folder+f"{model_type}.csv", "a") as f:
            f.write(f"{task_id},{task_name},{best_eval}\n")
        torch.save(best_state_dict, weight_folder+f"{task_name}_{model_type}_lr{lr}_lambda{lamda}_gamma{gamma}_stepsize{step_size}.pth")
        log(f"{task_name}_{model_type}_lr{lr}_lambda{lamda}_gamma{gamma}_stepsize{step_size}", str(model), log_folder)
        log(f"{task_name}_{model_type}_lr{lr}_lambda{lamda}_gamma{gamma}_stepsize{step_size}", str(summary(model)), log_folder)
