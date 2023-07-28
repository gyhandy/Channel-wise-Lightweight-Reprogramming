from dataset.custom_dataset import Custom_Dataset
from dataset.sample_dataset import ShellDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import math

def read_file(filename):
    container = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data = line.strip().split()
            container.append((data[0],int(data[1])))
    return container

# def custom_load_dataloader(dataset_name, dataset_id, class_num, batch_size=32):
#     train_files = read_file(f"/lab/tmpig8e/u/VTAB_data/path_files/VTAB_sample/{dataset_name}_train_sample.txt")
#     test_files = read_file(f"/lab/tmpig8e/u/VTAB_data/path_files/VTAB_sample/{dataset_name}_test_sample.txt")
#     train_dataset = Custom_Dataset(dataset_id, dataset_name, class_num, train_files)
#     test_dataset = Custom_Dataset(dataset_id, dataset_name, class_num, test_files)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size//2, shuffle=True)

#     return train_dataset, test_dataset, train_loader, test_loader

def load_dataloader(index, datapath, batch_size=32, full_dataset=False, shuffle=True):
    """
    function to load formal shell dataloader
    input:
        -index: the dataset id for the dataloader, does not acept 47, 56, 84, 93
        -batch_size: the batch_size of the dataloader
    output:
        the train dataset, test dataset, train dataloader and test dataloader
    """
    if index in [47, 56, 84, 93, 103] or index not in list(range(110)):
        print(f"task {index} does not exist, none will be returned, please use a reasonable input")
        return None
    if index in list(range(110)):
        transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
        if full_dataset:
            train_dataset = ShellDataset(datapath, index, "train", "original", pipeline=transform, threshold=0, large_size=1e100)
            val_dataset = ShellDataset(datapath, index, "validation", "original", label_dict=train_dataset.label_dict, pipeline=transform, threshold=0, large_size=1e100)
        else:
            train_dataset = ShellDataset(datapath, index, "train", "original", pipeline=transform)
            val_dataset = ShellDataset(datapath, index, "validation", "original", label_dict=train_dataset.label_dict, pipeline=transform)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_batch_size = batch_size//2
    if test_batch_size == 0:
        test_batch_size = 1
    test_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=shuffle)
    return train_dataset, val_dataset, train_loader, test_loader

if __name__ == "__main__":
    pass