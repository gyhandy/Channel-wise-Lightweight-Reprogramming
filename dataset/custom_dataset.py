import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class Custom_Dataset(Dataset):
    def __init__(self, id, task_name, num_classes, files, is_cuda=True) -> None:
        self.id = id
        self.dataset_name = task_name
        self.image_labels = files
        self.num_classes = num_classes
        
        self.transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.is_cuda = is_cuda

    @property
    def labels(self) -> list:
        return list(set([p[1] for p in self.image_labels]))

    def __len__(self) -> int:
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path, class_label = self.image_labels[idx]
        image: Image.Image = Image.open(img_path)
        image = image.convert("RGB")
        
        image_tensor: torch.Tensor = self.transform(image)
    
        return image_tensor, class_label, self.id
