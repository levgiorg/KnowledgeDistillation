import torch
import torch.nn as nn
from models import StudentModel
from knowledge_distilation import knowledge_distillation_train
from torch.utils.data import Dataset, DataLoader

# Define a dummy dataset
class DummyDataset(Dataset):
    def __init__(self, size=100, num_features=10):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return torch.randn(3, 32, 32), torch.randint(0,10,())



student = StudentModel()
# your pretrained large model
teacher  = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg19_bn", pretrained=True)

dummy_dataset = DummyDataset()
trainloader = DataLoader(dummy_dataset, batch_size=8, shuffle=True)
optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
epoch_loss =knowledge_distillation_train(teacher, 
                            student,
                            trainloader=trainloader,
                            criterion=nn.CrossEntropyLoss(),
                            optimizer=optimizer,
                            teacher_percentage=1,
                            temperature=1)
