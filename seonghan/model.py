import torch.nn as nn
import torch.nn.functional as F
import timm

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

#         [64, 2152] 
        
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2152, num_classes) # 원본
        self.fc1 = nn.Linear(2152, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)


    def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
        x = self.fc1(x)
#         x = self.conv2(x)
        x = F.relu(x)
#         x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        x = self.fc2(x)
#         x = self.conv3(x)
        x = F.relu(x)
#         x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.fc3(x)
#         x = self.avgpool(x)
#         x = x.view(-1, 128)
        return x
    

class ParallelModel(nn.Module):
    def __init__(self, classes_list):
        super().__init__()
        self.models = [BaseModel(num) for _, num in classes_list]
        for classes, model in zip(classes_list, self.models):
            self.add_module(classes[0], model)

    def forward(self, x):
        outs = [model(x) for model in self.models]
        return outs
        
def build_model(device):
    model = timm.create_model('efficientnetv2_rw_m', pretrained=True)
    model.classifier = ParallelModel([("mask", 3), ("gender", 2), ("age", 3)]).to(device) # output-d: 2+3+3
    model.to(device)
    return model

