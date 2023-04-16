import torch.nn as nn
import torch 
import torch.nn.functional as F
import timm 



class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)



class efficientnetv2_rw_m_Model(nn.Module):
    def __init__(self, num_classes, device=None, lr=1e-3):
        super(efficientnetv2_rw_m_Model, self).__init__()
        self.model = timm.create_model('efficientnetv2_rw_m', pretrained=True)
        self.model.classifier = nn.Sequential(
        nn.Linear(2152, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
        )

        self.train_params = [{'params': getattr(self.model, 'blocks').parameters(), 'lr': lr / 10, 'weight_decay':5e-4},
                    {'params': getattr(self.model, 'classifier').parameters(), 'lr': lr, 'weight_decay':5e-4}]
        
        if device:
            self.model.to(device)
    def forward(self, x):
        return self.model(x)

class resnet101d_Model(nn.Module):
    def __init__(self, num_classes, device=None, lr=1e-3):
        super(resnet101d_Model, self).__init__()
        self.model = timm.create_model('resnet101d', pretrained=True)
        self.model.classifier = nn.Sequential(
        nn.Linear(1000, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )   
        self.train_params = [{'params': getattr(self.model, 'layer1').parameters(), 'lr': lr / 10, 'weight_decay':5e-4},
                            {'params': getattr(self.model, 'layer2').parameters(), 'lr': lr / 10, 'weight_decay':5e-4},
                            {'params': getattr(self.model, 'layer3').parameters(), 'lr': lr / 10, 'weight_decay':5e-4},
                            {'params': getattr(self.model, 'layer4').parameters(), 'lr': lr / 10, 'weight_decay':5e-4},
                            {'params': getattr(self.model, 'fc').parameters(), 'lr': lr / 10, 'weight_decay':5e-4},
                            {'params': getattr(self.model, 'classifier').parameters(), 'lr': lr, 'weight_decay':5e-4}]
        if device:
            self.model.to(device)
    def forward(self, x):
        return self.model(x)
    
class tf_efficientnet_b7_Model(nn.Module):
    def __init__(self, num_classes, device=None, lr=1e-3):
        super(tf_efficientnet_b7_Model, self).__init__()
        self.model = timm.create_model('tf_efficientnet_b7', pretrained=True)
        self.model.classifier = nn.Sequential(
        nn.Linear(2560, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )   
        self.train_params = [{'params': getattr(self.model, 'blocks').parameters(), 'lr': lr / 10, 'weight_decay':5e-4},
                    {'params': getattr(self.model, 'classifier').parameters(), 'lr': lr, 'weight_decay':5e-4}]
        if device:
            self.model.to(device)
    def forward(self, x):
        return self.model(x)
    
class vgg19_bn_Model(nn.Module):
    def __init__(self, num_classes, device=None, lr=1e-3):
        super(vgg19_bn_Model, self).__init__()
        self.model = timm.create_model('vgg19_bn', pretrained=True)
        self.model.head = nn.Sequential(
        self.model.head.global_pool,
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )   
        
        self.train_params = [{'params': getattr(self.model, 'features').parameters(), 'lr': lr / 10, 'weight_decay':5e-4},
                             {'params': getattr(self.model, 'pre_logits').parameters(), 'lr': lr / 10, 'weight_decay':5e-4},
                             {'params': getattr(self.model, 'head').parameters(), 'lr': lr / 10, 'weight_decay':5e-4}]
        if device:
            self.model.to(device)
    def forward(self, x):
        return self.model(x)
    

# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class multi_efficientnetv2_rw_m_Model(nn.Module):
    def __init__(self, device=None, lr=1e-3):
        super(multi_efficientnetv2_rw_m_Model, self).__init__()
        self.model = timm.create_model('efficientnetv2_rw_m', pretrained=True)
        self.model.classifier = Identity()
        # mask
        self.fc1 = nn.Sequential(
            nn.Linear(2152, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 3),
        )
        # gender
        self.fc2 = nn.Sequential(
            nn.Linear(2152, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2),
        )
        # age
        self.fc3 = nn.Sequential(
            nn.Linear(2152, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 3),
        )

        self.train_params = [{'params': getattr(self.model, 'blocks').parameters(), 'lr': lr / 10, 'weight_decay':5e-4},
                            {'params': getattr(self, 'fc1').parameters(), 'lr': lr, 'weight_decay':5e-4},
                            {'params': getattr(self, 'fc2').parameters(), 'lr': lr, 'weight_decay':5e-4},
                            {'params': getattr(self, 'fc3').parameters(), 'lr': lr, 'weight_decay':5e-4}]
        
        if device:
            self.model.to(device)
    def forward(self, x):
        x = self.model(x)
        
        # mask, gender, age
        return self.fc1(x), self.fc2(x), self.fc3(x)
    
class multi_vgg19_bn_Model(nn.Module):
    def __init__(self, device=None, lr=1e-3):
        super(multi_vgg19_bn_Model, self).__init__()
        self.model = timm.create_model('vgg19_bn', pretrained=True)
        self.model.head = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 3),
        )   
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1)
        )   
        self.fc3 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 3),
        )   

        self.train_params = [{'params': getattr(self.model, 'features').parameters(), 'lr': lr / 10, 'weight_decay':5e-4},
                             {'params': getattr(self.model, 'pre_logits').parameters(), 'lr': lr / 10, 'weight_decay':5e-4},
                             {'params': getattr(self, 'fc1').parameters(), 'lr': lr, 'weight_decay':5e-4},
                             {'params': getattr(self, 'fc2').parameters(), 'lr': lr, 'weight_decay':5e-4},
                             {'params': getattr(self, 'fc3').parameters(), 'lr': lr, 'weight_decay':5e-4}]
        if device:
            self.model.to(device)
    def forward(self, x):
        x = self.model(x)
        
        return self.fc1(x), torch.sigmoid(self.fc2(x)), self.fc3(x)