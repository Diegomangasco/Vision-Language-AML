import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
    
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        return x.squeeze()

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, 7)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.category_encoder(x)
        x = self.classifier(x)
        return x

class DomainDisentangleModel(nn.Module):
    def __init__(self):
        super(DomainDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.domain_classifier = nn.Linear(64, 2)
        self.category_classifier = nn.Linear(64, 7)
        self.feature_reconstructor = nn.Sequential(
            # nn.Conv2d(512, 512)
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),

            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),

            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512)
        )

    def forward(self, x, label, experiment=None):
        # Feature extraction
        features = self.feature_extractor(x)
        # Disentanglement process
        if label != 2:
            domain_specific = self.domain_encoder(features)
            category_specific = self.category_encoder(features)
        else:
            # Testing
            category_specific = self.category_encoder(features)

        # Classification process
        if label == 0:
            # Training with source
            category_class_ce = self.category_classifier(category_specific) # Minimize loss
            domain_class_de = self.domain_classifier(domain_specific) # Minimize loss
            category_class_de = self.category_classifier(domain_specific) # Maximize loss
            domain_class_ce = self.domain_classifier(category_specific) # Maximize loss
            reconstructor = self.feature_reconstructor(torch.add(category_specific, domain_specific)) # Minimize loss
        elif label == 1:
            # Training with target
            domain_class_ce = self.domain_classifier(category_specific) # Maximize loss
            domain_class_de = self.domain_classifier(domain_specific) # Minimize loss
            reconstructor = self.feature_reconstructor(domain_specific) # Minimize loss
        else:
            # Testing
            category_class = self.category_classifier(category_specific)

        # Return objects
        if label == 0:
            # Training with source
            if experiment == None:
                return reconstructor, features, category_class_ce, domain_class_de, category_class_de, domain_class_ce
            else:
                return reconstructor, features, category_class_ce, domain_class_de, category_class_de, domain_class_ce, domain_specific
        elif label == 1:
            # Training with target
            if experiment == None:
                return reconstructor, features, domain_class_de, domain_class_ce
            else:
                return reconstructor, features, domain_class_de, domain_class_ce, domain_specific
        else:
            # Testing
            return category_class

