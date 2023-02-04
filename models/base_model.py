import torch
import torch.nn.functional as F
import torch.nn as nn
from models.components import * 

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
        self.category_encoder= CategoryEncoder()
        self.domain_encoder = DomainEncoder()
        self.category_classifier = CategoryClassifier()
        self.domain_classifier = DomainClassifier()
        self.reconstructor = Reconstructor()

    def forward(self, x, train, domain_generalization): 

        if train == True:
            f = self.feature_extractor(x)
            fcs = self.category_encoder(f)
            fds = self.domain_encoder(f)
            cc = self.category_classifier(fcs)
            dd = self.domain_classifier(fds, domain_generalization) 
            cd = self.category_classifier(fds)
            dc = self.domain_classifier(fcs, domain_generalization)
            r = self.reconstructor(torch.cat((fds, fcs), 1))
            return cc, dd, cd, dc, r, f
        else:
            f = self.feature_extractor(x)
            fcs = self.category_encoder(f)
            cc = self.category_classifier(fcs)
            return cc         

class CLIPDisentangleModel(nn.Module):
    def __init__(self):
        super(CLIPDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.category_encoder= CategoryEncoder()
        self.domain_encoder = DomainEncoder()
        self.category_classifier = CategoryClassifier()
        self.domain_classifier = DomainClassifier()
        self.reconstructor = Reconstructor()

    def forward(self, x, train, domain_generalization): 

        if train == True:
            f = self.feature_extractor(x)
            fcs = self.category_encoder(f)
            fds = self.domain_encoder(f)
            cc = self.category_classifier(fcs)
            dd = self.domain_classifier(fds, domain_generalization) 
            cd = self.category_classifier(fds)
            dc = self.domain_classifier(fcs, domain_generalization)
            r = self.reconstructor(torch.cat((fds, fcs), 1))
            return cc, dd, cd, dc, r, f, fds
        else:
            f = self.feature_extractor(x)
            fcs = self.category_encoder(f)
            cc = self.category_classifier(fcs)
            return cc            
