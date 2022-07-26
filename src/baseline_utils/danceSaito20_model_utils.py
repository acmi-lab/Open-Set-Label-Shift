import torch.nn as nn
import torchvision

import torch.optim.lr_scheduler as lr_sched
from models import * 
import logging 
import torch 

from collections import OrderedDict

from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from collections import Counter

from src.datasets.newsgroups_utils import *

log = logging.getLogger("app")

all_classifiers = {
    "Resnet18": ResNet18,
    "Densenet121": DenseNet121,
    # "ClipViTB32": ClipViTB32
}



def get_model_dance(arch, dataset, num_classes, pretrained, learning_rate, weight_decay, features = False, temp_scale=False, pretrained_model_dir=None): 

    if dataset.lower().startswith("cifar") and arch in all_classifiers: 
        net = all_classifiers[arch](num_classes=num_classes, features=features)

        parameters_net = net.parameters()

        optimizer_net = torch.optim.SGD(
            parameters_net,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )
    
        d_features = getattr(net, "linear").in_features
        classifier = ResClassifier_MME(num_classes, d_features, .05) 

        optimizer_classifier = torch.optim.SGD(
            classifier.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )     
        # scheduler_classifier = lr_sched.StepLR(optimizer_classifier, step_size=100, gamma=0.1)

        return net, classifier, optimizer_net, optimizer_classifier
          
    else: 
        raise NotImplementedError("Net %s is not implemented" % arch)

def get_linearaverage(net, num_data, temp, momentum): 
    d_features = getattr(net, "linear").in_features
    
    return LinearAverage(d_features, num_data, temp, momentum)