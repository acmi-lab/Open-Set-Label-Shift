import torch.nn as nn
import torchvision

import torch.optim.lr_scheduler as lr_sched
from models import * 
import logging 
import torch 

from collections import OrderedDict

from src.datasets.newsgroups_utils import *

log = logging.getLogger("app")

all_classifiers = {
    "Resnet18": ResNet18,
    "Densenet121": DenseNet121,
}

# DIR_PATH = "path/to/models"

def full_block(in_features, out_features, dropout):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(p=dropout),
    )


class Model_20(nn.Module):

    def __init__(self, vocab_size, dim, embeddings, num_classes, features=False):
        super(Model_20, self).__init__()
        self.vocab_size = vocab_size 
        self.dim = dim
        self.embedding = nn.Embedding(self.vocab_size, self.dim)
        self.convnet = nn.Sequential(OrderedDict([
            #('embed1', nn.Embedding(self.vocab_size, self.dim)),
            ('c1', nn.Conv1d(100, 128, 5)),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool1d(5)),
            ('c2', nn.Conv1d(128, 128, 5)),
            ('relu2', nn.ReLU()),
            ('maxpool2', nn.MaxPool1d(5)),
            ('c3', nn.Conv1d(128, 128, 5)),
            ('relu3', nn.ReLU()),
            ('maxpool3', nn.MaxPool1d(35)),
        ]))
    
        self.embedding.weight = nn.Parameter(torch.FloatTensor(embeddings))
        #copy_((embeddings))
        self.embedding.weight.requires_grad = True
    
        if not features:
            self.fc = nn.Sequential(OrderedDict([
                ('f4', nn.Linear(128, 128)),
                ('relu4', nn.ReLU()),
                ('f5', nn.Linear(128, num_classes)),
                ('sig5', nn.LogSoftmax(dim=-1))
            ]))
        else: 
            self.fc = nn.Sequential(OrderedDict([
                ('f4', nn.Linear(128, 128)),
                ('relu4', nn.ReLU()),
            ]))

    def forward(self, img):
        
        output = self.embedding(img)
        output.transpose_(1,2)
        output = self.convnet(output)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        
        return output

class FCNet(nn.Module):
    def __init__(self, x_dim, num_classes, hid_dim=64, z_dim=64, dropout=0.2, features=False):
        super(FCNet, self).__init__()

        if not features:
            self.encoder = nn.Sequential(
                full_block(x_dim, hid_dim, dropout),
                full_block(hid_dim, z_dim, dropout),
                nn.Linear(z_dim, num_classes)
            )
        else: 
            self.encoder = nn.Sequential(
                full_block(x_dim, hid_dim, dropout),
                full_block(hid_dim, z_dim, dropout),
            )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)



def get_model(arch, dataset, num_classes, pretrained, learning_rate, weight_decay, features = False, pretrained_model_dir= None): 

    if dataset.lower().startswith("cifar") and arch in all_classifiers: 
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")
        net = all_classifiers[arch](num_classes=num_classes, features=features)

        if pretrained: 
            log.debug(f"Loading SIMCLR pretrained model")
            checkpoint = torch.load(f"{pretrained_model_dir}/simclr/simclr_cifar-20.pth.tar", map_location='cpu')
            state_dict = {k[9:]: v for k, v in checkpoint.items()}
            net.load_state_dict(state_dict, strict=False)


        parameters_net = net.parameters()

        optimizer_net = torch.optim.SGD(
            parameters_net,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )

        return net, optimizer_net
    
    elif dataset.lower().startswith("tabula") and arch=="FCN": 

        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        net = FCNet(2866, num_classes)

        optimizer = torch.optim.Adam(net.parameters())

        return net, optimizer

    elif arch == "FCN" and dataset =="MNIST": 
        net = nn.Sequential(nn.Flatten(),
                nn.Linear(28*28, 5000, bias=True),
                nn.ReLU(),
                nn.Linear(5000, 5000, bias=True),
                nn.ReLU(),
                nn.Linear(5000, 50, bias=True),
                nn.ReLU(),
                nn.Linear(50, num_classes, bias=True)
            )
        return net 

    elif arch == "FCN" and  dataset.lower().startswith("cifar"): 
        net = nn.Sequential(nn.Flatten(),
                nn.Linear(32*32*3, 5000, bias=True),
                nn.ReLU(),
                nn.Linear(5000, 5000, bias=True),
                nn.ReLU(),
                nn.Linear(5000, 50, bias=True),
                nn.ReLU(),
                nn.Linear(50, num_classes, bias=True)
            )
        return net 

    
    elif dataset.lower().startswith("dermnet") and arch=="Resnet50":
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        log.debug(f"Loading {pretrained} pretrained model")

        net = torchvision.models.resnet50(pretrained=pretrained)
        last_layer_name = 'fc'
        
        d_features = getattr(net, last_layer_name).in_features
        last_layer = nn.Linear(d_features, num_classes)
        net.d_out = num_classes
        setattr(net, last_layer_name, last_layer)

        optimizer = torch.optim.Adam(
            net.parameters(), 
            lr=learning_rate
        )     

        return net, optimizer

    elif (dataset.lower().startswith("breakhis")  or dataset.lower().startswith("utkface")) and arch=="Resnet50":
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        log.debug(f"Loading {pretrained} pretrained model")

        net = torchvision.models.resnet50(pretrained=pretrained)
        last_layer_name = 'fc'

        d_features = getattr(net, last_layer_name).in_features
        last_layer = nn.Linear(d_features, num_classes)
        net.d_out = num_classes
        setattr(net, last_layer_name, last_layer)

        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=learning_rate
        )

        return net, optimizer

    elif dataset.lower().startswith("entity30") and arch=="Resnet18":
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        log.debug(f"Loading {pretrained} pretrained model")

        net = torchvision.models.resnet18(pretrained=False)

        if pretrained: 
            log.debug(f"Loading SIMCLR pretrained model")
            checkpoint = torch.load(f"{pretrained_model_dir}/simclr/pretrained_models/resnet50_imagenet_bs2k_epochs600.pth.tar", map_location='cpu')
            state_dict = {k[8:]: v for k, v in checkpoint['state_dict'].items()}
            net.load_state_dict(state_dict, strict=False)

        last_layer_name = 'fc'

        d_features = getattr(net, last_layer_name).in_features
        last_layer = nn.Linear(d_features, num_classes)
        net.d_out = num_classes
        setattr(net, last_layer_name, last_layer)

        if not pretrained:
            optimizer = torch.optim.SGD(
                net.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            optimizer = torch.optim.Adam(
                net.parameters(),
                lr=learning_rate
            )

        return net, optimizer

    elif dataset.lower().startswith("newsgroups"):
        arch= "Model_20"

        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        _, _, word_index = get_newsgroups()
        embedding_matrix = glove_embeddings(f"{pretrained_model_dir}/glove_embeddings/glove.6B.100d.txt", word_index)

        EMBEDDING_DIM = 100

        net = Model_20(embedding_matrix.shape[0], EMBEDDING_DIM, embedding_matrix, num_classes)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate)

        return net, optimizer

    elif dataset.lower().startswith("rxrx1") and arch=="Resnet50":
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        log.debug(f"Loading {pretrained} pretrained model")

        net = torchvision.models.resnet50(pretrained=pretrained)
        last_layer_name = 'fc'
        
        d_features = getattr(net, last_layer_name).in_features
        last_layer = nn.Linear(d_features, num_classes)
        net.d_out = num_classes
        setattr(net, last_layer_name, last_layer)

        optimizer = torch.optim.Adam(
            net.parameters(), 
            lr=learning_rate//10, 
            weight_decay=weight_decay
        )     

        return net, optimizer
    
    elif arch =="Densenet121":  
        net = torchvision.models.densenet121(pretrained=pretrained)
        last_layer_name = 'classifier'

    elif arch =="Resnet50": 
        net = torchvision.models.resnet50(pretrained=pretrained)
        last_layer_name = 'fc'

    else: 
        raise NotImplementedError("Net %s is not implemented" % arch)

    if arch in ('ResNet50', 'DenseNet121') :
        d_features = getattr(net, last_layer_name).in_features
        last_layer = nn.Linear(d_features, num_classes)
        net.d_out = num_classes
        setattr(net, last_layer_name, last_layer)

    return net	


def get_combined_model(arch, dataset, num_classes, pretrained, learning_rate, weight_decay, features = False, pretrained_model_dir=None): 

    if dataset.lower().startswith("cifar") and arch in all_classifiers: 
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")
        feature_extractor = all_classifiers[arch](num_classes=num_classes, features=features)

        d_features = getattr(feature_extractor, "linear").in_features

        linear_classifier = nn.Linear(d_features, num_classes)

        if pretrained: 
            log.debug(f"Loading SIMCLR pretrained model")
            checkpoint = torch.load(f"{pretrained_model_dir}/simclr/simclr_cifar-20.pth.tar", map_location='cpu')
            state_dict = {k[9:]: v for k, v in checkpoint.items()}
            feature_extractor.load_state_dict(state_dict, strict=False)

        linear_domain_discriminator = nn.Linear(d_features, 2)

        classifier = nn.Sequential(feature_extractor, linear_classifier)

        domain_discriminator = nn.Sequential(feature_extractor, linear_domain_discriminator)

        parameters_classifier = classifier.parameters()

        optimizer_classifier = torch.optim.SGD(
            parameters_classifier,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )

        parameters_domain_discriminator = domain_discriminator.parameters()

        optimizer_domain_discriminator = torch.optim.SGD(
            parameters_domain_discriminator,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )

        return classifier, domain_discriminator, optimizer_classifier, optimizer_domain_discriminator

    elif dataset.lower().startswith("entity30") and arch=="Resnet18":
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        log.debug(f"Loading {pretrained} pretrained model")

        feature_extractor = torchvision.models.resnet18(pretrained=False)

        if pretrained:
            log.debug(f"Loading SIMCLR pretrained model")
            checkpoint = torch.load(f"{pretrained_model_dir}/simclr/pretrained_models/resnet50_imagenet_bs2k_epochs600.pth.tar", map_location='cpu')
            state_dict = {k[8:]: v for k, v in checkpoint['state_dict'].items()}
            feature_extractor.load_state_dict(state_dict, strict=False)


        last_layer_name = 'fc'

        d_features = getattr(feature_extractor, last_layer_name).in_features
        last_layer = nn.Identity(d_features, d_features)
        feature_extractor.d_out = d_features

        setattr(feature_extractor, last_layer_name, last_layer)

        linear_classifier = nn.Linear(d_features, num_classes)

        linear_domain_discriminator = nn.Linear(d_features, 2)

        classifier = nn.Sequential(feature_extractor, linear_classifier)

        domain_discriminator = nn.Sequential(feature_extractor, linear_domain_discriminator)

        parameters_classifier = classifier.parameters()

        if not pretrained: 
            optimizer_classifier = torch.optim.SGD(
                parameters_classifier,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            optimizer_classifier = torch.optim.Adam(
                parameters_classifier,
                lr=learning_rate)
        
        parameters_domain_discriminator = domain_discriminator.parameters()

        if not pretrained: 
            optimizer_domain_discriminator = torch.optim.SGD(
                parameters_domain_discriminator,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            optimizer_domain_discriminator = torch.optim.Adam(
                parameters_domain_discriminator,
                lr=learning_rate)

        return classifier, domain_discriminator, optimizer_classifier, optimizer_domain_discriminator

    elif dataset.lower().startswith("dermnet") and arch=="Resnet50":
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        log.debug(f"Loading {pretrained} pretrained model")

        feature_extractor = torchvision.models.resnet50(pretrained=pretrained)
        last_layer_name = 'fc'
        
        d_features = getattr(feature_extractor, last_layer_name).in_features
        last_layer = nn.Identity(d_features, d_features)
        feature_extractor.d_out = d_features

        setattr(feature_extractor, last_layer_name, last_layer)

        linear_classifier = nn.Linear(d_features, num_classes)

        linear_domain_discriminator = nn.Linear(d_features, 2)

        classifier = nn.Sequential(feature_extractor, linear_classifier)

        domain_discriminator = nn.Sequential(feature_extractor, linear_domain_discriminator)

        parameters_classifier = classifier.parameters()

        optimizer_classifier = torch.optim.SGD(
            parameters_classifier,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )

        parameters_domain_discriminator = domain_discriminator.parameters()

        optimizer_domain_discriminator = torch.optim.SGD(
            parameters_domain_discriminator,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )     

        return classifier, domain_discriminator, optimizer_classifier, optimizer_domain_discriminator

    elif (dataset.lower().startswith("breakhis")  or dataset.lower().startswith("utkface")) and arch=="Resnet50":
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        log.debug(f"Loading {pretrained} pretrained model")

        feature_extractor = torchvision.models.resnet50(pretrained=pretrained)
        last_layer_name = 'fc'

        d_features = getattr(feature_extractor, last_layer_name).in_features
        last_layer = nn.Identity(d_features, d_features)
        feature_extractor.d_out = d_features

        setattr(feature_extractor, last_layer_name, last_layer)

        linear_classifier = nn.Linear(d_features, num_classes)

        linear_domain_discriminator = nn.Linear(d_features, 2)

        classifier = nn.Sequential(feature_extractor, linear_classifier)

        domain_discriminator = nn.Sequential(feature_extractor, linear_domain_discriminator)

        parameters_classifier = classifier.parameters()

        optimizer_classifier = torch.optim.SGD(
            parameters_classifier,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )

        parameters_domain_discriminator = domain_discriminator.parameters()

        optimizer_domain_discriminator = torch.optim.SGD(
            parameters_domain_discriminator,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )     

        return classifier, domain_discriminator, optimizer_classifier, optimizer_domain_discriminator

    elif dataset.lower().startswith("newsgroups"):
        arch= "Model_20"

        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        _, _, word_index = get_newsgroups()
        embedding_matrix = glove_embeddings(f"{pretrained_model_dir}/glove_embeddings/glove.6B.100d.txt", word_index)

        EMBEDDING_DIM = 100

        feature_extractor = Model_20(embedding_matrix.shape[0], EMBEDDING_DIM, embedding_matrix, num_classes, features=features)

        d_features = 128

        linear_classifier = nn.Linear(d_features, num_classes)

        linear_domain_discriminator = nn.Linear(d_features, 2)

        classifier = nn.Sequential(feature_extractor, linear_classifier)

        domain_discriminator = nn.Sequential(feature_extractor, linear_domain_discriminator)

        optimizer_classifier = torch.optim.Adam(\
            filter(lambda p: p.requires_grad, classifier.parameters()), \
            lr=learning_rate)

        optimizer_domain_discriminator = torch.optim.Adam(\
            filter(lambda p: p.requires_grad, domain_discriminator.parameters()), \
            lr=learning_rate)

        return classifier, domain_discriminator, optimizer_classifier, optimizer_domain_discriminator

    elif dataset.lower().startswith("tabula") and arch=="FCN": 

        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        feature_extractor = FCNet(2866, num_classes, features=features)
        d_features = 64

        linear_classifier = nn.Linear(d_features, num_classes)

        linear_domain_discriminator = nn.Linear(d_features, 2)

        classifier = nn.Sequential(feature_extractor, linear_classifier)

        domain_discriminator = nn.Sequential(feature_extractor, linear_domain_discriminator)

        optimizer_classifier = torch.optim.Adam(classifier.parameters())

        optimizer_domain_discriminator = torch.optim.Adam(domain_discriminator.parameters())

        return classifier, domain_discriminator, optimizer_classifier, optimizer_domain_discriminator

def update_optimizer(epoch, opt, data, lr): 

    if data.lower().startswith("cifar"): 
        if epoch>=70: 
            for g in opt.param_groups:
                g['lr'] = 0.1*lr			
        if epoch>=140: 
            for g in opt.param_groups:
                g['lr'] = 0.01*lr

    elif data.lower().startswith("entity30"): 
        if epoch>=100: 
            for g in opt.param_groups:
                g['lr'] = 0.1*lr			
        if epoch>=200: 
            for g in opt.param_groups:
                g['lr'] = 0.01*lr

    elif data.lower().startswith("breakhis") or data.lower().startswith("dermnet"):
        for g in opt.param_groups:
            g['lr'] = lr*((0.96)**(epoch))

    # elif data.lower().startswith("newsgroups"):
    #     for g in opt.param_groups:
    #         g['lr'] = lr*((0.96)**(epoch))

    elif data.lower().startswith("rxrx1"):
        if epoch <10: 
            for g in opt.param_groups:
                g['lr'] = (epoch+1)*lr / 10.0
        else: 
            for g in opt.param_groups:
                g['lr'] = max(0.0, 0.5*(1.0  + math.cos(math.pi *(epoch - 10.0/(80.0)))))*lr

    return opt
