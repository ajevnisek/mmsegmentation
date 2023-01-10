import torch


def create_vgg11_classifier():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11',
                           pretrained=True)
    # adapt vgg to binary classification
    model.classifier[6] = torch.nn.Linear(in_features=4096,
                                          out_features=2, bias=True)
    return model
