import os
import time
import copy
import json
import pickle
import argparse

import mmcv
import matplotlib.pyplot as plt

import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from datetime import datetime
from torchvision import datasets, models, transforms


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="HCOCO",
                    choices=['HCOCO', 'HAdobe5k', 'HFlickr', 'Hday2night'])
parser.add_argument('--model', default="resnet",
                    choices=['resnet', 'alexnet', 'vgg', 'squeezenet',
                             'densenet', 'inception'],
                    help="which model to train, default: resnet.")
parser.add_argument('--epochs', default=15, type=int,
                    help='Number of epochs, default: 15.')
parser.add_argument('--batch-size', default=128, type=int,
                    help='Batch size, default: 128.')
parser.add_argument('--freeze-backbone', action='store_true',
                    help='If set, freezes the backbone parameters and trains '
                         'only the FC head.')
parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                    help='which optimizer to use, default: Adam.')
parser.add_argument('--is-cluster', default=False, action='store_true',
                    help="running on TAU cluster or on personal GPU.")

args = parser.parse_args()
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = args.model

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = args.batch_size

# Number of epochs to train for
num_epochs = args.epochs

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
freeze_backbone = args.freeze_backbone
DATASET = args.dataset

if args.is_cluster == True:
    DATASET_ROOT = os.path.join('/storage/jevnisek/ImageHarmonizationDataset/',
                                 DATASET)
    TARGET_ROOT = os.path.join(
        '/storage/jevnisek/CompositeAndRealImagesClassifier', DATASET,
        datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
else:
    DATASET_ROOT = os.path.join('..', 'data', 'Image_Harmonization_Dataset',
                                DATASET,)
    TARGET_ROOT = os.path.join(
        'CompositeAndRealImagesClassifier', DATASET,
        datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))

os.makedirs(TARGET_ROOT, exist_ok=True)


def convert_composite_image_name_to_real_image_name(composite_image_name,
                                              real_image_suffix='.jpg'):
    real_image_filename_without_suffix = composite_image_name.split("_")[0]
    real_image = real_image_filename_without_suffix + real_image_suffix
    return real_image


def set_parameter_requires_grad(model, freeze_backbone):
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25,
                is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def initialize_model(model_name, num_classes, freeze_backbone,
                     use_pretrained=True, in_channels=3):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_backbone)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.conv1.in_channels = in_channels
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_backbone)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_backbone)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_backbone)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_backbone)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_backbone)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# Create training and validation datasets
class RealFakeDataset(torch.utils.data.Dataset):
    def __init__(self, real_images, fake_images, transform=None):
        super().__init__()
        self.real_images = real_images
        self.fake_images = fake_images
        self.transform = transform

        self.all_images = self.real_images + self.fake_images
        self.num_real_images = len(self.real_images)

    def __getitem__(self, item):
        image = Image.open(self.all_images[item])
        if self.transform:
            image = self.transform(image)
        label = 0 if item < self.num_real_images else 1
        return image, label

    def __len__(self):
        return len(self.all_images)


# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes,
                                        freeze_backbone=freeze_backbone,
                                        use_pretrained=True)

# Print the model we just instantiated
print(model_ft)


class RGB2LAB(object):
    def __call__(self, pil_image):
        return mmcv.rgb2ycbcr(np.array(pil_image))


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        # transforms.RandomHorizontalFlip(),
        RGB2LAB(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        # transforms.CenterCrop(input_size),
        RGB2LAB(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")


fake_images = {}
real_images = {}
for split in ['train', 'test']:
    split_path = os.path.join(DATASET_ROOT, f'{DATASET}_{split}.txt')
    fake_images_names = mmcv.list_from_file(split_path)
    fake_images[split] = [os.path.join(DATASET_ROOT,
                                       'composite_images', image_name)
                          for image_name in fake_images_names]
    real_images[split] = [os.path.join(
        DATASET_ROOT, 'real_images',
        convert_composite_image_name_to_real_image_name(image_name))
                          for image_name in fake_images_names]


image_datasets = {x: RealFakeDataset(real_images[x], fake_images[x],
                                     data_transforms[x])
                  for x in ['train', 'test']}

# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)
                    for x in ['train', 'test']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if freeze_backbone:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
if args.optimizer == 'SGD':
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
else:
    optimizer_ft = optim.Adam(params_to_update, lr=0.001)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion,
                             optimizer_ft, num_epochs=num_epochs,
                             is_inception=(model_name=="inception"))
# save training history
with open(os.path.join(TARGET_ROOT, 'val_acc_history.json'), 'w') as f:
    json.dump({'val_acc_history': [x.item() for x in hist]}, f, indent=2)

# save trained model.
torch.save(model_ft, os.path.join(TARGET_ROOT, f'model.ckpt'))

# save training script parameters.
json_path = os.path.join(TARGET_ROOT, f'args.json')
with open(json_path, 'wt') as f:
    json.dump(vars(args), f, indent=4)
