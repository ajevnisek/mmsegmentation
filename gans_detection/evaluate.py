import os
import torch
import numpy as np
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import average_precision_score, precision_recall_curve, \
    accuracy_score, roc_curve, auc
from tools.real_and_composite_images_dataloaders import create_dataloaders
from tools.real_and_composite_images_models import create_vgg11_classifier



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_image_harmonization_dataloader(data_dir, dataset,
                                          train_batch_size, test_batch_size):
    train_loader, test_dataloader = create_dataloaders(
        data_dir=data_dir,
        dataset=dataset,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size)
    return train_loader, test_dataloader


def create_gan_dataloader(gan_name='biggan'):
    dataset = ImageFolder(root=os.path.join('/mnt/data/CNN_synth_testset/',
                                            gan_name),
                          transform=torchvision.models.VGG16_Weights
                          .IMAGENET1K_V1.transforms())

    dataloader = DataLoader(dataset, batch_size=128, num_workers=10)
    return dataloader


def validate(model, test_dataloader, is_iharm=True, is_tqdm=False):
    with torch.no_grad():
        y_true, y_pred = [], []
        loader = tqdm(test_dataloader) if is_tqdm else test_dataloader
        for batch in loader:
            if is_iharm:
                in_tens = batch['image'].to(device)
                label = batch['label'].to(device)
            else:
                in_tens = batch[0].to(device)
                label = batch[1].to(device)
            pred = model(in_tens)[..., 1] - model(in_tens)[..., 0]
            y_pred.extend(pred.flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_result = auc(fpr, tpr)
    return acc, auc_result, ap, r_acc, f_acc, y_true, y_pred


def main():
    # PATH = '/home/uriel/research/mmsegmentation/results/vgg_training/2023_01_06__12_16_53/best_model_epoch_005.pth'
    PATH = '/home/uriel/research/mmsegmentation/results/hadobe5k.ckpt'
    data_dir = '/home/uriel/research/data/Image_Harmonization_Dataset'
    dataset_name = 'HAdobe5k'

    model = create_vgg11_classifier().to(device)
    checkpoint = torch.load(PATH)

    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint['model']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[len('model.'):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    train_dataloader, test_dataloader = create_image_harmonization_dataloader(
        data_dir=data_dir, dataset=dataset_name, train_batch_size=50,
        test_batch_size=128)
    acc, auc_result, ap, r_acc, f_acc, y_true, y_pred = validate(
        model, test_dataloader, is_tqdm=True)
    print(f"[{dataset_name}] Accuracy: {acc * 100.0} [%]")
    print(f"[{dataset_name}] AP: {ap * 100.0} [%]")
    print(f"[{dataset_name}] AuC: {auc_result * 100.0} [%]")

    gan_to_performance = {}
    for gan_name in ['biggan', 'crn', 'deepfake', 'gaugan', 'imle', 'san',
                     'seeindark', 'stargan', 'whichfaceisreal']:
        gan_dataloader = create_gan_dataloader()
        acc2, auc_result2, ap2, r_acc2, f_acc2, y_true2, y_pred2 = validate(
            model, gan_dataloader, is_iharm=False, is_tqdm=True)
        print(f"[{gan_name}] Accuracy: {acc2 * 100.0:.2f} [%]")
        print(f"[{gan_name}] AP: {ap2 * 100.0:.2f} [%]")
        print(f"[{gan_name}] AuC: {auc_result2 * 100.0:.2f} [%]")
        gan_to_performance[gan_name] = {
            'accuracy': acc2,
            'auc': auc_result2,
            'ap': ap2
        }
    return {'accuracy': acc, 'ap': ap, 'auc': auc_result}, gan_to_performance
