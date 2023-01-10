import os
import json
import argparse

import torch
import torchvision
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.real_and_composite_images_dataloaders import create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser('Train Real/composite Images classifier '
                                     'based on VGG')
    parser.add_argument('--dataset',
                        choices=['HCOCO', 'HAdobe5k', 'HFlickr',
                                 'Hday2night', 'IHD', 'LabelMe_all',
                                 'LabelMe_15categories'],
                        default='Hday2night',
                        help='dataset name.')
    parser.add_argument('--epochs',
                        type=int,
                        default=0,
                        help='Number of train epochs. Default is -1 which '
                             'means that the number of epochs will be a bit '
                             'higher than 25K iterations.')
    parser.add_argument('--batch-size',
                        type=int,
                        default=50,
                        help='Batch size, set to labelMe_all default: 50.')
    parser.add_argument('--optimizer-type',
                        type=str,
                        default='SGD',
                        choices=['SGD', 'Adam'],
                        help='Which optimizer, default: SGD.')
    parser.add_argument('--data-dir',
                        default='../data/Image_Harmonization_Dataset/',
                        choices=[
                            '/storage/jevnisek/ImageHarmonizationDataset/',
                            '../data/Image_Harmonization_Dataset/',
                            '../data/realism_datasets/',
                            '/storage/jevnisek/realism_datasets/'
                        ])
    parser.add_argument('--target-dir', default='results/vgg_training')
    return parser.parse_args()


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11',
                                    pretrained=True)

        # adapt vgg to binary classification
        self.model.classifier[6] = torch.nn.Linear(in_features=4096,
                                                   out_features=2, bias=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task='multiclass', num_classes=2)
        return loss, acc

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log("val_accuracy", acc, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log("test_accuracy", acc, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=10)
        return [optimizer], [scheduler]


args = parse_args()


target_dir = os.path.join(args.target_dir,
                          datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
os.makedirs(target_dir, exist_ok=True)
with open(os.path.join(target_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)

train_loader, test_dataloader = create_dataloaders(
    data_dir=args.data_dir,
    dataset=args.dataset,
    train_batch_size=args.batch_size,
    test_batch_size=128, num_workers=10)

checkpoint_callback = ModelCheckpoint(
    dirpath=target_dir,
    filename="checkpoint-{epoch:02d}-{train_loss:.2f}",
    save_top_k=-1, every_n_epochs=1, save_last=True
)
trainer = pl.Trainer(accelerator='gpu',
                     max_epochs=args.epochs, check_val_every_n_epoch=1,
                     enable_checkpointing=True, callbacks=[checkpoint_callback])
model = LitModel()

trainer.fit(model, train_dataloaders=train_loader,
            val_dataloaders=test_dataloader)
trainer.validate(model, dataloaders=test_dataloader)
# call after training
trainer.test(model, dataloaders=test_dataloader)
