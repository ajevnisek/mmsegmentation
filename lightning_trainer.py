"""
Example template for defining a system.
"""
from datetime import datetime
import os

import numpy as np

import pytorch_lightning as pl

from argparse import ArgumentParser

import torch
import torchmetrics
import torch.nn.functional as F

from torch import optim
from torch.utils.data import DataLoader

from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

from tools.real_and_composite_images_dataloaders import create_datasets


SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


class LightningTemplateModel(LightningModule):
    """
    Sample model to show how to define a template.
    Example:
        >>> # define simple Net for MNIST dataset
        >>> params = dict(
        ...     drop_prob=0.2,
        ...     batch_size=2,
        ...     in_features=28 * 28,
        ...     learning_rate=0.001 * 8,
        ...     optimizer_name='adam',
        ...     data_root='./datasets',
        ...     out_features=10,
        ...     hidden_dim=1000,
        ... )
        >>> from argparse import Namespace
        >>> hparams = Namespace(**params)
        >>> model = LightningTemplateModel(hparams)
    """

    def __init__(self, **kwargs):
        """
        Pass in hyperparameters as a `argparse.Namespace` or a `dict` to the model.
        """
        # init superclass
        super().__init__()
        for key in kwargs.keys():
            self.hparams[key] = kwargs[key]
        self.save_hyperparameters()
        if self.hparams.model_name == 'vgg11':
            self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11',
                                           pretrained=True)

            # adapt vgg to binary classification
            self.backbone.classifier[6] = torch.nn.Linear(in_features=4096,
                                                          out_features=2,
                                                          bias=True)
        elif self.hparams.model_name == 'resnet18':
            self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18',
                                           pretrained=True)

            # adapt vgg to binary classification
            self.backbone.fc = torch.nn.Linear(in_features=512,
                                               out_features=2,
                                               bias=True)
        else:
            assert False, f"model {self.hparams.model_name} not supported"

    def forward(self, x):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        x = self.backbone(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        preds = y_hat[..., 1] - y_hat[..., 0]
        labels = y.cpu()
        return {'loss': loss, 'log': tensorboard_logs,
                'train_preds': preds.cpu().detach(),
                'train_labels': labels,
                'train_loss': loss,
                }

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        preds = y_hat[..., 1] - y_hat[..., 0]
        labels = y.cpu()
        return {'val_loss': val_loss, "n_correct_pred": n_correct_pred,
                "n_pred": len(x),
                'val_preds': preds.cpu().detach(),
                'val_labels': labels,

                }

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        preds = y_hat[..., 1] - y_hat[..., 0]
        labels = y.cpu()
        return {'test_loss': test_loss, "n_correct_pred": n_correct_pred,
                "n_pred": len(x),
                'test_preds': preds.cpu().detach(),
                'test_labels': labels,
                }

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        all_preds = torch.cat([x['val_preds'] for x in outputs])
        all_labels = torch.cat([x['val_labels'] for x in outputs])
        auroc = torchmetrics.functional.auroc(all_preds, all_labels,
                                              num_classes=2, task='binary',
                                              average="micro")
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': val_acc,
                            'val_auroc': auroc}
        return {'val_loss': avg_loss, 'val_auroc': auroc,
                'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        all_preds = torch.cat([x['test_preds'] for x in outputs])
        all_labels = torch.cat([x['test_labels'] for x in outputs])
        auroc = torchmetrics.functional.auroc(all_preds, all_labels,
                                              num_classes=2, task='binary',
                                              average="micro")
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': test_acc}
        return {'test_loss': avg_loss, 'log': tensorboard_logs,
                'test_auroc': auroc}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        if self.hparams.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(),
                                   lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(),
                                  lr=self.hparams.learning_rate)
        else:
            assert False, f'optimizer {self.hparams.optimizer_name} not ' \
                          f'supported'
        return optimizer

    def prepare_data(self):
        self.train_set, self.test_set = create_datasets(
            data_dir=self.hparams.data_dir,
            dataset=self.hparams.dataset,
            model_name=self.hparams.model_name,)

    def train_dataloader(self):
        log.info('Training data loader called.')
        return DataLoader(self.train_set,
                          batch_size=self.hparams.batch_size, num_workers=10,
                          shuffle=True)

    def val_dataloader(self):
        log.info('Validation data loader called.')
        return DataLoader(self.test_set,
                          batch_size=self.hparams.batch_size, num_workers=10,
                          shuffle=True)

    def test_dataloader(self):
        log.info('Test data loader called.')
        return DataLoader(self.test_set,
                          batch_size=self.hparams.batch_size, num_workers=10,
                          shuffle=False)

    @staticmethod
    def add_model_specific_args():  # pragma: no-cover
        """
        Parameters you define here will be available to your model through `self.hparams`.
        """
        parser = ArgumentParser()

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)
        # gpu args
        parser.add_argument(
            '--accelerator',
            type=str,
            default='gpu',
            help='which machine the trainer runs on.'
        )
        parser.add_argument(
            '--devices',
            type=int,
            default=1,
            help='how many gpus'
        )

        # network params
        parser.add_argument('--model_name', default='vgg11', type=str,
                            choices=['vgg11', 'resnet18'])
        parser.add_argument('--learning_rate', default=0.001, type=float)

        # data
        parser.add_argument('--dataset',
                           choices=['HCOCO', 'HAdobe5k', 'HFlickr',
                                    'Hday2night', 'IHD', 'LabelMe_all',
                                    'LabelMe_15categories'],
                           default='Hday2night',
                           help='dataset name.')
        parser.add_argument('--data-dir',
                            default='../data/Image_Harmonization_Dataset/',
                            choices=[
                                '/storage/jevnisek/ImageHarmonizationDataset/',
                                '../data/Image_Harmonization_Dataset/',
                                '../data/realism_datasets/',
                                '/storage/jevnisek/realism_datasets/'
                            ])
        parser.add_argument('--target-dir', default='results/vgg_training')

        # training params (opt)
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--optimizer', default='adam', type=str,
                            choices=['adam', 'sgd'])
        parser.add_argument('--batch_size', default=64, type=int)
        return parser


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LightningTemplateModel(**vars(hparams))
    target_dir = os.path.join(hparams.target_dir,
                              datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
    os.makedirs(target_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=target_dir,
        filename="checkpoint-{epoch:02d}-{train_loss:.2f}",
        save_top_k=-1, every_n_epochs=1, save_last=True
    )

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        devices=hparams.devices,
        accelerator=hparams.accelerator,
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback]
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = os.path.dirname(os.path.realpath(__file__))

    # each LightningModule defines arguments relevant to it
    parser = LightningTemplateModel.add_model_specific_args()
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
""" python lightning_trainer.py --model_name resnet18 --dataset HAdobe5k --data-dir ../data/Image_Harmonization_Dataset/ --target-dir /mnt/data/pl_classifier/HAdobe5k/ --epochs 5 --batch_size 64"""
