import os
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.l1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task='multiclass', num_classes=10)
        return loss, acc

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


train_loader = DataLoader(MNIST(os.getcwd(), download=True,
                                transform=transforms.ToTensor()),
                          batch_size=32)
test_dataloader = DataLoader(MNIST(os.getcwd(), train=False, download=True,
                                   transform=transforms.ToTensor()),
                             batch_size=128)
checkpoint_callback = ModelCheckpoint(
    dirpath="pl_checkpoints/",
    filename="sample-mnist-{epoch:02d}-{train_loss:.2f}",
)
trainer = pl.Trainer(accelerator='gpu',
                     max_epochs=20, check_val_every_n_epoch=2,
                     enable_checkpointing=True, callbacks=[checkpoint_callback])
model = LitModel()

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_dataloader)
trainer.validate(model, dataloaders=test_dataloader)
# call after training
trainer.test(model, dataloaders=test_dataloader)