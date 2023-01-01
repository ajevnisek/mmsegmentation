import os
import json
import time
import mmcv

from tqdm import tqdm

import torch
import torch.nn as nn

from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

import dove_discriminator.networks as networks

from tools.discriminator_args import DiscriminatorArgs
from dove_discriminator.dovenet_model import create_discriminator
from dove_discriminator.discriminator_dataset import get_loader
from tools.discriminator_test_utils import calc_metrics, log_metrics,\
    generate_graphs, get_logger


TEXT_LOGS = 'text-logs'
ARGUMENTS_LOGS = 'arguments-logs'
CHECKPOINTS_ROOT = 'checkpoints'
TENSORBOARD_LOGS = 'tensorboard-logs'
DEFAULT_DISCRIMINATOR_ARGS = DiscriminatorArgs()


class GlobalAndLocalFeaturesDiscriminator:
    def __init__(self, train_loader, test_loader, root_results_dir,
                 discriminator_args=DEFAULT_DISCRIMINATOR_ARGS):
        self.netD = create_discriminator()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gan_mode = discriminator_args.gan_mode
        self.criterionGAN = networks.GANLoss(self.gan_mode).to(self.device)
        self.lr = discriminator_args.learning_rate
        self.beta1 = discriminator_args.beta1
        self.gp_ratio = discriminator_args.gp_ratio
        self.relu = nn.ReLU()
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=self.lr,
                                            betas=(self.beta1, 0.999))
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.root_results_dir = root_results_dir
        self._create_artifact_folders()
        self.init_losses()
        self.logger = get_logger(self.logfile_path)
        self.tb_writer = SummaryWriter(log_dir=self.tensorboard_log_dir)
        self.arguments = {'lr': self.lr, 'gan_mode': self.gan_mode,
                          'beta1': self.beta1, 'gp_ratio': self.gp_ratio,
                          'train_batch_size': self.train_loader.batch_size,
                          'train_dataset': train_loader.dataset.dataset_name}
        self.store_arguments_in_json()
        self.epoch = 0

    def init_losses(self):
        self.train_loss = 0
        self.train_loss_D_fake = 0
        self.train_loss_D_real = 0
        self.train_gradient_penalty = 0
        self.test_loss = 0
        self.test_loss_D_fake = 0
        self.test_loss_D_real = 0

    def _create_artifact_folders(self):
        self.checkpoint_path = os.path.join(self.root_results_dir,
                                            CHECKPOINTS_ROOT)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.tensorboard_log_dir = os.path.join(self.root_results_dir,
                                                TENSORBOARD_LOGS,
                                                )
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        self.logfile_path = os.path.join(self.root_results_dir, TEXT_LOGS,
                                         'trainer_log.log')
        os.makedirs(os.path.dirname(self.logfile_path), exist_ok=True)
        self.arguments_log_path = os.path.join(self.root_results_dir,
                                               ARGUMENTS_LOGS,
                                               'arguments.json')
        os.makedirs(os.path.dirname(self.arguments_log_path), exist_ok=True)

    def store_arguments_in_json(self):
        with open(self.arguments_log_path, 'w') as f:
            json.dump(self.arguments, f, indent=4)

    def set_input(self, input):
        self.compoisite = input['composite_image'].to(self.device)
        self.real = input['real_image'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.real_f = self.real * self.mask
        self.bg = self.real * (1 - self.mask)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake;
        fake_AB = self.compoisite
        pred_fake, ver_fake = self.netD(fake_AB.detach(), self.mask)
        if self.gan_mode == 'wgangp':
            global_fake = self.relu(1 + pred_fake).mean()
            local_fake = self.relu(1 + ver_fake).mean()
        else:
            global_fake = self.criterionGAN(pred_fake, False)
            local_fake = self.criterionGAN(ver_fake, False)
        self.loss_D_fake = global_fake + local_fake

        # Real
        real_AB = self.real
        pred_real, ver_real = self.netD(real_AB, self.mask)
        if self.gan_mode == 'wgangp':
            global_real = self.relu(1 - pred_real).mean()
            local_real = self.relu(1 - ver_real).mean()
        else:
            global_real = self.criterionGAN(pred_real, True)
            local_real = self.criterionGAN(ver_real, True)

        self.loss_D_real = global_real + local_real
        self.loss_D_global = global_fake + global_real
        self.loss_D_local = local_fake + local_real

        gradient_penalty, gradients = networks.cal_gradient_penalty(
            self.netD, real_AB.detach(), fake_AB.detach(), 'cuda',
            mask=self.mask)
        self.loss_D_gp = gradient_penalty

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real + self.gp_ratio * gradient_penalty)
        self.loss_D.backward(retain_graph=True)

        # track losses:
        self.train_loss += self.loss_D.item()
        self.train_loss_D_fake += self.loss_D_fake.item()
        self.train_loss_D_real += self.loss_D_real.item()
        self.train_gradient_penalty += gradient_penalty.item()

    def train_one_epoch(self):
        self.netD.train()
        for batch in tqdm(self.train_loader):
            self.set_input(batch)
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D()  # calculate gradients for D
            self.optimizer_D.step()  # update D's weights

        # update train losses:
        self.train_loss /= (len(self.train_loader) *
            self.train_loader.batch_size)
        self.train_loss_D_fake /= (len(self.train_loader) *
                                   self.train_loader.batch_size)
        self.train_loss_D_real /= (len(self.train_loader) *
                                   self.train_loader.batch_size)
        self.train_gradient_penalty /= (len(self.train_loader) *
                                        self.train_loader.batch_size)

        # log train losses in tensorboard:
        self.tb_writer.add_scalar('Loss/train/loss', self.train_loss, self.epoch)
        self.tb_writer.add_scalar('Loss/train/loss_D_fake', self.train_loss_D_fake, self.epoch)
        self.tb_writer.add_scalar('Loss/train/loss_D_real', self.train_loss_D_real, self.epoch)
        self.tb_writer.add_scalar('Loss/train/gradient_penalty', self.train_gradient_penalty, self.epoch)
        message = (f"[{self.epoch}] | "
                   f"Train Loss: {self.train_loss:.3f} | ")
        print(message)
        self.logger.debug(message)

    def test_one_epoch(self, loader):
        self.netD.eval()
        results = defaultdict(list)
        self.netD.eval()
        for batch in tqdm(loader):
            self.set_input(batch)
            with torch.no_grad():
                fake_AB = self.compoisite
                pred_fake, ver_fake = self.netD(fake_AB.detach(), self.mask)
                if self.gan_mode == 'wgangp':
                    global_fake = self.relu(1 + pred_fake).mean()
                    local_fake = self.relu(1 + ver_fake).mean()
                else:
                    global_fake = self.criterionGAN(pred_fake, False)
                    local_fake = self.criterionGAN(ver_fake, False)
                self.loss_D_fake = global_fake + local_fake
                results['composite_image_path'].append(
                    batch['composite_image_path'])
                results['composite_image_global_score'].append(
                    pred_fake.detach().cpu())
                results['composite_image_domain_similarity_score'].append(
                    ver_fake.detach().cpu())

                real_AB = self.real
                pred_real, ver_real = self.netD(real_AB.detach(), self.mask)
                if self.gan_mode == 'wgangp':
                    global_real = self.relu(1 - pred_real).mean()
                    local_real = self.relu(1 - ver_real).mean()
                else:
                    global_real = self.criterionGAN(pred_real, True)
                    local_real = self.criterionGAN(ver_real, True)

                self.loss_D_real = global_real + local_real
                self.loss_D_global = global_fake + global_real
                self.loss_D_local = local_fake + local_real

                # combine loss and calculate gradients
                self.loss_D = self.loss_D_fake + self.loss_D_real
                # track losses:
                self.test_loss += self.loss_D.item()
                self.test_loss_D_fake += self.loss_D_fake.item()
                self.test_loss_D_real += self.loss_D_real.item()
                results['real_image_path'].append(batch['real_image_path'])
                results['real_image_global_score'].append(
                    pred_real.detach().cpu())
                results['real_image_domain_similarity_score'].append(
                    ver_real.detach().cpu())

        # update train losses:
        self.test_loss /= (len(loader) * loader.batch_size)
        self.test_loss_D_fake /= (len(loader) * loader.batch_size)
        self.test_loss_D_real /= (len(loader) * loader.batch_size)

        # log train losses in tensorboard:
        self.tb_writer.add_scalar('Loss/test/loss', self.test_loss,
                                  self.epoch)
        self.tb_writer.add_scalar('Loss/test/loss_D_fake',
                                  self.test_loss_D_fake, self.epoch)
        self.tb_writer.add_scalar('Loss/test/loss_D_real',
                                  self.test_loss_D_real, self.epoch)

        results['composite_image_path'] = sum(results[
            'composite_image_path'], [])
        results['real_image_path'] = sum(results['real_image_path'], [])
        for key in ['composite_image_global_score', 'real_image_global_score',
                    'composite_image_domain_similarity_score',
                    'real_image_domain_similarity_score']:
            results[key] = torch.cat(results[key]).squeeze()
        return results

    def save_checkpoint(self):
        """Save model checkpoint along with the epoch and train accuracy."""
        state = {
            'net': self.netD.state_dict(),
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
            'epoch': self.epoch,
        }
        torch.save(state,
                   os.path.join(self.checkpoint_path,
                                f'classifier_epoch_{self.epoch:03d}.pth'))

    def run(self, epochs):
        for self.epoch in range(1, 1 + epochs):
            self.init_losses()
            self.train_one_epoch()
            results = self.test_one_epoch(self.train_loader)
            curr_metrics = calc_metrics(results)
            self.tb_writer.add_scalar('AuC/train_auc_dove_score',
                                      curr_metrics['auc_dove_score'],
                                      self.epoch)
            self.tb_writer.add_scalar('AuC/train_auc_global_score',
                                      curr_metrics['auc_global_score'],
                                      self.epoch)
            self.tb_writer.add_scalar('AuC/train_auc_combined_score',
                                      curr_metrics['auc_combined_score'],
                                      self.epoch)
            message = (f"[{self.epoch}] | "
                       f"Train Loss: {self.train_loss:.3f} | "
                       f"Train AuC Dove Score: "
                       f"{curr_metrics['auc_dove_score']:.3f} | "
                       f"Train AuC Global Score: "
                       f"{curr_metrics['auc_global_score']:.3f} | "
                       f"Train AuC Combined Score: "
                       f"{curr_metrics['auc_combined_score']:.3f} | ")
            print(message)
            self.logger.debug(message)
            generate_graphs(results, curr_metrics, self.epoch,
                            os.path.join(self.root_results_dir,
                                         'train_results'))
            log_metrics(curr_metrics, self.epoch,
                        os.path.join(self.root_results_dir, 'train_results'))
            results = self.test_one_epoch(self.test_loader)
            curr_metrics = calc_metrics(results)
            self.tb_writer.add_scalar('AuC/test_auc_dove_score',
                                      curr_metrics['auc_dove_score'],
                                      self.epoch)
            self.tb_writer.add_scalar('AuC/test_auc_global_score',
                                      curr_metrics['auc_global_score'],
                                      self.epoch)
            self.tb_writer.add_scalar('AuC/test_auc_combined_score',
                                      curr_metrics['auc_combined_score'],
                                      self.epoch)
            message = (f"[{self.epoch}] | "
                       f"Test Loss: {self.test_loss:.3f} | "
                       f"Test AuC Dove Score: "
                       f"{curr_metrics['auc_dove_score']:.3f} | "
                       f"Test AuC Global Score: "
                       f"{curr_metrics['auc_global_score']:.3f} | "
                       f"Test AuC Combined Score: "
                       f"{curr_metrics['auc_combined_score']:.3f} | ")
            print(message)
            self.logger.debug(message)
            generate_graphs(results, curr_metrics, self.epoch,
                            os.path.join(self.root_results_dir, 'test_results'))
            log_metrics(curr_metrics, self.epoch,
                        os.path.join(self.root_results_dir, 'test_results'))
            print(f"Epoch [{self.epoch:03d}]: Domain Verification Score: "
                  f"{curr_metrics['auc_dove_score'] * 100:.2f} [%]")
            print(f"Epoch [{self.epoch:03d}]: Global Score: "
                  f"{curr_metrics['auc_global_score'] * 100:.2f} [%]")
            print(f"Epoch [{self.epoch:03d}]: Combined Score: "
                  f"{curr_metrics['auc_combined_score'] * 100:.2f} [%]")
            print("~" * 30)
            if self.epoch % 5 == 0 or self.epoch == epochs:
                self.save_checkpoint()