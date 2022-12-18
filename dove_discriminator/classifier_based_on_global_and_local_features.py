import mmcv
import torch
import torch.nn as nn
from collections import defaultdict
import dove_discriminator.networks as networks
from dove_discriminator.dovenet_model import create_discriminator
from dove_discriminator.discriminator_dataset import get_loader


class GlobalAndLocalFeaturesDiscriminator:
    def __init__(self, train_loader, test_loader):
        self.netD = create_discriminator()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gan_mode = 'wgangp'
        self.criterionGAN = networks.GANLoss(self.gan_mode).to(self.device)
        self.lr = 0.0002
        self.beta1 = 0.5
        self.gp_ratio = 1.0
        self.relu = nn.ReLU()
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=self.lr,
                                            betas=(self.beta1, 0.999))
        self.train_loader = train_loader
        self.test_loader = test_loader

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

    def train_one_epoch(self):
        for batch in self.train_loader:
            self.set_input(batch)
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D()  # calculate gradients for D
            self.optimizer_D.step()  # update D's weights

    def test_one_epoch(self):
        results = defaultdict(list)
        self.netD.eval()
        for batch in self.test_loader:
            self.set_input(batch)
            with torch.no_grad():
                fake_AB = self.compoisite
                pred_fake, ver_fake = self.netD(fake_AB.detach(), self.mask)
                results['composite_image_path'].append(
                    batch['composite_image_path'])
                results['composite_image_global_score'].append(
                    pred_fake.detach().cpu())
                results['composite_image_domain_similarity_score'].append(
                    ver_fake.detach().cpu())

                real_AB = self.real
                pred_real, ver_real = self.netD(real_AB.detach(), self.mask)
                results['real_image_path'].append(batch['real_image_path'])
                results['real_image_global_score'].append(
                    pred_real.detach().cpu())
                results['real_image_domain_similarity_score'].append(
                    ver_real.detach().cpu())
        results['composite_image_path'] = sum(results[
            'composite_image_path'], [])
        results['real_image_path'] = sum(results['real_image_path'], [])
        for key in ['composite_image_global_score', 'real_image_global_score',
                    'composite_image_domain_similarity_score',
                    'real_image_domain_similarity_score']:
            results[key] = torch.cat(results[key]).squeeze()
        return results

