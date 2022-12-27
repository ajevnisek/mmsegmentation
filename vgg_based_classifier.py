import os
import torch
from math import ceil
from PIL import Image
try:
    from torchvision.models import vgg, VGG11_Weights
    IS_CLUSTER = False
except:
    from torchvision.models import vgg11
    IS_CLUSTER = True
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tools.discriminator_test_utils import get_logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FakesAndRealsDataset(torch.utils.data.Dataset):
    def __init__(self, real_images_paths, fake_images_paths, transform):
        self.real_images_paths = real_images_paths
        self.fake_images_paths = fake_images_paths
        self.transform = transform

    def __getitem__(self, item):
        if item < len(self.real_images_paths):
            image_path = self.real_images_paths[item]
            label = 0
        else:
            image_path = self.fake_images_paths[item - len(self.real_images_paths)]
            label = 1
        image = Image.open(image_path)
        image_tensor = self.transform(image)
        sample = {'path': image_path,
                  'label': label,
                  'image': image_tensor}
        return sample

    def __len__(self):
        return len(self.real_images_paths) + len(self.fake_images_paths)


class Trainer:
    def __init__(self, train_images_paths, test_images_paths, artifacts_dir):
        self.train_images_paths = train_images_paths
        self.test_images_paths = test_images_paths

        self.model = self.initialize_network().cuda()
        self.initialize_training()
        self.initialize_datasets()
        self.artifacts_dir = artifacts_dir
        os.makedirs(os.path.join(self.artifacts_dir, 'text-logs'),
                    exist_ok=True)
        self.logfile_path = os.path.join(self.artifacts_dir, 'text-logs',
                                         'training.log')
        os.makedirs(os.path.join(self.artifacts_dir, 'tensorboard'),
                    exist_ok=True)
        self.tensorboard_log_dir = os.path.join(self.artifacts_dir,
                                                'tensorboard', )
        self.logger = get_logger(self.logfile_path)
        self.tb_writer = SummaryWriter(log_dir=self.tensorboard_log_dir)
        self.epochs = ceil(25 * 1e3 / self.batch_size / len(
            self.train_dataloader) * 1.0)

    def initialize_network(self,):
        if IS_CLUSTER:
            model = vgg11(pretrained=True)
        else:
            weights = VGG11_Weights.IMAGENET1K_V1
            model = vgg.vgg11(weights=weights)
        # adapt vgg to binary classification
        model.classifier[6] = torch.nn.Linear(in_features=4096,
                                              out_features=2, bias=True)
        return model

    def initialize_training(self):
        self.learning_rate = 1e-4  # 0.0001
        self.optimizer = torch.optim.SGD([
            {'params': [x for i, x in enumerate(self.model.parameters())
                        if i < 20]},
            {'params': [x for i, x in enumerate(self.model.parameters())
                        if i >= 20],
             'lr': self.learning_rate * 10.0}],
            lr=self.learning_rate, momentum=0.9)
        # self.optimizer = torch.optim.SGD(self.model.parameters(),
        #     lr=self.learning_rate, momentum=0.9)
        self.batch_size = 50
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=int(1e4 / self.batch_size), gamma=0.1)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.curr_iters = 0
        self.num_iters = 25 * 1e3



    def initialize_datasets(self):
        self.train_set = FakesAndRealsDataset(
            self.train_images_paths['real_images'],
            self.train_images_paths['fake_images'],
            transforms.Compose([transforms.Resize((256, 256)),
                                transforms.ToTensor()])
        )
        self.test_set = FakesAndRealsDataset(
            self.test_images_paths['real_images'],
            self.test_images_paths['fake_images'],
            transforms.Compose([transforms.Resize((256, 256)),
                                transforms.ToTensor()])
        )
        self.train_dataloader = DataLoader(self.train_set,
                                           batch_size=self.batch_size,
                                           shuffle=True)
        self.test_dataloader = DataLoader(self.test_set,
                                           batch_size=self.batch_size,
                                           shuffle=False)

    def train(self):
        for batch in self.train_dataloader:
            images, labels = batch['image'].cuda(), batch['label'].cuda()
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

    def test(self, dataloader):
        correct = 0
        total = 0
        total_loss = 0
        # since we're not training, we don't need to calculate the gradients
        # for our outputs
        with torch.no_grad():
            for data in dataloader:
                images, labels = data['image'].cuda(), data['label'].cuda()
                # calculate outputs by running images through the network
                outputs = self.model(images)
                curr_loss = self.criterion(outputs, labels)
                total_loss += curr_loss.item()
                # the class with the highest energy is what we choose as
                # prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        mean_loss = total_loss / total * 1.0
        test_metrics = {'accuracy': accuracy, 'mean_loss': mean_loss}
        return test_metrics

    def cache_model_and_stats(self, epoch, train_metrics, test_metrics):
        cache_dict = {'epoch': epoch}
        for k in train_metrics:
            cache_dict[f"train_{k}"] = train_metrics[k]
        for k in train_metrics:
            cache_dict[f"test_{k}"] = test_metrics[k]
        cache_dict['model'] = self.model.state_dict()
        torch.save(cache_dict, os.path.join(
            self.artifacts_dir, f"best_model_epoch_{epoch:03d}.pth"))

    def write_stats_for_mode(self, epoch, metrics, mode='train'):
        message = (f"[{epoch:04d} / {self.epochs:04d}] | "
                   f"{mode} Loss: {metrics['mean_loss']:.4f} | "
                   f"{mode} Accuracy: {metrics['accuracy']:.2f} [%]")
        print(message)
        self.logger.debug(message)
        self.tb_writer.add_scalar(f'Loss/{mode}/loss', metrics['mean_loss'],
                                  epoch)
        self.tb_writer.add_scalar(f'Accuracy/{mode}/accuracy',
                                  metrics['accuracy'], epoch)

    def run(self):
        best_train_accuracy = 0
        for epoch in range(1, 1 + self.epochs):
            self.train()
            train_metrics = self.test(self.train_dataloader)
            self.write_stats_for_mode(epoch, train_metrics, mode='train')
            test_metrics = self.test(self.train_dataloader)
            self.write_stats_for_mode(epoch, test_metrics, mode='test')
            if train_metrics['accuracy'] > best_train_accuracy:
                self.cache_model_and_stats(epoch, train_metrics, test_metrics)
                best_train_accuracy = train_metrics['accuracy']

