import os

from tqdm import tqdm


import torch
from math import ceil
from PIL import Image

import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tools.discriminator_test_utils import get_logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LANDONE_DATA_ROOT = '/mnt/data/data/realism_datasets/human_evaluation/'\
                    'lalonde_and_efros_dataset'


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
    def __init__(self, train_images_paths, test_images_paths, artifacts_dir,
                 epochs=0, batch_size=50, landone_root=LANDONE_DATA_ROOT,
                 optimizer_type='SGD'):
        self.train_images_paths = train_images_paths
        self.test_images_paths = test_images_paths
        self.landone_root = landone_root
        self.batch_size = batch_size

        self.model = self.initialize_network().cuda()
        self.initialize_training(optimizer_type=optimizer_type)
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
        os.makedirs(os.path.join(self.artifacts_dir, 'landone-stats'),
                    exist_ok=True)
        self.logger = get_logger(self.logfile_path)
        self.tb_writer = SummaryWriter(log_dir=self.tensorboard_log_dir)
        if epochs == 0:
            self.epochs = ceil(25 * 1e3 / self.batch_size / len(
                self.train_dataloader) * 1.0)
        else:
            self.epochs = epochs

    def initialize_network(self,):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11',
                               pretrained=True)

        # adapt vgg to binary classification
        model.classifier[6] = torch.nn.Linear(in_features=4096,
                                              out_features=2, bias=True)
        return model

    def initialize_training(self, optimizer_type='SGD'):
        # self.learning_rate = 1e-4  # 0.0001
        self.learning_rate = 1e-3  # 0.001
        # self.optimizer = torch.optim.SGD([
        #     {'params': [x for i, x in enumerate(self.model.parameters())
        #                 if i < 20]},
        #     {'params': [x for i, x in enumerate(self.model.parameters())
        #                 if i >= 20],
        #      'lr': self.learning_rate * 10.0}],
        #     lr=self.learning_rate, momentum=0.9)
        if optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.learning_rate,
                                             momentum=0.9)
        elif optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                             lr=self.learning_rate)
        else:
            assert False, f"optimizer {optimizer_type} not supported."
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=int(1e4 / self.batch_size), gamma=0.1)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.curr_iters = 0
        self.num_iters = 25 * 1e3


    def initialize_landone_dataset(self):
        import scipy
        path_to_mat = os.path.join(self.landone_root, 'human_labels.mat')
        human_labels = scipy.io.loadmat(path_to_mat)

        composite_images = [
            os.path.join(self.landone_root, 'images', name[0][0])
            for name, label in
            zip(human_labels['imgList'], human_labels['labels'])
            if human_labels['label_strs'][0][label[0]][0] != 'natural photos']
        real_images = [
            os.path.join(self.landone_root, 'images', name[0][0])
            for name, label in
            zip(human_labels['imgList'], human_labels['labels'])
            if human_labels['label_strs'][0][label[0]][0] == 'natural photos']
        import torchvision

        self.landone_dataset = FakesAndRealsDataset(
            real_images, composite_images,
            torchvision.models.VGG16_Weights.IMAGENET1K_V1.transforms()
        )
        self.test_landone_dataloader = DataLoader(self.landone_dataset,
                                                  batch_size=64, shuffle=False)

    def initialize_datasets(self):
        self.initialize_landone_dataset()
        self.train_set = FakesAndRealsDataset(
            self.train_images_paths['real_images'],
            self.train_images_paths['fake_images'],
            torchvision.models.VGG16_Weights.IMAGENET1K_V1.transforms()
        )
        self.test_set = FakesAndRealsDataset(
            self.test_images_paths['real_images'],
            self.test_images_paths['fake_images'],
            torchvision.models.VGG16_Weights.IMAGENET1K_V1.transforms()
        )
        self.train_dataloader = DataLoader(self.train_set,
                                           batch_size=self.batch_size,
                                           shuffle=True)
        self.test_dataloader = DataLoader(self.test_set,
                                           batch_size=self.batch_size,
                                           shuffle=False)

    def train(self):
        self.model.train()
        for batch in tqdm(self.train_dataloader):
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
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        # since we're not training, we don't need to calculate the gradients
        # for our outputs
        with torch.no_grad():
            for data in tqdm(dataloader):
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

    def test_with_auc(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        # since we're not training, we don't need to calculate the gradients
        # for our outputs
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for data in tqdm(dataloader):
                images, labels = data['image'].cuda(), data['label'].cuda()
                # calculate outputs by running images through the network
                outputs = self.model(images)
                all_scores.append(outputs[..., 0].cpu().detach())
                all_labels.append(labels.cpu().detach())
                curr_loss = self.criterion(outputs, labels)
                total_loss += curr_loss.item()
                # the class with the highest energy is what we choose as
                # prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        mean_loss = total_loss / total * 1.0
        auc = roc_auc_score(
                            torch.cat(all_labels),
                            torch.cat(all_scores))
        auc = auc if auc > 0.5 else 1 - auc
        test_metrics = {'accuracy': accuracy, 'mean_loss': mean_loss,
                        'auc': 100.0 * auc}
        return test_metrics

    def test_landone_dataset(self, epoch):
        from sklearn import svm
        from sklearn.metrics import roc_auc_score

        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn import svm
        from sklearn.metrics import RocCurveDisplay
        from sklearn.model_selection import StratifiedKFold
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        all_scores = []
        all_labels = []
        features = []
        with torch.no_grad():
            for data in tqdm(self.test_landone_dataloader):
                images, labels = data['image'].cuda(), data['label'].cuda()
                all_labels.append(labels.tolist())
                outputs = self.model(images)

                # extract fc7 features
                feat = self.model.features(images)
                feat_avg = self.model.avgpool(feat)
                feat_cls = self.model.classifier[:4](feat_avg.flatten(1))
                features.append(feat_cls.cpu().detach())

                all_scores.append(outputs[..., 1].tolist())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        accuracy = accuracy if accuracy > 0.5 else 1 - accuracy
        print(f"Realism Accuracy: {accuracy * 100.0:.2f} [%]")
        auc = roc_auc_score(sum(all_labels, []), sum(all_scores, []))
        auc = auc if auc > 0.5 else 1 - auc
        print(f"Realism AuC score: {auc * 100.0:.2f} [%]")
        X = torch.cat(features, axis=0)
        y = np.array(sum(all_labels, []))

        cv = StratifiedKFold(n_splits=10)
        classifier = svm.SVC(kernel="linear", probability=True, random_state=42)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots(figsize=(6, 6))
        for fold, (train, test) in enumerate(cv.split(X, y)):
            classifier.fit(X[train], y[train])
            viz = RocCurveDisplay.from_estimator(
                classifier,
                X[test],
                y[test],
                name=f"ROC fold {fold}",
                alpha=0.3,
                lw=1,
                ax=ax,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
        ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        from sklearn.metrics import auc
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        print(
            f"10 fold cross-validation, mean AuC: {mean_auc * 100.0:.2f} [%] "
            f"+- "
            f"std(AuC): {std_auc * 100.0:.2f}")
        plt.title(f'AuC = {mean_auc * 100.0:.2f} [%]')
        plt.savefig(os.path.join(self.artifacts_dir,'landone-stats',
                                 f'epoch_{epoch:03d}_landone_auc.png'))
        return mean_auc

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
                   f"{mode} Loss: {metrics['mean_loss']:.6f} | "
                   f"{mode} Accuracy: {metrics['accuracy']:.2f} [%]")
        print(message)
        self.logger.debug(message)
        self.tb_writer.add_scalar(f'Loss/{mode}/loss', metrics['mean_loss'],
                                  epoch)
        self.tb_writer.add_scalar(f'Accuracy/{mode}/accuracy',
                                  metrics['accuracy'], epoch)
        if mode == 'test':
            self.tb_writer.add_scalar(f'AuC/{mode}/landone-auc',
                                      metrics['landone_auc'], epoch)
            self.tb_writer.add_scalar(f'AuC/{mode}/test-auc',
                                      metrics['auc'], epoch)
            message = (f"[{epoch:04d} / {self.epochs:04d}] | "
                       f"{mode} AuC: {metrics['auc']:.2f} [%] ")
            print(message)
            self.logger.debug(message)

    def run(self):
        best_train_accuracy = 0
        for epoch in range(1, 1 + self.epochs):
            self.train()
            train_metrics = self.test(self.train_dataloader)
            self.write_stats_for_mode(epoch, train_metrics, mode='train')
            test_metrics = self.test_with_auc(self.test_dataloader)
            landone_auc = float(self.test_landone_dataset(epoch))
            test_metrics['landone_auc'] = landone_auc
            self.write_stats_for_mode(epoch, test_metrics, mode='test')
            if train_metrics['accuracy'] > best_train_accuracy:
                self.cache_model_and_stats(epoch, train_metrics, test_metrics)
                best_train_accuracy = train_metrics['accuracy']

