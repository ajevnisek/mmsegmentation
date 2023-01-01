import os
import torch
import scipy

import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader

from sklearn import svm
from sklearn.metrics import roc_auc_score


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold



model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
model.classifier[6] = torch.nn.Linear(in_features=4096,
                                              out_features=2, bias=True)
# checkpoint_path = 'results/vgg_training/2023_01_01__08_44_30/best_model_epoch_018.pth'
# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint['model'])

path_to_mat = '/mnt/data/data/realism_datasets/human_evaluation/lalonde_and_efros_dataset/human_labels.mat'
human_labels = scipy.io.loadmat(path_to_mat)

composite_images = [
    os.path.join('/mnt/data/data/realism_datasets/human_evaluation/'
                 'lalonde_and_efros_dataset/images/', name[0][0])
    for name, label in zip(human_labels['imgList'], human_labels['labels'])
    if human_labels['label_strs'][0][label[0]][0] != 'natural photos']
real_images = [
    os.path.join('/mnt/data/data/realism_datasets/human_evaluation/'
                 'lalonde_and_efros_dataset/images/', name[0][0])
    for name, label in zip(human_labels['imgList'], human_labels['labels'])
    if human_labels['label_strs'][0][label[0]][0] == 'natural photos']


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


import torchvision

landone_dataset = FakesAndRealsDataset(real_images, composite_images,
                                       torchvision.models. VGG16_Weights.IMAGENET1K_V1.transforms()
                                       )
test_dataloader = DataLoader(landone_dataset, batch_size=64, shuffle=False)
model.eval()
model = model.cuda()
correct = 0
total = 0
total_loss = 0
all_scores = []
all_labels = []
features = []
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data['image'].cuda(), data['label'].cuda()
        all_labels.append(labels.tolist())
        outputs = model(images)

        # extract fc7 features
        feat = model.features(images)
        feat_avg = model.avgpool(feat)
        feat_cls = model.classifier[:4](feat_avg.flatten(1))
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
print(f"10 fold cross-validation, mean AuC: {mean_auc * 100.0:.2f} [%] +- "
      f"std(AuC): {std_auc * 100.0:.2f}")
plt.title(f'AuC = {mean_auc * 100.0:.2f} [%]')
plt.show()
