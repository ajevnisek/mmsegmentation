import mmcv
import json
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

from dove_discriminator.discriminator_dataset import ImageHarmonizationDataset
from scipy.stats import wasserstein_distance


K = 5  # number of GMM components.
H = W = 256


class HarmonizationDataset(ImageHarmonizationDataset):
    def __init__(self, dataset_root, dataset_name, dataset_type='train',
                 image_transform=None, mask_transform=None):
        super().__init__(dataset_root, dataset_name, dataset_type,
                         image_transform, mask_transform)
    def __getitem__(self, item):
        real_image_path = self.real_images[item]
        composite_image_path = self.composite_images[item]
        mask_path = self.masks[item]

        real_image = mmcv.imread(real_image_path)
        composite_image = mmcv.imread(composite_image_path)
        mask = mmcv.imread(mask_path, 'grayscale')

        sample = {'real_image': real_image,
                  'composite_image': composite_image,
                  'mask': mask,
                  'real_image_path': real_image_path,
                  'composite_image_path': composite_image_path,
                  'mask_path': mask_path}
        return sample

    def __len__(self):
        return len(self.real_images)


for dataset in [
    'Hday2night',
    'HFlickr',
    'HAdobe5k',
    'HCOCO',
]:

    test_loader = HarmonizationDataset(
        '../data/Image_Harmonization_Dataset/', dataset, dataset_type='test',)
    stats = []
    print(f'Detecting fake regions for {dataset} dataset')
    wasserstein_dist_scores_real_all_dataset = []
    wasserstein_dist_scores_composite_all_dataset = []
    mean_scores_real_all_dataset = []
    mean_scores_composite_all_dataset = []
    correct = 0
    for sample in tqdm(test_loader):
        composite_image = sample['composite_image']
        real_image = sample['real_image']
        mask = sample['mask']

        composite_image = mmcv.imresize(composite_image, (H, W), return_scale=False)
        real_image = mmcv.imresize(real_image, (H, W), return_scale=False)
        mask = mmcv.imresize(mask, (H, W), return_scale=False)
        mask[mask != 0] = 255

        ycbcr_real_image = mmcv.bgr2ycbcr(real_image)
        ycbcr_composite_image = mmcv.bgr2ycbcr(composite_image)
        data = ycbcr_composite_image[mask == 0]
        gm = GaussianMixture(n_components=K, random_state=0).fit(data)

        composite_test_data = ycbcr_composite_image[mask != 0]
        real_test_data = ycbcr_real_image[mask != 0]

        real_scores = gm.score_samples(real_test_data)
        composite_scores = gm.score_samples(composite_test_data)
        bg_scores = gm.score_samples(data)

        wasserstein_dist_real = wasserstein_distance(real_scores, bg_scores)
        wasserstein_dist_composite = wasserstein_distance(composite_scores, bg_scores)

        wasserstein_dist_scores_real_all_dataset.append(wasserstein_dist_real)
        wasserstein_dist_scores_composite_all_dataset.append(wasserstein_dist_composite)

        mean_scores_real_all_dataset.append(real_scores.mean())
        mean_scores_composite_all_dataset.append(composite_scores.mean())

        if real_scores.mean() > composite_scores.mean():
            correct += 1

    auc = roc_auc_score(
        np.array([1] * len(wasserstein_dist_scores_composite_all_dataset) +
                 [0] * len(wasserstein_dist_scores_real_all_dataset)),
        np.concatenate([np.array(wasserstein_dist_scores_composite_all_dataset),
                        np.array(wasserstein_dist_scores_real_all_dataset)])
    )
    print(f"Wasserstein Distance Dataset AuC: {auc * 100:.2f} [%]")
    auc = roc_auc_score(
        np.array([1] * len(mean_scores_real_all_dataset) +
                 [0] * len(mean_scores_composite_all_dataset)),
        np.concatenate([np.array(mean_scores_real_all_dataset),
                        np.array(mean_scores_composite_all_dataset)])
    )
    print(f"Mean scores Dataset AuC: {auc * 100:.2f} [%]")
    print(f"Accuracy: {correct * 1.0 / len(test_loader) * 100.0} [%]")
