import mmcv
import json
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

from dove_discriminator.discriminator_dataset import ImageHarmonizationDataset


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
    'HAdobe5k',
    'HCOCO',
    'Hday2night',
    'HFlickr'
]:

    test_loader = HarmonizationDataset(
        '../data/Image_Harmonization_Dataset/', dataset, dataset_type='test',)
    stats = []
    print(f'Detecting fake regions for {dataset} dataset')
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

        auc = roc_auc_score(
            np.array([1] * len(real_scores) + [0] * len(composite_scores)),
            np.concatenate([real_scores, composite_scores])
        )
        # print(f"{auc * 100:.2f}")
        d = {'auc': f"{auc:.4f}",
             'composite_image': sample['composite_image_path'],
             'real_image': sample['real_image_path'],
             'mask': sample['mask_path'],
             }
        with open(f'aucs_{dataset}.json', 'a+') as f:
            json.dump(d, f, indent=2)

