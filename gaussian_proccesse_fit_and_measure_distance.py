import mmcv
import numpy as np
import matplotlib.pyplot as plt

import scipy
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity

from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import jensenshannon


K = 3  # number of GMM components.
H = W = 256

composite_image = mmcv.imread('../data/Image_Harmonization_Dataset/HAdobe5k/composite_images/a0001_1_1.jpg')
real_image = mmcv.imread('../data/Image_Harmonization_Dataset/HAdobe5k/real_images/a0001.jpg')
mask = mmcv.imread('../data/Image_Harmonization_Dataset/HAdobe5k/masks/a0001_1.png', 'grayscale')

composite_image = mmcv.imread('../data/Image_Harmonization_Dataset/HCOCO/composite_images/c100300_1602094_1.jpg')
real_image = mmcv.imread('../data/Image_Harmonization_Dataset/HCOCO/real_images/c100300.jpg')
mask = mmcv.imread('../data/Image_Harmonization_Dataset/HCOCO/masks/c100300_1602094.png', 'grayscale')

composite_image = mmcv.imresize(composite_image, (H, W), return_scale=False)
real_image = mmcv.imresize(real_image, (H, W), return_scale=False)
mask = mmcv.imresize(mask, (H, W), return_scale=False)
mask[mask != 0] = 255

plt.subplot(2, 3, 1)
plt.title("real image")
plt.imshow(mmcv.bgr2rgb(real_image))
plt.subplot(2, 3, 2)
plt.title("mask")
plt.imshow(mask)
plt.colorbar()
plt.subplot(2, 3, 3)
plt.title("composite image")
plt.imshow(mmcv.bgr2rgb(composite_image))
titles = ['Y', 'Cb', 'Cr']
for i in range(3):
    plt.subplot(2, 3, 4 + i)
    plt.title(f'{titles[i]}-channel')
    plt.imshow(mmcv.bgr2ycbcr(composite_image)[..., i])
    plt.colorbar()
plt.gcf().set_size_inches((12, 6))
plt.show()

ycbcr_real_image = mmcv.bgr2ycbcr(real_image)
ycbcr_composite_image = mmcv.bgr2ycbcr(composite_image)
data = ycbcr_composite_image[mask == 0]
gm = GaussianMixture(n_components=K, random_state=0).fit(data)

composite_test_data = ycbcr_composite_image[mask != 0]
real_test_data = ycbcr_real_image[mask != 0]

real_scores = gm.score_samples(real_test_data)
composite_scores = gm.score_samples(composite_test_data)
real_bg_scores = gm.score_samples(ycbcr_real_image[mask == 0])

kde_bg_scores = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
    real_bg_scores[:, np.newaxis])
kde_real_fg_scores = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
    real_scores[:, np.newaxis])
kde_composite_scores = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
    composite_scores[:, np.newaxis])

x = np.linspace(-100, 100, 1000)[:, np.newaxis]
plt.plot(x, np.exp(kde_bg_scores.score_samples(x)))
plt.plot(x, np.exp(kde_real_fg_scores.score_samples(x)))
plt.plot(x, np.exp(kde_composite_scores.score_samples(x)))
plt.legend(['bg-scores', 'fg-real', 'fg-composite'])
plt.show()
from scipy.spatial import distance

real_distance = distance.jensenshannon(
    np.exp(kde_bg_scores.score_samples(x)),
    np.exp(kde_real_fg_scores.score_samples(x)))
composite_distance = distance.jensenshannon(
    np.exp(kde_bg_scores.score_samples(x)),
    np.exp(kde_composite_scores.score_samples(x)))
print(f"Jensen-Shannon: real_distance:{real_distance} < composite_distance:"
      f" {composite_distance}")

from scipy.stats import wasserstein_distance
real_wasserstein_distance = wasserstein_distance(real_scores, real_bg_scores)
composite_wasserstein_distance = wasserstein_distance(composite_scores,
                                                      real_bg_scores)
print(f"Wasserstein: real_distance: {real_wasserstein_distance} < "
      f"composite_distance:"
      f" {composite_wasserstein_distance}")
