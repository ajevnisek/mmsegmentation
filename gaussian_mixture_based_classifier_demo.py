import mmcv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture


K = 9  # number of GMM components.
H = W = 256

composite_image = mmcv.imread('../data/Image_Harmonization_Dataset/HAdobe5k/composite_images/a0001_1_1.jpg')
real_image = mmcv.imread('../data/Image_Harmonization_Dataset/HAdobe5k/real_images/a0001.jpg')
mask = mmcv.imread('../data/Image_Harmonization_Dataset/HAdobe5k/masks/a0001_1.png', 'grayscale')

# composite_image = mmcv.imread('../data/Image_Harmonization_Dataset/HCOCO/composite_images/c100300_1602094_1.jpg')
# real_image = mmcv.imread('../data/Image_Harmonization_Dataset/HCOCO/real_images/c100300.jpg')
# mask = mmcv.imread('../data/Image_Harmonization_Dataset/HCOCO/masks/c100300_1602094.png', 'grayscale')

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

auc = roc_auc_score(
    np.array([1] * len(real_scores) + [0] * len(composite_scores)),
    np.concatenate([real_scores, composite_scores])
)

print(f"Area Under the Curve: {auc * 100:.2f} [%]")

plt.clf()
plt.title('GMM scores histogram\n'
          f"Area Under the Curve: {auc * 100:.2f} [%]")
plt.hist(real_bg_scores, bins=int(np.sqrt(real_bg_scores.shape[0])),
         alpha=0.5, label='real-bg-scores')
plt.hist(real_scores, bins=int(np.sqrt(real_scores.shape[0])),
         alpha=0.5, label='real-scores')
plt.hist(composite_scores, bins=int(np.sqrt(real_scores.shape[0])),
         alpha=0.5, label='composite-scores')
plt.legend()
plt.grid(True)
plt.show()


