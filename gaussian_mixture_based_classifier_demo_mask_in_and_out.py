import cv2
import mmcv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture


K = 9  # number of GMM components.
H = W = 512

# composite_image = mmcv.imread('../data/Image_Harmonization_Dataset/HAdobe5k/composite_images/a0001_1_1.jpg')
# real_image = mmcv.imread('../data/Image_Harmonization_Dataset/HAdobe5k/real_images/a0001.jpg')
# mask = mmcv.imread('../data/Image_Harmonization_Dataset/HAdobe5k/masks/a0001_1.png', 'grayscale')

# composite_image = mmcv.imread('../data/Image_Harmonization_Dataset/HCOCO/composite_images/c100300_1602094_1.jpg')
# real_image = mmcv.imread('../data/Image_Harmonization_Dataset/HCOCO/real_images/c100300.jpg')
# mask = mmcv.imread('../data/Image_Harmonization_Dataset/HCOCO/masks/c100300_1602094.png', 'grayscale')

composite_image = mmcv.imread(
    '../data/Image_Harmonization_Dataset/HCOCO/composite_images'
    '/c102004_1610456_1.jpg')
real_image = mmcv.imread('../data/Image_Harmonization_Dataset/HCOCO/real_images/c102004.jpg')
mask = mmcv.imread('../data/Image_Harmonization_Dataset/HCOCO/masks/c102004_1610456.png', 'grayscale')


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

# ycbcr_real_image = mmcv.bgr2ycbcr(real_image)
# ycbcr_composite_image = mmcv.bgr2ycbcr(composite_image)
ycbcr_real_image = real_image
ycbcr_composite_image = composite_image
data = ycbcr_composite_image[mask == 0]
gm = GaussianMixture(n_components=K, random_state=0).fit(data)

composite_test_data = ycbcr_composite_image[mask != 0]
real_test_data = ycbcr_real_image[mask != 0]

real_bg_scores = gm.score_samples(ycbcr_real_image[mask == 0])
real_scores = gm.score_samples(real_test_data)
composite_scores = gm.score_samples(composite_test_data)

A = min([real_bg_scores.min(), real_scores.min(), composite_scores.min()])
B = max([real_bg_scores.max(), real_scores.max(), composite_scores.max()])
C = 0
D = 255

heatmap_real = np.zeros_like(mask, dtype=np.float64)
heatmap_real[mask == 0] = real_bg_scores
heatmap_real[mask != 0] = real_scores
heatmap_real_normalized = (D-C) * (heatmap_real-A)/(B-A) + C
heatmap_real_img = cv2.applyColorMap(heatmap_real_normalized.astype(np.uint8),
                                     cv2.COLORMAP_JET)
super_imposed_real_img = cv2.addWeighted(heatmap_real_img, 0.5, mmcv.bgr2rgb(
    real_image), 0.5, 0)

heatmap_fake = np.zeros_like(mask, dtype=np.float64)
heatmap_fake[mask == 0] = real_bg_scores
heatmap_fake[mask != 0] = composite_scores
heatmap_fake_normalized = (D-C) * (heatmap_fake-A)/(B-A) + C
heatmap_fake_img = cv2.applyColorMap(heatmap_fake_normalized.astype(np.uint8),
                                     cv2.COLORMAP_JET)
super_imposed_fake_img = cv2.addWeighted(heatmap_fake_img, 0.5, mmcv.bgr2rgb(
    composite_image), 0.5, 0)

plt.clf()
plt.subplot(2, 3, 1)
plt.title('mask, GMM trained on BG pixels (=0)')
plt.imshow(mask)
plt.colorbar()
plt.subplot(2, 3, 2)
plt.title('scores for real image')
plt.imshow(heatmap_real_normalized)
plt.colorbar()
plt.subplot(2, 3, 3)
plt.title('scores for fake image')
plt.imshow(heatmap_fake_normalized)
plt.colorbar()
plt.subplot(2, 3, 5)
plt.title('scores overlaid for real image')
plt.imshow(super_imposed_real_img)
plt.colorbar()
plt.subplot(2, 3, 6)
plt.title('scores overlaid for fake image')
plt.imshow(super_imposed_fake_img)
plt.colorbar()
plt.show()

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
plt.hist(composite_scores, bins=int(np.sqrt(composite_scores.shape[0])),
         alpha=0.5, label='composite-scores')
plt.legend()
plt.grid(True)
plt.show()


plt.subplot(2, 2, 1)
plt.title('real image heatmap')
plt.imshow(heatmap_real_normalized)
plt.colorbar()
plt.subplot(2, 2, 2)
plt.title('fake image heatmap')
plt.imshow(heatmap_fake_normalized)
plt.colorbar()
plt.subplot(2, 2, 3)
plt.title('real image - fake image scores heatmap')
plt.imshow(heatmap_real_normalized - heatmap_fake_normalized)
plt.colorbar()
plt.subplot(2, 2, 4)
accuracy = (heatmap_real_normalized[mask != 0] -
            heatmap_fake_normalized[mask != 0] > 0).astype(
    np.float64).sum() / (mask != 0).sum()

plt.title('is real scores higher than fake scores?\n'
          f'accuracy {accuracy * 100.0:.2f} [%]')
plt.imshow((heatmap_real_normalized - heatmap_fake_normalized > 0).astype(
    np.float64))
plt.colorbar()
plt.tight_layout()
plt.show()


