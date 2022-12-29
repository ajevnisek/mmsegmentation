# create results directories:
mkdir -p /storage/jevnisek/vgg_classifier/${DATASET}
# generate configs:
python main_vgg_classifier_for_image_harmonization_datasets.py \
  --dataset=${DATASET} --epochs=${EPCOHS}\
  --data-dir /storage/jevnisek/ImageHarmonizationDataset/ \
  --target-dir /storage/jevnisek/vgg_classifier/${DATASET}
