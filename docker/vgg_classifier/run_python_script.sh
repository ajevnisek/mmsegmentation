# create results directories:
mkdir -p /storage/jevnisek/vgg_classifier/${DATASET}
# generate configs:
python main_vgg_classifier_for_image_harmonization_datasets.py \
  --dataset=${DATASET} --epochs=${EPOCHS} \
  --data-dir ${DATA_DIR} \
  --target-dir /storage/jevnisek/vgg_classifier/${DATASET}
