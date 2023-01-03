# create results directories:
mkdir -p /storage/jevnisek/vgg_classifier/${DATASET}
# generate configs:
python main_vgg_classifier_for_image_harmonization_datasets.py \
  --dataset=${DATASET} --epochs=${EPOCHS} \
  --data-dir ${DATA_DIR} --batch-size=${BATCH_SIZE} \
  --target-dir /storage/jevnisek/vgg_classifier/${DATASET} \
  --landone-root /storage/jevnisek/realism_datasets/human_evaluation/lalonde_and_efros_dataset/
