# create results directories:
mkdir -p /storage/jevnisek/vgg_classifier/${DATASET}
# generate configs:
python main_test_vgg_classifier.py \
  --dataset=${DATASET} --epochs=${EPOCHS} \
  --data-dir ${DATA_DIR} --batch-size=${BATCH_SIZE} \
  --target-dir results/ \
  --optimizer-type='SGD' \
  --models-root-path ${MODELS_ROOT_PATH} \
  --landone-root /storage/jevnisek/realism_datasets/human_evaluation/lalonde_and_efros_dataset/
