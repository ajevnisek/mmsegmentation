# create results directories:
mkdir -p ${BASE_DIR}/results/${DATASET}

# generate configs:
python global_and_local_features_classifier.py \
  --dataset-root ${DATASET_ROOT} --dataset ${DATASET} --epochs ${EPOCHS}  \
  --batch-size ${BATCH_SIZE} --gan-model=${GAN_MODE} \
  --learning-rate ${LEARNING_RATE} \
  --results-dir ${BASE_DIR}/results/${DATASET}
