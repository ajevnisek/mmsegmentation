# create results directories:
mkdir -p ${BASE_DIR}/results/${DATASET}

# generate configs:
python global_and_local_features_classifier.py --dataset-root ${DATASET_ROOT} --dataset ${DATASET} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --results-dir ${BASE_DIR}/results/${DATASET}
