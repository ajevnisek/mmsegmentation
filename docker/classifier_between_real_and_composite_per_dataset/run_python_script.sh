# create results directories:
mkdir -p ${BASE_DIR}/CompositeAndRealImagesClassifier/AllDatasets
python classifier_between_real_and_composite_all_datasets.py --is-cluster \
  --optimizer Adam --model ${MODEL} \
  --epochs ${EPOCHS} --batch-size ${BATCH_SIZE}

