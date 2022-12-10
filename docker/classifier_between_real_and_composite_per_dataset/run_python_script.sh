
# create results directories:

python classifier_between_real_and_composite_per_dataset.py --is-cluster \
  --optimizer Adam --model ${MODEL} \
  --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --dataset ${DATASET}

