git pull
python cache_predictions.py --dataset ${DATASET} --is-cluster \
  --process-all-images --longer-mask-prediction-training
python classifier_between_real_and_composite_with_prediction_map_per_dataset.py --is-cluster \
  --optimizer Adam --model ${MODEL} --longer-mask-prediction-training\
  --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --dataset ${DATASET}

