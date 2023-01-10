sleep 360

# create results directories:
mkdir -p /storage/jevnisek/pl_better_real_fake_classifier/${DATASET}

python lightning_trainer.py --model_name ${MODEL_NAME} \
  --dataset ${DATASET} --data-dir ${DATA_DIR} \
  --target-dir /storage/jevnisek/pl_better_real_fake_classifier/${DATASET} \
   --epochs 15 --batch_size 64
