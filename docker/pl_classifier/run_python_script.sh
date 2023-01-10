sleep 360

# create results directories:
mkdir -p /storage/jevnisek/pl_real_fake_classifier/${DATASET}

python main_real_and_fake_classifier_pl.py --dataset ${DATASET} \
 --epochs 15 --batch-size 50 \
 --target-dir /storage/jevnisek/pl_real_fake_classifier/${DATASET} \
 --data-dir ${DATA_DIR}
