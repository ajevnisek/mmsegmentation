# create results directories:
mkdir -p ${BASE_DIR}/CompositeAndRealImagesClassifier/${DATASET}
python classifier_between_real_and_composite_per_dataset.py --is-cluster --optimizer Adam --model ${MODEL}  --epochs 20 --batch-size 128
