# create results directories:
mkdir -p /storage/jevnisek/vgg_classifier/${DATASET}
# generate configs:
python main_test_vgg_classifier.py \
  --dataset=${DATASET} --epochs=${EPOCHS} \
  --data-dir ${DATA_DIR} --batch-size=${BATCH_SIZE} \
  --target-dir /storage/jevnisek/vgg_classifier/${DATASET} \
  --landone-root /storage/jevnisek/realism_datasets/human_evaluation/lalonde_and_efros_dataset/
python main_test_vgg_classifier.py --dataset HAdobe5k --epochs 1
--batch-size 100 --data-dir ../data/Image_Harmonization_Dataset/ --target-dir results/ --model-path results/vgg_classifier/HAdobe5k/2023_01_01__21_17_40/