#docker image build -t ajevnisek/vgg-classifier:latest . --no-cache
docker image build -t ajevnisek/vgg-classifier-tester:latest .
docker push  ajevnisek/vgg-classifier-tester:latest


runai submit -g 1 -e DATASET='Hday2night' -e EPOCHS=1 \
  --name vgg-classifier-tester-hday2night -e BATCH_SIZE=256 \
  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e MODELS_ROOT_PATH='/storage/jevnisek/vgg_classifier/Hday2night/2023_01_01__21_17_38/' \
  -i ajevnisek/vgg-classifier-tester:latest --pvc=storage:/storage \
  --large-shm

runai submit -g 1 -e DATASET='HCOCO' -e EPOCHS=1 \
  --name vgg-classifier-tester-hcoco -e BATCH_SIZE=256 \
  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e MODELS_ROOT_PATH='/storage/jevnisek/vgg_classifier/HCOCO/2023_01_01__21_17_25/' \
  -i ajevnisek/vgg-classifier-tester:latest --pvc=storage:/storage \
  --large-shm


runai submit -g 1 -e DATASET='HFlickr' -e EPOCHS=1 \
  --name vgg-classifier-tester-hflickr -e BATCH_SIZE=256 \
  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e MODELS_ROOT_PATH='/storage/jevnisek/vgg_classifier/HFlickr/2023_01_01__21_17_23/' \
  -i ajevnisek/vgg-classifier-tester:latest --pvc=storage:/storage \
  --large-shm

runai submit -g 1 -e DATASET='HAdobe5k' -e EPOCHS=1 \
  --name vgg-classifier-tester-hadobe5k -e BATCH_SIZE=256 \
  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e MODELS_ROOT_PATH='/storage/jevnisek/vgg_classifier/HAdobe5k/2023_01_01__21_17_40/' \
  -i ajevnisek/vgg-classifier-tester:latest --pvc=storage:/storage \
  --large-shm

runai submit -g 1 -e DATASET='HAdobe5k' -e EPOCHS=1 \
  --name vgg-classifier-tester-hadobe5k-small-bs -e BATCH_SIZE=256 \
  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e MODELS_ROOT_PATH='/storage/jevnisek/vgg_classifier/HAdobe5k/2023_01_03__13_49_48/' \
  -i ajevnisek/vgg-classifier-tester:latest --pvc=storage:/storage \
  --large-shm

runai submit -g 1 -e DATASET='LabelMe_all' -e EPOCHS=1 \
  --name vgg-classifier-tester-label-me-my-machine -e BATCH_SIZE=256 \
  -e DATA_DIR='/storage/jevnisek/realism_datasets/' \
  -e MODELS_ROOT_PATH='/storage/jevnisek/vgg_classifier/LabelMe_all/2023_01_01__19_26_17' \
  -i ajevnisek/vgg-classifier-tester:latest --pvc=storage:/storage \
  --large-shm

