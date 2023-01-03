#docker image build -t ajevnisek/vgg-classifier:latest . --no-cache
docker image build -t ajevnisek/vgg-classifier-tester:latest .
docker push  ajevnisek/vgg-classifier-tester:latest


runai submit -g 1 -e DATASET='Hday2night' -e EPOCHS=1 \
  --name vgg-classifier-tester-hday2night -e BATCH_SIZE=256 \
  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e MODELS_ROOT_PATH='/storage/jevnisek/vgg_classifier/Hday2night/2023_01_01__21_17_38/' \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm

runai submit -g 1 -e DATASET='HCOCO' -e EPOCHS=1 \
  --name vgg-classifier-tester-hcoco -e BATCH_SIZE=256 \
  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e MODELS_ROOT_PATH='/storage/jevnisek/vgg_classifier/HCOCO/2023_01_01__21_17_25/' \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm


runai submit -g 1 -e DATASET='HFlickr' -e EPOCHS=1 \
  --name vgg-classifier-tester-hflickr -e BATCH_SIZE=256 \
  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e MODELS_ROOT_PATH='/storage/jevnisek/vgg_classifier/HFlickr/2023_01_01__21_17_23/' \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm

runai submit -g 1 -e DATASET='HFlickr' -e EPOCHS=1 \
  --name vgg-classifier-tester-hflickr -e BATCH_SIZE=256 \
  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e MODELS_ROOT_PATH='/storage/jevnisek/vgg_classifier/HAdobe5k/2023_01_01__21_17_40/' \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm

