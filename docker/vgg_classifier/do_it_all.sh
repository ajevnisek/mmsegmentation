docker image build -t ajevnisek/vgg-classifier:latest . --no-cache
docker push  ajevnisek/vgg-classifier:latest


runai submit -g 1 -e DATASET='HAdobe5k' -e EPOCHS=50 \
  --name vgg-classifier-hadobe5k \
  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm


runai submit -g 1 -e DATASET='Hday2night' -e EPOCHS=50 \
  --name vgg-classifier-hday2night \
  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm


runai submit -g 1 -e DATASET='HFlickr' -e EPOCHS=50 \
  --name vgg-classifier-hflickr \
  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm



runai submit -g 1 -e DATASET='HCOCO' -e EPOCHS=50 \
  --name vgg-classifier-hcoco \
  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm

runai submit -g 1 -e DATASET='LabelMe' -e EPOCHS=50 \
  --name vgg-classifier-labelme \
  -e DATA_DIR='/storage/jevnisek/realism_datasets/' \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm

