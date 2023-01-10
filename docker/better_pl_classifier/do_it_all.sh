docker image build -t ajevnisek/pl-better-classifier:latest .
docker push  ajevnisek/pl-better-classifier:latest


runai submit -g 1 -e  DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e DATASET='HAdobe5k' --name pl-better-classifier-hadobe5k-resnet18 \
  -e MODEL_NAME=resnet18 \
  -i ajevnisek/pl-better-classifier:latest --pvc=storage:/storage \
  --large-shm

runai submit -g 1 -e  DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e DATASET='HFlickr' --name pl-better-classifier-hflickr-resnet18 \
  -e MODEL_NAME=resnet18 \
  -i ajevnisek/pl-better-classifier:latest --pvc=storage:/storage \
  --large-shm

runai submit -g 1 -e  DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e DATASET='HCOCO' --name pl-better-classifier-hcoco-resnet18 \
  -e MODEL_NAME=resnet18 \
  -i ajevnisek/pl-better-classifier:latest --pvc=storage:/storage \
  --large-shm

runai submit -g 1 -e  DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e DATASET='Hday2night' --name pl-better-classifier-hday2night-resnet18 \
  -e MODEL_NAME=resnet18 \
  -i ajevnisek/pl-better-classifier:latest --pvc=storage:/storage \
  --large-shm


                     
