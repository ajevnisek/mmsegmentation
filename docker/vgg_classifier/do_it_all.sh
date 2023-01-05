docker image build -t ajevnisek/vgg-classifier:latest . --no-cache
#docker image build -t ajevnisek/vgg-classifier:latest .
docker push  ajevnisek/vgg-classifier:latest


runai submit -g 1 -e DATASET='HAdobe5k' -e EPOCHS=50 \
  --name vgg-classifier-hadobe5k-smallest-bs -e BATCH_SIZE=16 \
  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e OPTIMIZER_TYPE='SGD' \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm


runai submit -g 1 -e DATASET='HAdobe5k' -e EPOCHS=50 \
  --name vgg-classifier-hadobe5k -e BATCH_SIZE=50 \
  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e OPTIMIZER_TYPE='SGD' \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm

runai submit -g 1 -e DATASET='HAdobe5k' -e EPOCHS=50 \
  --name vgg-classifier-hadobe5k-adam -e BATCH_SIZE=50 \
  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e OPTIMIZER_TYPE='Adam' \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm


runai submit -g 1 -e DATASET='HCOCO' -e EPOCHS=50 \
  --name vgg-classifier-hcoco -e BATCH_SIZE=50 \
  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e OPTIMIZER_TYPE='SGD' \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm


runai submit -g 1 -e DATASET='HCOCO' -e EPOCHS=50 \
  --name vgg-classifier-hcoco-adam -e BATCH_SIZE=50 \
  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e OPTIMIZER_TYPE='Adam' \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm


runai submit -g 1 -e DATASET='LabelMe_all' -e EPOCHS=50 \
  --name vgg-classifier-labelme-adam -e BATCH_SIZE=50 \
  -e DATA_DIR='/storage/jevnisek/realism_datasets/' \
  -e OPTIMIZER_TYPE='Adam' \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm


runai submit -g 1 -e DATASET='LabelMe_all' -e EPOCHS=50 \
  --name vgg-classifier-labelme-small-bs -e BATCH_SIZE=16 \
  -e DATA_DIR='/storage/jevnisek/realism_datasets/' \
  -e OPTIMIZER_TYPE='SGD' \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm

#
#runai submit -g 1 -e DATASET='Hday2night' -e EPOCHS=50 \
#  --name vgg-classifier-hday2night -e BATCH_SIZE=50 \
#  -e OPTIMIZER_TYPE='SGD' \
#  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
#  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
#  --large-shm
#
#
#runai submit -g 1 -e DATASET='HFlickr' -e EPOCHS=50 \
#  --name vgg-classifier-hflickr -e BATCH_SIZE=50 \
#  -e OPTIMIZER_TYPE='SGD' \
#  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
#  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
#  --large-shm
#
#
#
#runai submit -g 1 -e DATASET='HCOCO' -e EPOCHS=50 \
#  --name vgg-classifier-hcoco -e BATCH_SIZE=50 \
#  -e OPTIMIZER_TYPE='SGD' \
#  -e DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
#  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
#  --large-shm
#
#runai submit -g 1 -e DATASET='LabelMe_all' -e EPOCHS=50 \
#  --name vgg-classifier-labelme-small-train -e BATCH_SIZE=50 \
#  -e OPTIMIZER_TYPE='Adam' \
#  -e DATA_DIR='/storage/jevnisek/realism_datasets/' \
#  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
#  --large-shm
#
#sleep 10
#
#runai submit -g 1 -e DATASET='LabelMe_all' -e EPOCHS=200 \
#  --name vgg-classifier-labelme-longer-train -e BATCH_SIZE=50 \
#  -e OPTIMIZER_TYPE='Adam' \
#  -e DATA_DIR='/storage/jevnisek/realism_datasets/' \
#  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
#  --large-shm
#
