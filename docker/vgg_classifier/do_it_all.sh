docker image build -t ajevnisek/vgg-classifier:latest . --no-cache
docker push  ajevnisek/vgg-classifier:latest


runai submit -g 1 -e DATASET='HAdobe5k' -e EPOCHS=50 \
  --name vgg-classifier-hadobe5k \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm


runai submit -g 1 -e DATASET='Hday2night' -e EPOCHS=50 \
  --name vgg-classifier-hday2night \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm


runai submit -g 1 -e DATASET='HFlickr' -e EPOCHS=50 \
  --name vgg-classifier-hflickr \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm



runai submit -g 1 -e DATASET='HCOCO' -e EPOCHS=50\
  --name vgg-classifier-hcoco \
  -i ajevnisek/vgg-classifier:latest --pvc=storage:/storage \
  --large-shm

