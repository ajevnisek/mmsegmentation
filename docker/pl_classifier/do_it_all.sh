docker image build -t ajevnisek/pl-classifier:latest .
docker push  ajevnisek/pl-classifier:latest


runai submit -g 1 -e  DATA_DIR='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e DATASET='HAdobe5k' --name pl-classifier-hadobe5k \
  -i ajevnisek/pl-classifier:latest --pvc=storage:/storage \
  --large-shm
