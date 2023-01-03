docker image build -t ajevnisek/discriminator-classifier:latest .
docker push  ajevnisek/discriminator-classifier:latest

#
#runai submit -g 1 -e BASE_DIR='/storage/jevnisek/discriminator-classifier' \
#  -e  DATASET_ROOT='/storage/jevnisek/ImageHarmonizationDataset/' \
#  -e DATASET='Hday2night' -e EPOCHS=20 -e BATCH_SIZE=32 \
#  -e GAN_MODE='vanilla' -e LEARNING_RATE=1e-3 \
#  --name discriminator-classifier-hday2night-vanilla-gan \
#  -i ajevnisek/discriminator-classifier:latest --pvc=storage:/storage \
#  --large-shm


#runai submit -g 1 -e BASE_DIR='/storage/jevnisek/discriminator-classifier' \
#  -e  DATASET_ROOT='/storage/jevnisek/ImageHarmonizationDataset/' \
#  -e DATASET='HFlickr' -e EPOCHS=20 -e BATCH_SIZE=32 \
#  -e GAN_MODE='vanilla' -e LEARNING_RATE=1e-3 \
#  --name discriminator-classifier-hflickr-vanilla-gan \
#  -i ajevnisek/discriminator-classifier:latest --pvc=storage:/storage \
#  --large-shm
runai submit -g 1 -e BASE_DIR='/storage/jevnisek/discriminator-classifier' \
  -e  DATASET_ROOT='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e DATASET='HAdobe5k' -e EPOCHS=20 -e BATCH_SIZE=32 \
  -e GAN_MODE='vanilla' -e LEARNING_RATE=1e-3 \
  --name discriminator-classifier-hadobe5k-vanilla-gan \
  -i ajevnisek/discriminator-classifier:latest --pvc=storage:/storage \
  --large-shm
runai submit -g 1 -e BASE_DIR='/storage/jevnisek/discriminator-classifier' \
  -e  DATASET_ROOT='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e DATASET='HCOCO' -e EPOCHS=20 -e BATCH_SIZE=32 \
  -e GAN_MODE='vanilla' -e LEARNING_RATE=1e-3 \
  --name discriminator-classifier-hcoco-vanilla-gan \
  -i ajevnisek/discriminator-classifier:latest --pvc=storage:/storage \
  --large-shm


runai submit -g 1 -e BASE_DIR='/storage/jevnisek/discriminator-classifier' \
  -e  DATASET_ROOT='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e DATASET='IHD' -e EPOCHS=20 -e BATCH_SIZE=32 \
  -e GAN_MODE='vanilla' -e LEARNING_RATE=1e-3 \
  --name discriminator-classifier-ihd-vanilla-gan \
  -i ajevnisek/discriminator-classifier:latest --pvc=storage:/storage \
  --large-shm

runai submit -g 1 -e BASE_DIR='/storage/jevnisek/discriminator-classifier' \
  -e  DATASET_ROOT='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e DATASET='HAdobe5k' -e EPOCHS=20 -e BATCH_SIZE=1 \
  -e GAN_MODE='vanilla' -e LEARNING_RATE=1e-3 \
  --name discriminator-classifier-hadobe5k-vanilla-gan-bs-1 \
  -i ajevnisek/discriminator-classifier:latest --pvc=storage:/storage \
  --large-shm



#runai submit -g 1 -e BASE_DIR='/storage/jevnisek/discriminator-classifier' \
#  -e  DATASET_ROOT='/storage/jevnisek/ImageHarmonizationDataset/' \
#  -e DATASET='Hday2night' -e EPOCHS=20 -e BATCH_SIZE=32 \
#  -e GAN_MODE='lsgan' -e LEARNING_RATE=1e-3 \
#  --name discriminator-classifier-hday2night-lsgan-gan \
#  -i ajevnisek/discriminator-classifier:latest --pvc=storage:/storage \
#  --large-shm
#
#
#runai submit -g 1 -e BASE_DIR='/storage/jevnisek/discriminator-classifier' \
#  -e  DATASET_ROOT='/storage/jevnisek/ImageHarmonizationDataset/' \
#  -e DATASET='HFlickr' -e EPOCHS=20 -e BATCH_SIZE=32 \
#  -e GAN_MODE='lsgan' -e LEARNING_RATE=1e-3 \
#  --name discriminator-classifier-hflickr-lsgan-gan \
#  -i ajevnisek/discriminator-classifier:latest --pvc=storage:/storage \
#  --large-shm
runai submit -g 1 -e BASE_DIR='/storage/jevnisek/discriminator-classifier' \
  -e  DATASET_ROOT='/storage/jevnisek/ImageHarmonizationDataset/' \
  -e DATASET='HAdobe5k' -e EPOCHS=20 -e BATCH_SIZE=32 \
  -e GAN_MODE='lsgan' -e LEARNING_RATE=1e-3 \
  --name discriminator-classifier-hadobe5k-lsgan-gan \
  -i ajevnisek/discriminator-classifier:latest --pvc=storage:/storage \
  --large-shm
#runai submit -g 1 -e BASE_DIR='/storage/jevnisek/discriminator-classifier' \
#  -e  DATASET_ROOT='/storage/jevnisek/ImageHarmonizationDataset/' \
#  -e DATASET='HCOCO' -e EPOCHS=20 -e BATCH_SIZE=32 \
#  -e GAN_MODE='lsgan' -e LEARNING_RATE=1e-3 \
#  --name discriminator-classifier-hcoco-lsgan-gan \
#  -i ajevnisek/discriminator-classifier:latest --pvc=storage:/storage \
#  --large-shm
