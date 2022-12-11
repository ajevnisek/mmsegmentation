# docker image build -t ajevnisek/mmsegmentation-classifier-between-real-and-composite-per-dataset:latest .  --no-cache
# docker push  ajevnisek/mmsegmentation-classifier-between-real-and-composite-per-dataset:latest

runai submit -g 1 -e BASE_DIR='/storage/jevnisek/CompositeAndRealImagesClassifier'  -e DATASET='HAdobe5k' -e MODEL='resnet' -e EPOCHS=20 -e BATCH_SIZE=16 --name classifier-real-composite-hadobe5k-small-batch-size -i ajevnisek/mmsegmentation-classifier-between-real-and-composite-per-dataset:latest --pvc=storage:/storage --large-shm


runai submit -g 1 -e BASE_DIR='/storage/jevnisek/CompositeAndRealImagesClassifier'  -e DATASET='HAdobe5k' -e MODEL='resnet' -e EPOCHS=20 -e BATCH_SIZE=128 --name mmsegmentation-classifier-real-composite-hadobe5k -i ajevnisek/mmsegmentation-classifier-between-real-and-composite-per-dataset:latest --pvc=storage:/storage --large-shm

 runai submit -g 1 -e BASE_DIR='/storage/jevnisek/CompositeAndRealImagesClassifier'  -e DATASET='Hday2night' -e MODEL='resnet' -e EPOCHS=20 -e BATCH_SIZE=128 --name mmsegmentation-classifier-real-composite-hday2night -i ajevnisek/mmsegmentation-classifier-between-real-and-composite-per-dataset:latest --pvc=storage:/storage --large-shm


unai submit -g 1 -e BASE_DIR='/storage/jevnisek/CompositeAndRealImagesClassifier'  -e DATASET='HFlickr' -e MODEL='resnet' -e EPOCHS=20 -e BATCH_SIZE=128 --name mmsegmentation-classifier-real-composite-hflickr -i ajevnisek/mmsegmentation-classifier-between-real-and-composite-per-dataset:latest --pvc=storage:/storage --large-shm

runai submit -g 1 -e BASE_DIR='/storage/jevnisek/CompositeAndRealImagesClassifier'  -e DATASET='HCOCO' -e MODEL='resnet' -e EPOCHS=20 -e BATCH_SIZE=128 --name mmsegmentation-classifier-real-composite-hcoco -i ajevnisek/mmsegmentation-classifier-between-real-and-composite-per-dataset:latest --pvc=storage:/storage --large-shm



