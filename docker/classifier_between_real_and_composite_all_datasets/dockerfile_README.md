# Building docker image and running with runai

# Setup
Train a per-dataset classifier between real and composite images.



# Build docker Image
```bash
docker image build -t ajevnisek/mmsegmentation-classifier-between-real-and-composite-per-dataset:latest .  
docker push  ajevnisek/mmsegmentation-classifier-between-real-and-composite-per-dataset:latest
```


# Run docker Container
```bash
docker run --name container-demo -dit ajevnisek/mmsegmentation-classifier-between-real-and-composite-per-dataset:latest
```
## Debug
### watch logs
```bash
docker logs container-demo -f
```

### bash to the container
```bash
docker exec -it container-demo bash
```

# Misc
Clear all docker cache:
```bash
docker system prune -a
```

# Run this image with run:ai:
```bash
runai submit -g 1 -e BASE_DIR='/storage/jevnisek/CompositeAndRealImagesClassifier'  -e DATASET='HAdobe5k' -e MODEL='resnet' --name mmsegmentation-classifier-real-composite-hadobe5k -i ajevnisek/mmsegmentation-classifier-between-real-and-composite-per-dataset:latest --pvc=storage:/storage --large-shm
runai submit -g 1 -e BASE_DIR='/storage/jevnisek/CompositeAndRealImagesClassifier'  -e DATASET='HCOCO' -e MODEL='resnet' --name mmsegmentation-classifier-real-composite-hcoco -i ajevnisek/mmsegmentation-classifier-between-real-and-composite-per-dataset:latest --pvc=storage:/storage --large-shm
runai submit -g 1 -e BASE_DIR='/storage/jevnisek/CompositeAndRealImagesClassifier'  -e DATASET='HFlickr' -e MODEL='resnet' --name mmsegmentation-classifier-real-composite-hflickr -i ajevnisek/mmsegmentation-classifier-between-real-and-composite-per-dataset:latest --pvc=storage:/storage --large-shm
runai submit -g 1 -e BASE_DIR='/storage/jevnisek/CompositeAndRealImagesClassifier'  -e DATASET='Hday2night' -e MODEL='resnet' --name mmsegmentation-classifier-real-composite-hday2night -i ajevnisek/mmsegmentation-classifier-between-real-and-composite-per-dataset:latest --pvc=storage:/storage --large-shm


```
# Show the logs and telemetries:
```bash
runai logs mmsegmentation-classifier-between-real-and-composite-per-dataset -f

```