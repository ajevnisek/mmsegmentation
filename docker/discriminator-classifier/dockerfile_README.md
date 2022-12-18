# Building docker image and running with runai

# Setup
Generate mask predictions on:
- real images
- composite images
- Harmonizer outputs 


# Build docker Image
```bash
docker image build -t ajevnisek/discriminator-classifier:latest .  
docker push  ajevnisek/discriminator-classifier:latest
```


# Run docker Container
```bash
docker run --name container-demo -dit ajevnisek/discriminator-classifier:latest
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
runai submit -g 1 -e BASE_DIR='/storage/jevnisek/discriminator-classifier' -e  DATASET_ROOT='/storage/jevnisek/ImageHarmonizationDataset/' -e DATASET='Hday2night' -e EPOCHS=20 -e BATCH_SIZE=1 --name discriminator-classifier-hday2night -i ajevnisek/discriminator-classifier:latest --pvc=storage:/storage --large-shm
runai submit -g 1 -e BASE_DIR='/storage/jevnisek/discriminator-classifier' -e  DATASET_ROOT='/storage/jevnisek/ImageHarmonizationDataset/' -e DATASET='HFlickr' -e EPOCHS=20 -e BATCH_SIZE=1 --name discriminator-classifier-hflickr -i ajevnisek/discriminator-classifier:latest --pvc=storage:/storage --large-shm
runai submit -g 1 -e BASE_DIR='/storage/jevnisek/discriminator-classifier' -e  DATASET_ROOT='/storage/jevnisek/ImageHarmonizationDataset/' -e DATASET='HAdobe5k' -e EPOCHS=20 -e BATCH_SIZE=1 --name discriminator-classifier-hadobe5k -i ajevnisek/discriminator-classifier:latest --pvc=storage:/storage --large-shm
runai submit -g 1 -e BASE_DIR='/storage/jevnisek/discriminator-classifier' -e  DATASET_ROOT='/storage/jevnisek/ImageHarmonizationDataset/' -e DATASET='HCOCO' -e EPOCHS=20 -e BATCH_SIZE=1 --name discriminator-classifier-hcoco -i ajevnisek/discriminator-classifier:latest --pvc=storage:/storage --large-shm

```
# Show the logs and telemetries:
```bash
runai logs discriminator-classifier -f

```