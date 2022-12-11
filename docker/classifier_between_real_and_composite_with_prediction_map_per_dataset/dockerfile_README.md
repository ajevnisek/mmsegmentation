# Building docker image and running with runai

# Setup
Train a per-dataset classifier between real and composite images.



# Build docker Image
```bash
docker image build -t ajevnisek/classifier-real-and-composite-with-preds:latest .  --no-cache
docker push  ajevnisek/classifier-real-and-composite-with-preds:latest
```


# Run docker Container
```bash
docker run --name container-demo -dit ajevnisek/classifier-real-and-composite-with-preds:latest
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

HFlickr
Hday2night
HCOCO
hflickr
hday2night
hcoco

# Run this image with run:ai:
```bash
runai submit -g 1 -e MODEL='resnet' -e EPOCHS=20 -e BATCH_SIZE=128 -e DATASET='HAdobe5k' --name classifier-real-composite-hadobe5k-with-preds -i ajevnisek/classifier-real-and-composite-with-preds:latest --pvc=storage:/storage --large-shm
runai submit -g 1 -e MODEL='resnet' -e EPOCHS=20 -e BATCH_SIZE=128 -e DATASET='HFlickr' --name classifier-real-composite-hflickr-with-preds -i ajevnisek/classifier-real-and-composite-with-preds:latest --pvc=storage:/storage --large-shm
runai submit -g 1 -e MODEL='resnet' -e EPOCHS=20 -e BATCH_SIZE=128 -e DATASET='HCOCO' --name classifier-real-composite-hcoco-with-preds -i ajevnisek/classifier-real-and-composite-with-preds:latest --pvc=storage:/storage --large-shm
runai submit -g 1 -e MODEL='resnet' -e EPOCHS=20 -e BATCH_SIZE=128 -e DATASET='Hday2night' --name classifier-real-composite-hday2night-with-preds -i ajevnisek/classifier-real-and-composite-with-preds:latest --pvc=storage:/storage --large-shm



```
# Show the logs and telemetries:
```bash
runai logs mmsegmentation-classifier-real-composite-all-datasets -f

```