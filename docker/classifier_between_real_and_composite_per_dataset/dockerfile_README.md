# Building docker image and running with runai

# Setup
Train a per-dataset classifier between real and composite images.



# Build docker Image
```bash
docker image build -t ajevnisek/mmsegmentation-classifier-between-real-and-composite-all-datasets:latest .  --no-cache
docker push  ajevnisek/mmsegmentation-classifier-between-real-and-composite-all-datasets:latest
```


# Run docker Container
```bash
docker run --name container-demo -dit ajevnisek/mmsegmentation-classifier-between-real-and-composite-all-datasets:latest
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
runai submit -g 1 -e BASE_DIR='/storage/jevnisek/CompositeAndRealImagesClassifier' -e MODEL='resnet' -e EPOCHS=20 -e BATCH_SIZE=128 --name mmsegmentation-classifier-real-composite-all-datasets -i ajevnisek/mmsegmentation-classifier-between-real-and-composite-all-datasets:latest --pvc=storage:/storage --large-shm

```
# Show the logs and telemetries:
```bash
runai logs mmsegmentation-classifier-real-composite-all-datasets -f

```