# Building docker image and running with runai

# Setup
Generate mask predictions on:
- real images
- composite images
- Harmonizer outputs 


# Build docker Image
```bash
docker image build -t ajevnisek/mmsegmentation-generate-predictions:latest .  
docker push  ajevnisek/mmsegmentation-generate-predictions:latest
```


# Run docker Container
```bash
docker run --name container-demo -dit ajevnisek/mmsegmentation-generate-predictions:latest
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
runai submit -g 1 -e BASE_DIR='/storage/jevnisek/MaskPredictionResults' --name mmsegmentation-generate-predictions -i ajevnisek/mmsegmentation-generate-predictions:latest --pvc=storage:/storage --large-shm

```
# Show the logs and telemetries:
```bash
runai logs mmsegmentation-generate-predictions -f

```