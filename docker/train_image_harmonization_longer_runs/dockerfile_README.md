# Building docker image and running with runai

# Setup
Create a directory shared with the docker

# Build docker Image
```bash
docker image build -t ajevnisek/mmsegmentation-mask-prediction-longer-runs .
docker push  ajevnisek/mmsegmentation-mask-prediction-longer-runs
```


# Run docker Container
```bash
docker run --rm --name container-demo -dit ajevnisek/mmsegmentation-mask-prediction-longer-runs
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
runai submit -g 1 -e DATASET=HCOCO --name mmsegmentation-hcoco -i ajevnisek/mmsegmentation-mask-prediction-longer-runs:latest --pvc=storage:/storage --large-shm
runai submit -g 1 -e DATASET=HAdobe5k --name mmsegmentation-hadobe5k -i ajevnisek/mmsegmentation-mask-prediction-longer-runs:latest --pvc=storage:/storage --large-shm
runai submit -g 1 -e DATASET=HFlickr --name mmsegmentation-hflickr -i ajevnisek/mmsegmentation-mask-prediction-longer-runs:latest --pvc=storage:/storage --large-shm
runai submit -g 1 -e DATASET=Hday2night --name mmsegmentation-hday2night -i ajevnisek/mmsegmentation-mask-prediction-longer-runs:latest --pvc=storage:/storage --large-shm

```
# Show the logs and telemetries:
```bash
runai logs mmsegmentation-hcoco -f
runai logs mmsegmentation-hadobe5k -f
runai logs mmsegmentation-hflickr -f
runai logs mmsegmentation-hday2night -f
```