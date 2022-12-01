# Building docker image and running with runai

# Setup
Create a directory shared with the docker

# Build docker Image
```bash
docker image build -t ajevnisek/mmsegmentation-test-for-image-harmonization-mask-prediction:latest .  
docker push  ajevnisek/mmsegmentation-test-for-image-harmonization-mask-prediction:latest
```


# Run docker Container
```bash
docker run --name container-demo -dit ajevnisek/mmsegmentation-test-for-image-harmonization-mask-prediction:latest
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
runai submit -g 1 -e DATASET=HCOCO --name mmsegmentation-test-hcoco -i ajevnisek/mmsegmentation-test-for-image-harmonization-mask-prediction:latest --pvc=storage:/storage --large-shm
runai submit -g 1 -e DATASET=HAdobe5k --name mmsegmentation-test-hadobe5k -i ajevnisek/mmsegmentation-test-for-image-harmonization-mask-prediction:latest --pvc=storage:/storage --large-shm
runai submit -g 1 -e DATASET=HFlickr --name mmsegmentation-test-hflickr -i ajevnisek/mmsegmentation-test-for-image-harmonization-mask-prediction:latest --pvc=storage:/storage --large-shm
runai submit -g 1 -e DATASET=Hday2night --name mmsegmentation-test-hday2night -i ajevnisek/mmsegmentation-test-for-image-harmonization-mask-prediction:latest --pvc=storage:/storage --large-shm

```
# Show the logs and telemetries:
```bash
runai logs mmsegmentation-hcoco -f
runai logs mmsegmentation-hadobe5k -f
runai logs mmsegmentation-hflickr -f
runai logs mmsegmentation-hday2night -f
```