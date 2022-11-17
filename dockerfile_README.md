# Building docker image and running with runai

# Setup
Create a directory shared with the docker

# Build docker Image
```bash
docker image build -t ajevnisek/mmsegmentation-for-image-harmonization-mask-prediction .
```


# Run docker Container
```bash
docker run --rm --name container-demo -dit ajevnisek/mmsegmentation-for-image-harmonization-mask-prediction
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
runai submit -g 1 -e DATASET=HCOCO --name mmsegmentation-hcoco -i ajevnisek/mmsegmentation-for-image-harmonization-mask-prediction:v3 --pvc=storage:/storage --large-shm
runai submit -g 1 -e DATASET=HAdobe5k --name mmsegmentation-hadobe5k -i ajevnisek/mmsegmentation-for-image-harmonization-mask-prediction:v3 --pvc=storage:/storage --large-shm
runai submit -g 1 -e DATASET=HFlickr --name mmsegmentation-hflickr -i ajevnisek/mmsegmentation-for-image-harmonization-mask-prediction:v3 --pvc=storage:/storage --large-shm
runai submit -g 1 -e DATASET=Hday2night --name mmsegmentation-hday2night -i ajevnisek/mmsegmentation-for-image-harmonization-mask-prediction:v3 --pvc=storage:/storage --large-shm

```
