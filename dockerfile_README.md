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
runai submit --rm -g 1 -e DATASET=HCOCO --name mmsegmentation-HCOCO -i ajevnisek/mmsegmentation-for-image-harmonization-mask-prediction --pvc=storage:/storage
runai submit --rm -g 1 -e DATASET=HAdobe5k --name mmsegmentation-HAdobe5k -i ajevnisek/mmsegmentation-for-image-harmonization-mask-prediction --pvc=storage:/storage
runai submit --rm -g 1 -e DATASET=HFlickr --name mmsegmentation-HFlickr -i ajevnisek/mmsegmentation-for-image-harmonization-mask-prediction --pvc=storage:/storage
runai submit --rm -g 1 -e DATASET=Hday2night --name mmsegmentation-Hday2night -i ajevnisek/mmsegmentation-for-image-harmonization-mask-prediction --pvc=storage:/storage
```
