cd /mmsegmentation/

python generate_configs.py --dataset=$DATASET --is-cluster
python test_image_harmonization_mask_predictor.py --dataset=$DATASET --is-cluster --all-test-images

