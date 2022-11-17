sleep 360

cd /mmsegmentation/

python image_harmonization_mask_predictor.py --dataset=$DATASET --is-cluster

