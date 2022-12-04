cd /mmsegmentation/

for DATASET in HAdobe5k HCOCO HFlickr Hday2night
do
   # create results directories:
    mkdir ${BASE_DIR}/mask_prediction_on_harmonizer_output_results/${DATASET}
    mkdir ${BASE_DIR}/mask_prediction_on_real_images_results/${DATASET}
    mkdir ${BASE_DIR}/mask_prediction_on_composite_images_results/${DATASET}

    # create images:
    python test_image_harmonization_mask_predictor_on_harmonizer_images.py --dataset ${DATASET} --all-test-images
    python test_image_harmonization_mask_predictor_on_real_images.py --dataset ${DATASET} --all-test-images
    python test_image_harmonization_mask_predictor_on_composite_image.py --dataset ${DATASET} --all-test-images

    # create videos:
    python tools/images_to_video.py --images ${BASE_DIR}/mask_prediction_on_harmonizer_output_results/${DATASET} --video ${BASE_DIR}/mask_prediction_on_harmonizer_output_results/${DATASET}_on_harmonizer_output.avi
    python tools/images_to_video.py --images ${BASE_DIR}/mask_prediction_on_real_images_results/${DATASET} --video ${BASE_DIR}/mask_prediction_on_real_images_results/${DATASET}_on_real_images.avi
    python tools/images_to_video.py --images ${BASE_DIR}/mask_prediction_on_composite_images_results/${DATASET} --video ${BASE_DIR}/mask_prediction_on_composite_images_results/${DATASET}_on_composite_images.avi
    
done

