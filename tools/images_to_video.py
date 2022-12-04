import cv2
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--images", type=str, help='path to images folder')
parser.add_argument("--video", type=str, default='viedo.avi',
                    help='path to target video')
args = parser.parse_args()


image_folder = args.images
video_name = args.video

shape = 960, 720
fps = 1

images = [f for f in os.listdir(image_folder)]
images.sort()
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video = cv2.VideoWriter(video_name, fourcc, fps, shape)

for image in images:
    image_path = os.path.join(image_folder, image)
    image = cv2.imread(image_path)
    resized=cv2.resize(image,shape)
    video.write(resized)

video.release()
