# -*- coding: utf-8 -*-
"""
Created on ~

@author: Gerardo Cervantes
"""

import cv2
import argparse
import os


# video_path is a path to the video file
# output_image_dir is the path to the directory where it will save all the images
# num_frames_to_skip means it should skip this many frames until saving another frame, if 1 then will save every frame
def save_video_frames(video_path, output_image_dir):
    video_path = os.path.normpath(video_path)
    output_image_dir = os.path.normpath(output_image_dir)
    print(video_path)
    print(output_image_dir)
    vidcap = cv2.VideoCapture(video_path)

    if not vidcap:
        return False
    has_next_frame = True
    i = 0
    while has_next_frame:
        has_next_frame, image = vidcap.read()
        if has_next_frame:
            frame_output_path = output_image_dir + '/' + str(i) + '.png'
            cv2.imwrite(frame_output_path, image)  # Save image
            i += 1
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts the video into images')
    parser.add_argument('video_path', type=str,
                        help='path to the video file')
    parser.add_argument('image_output_dir', type=str,
                        help='a directory to put the images')

    args = parser.parse_args()
    save_video_frames(args.video_path, args.image_output_dir)
